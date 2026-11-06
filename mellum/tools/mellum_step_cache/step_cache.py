"""Per-forward BF16 Mellum decode reuse. All arithmetic stays in the original RoPE."""
import aiter
from atom.models.mellum_naive_ops import TorchPagedKV
from atom.utils.forward_context import get_forward_context

# The address-only and native variants replace the same cache-write operation.
MODES = {
    'control': ('mixed_packed', False, 'original'),
    'rope': ('mixed_packed', True, 'original'),
    'address': ('mixed_packed', False, 'address'),
    'native': ('mixed_packed', False, 'native'),
    'combined': ('mixed_packed', True, 'native'),
    'fast_control': ('paged_grouped', False, 'original'),
    'fast_combined': ('paged_grouped', True, 'native'),
}

def slot_plan(slots, block_size):
    rows = (slots >= 0).nonzero(as_tuple=True)[0]
    valid = slots.index_select(0, rows).long()
    return rows, valid // block_size, valid % block_size

def write_planned(cache, plan, key, value):
    rows, blocks, offsets = plan
    keys = key.index_select(0, rows).reshape(len(rows), cache.heads,
                                              cache.width // cache.pack, cache.pack)
    cache.key[blocks, :, :, offsets, :] = keys
    cache.value[blocks, :, :, offsets] = value.index_select(0, rows)

def write_native(cache, slots, key, value):
    # The installed HIP kernel accepts a per-token stride but assumes dense head axes.
    def head_dense(x):
        return x if x.stride(2) == 1 and x.stride(1) == x.size(2) else x.contiguous()
    aiter.reshape_and_cache(head_dense(key), head_dense(value), cache.key, cache.value,
                            slots, 'auto', None, None, False)

def install(model, select):
    import adapter
    layer_attentions = [layer.self_attn for layer in model.model.layers]
    rope = layer_attentions[0].rope
    assert all(layer.rope is rope for layer in layer_attentions)
    original_rope = rope.forward
    original_write = TorchPagedKV.write
    adapter.install(lambda: MODES[select()][0])

    def begin(_model, _args):
        ctx = get_forward_context()
        _, share_rope, writer = MODES[select()]
        enabled = (not ctx.context.is_prefill and not ctx.context.is_dummy_run
                   and ctx.attn_metadata.max_seqlen_q == 1)
        ctx.step_rope_enabled = enabled and share_rope
        ctx.step_rope = None
        ctx.step_writer = writer if enabled else 'original'
        ctx.step_slots = None

    def shared_rope(x, positions):
        ctx = get_forward_context()
        if not ctx.step_rope_enabled:
            return original_rope(x, positions)
        if ctx.step_rope is None:
            ctx.step_rope = original_rope(x, positions)
        return ctx.step_rope

    def write(cache, slots, key, value):
        ctx = get_forward_context()
        if ctx.step_writer == 'original' or not cache.key.is_contiguous() or not cache.value.is_contiguous():
            return original_write(cache, slots, key, value)
        if ctx.step_slots is None:
            if ctx.step_writer == 'native':
                ctx.step_slots = slots.long().contiguous()
            else:
                ctx.step_slots = slot_plan(slots, cache.block_size)
        if ctx.step_writer == 'native':
            return write_native(cache, ctx.step_slots, key, value)
        return write_planned(cache, ctx.step_slots, key, value)

    model.register_forward_pre_hook(begin)
    rope.forward = shared_rope
    TorchPagedKV.write = write
