"""Keep per-expert GEMM shapes and BF16 boundaries, batch activation and reduction."""
import torch
import triton
import triton.language as tl
from torch.nn import functional as F

def packing(ids, experts):
    n = ids.shape[0]
    flat = ids.T.reshape(-1)
    order = flat.argsort(stable=True)
    tokens = order.remainder(n)
    counts = torch.bincount(flat, minlength=experts).tolist()
    segments = []
    start = 0
    for expert, count in enumerate(counts):
        if count:
            segments.append((expert, start, start + count))
            start += count
    return order, tokens, segments

def projected(x, weights, segments):
    return torch.cat([F.linear(x[start:stop], weights[expert])
                      for expert, start, stop in segments])

@triton.jit
def _reduce(states, weights, rows, slots, out, N:tl.constexpr, K:tl.constexpr,
            D:tl.constexpr, WS0, WS1, BLOCK:tl.constexpr):
    tok = tl.program_id(0)
    col = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for i in tl.static_range(K):
        slot = tl.load(slots + tok*K + i)
        row = tl.load(rows + slot*N + tok)
        weight = tl.load(weights + tok*WS0 + slot*WS1).to(tl.float32)
        state = tl.load(states + row*D + col, col<D, other=0).to(tl.float32)
        weighted = (state * weight).to(tl.bfloat16).to(tl.float32)
        acc = (acc + weighted).to(tl.bfloat16).to(tl.float32)
    tl.store(out + tok*D + col, acc, col<D)

def experts(x, w13, w2, ids, weights):
    n, top = ids.shape
    order, tokens, segments = packing(ids, w13.shape[0])
    gate, up = projected(x[tokens], w13, segments).chunk(2, dim=-1)
    down = projected(F.silu(gate)*up, w2, segments)
    inverse = order.argsort()
    slots = ids.argsort(dim=-1).contiguous()
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    width = x.shape[-1]
    _reduce[(n, triton.cdiv(width, 256))](down, weights, inverse, slots, out,
        n, top, width, *weights.stride(), 256, enable_fp_fusion=False)
    return out
