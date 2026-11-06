"""Batch driver for source-attributed HF Mellum reference. No package upgrades."""
import torch
from transformers.cache_utils import DynamicCache


@torch.inference_mode()
def generate(model, sequences, max_tokens, eos_id=None):
    device = next(model.parameters()).device
    maximum = max(map(len, sequences))
    ids = torch.zeros((len(sequences), maximum), dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        ids[i, -len(seq):] = torch.tensor(seq, device=device)
        valid[i, -len(seq):] = True
    lengths = torch.tensor(list(map(len, sequences)), device=device)
    active = list(range(len(sequences)))
    result = [[] for _ in sequences]
    cache = DynamicCache()
    for step in range(max_tokens):
        start, qlen = cache.get_seq_length(), ids.shape[1]
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0) if step == 0 else (lengths + step - 1)[:, None]
        hidden = model.embed_tokens(ids)
        rotary = {typ: model.rotary_emb(hidden, positions, typ) for typ in set(model.config.layer_types)}
        keypos = torch.arange(valid.shape[1], device=device)
        querypos = torch.arange(start, start + qlen, device=device)
        causal = keypos[None, :] <= querypos[:, None]
        masks = {}
        for typ in rotary:
            allowed = causal[None] & valid[:, None, :]
            if typ == 'sliding_attention':
                allowed = allowed & (keypos[None, None, :] > querypos[None, :, None] - model.config.sliding_window)
            masks[typ] = torch.where(allowed, 0.0, float('-inf')).to(hidden.dtype)[:, None]
        for i, layer in enumerate(model.layers):
            typ = model.config.layer_types[i]
            hidden = layer(hidden, attention_mask=masks[typ], position_embeddings=rotary[typ], past_key_values=cache)
        ids = model.lm_head(model.norm(hidden[:, -1:])).argmax(-1)
        tokens = ids[:, 0].tolist()
        keep = []
        for i, (original, token) in enumerate(zip(active, tokens)):
            result[original].append(token)
            if eos_id is None or token != eos_id:
                keep.append(i)
        if not keep:
            break
        valid = torch.cat((valid, torch.ones((len(active), 1), dtype=torch.bool, device=device)), dim=1)
        if len(keep) != len(active):
            indices = torch.tensor(keep, device=device)
            cache.batch_select_indices(indices)
            ids, valid, lengths = ids[indices], valid[indices], lengths[indices]
            active = [active[i] for i in keep]
    return result
