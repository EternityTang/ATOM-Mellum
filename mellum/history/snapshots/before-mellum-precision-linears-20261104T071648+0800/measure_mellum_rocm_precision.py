"""Frozen ten-prompt ablations of the Mellum-local ROCm precision adapter."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

W = Path('/workspace')
OUT = Path(os.environ['MELLUM_ROCM_OUT'])
CONTROL = OUT / 'profile.json'


def write(name, value):
    p = OUT / name
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)


def install():
    import atom.utils.debug_helper as helper

    def audit(model):
        import torch
        from transformers.models.mellum.modeling_mellum import MellumRMSNorm, MellumRotaryEmbedding
        from atom.models.mellum_rocm_precision import (
            MellumRocmPrecisionForCausalLM, MellumPrecisionNorm, MellumPrecisionExpertMethod)
        assert isinstance(model, MellumRocmPrecisionForCausalLM)
        assert model.precision_parameter_identity_preserved and model.precision_norm_count == 113
        assert all(type(layer.mlp.experts.quant_method) is MellumPrecisionExpertMethod
                   for layer in model.model.layers)
        # Independent upstream HF modules, real checkpoint norm weights.
        norm_checks = []
        model.configure_precision_profile('correctness')
        with torch.inference_mode():
            torch.manual_seed(0)
            for name, norm in model.named_modules():
                if not isinstance(norm, MellumPrecisionNorm):
                    continue
                width = norm.weight.numel()
                x = torch.randn((7, width), device='cuda', dtype=torch.bfloat16)
                residual = torch.randn_like(x)
                hf = MellumRMSNorm(width, eps=norm.eps).cuda().to(torch.bfloat16)
                hf.weight = norm.weight
                expected = hf(x)
                value, saved = norm(x, residual)
                row = dict(name=name, standalone_exact=torch.equal(norm(x), expected),
                    residual_exact=torch.equal(saved, x + residual),
                    fused_exact=torch.equal(value, hf(x + residual)))
                model.configure_precision_profile('ck_native')
                ck = norm(x)
                row.update(ck_finite=bool(torch.isfinite(ck).all()),
                    ck_relative_l2=float((ck.float() - expected.float()).norm()
                        / expected.float().norm().clamp_min(1e-12)))
                model.configure_precision_profile('correctness')
                norm_checks.append(row)
            rope_checks = []
            positions = torch.tensor([0, 7, 8191, 8192, 16383, 32767], device='cuda')
            x = torch.zeros((len(positions), model.config.hidden_size), device='cuda', dtype=torch.bfloat16)
            reference = MellumRotaryEmbedding(model.config).cuda()
            local = model.model.layers[0].self_attn.rope(x, positions)
            for kind in local:
                cos, sin = reference(x[None], positions[None], layer_type=kind)
                rope_checks.append(dict(kind=kind,
                    cos_exact=torch.equal(local[kind][0], cos[0, :, None, :]),
                    sin_exact=torch.equal(local[kind][1], sin[0, :, None, :])))
        checks = dict(norm_count=len(norm_checks),norms=norm_checks,rope=rope_checks)
        write('component_checks.json', checks)
        assert all(r['standalone_exact'] and r['residual_exact'] and r['fused_exact']
                   and r['ck_finite'] and r['ck_relative_l2'] < .02 for r in norm_checks)
        assert all(r['cos_exact'] and r['sin_exact'] for r in rope_checks)
        spec = importlib.util.spec_from_file_location('weight_audit', W / 'tools/compare_mellum_variants.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.audit_weights(model, 'native_torch')
        write('model_audit.json', dict(model_class=type(model).__name__,norms=113,
            parameter_identity_preserved=True,expert_method='MellumPrecisionExpertMethod',
            level=0,eager=True,mtp=False,prefix_cache=False,parameter_count=len(dict(model.named_parameters()))))

        def select_profile(mod, args):
            wanted = json.loads(CONTROL.read_text())['profile']
            if mod.precision_profile != wanted:
                mod.configure_precision_profile(wanted)
        model.register_forward_pre_hook(select_profile)
        return 1
    helper.install_block_forward_hooks = audit


def main():
    import torch
    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs
    torch.set_num_threads(8)
    torch.manual_seed(0)
    selection_path = W / 'results/mellum_naive_baseline/quick10_selection.json'
    selection = json.loads(selection_path.read_text())
    reference_path = Path(selection['source'])
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == selection['source_sha256']
    reference = json.loads(reference_path.read_text())
    assert reference['metadata']['hf']['attention'] == 'sdpa'
    assert reference['metadata']['hf']['experts'] == 'eager'
    rows = {r['id']: r for r in reference['results']}
    chosen = [r['id'] for r in selection['samples']]
    assert len(chosen) == len(set(chosen)) == 10
    write('selection.json', selection)
    state = dict(status='loading',profiles={},selected_ids=chosen,
        reference_sha256=selection['source_sha256'],reference_metadata=reference['metadata']['hf'])
    write('validation.json', state)
    write('profile.json', dict(profile='correctness'))
    engine = EngineArgs(model='/data/models/Mellum2.1-12B-A2.5B-Thinking',port=8073,
        level=0,enforce_eager=True,enable_prefix_caching=False,kv_cache_dtype='bf16',
        max_model_len=32768,max_num_seqs=512,gpu_memory_utilization=.65).create_engine()
    params = SamplingParams(temperature=0,max_tokens=64,ignore_eos=True)
    try:
        for profile in ['ck_native','torch_native','torch_sdpa_packed','correctness']:
            write('profile.json', dict(profile=profile))
            entry = dict(samples=[],repeats=[])
            state['profiles'][profile] = entry
            state.update(status='running',current_profile=profile)
            write('validation.json', state)
            for repeat in range(2 if profile == 'correctness' else 1):
                for identity in chosen + ['HumanEval/61']:
                    row = rows[identity]
                    started = time.monotonic()
                    output = engine.generate([row['input_ids']],params)[0]['token_ids']
                    assert len(output) == 64
                    sample = dict(id=identity,output_ids=output,reference_ids=row['reference_ids'],
                        input_sha256=row['input_sha256'],input_ids=row['input_ids'],
                        matching_tokens=sum(a == b for a,b in zip(output,row['reference_ids'])),
                        exact=output == row['reference_ids'],elapsed_s=time.monotonic()-started)
                    if repeat:
                        first = next(r for r in entry['samples'] if r['id'] == identity)
                        sample['equals_first'] = output == first['output_ids']
                    entry['repeats' if repeat else 'samples'].append(sample)
                    write('validation.json', state)
                    print(json.dumps(dict(profile=profile,round=repeat+1,id=identity,
                        matching_tokens=sample['matching_tokens'])),flush=True)
            quick = [r for r in entry['samples'] if r['id'] in chosen]
            entry.update(matching_tokens=sum(r['matching_tokens'] for r in quick),total_tokens=640,
                exact_sequences=sum(r['exact'] for r in quick),
                stable_sequences=sum(r.get('equals_first',False) for r in entry['repeats'] if r['id'] in chosen))
            write('validation.json', state)
        assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == selection['source_sha256']
        state['status'] = 'complete'
        write('validation.json', state)
    finally:
        engine.close()


if os.environ.get('MELLUM_ROCM_AUDIT') == '1':
    install()
if __name__ == '__main__':
    main()
