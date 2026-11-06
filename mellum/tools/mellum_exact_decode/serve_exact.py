"""Serve the precision-preserving Mellum decode path through ATOM."""
import torch
import adapter
import batched_sdpa
import step_cache
from atom.models.mellum_rocm_precision import MellumRocmPrecisionForCausalLM
from atom.utils import debug_helper
from atom.entrypoints.openai import api_server
from exact import experts
from gathered import batched

original_hooks = debug_helper.install_block_forward_hooks

def configure(model):
    if not isinstance(model,MellumRocmPrecisionForCausalLM) or model.precision_profile!='correctness':
        raise ValueError('The exact decode path requires the Mellum correctness profile')
    cfg = model.atom_config
    if torch.cuda.device_count()!=1 or cfg.speculative_config is not None or cfg.compilation_config.level!=0 or not cfg.enforce_eager:
        raise ValueError('This path requires single GPU, eager level 0, MTP disabled')
    batched_sdpa.batched = batched
    adapter.grouped_experts = experts
    step_cache.MODES['exact_decode'] = ('mixed_grouped',True,'native')
    step_cache.install(model,lambda:'exact_decode')
    print('Mellum exact decode active',flush=True)
    return original_hooks(model)

debug_helper.install_block_forward_hooks = configure
if __name__=='__main__':
    api_server.main()
