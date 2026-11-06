"""Serve an opt-in single-GPU Mellum decode experiment through the standard ATOM API."""
import os
import torch
import step_cache
from atom.models.mellum_rocm_precision import MellumRocmPrecisionForCausalLM
from atom.utils import debug_helper
from atom.entrypoints.openai.api_server import main

mode=os.environ["MELLUM_STEP_IMPL"]
if mode not in step_cache.MODES:
    raise ValueError(f"Unknown Mellum step-cache implementation {mode!r}.")
original_hooks=debug_helper.install_block_forward_hooks

def configure(model):
    if not isinstance(model,MellumRocmPrecisionForCausalLM) or model.precision_profile!="correctness":
        raise ValueError("The decode experiment requires the Mellum ROCm correctness profile.")
    if torch.cuda.device_count()!=1:
        raise ValueError("Select exactly one GPU for this experiment.")
    if model.atom_config.speculative_config is not None or model.atom_config.compilation_config.level != 0 or not model.atom_config.enforce_eager:
        raise ValueError("Step-cache experiment requires eager level 0 and no MTP.")
    step_cache.install(model, lambda:mode)
    print(f"Mellum step-cache implementation: {mode}",flush=True)
    return original_hooks(model)

debug_helper.install_block_forward_hooks=configure
if __name__=="__main__":
    main()
