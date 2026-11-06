"""Serve an opt-in single-GPU Mellum decode experiment through the standard ATOM API."""
import os
import torch
import adapter
from atom.models.mellum_rocm_precision import MellumRocmPrecisionForCausalLM
from atom.utils import debug_helper
from atom.entrypoints.openai.api_server import main

mode=os.environ["MELLUM_DECODE_IMPL"]
choices=("batched","batched_packed","mixed_packed","batched_grouped","paged","grouped","paged_grouped")
if mode not in choices:
    raise ValueError(f"MELLUM_DECODE_IMPL must be one of {choices}.")
original_hooks=debug_helper.install_block_forward_hooks

def configure(model):
    if not isinstance(model,MellumRocmPrecisionForCausalLM) or model.precision_profile!="correctness":
        raise ValueError("The decode experiment requires the Mellum ROCm correctness profile.")
    if torch.cuda.device_count()!=1:
        raise ValueError("Select exactly one GPU for this experiment.")
    adapter.install(lambda:mode)
    print(f"Mellum decode implementation: {mode}",flush=True)
    return original_hooks(model)

debug_helper.install_block_forward_hooks=configure
if __name__=="__main__":
    main()
