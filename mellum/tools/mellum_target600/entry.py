import os
import torch
import target
from atom.models.mellum_rocm_precision import MellumRocmPrecisionForCausalLM
from atom.utils import debug_helper
from atom.entrypoints.openai import api_server
mode=os.getenv('MELLUM_DECODE_IMPL','packed')
if mode not in ('packed','post','adaptive','fast_combined'):
    raise ValueError(f'Unsupported Mellum implementation {mode}')
original=debug_helper.install_block_forward_hooks
def configure(model):
    cfg=model.atom_config
    if not isinstance(model,MellumRocmPrecisionForCausalLM) or model.precision_profile!='correctness':
        raise ValueError('Mellum correctness backbone required')
    if torch.cuda.device_count()!=1 or cfg.speculative_config is not None or cfg.compilation_config.level!=0 or not cfg.enforce_eager:
        raise ValueError('Requires single GPU BF16 eager, compile level 0, MTP disabled')
    target.install(model,lambda:mode)
    print(f'Active Mellum decode implementation {mode}',flush=True)
    return original(model)
debug_helper.install_block_forward_hooks=configure
if __name__=='__main__':api_server.main()
