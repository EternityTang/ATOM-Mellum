import os
from atom.utils import debug_helper
from atom.entrypoints.openai import api_server
from mellum_parallel_precision import install
original=debug_helper.install_block_forward_hooks
def configure(model):
    metadata=install(model,os.environ.get('MELLUM_PARALLEL_MODE','norm_cap8'))
    print('Mellum parallel precision:',metadata,flush=True)
    return original(model)
debug_helper.install_block_forward_hooks=configure
if __name__=='__main__':api_server.main()
