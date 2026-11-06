import os
from atom.utils import debug_helper
from atom.entrypoints.openai import api_server
from .install import install
original=debug_helper.install_block_forward_hooks
def configure(model):
    metadata=install(model,attention=os.environ.get('OPT_ATTENTION','bounded'),moe=os.environ.get('OPT_MOE','off'),dispatch=os.environ.get('OPT_DISPATCH','graph'))
    print('Mellum optimization points:',metadata,flush=True)
    return original(model)
debug_helper.install_block_forward_hooks=configure
if __name__=='__main__':api_server.main()
