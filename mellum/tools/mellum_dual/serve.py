from atom.utils import debug_helper
from atom.entrypoints.openai import api_server
from .precision import install
original=debug_helper.install_block_forward_hooks
def configure(model):
    metadata=install(model)
    print('Mellum validated precision optimization:',metadata,flush=True)
    return original(model)
debug_helper.install_block_forward_hooks=configure
if __name__=='__main__':api_server.main()
