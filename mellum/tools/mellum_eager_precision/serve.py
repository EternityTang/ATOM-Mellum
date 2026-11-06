from mellum_eager_precision import install
from atom.utils import debug_helper
from atom.entrypoints.openai import api_server
original=debug_helper.install_block_forward_hooks
def configure(model):
    print('Mellum eager precision:',install(model),flush=True)
    return original(model)
debug_helper.install_block_forward_hooks=configure
if __name__=='__main__':api_server.main()
