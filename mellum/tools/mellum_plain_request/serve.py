from mellum_plain_request import install
from atom.utils import debug_helper
from atom.entrypoints.openai import api_server
original=debug_helper.install_block_forward_hooks
def configure(model):
    print('Mellum plain per-request:',install(model),flush=True)
    return original(model)
debug_helper.install_block_forward_hooks=configure
if __name__=='__main__':api_server.main()
