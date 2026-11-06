"""Production startup with generic graph warmup; no benchmark inputs or references."""
import os
from .serve import api_server
from atom.model_engine.arg_utils import EngineArgs
from atom import SamplingParams
original_create=EngineArgs.create_engine
def create(self,*args,**kwargs):
    engine=original_create(self,*args,**kwargs)
    if os.environ.get('OPT_DISPATCH','graph_bucket').startswith('graph'):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking',local_files_only=True)
        seed=tokenizer.encode('def warmup(x):\n    return x + 1\n')
        ids=(seed*8)[:32]
        sizes=(1,2,4,8,16,32) if os.environ.get('OPT_DISPATCH')=='graph_bucket' else (1,32)
        for n in sizes:
            out=engine.generate([ids]*n,SamplingParams(temperature=0,max_tokens=2,ignore_eos=True))
            assert len(out)==n and all(len(x['token_ids'])==2 for x in out)
        print('Mellum generic graph warmup complete:',sizes,flush=True)
    return engine
EngineArgs.create_engine=create
if __name__=='__main__':api_server.main()
