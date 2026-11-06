"""Warm independent operation graphs using generic tokens before serving."""
from .serve import api_server
from atom.model_engine.arg_utils import EngineArgs
from atom import SamplingParams

original_create=EngineArgs.create_engine
def create(self,*args,**kwargs):
    engine=original_create(self,*args,**kwargs)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(self.model,local_files_only=True)
    seed=tokenizer.encode('def warmup(x):\n    return x + 1\n')
    ids=(seed*8)[:32]
    sizes=(1,2,4,8,16,32)
    for n in sizes:
        out=engine.generate([ids]*n,SamplingParams(temperature=0,max_tokens=2,ignore_eos=True))
        assert len(out)==n and all(len(x['token_ids'])==2 for x in out)
    print('Mellum generic graph warmup complete:',sizes,flush=True)
    return engine

EngineArgs.create_engine=create
if __name__=='__main__':api_server.main()
