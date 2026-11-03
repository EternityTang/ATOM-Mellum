import sys,json,time,urllib.request
from pathlib import Path
import torch
sys.path.insert(0,'/workspace/reference')
from mellum_reference import ReferenceMellum,AutoTokenizer

def main():
    torch.set_num_threads(8)
    path='/data/models/Mellum2.1-12B-A2.5B-Thinking'
    tok=AutoTokenizer.from_pretrained(path)
    model=ReferenceMellum(path)
    source=tok.encode('def identity(x):\n    return x\n\n'*12000,add_special_tokens=False)
    results=[]
    for length in [32,1023,1024,1025,8191,8192,8193,16384]:
        prompt=tok.decode(source[:length])
        ids=tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
        assert ids.shape[1]==length
        started=time.time(); reference=model.greedy(ids,64)
        request=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps({'model':path,'prompt':prompt,'max_tokens':64,'temperature':0,'ignore_eos':True}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=300) as response: atom=json.load(response)
        got=atom['token_ids']; correct=sum(a==b for a,b in zip(reference,got))
        item={'length':length,'reference_ids':reference,'atom_ids':got,'matching_tokens':correct,'expected_tokens':64,'exact_match':reference==got,'elapsed_s':time.time()-started}
        results.append(item)
        Path('/workspace/results/boundary_correctness.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(item),flush=True)
    if not all(x['exact_match'] for x in results): raise SystemExit(1)

if __name__=='__main__':main()
