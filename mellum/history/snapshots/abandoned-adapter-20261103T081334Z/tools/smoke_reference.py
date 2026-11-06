import sys,json,time,urllib.request
from pathlib import Path
import torch
sys.path.insert(0,'/workspace/reference')
from mellum_reference import ReferenceMellum,AutoTokenizer
path='/data/models/Mellum2.1-12B-A2.5B-Thinking'
tok=AutoTokenizer.from_pretrained(path)
for attempt in range(90):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=2).close()
        break
    except Exception:
        time.sleep(1)
else: raise RuntimeError('ATOM startup timeout')
m=ReferenceMellum(path)
prompts=['def add(a, b):\n    return', 'The capital of France is', tok.apply_chat_template([{'role':'user','content':'Write a Python function that adds two numbers.'}],tokenize=False,add_generation_prompt=True)]
results=[]
for prompt in prompts:
    ids=tok(prompt,return_tensors='pt').input_ids.cuda()
    t=time.time(); tokens=m.greedy(ids,16)
    result={'prompt':prompt,'input_ids':ids[0].tolist(),'reference_ids':tokens,'reference_text':tok.decode(tokens),'reference_s':time.time()-t}
    request=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps({'model':path,'prompt':prompt,'max_tokens':16,'temperature':0,'ignore_eos':True}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response: result['atom']=json.load(response)
    result['exact_ids_match']=result['atom'].get('token_ids') == tokens
    print(json.dumps(result,ensure_ascii=False),flush=True);results.append(result)
Path('/workspace/results/smoke_reference.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
