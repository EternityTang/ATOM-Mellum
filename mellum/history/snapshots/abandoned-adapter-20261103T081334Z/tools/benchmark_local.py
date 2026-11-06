"""Local diagnostic benchmark. Not the unavailable official evalscope runner.
A/C use provided JSONL verbatim. B uses 50 distinct seeded 1024-token prompts.
C arrival schedule is independent of response completion; timeout is per-request.
"""
import argparse,asyncio,json,time,random
from pathlib import Path
import aiohttp,numpy as np
from transformers import AutoTokenizer

async def main(args):
    model='/data/models/Mellum2.1-12B-A2.5B-Thinking'
    tok=AutoTokenizer.from_pretrained(model)
    n=args.gpus; rng=random.Random(args.seed)
    if args.scenario in ['A','C']:
        source=Path('/data/datasets')/('scenario_a_prompts.jsonl' if args.scenario=='A' else 'scenario_c_prompts.jsonl')
        rows=[json.loads(x) for x in source.read_text().splitlines()]*n
    else:
        words='python code function return input output test data string list model value error request service'.split()
        rows=[]
        for i in range(50):
            text=' '.join(rng.choices(words,k=2000))
            ids=tok.encode(text,add_special_tokens=False)[:1024]
            prompt=tok.decode(ids)
            assert len(tok.encode(prompt,add_special_tokens=False))==1024
            rows.append({'id':f'B-{i:04d}','prompt':prompt})
        assert len({row['prompt'] for row in rows})==50
    if args.limit: rows=rows[:args.limit]
    sem=asyncio.Semaphore(32*n if args.scenario=='A' else 1)
    records=[]
    timeout=aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def request(row):
            started=time.perf_counter(); first=None; last=None; output_tokens=0; chunks=[]; error=None
            try:
                body={'model':model,'prompt':row['prompt'],'max_tokens':args.output_tokens,'temperature':0.0,'ignore_eos':True,'stream':True}
                async with session.post(args.url+'/v1/completions',json=body) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        line=line.decode().strip()
                        if not line.startswith('data: '): continue
                        data=line[6:]
                        if data=='[DONE]': break
                        item=json.loads(data)
                        text=''.join(c.get('text','') for c in item.get('choices',[]))
                        if text:
                            now=time.perf_counter()
                            if first is None: first=now
                            last=now; chunks.append({'time':now-started,'text':text})
                        usage=item.get('usage',{})
                        if 'completion_tokens' in usage: output_tokens=usage['completion_tokens']
                if first is None: raise RuntimeError('No nonempty token stream')
                if output_tokens!=args.output_tokens: raise RuntimeError(f'Expected {args.output_tokens} tokens, got {output_tokens}')
            except Exception as exc: error=f'{type(exc).__name__}: {exc}'
            end=time.perf_counter()
            record={'id':row['id'],'input_tokens':len(tok.encode(row['prompt'])),'output_tokens':output_tokens,'ttft_s':None if first is None else first-started,'tpot_s':None if first is None or output_tokens<2 else (last-first)/(output_tokens-1),'latency_s':end-started,'error':error,'chunks':chunks}
            records.append(record)
            with open(args.output+'.requests.jsonl','a') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k!='chunks'}),flush=True)
        async def limited(row):
            async with sem: await request(row)
        # Warmup outside timed interval, using a neutral prompt; never caches outputs.
        async with session.post(args.url+'/v1/completions',json={'model':model,'prompt':'Warmup.','max_tokens':2,'temperature':0}) as response:
            response.raise_for_status(); await response.read()
        start=time.perf_counter(); tasks=[]; due=start
        for i,row in enumerate(rows):
            if args.scenario=='C':
                if i: due+=rng.expovariate(.30*n)
                await asyncio.sleep(max(0,due-time.perf_counter()))
                tasks.append(asyncio.create_task(request(row)))
            else: tasks.append(asyncio.create_task(limited(row)))
        await asyncio.gather(*tasks)
        elapsed=time.perf_counter()-start
    good=[r for r in records if r['error'] is None]
    summary={'runner':'local diagnostic, NOT official evalscope','scenario':args.scenario,'gpus':n,'seed':args.seed,'count':len(rows),'successful':len(good),'elapsed_s':elapsed,'output_tokens_per_s_gpu':sum(r['output_tokens'] for r in good)/elapsed/n,'request_timeout_s':args.request_timeout,'output_tokens':args.output_tokens}
    for metric in ['ttft_s','tpot_s','latency_s']:
        values=[r[metric] for r in good if r[metric] is not None]
        summary[metric]={str(p):float(np.percentile(values,p)) for p in [50,99]} if values else {}
    Path(args.output).write_text(json.dumps({'summary':summary,'records':records},ensure_ascii=False,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(good)!=len(rows): raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scenario',choices=['A','B','C']);p.add_argument('--gpus',type=int,choices=[1,2],default=1);p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--seed',type=int,default=20261103);p.add_argument('--limit',type=int);p.add_argument('--output-tokens',type=int,default=256);p.add_argument('--request-timeout',type=float,default=180);p.add_argument('--output',required=True)
    asyncio.run(main(p.parse_args()))
