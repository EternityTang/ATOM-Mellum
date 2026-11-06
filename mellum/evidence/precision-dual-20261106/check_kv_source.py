import torch,json
from pathlib import Path
from types import SimpleNamespace
from mellum_dual.kv import plan,read
torch.set_default_device('cuda')
torch.manual_seed(142)
cases=[]
for lengths in ([15,16,17],[1023,1024,1025],[4099,512,1400],[15194,16019]):
 bs=16;h=4;d=128;pack=8;pages_per=(max(lengths)+15)//16;n=len(lengths)
 physical=torch.randperm(n*pages_per).reshape(n,pages_per).int()
 k=torch.randn((n*pages_per,h,d//pack,bs,pack),dtype=torch.bfloat16)
 v=torch.randn((n*pages_per,h,d,bs),dtype=torch.bfloat16)
 c=SimpleNamespace(key=k,value=v,heads=h,width=d,pack=pack,block_size=bs)
 for window in (None,1024):
  p=plan(lengths,window,32,d,k.device);ko,vo=read(c,physical,p,32)
  for i,(length,base) in enumerate(zip(p[0],p[1])):
   positions=torch.arange(lengths[i]-length,lengths[i]);blocks=physical[i,positions//bs].long()
   ki=k[blocks,:,:,positions%bs,:].reshape(length,h,d)
   vi=v[blocks,:,:,positions%bs]
   def repeat(x):return x.transpose(0,1)[None,:,None].expand(1,h,8,length,d).reshape(1,32,length,d)
   gotk=ko[base:base+32*length*d].view(1,32,length,d);gotv=vo[base:base+32*length*d].view_as(gotk)
   assert torch.equal(gotk,repeat(ki)) and torch.equal(gotv,repeat(vi))
  cases.append(dict(lengths=lengths,window=window,bitwise_equal=True))
Path(__file__).resolve().parent.parent.joinpath('kv-component.json').write_text(json.dumps(cases,indent=2))
print('KV bitwise tests passed',len(cases),flush=True)
