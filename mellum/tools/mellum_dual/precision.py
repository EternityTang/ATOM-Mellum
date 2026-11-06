import os
from mellum_opt_points import install as base_install
def install(model):
 base_install(model,attention='bounded',moe='off',dispatch='graph_bucket')
 counts=dict(kv_gathers=0)
 if os.environ.get('KV_MODE','off')=='direct':
  from .kv import install as install_kv
  install_kv(model,counts)
 model.dual_metadata=dict(track='precision',kv=os.environ.get('KV_MODE','off'),dense_cap=int(os.environ.get('DENSE_CAP','1')),counts=counts,base=model.opt_points_metadata)
 return model.dual_metadata
