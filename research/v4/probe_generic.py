"""Direct MPS check of the generic multi-axis norm bucket."""
import torch,json
from pathlib import Path
from cotangent.model import ModelConfig
from cotangent.runtime import batch,load_bytes,make_schedule
from research.v3.pilot import build
from collections import defaultdict
m,_,_,_=build(ModelConfig(),101,'incidence-fused'); d=load_bytes(Path('data/train.bin')); t=d[:int(len(d)*.95)]
x,y=batch(t,make_schedule(len(t),1,8,256,101)[0],256,'mps')
with torch.autocast('mps',dtype=torch.bfloat16): _,loss=m(x,y)
loss.backward(); grads=[p.grad for p in m.parameters()]; refs=torch.stack([torch.linalg.vector_norm(g,2.) for g in grads]); buckets=defaultdict(list)
for i,g in enumerate(grads): buckets[g.numel()].append(i)
values=[None]*len(grads)
for ids in buckets.values():
 a=torch.stack([grads[i].reshape(-1) for i in ids]); n=torch.linalg.vector_norm(a.reshape(len(ids),a.shape[1]//2,2),2.,dim=(1,2))
 for i,v in zip(ids,n.unbind()): values[i]=v
out=torch.stack(values); result=dict(buckets=len(buckets),all_norms_bitwise_equal=torch.equal(refs,out),max_norm_difference=float((refs-out).abs().max()),total_norm_bitwise_equal=torch.equal(torch.linalg.vector_norm(refs),torch.linalg.vector_norm(out)))
Path('artifacts/v4/probe-generic-bucket-norms.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
