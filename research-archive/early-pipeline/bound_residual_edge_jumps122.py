"""Analytic interval exclusion for frozen point triples, no parameter search or fit."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
D=R/'residual-distortion-direction';O=R/'residual-edge-identity'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'interval-bound.json').exists()
P=read(D/'plan.json');M=read(D/'measurements.json');K=np.array(P['model']['K']);f=K.diagonal()[:2];pp=K[:2,2];bounds=np.array(P['fit']['bounds'])
def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
def basis(uv):
 x,y=((uv-pp)/f).T;r=x*x+y*y
 return np.stack([np.stack([x*r,x*r*r,2*x*y,r+2*x*x],-1)*f[0],np.stack([y*r,y*r*r,r+2*y*y,2*x*y],-1)*f[1]],1)
results=[];checks=0;minmargin=float('inf');rng=np.random.default_rng(122)
for line in M['records']:
 uv=np.array(line['points']);valid=np.array(line['valid']);B=basis(uv);triples=[]
 for i in range(1,len(uv)-1):
  if not valid[i-1:i+2].all():continue
  a=uv[i]-uv[i-1];b=uv[i+1]-uv[i];A=B[i]-B[i-1];BB=B[i+1]-B[i]
  c0=float(cross(a,b));linear=np.array([cross(A[:,j],b)+cross(a,BB[:,j]) for j in range(4)])
  Q=np.array([[cross(A[:,j],BB[:,k]) for k in range(4)] for j in range(4)])
  # Combine symmetric cross terms to tighten the bound without any optimization.
  Q=(Q+Q.T)/2
  variation=float(np.abs(linear)@bounds+np.sum(np.abs(Q)*bounds[:,None]*bounds[None,:]))
  area_lower=max(0,abs(c0)-variation)
  chord=uv[i+1]-uv[i-1];CB=B[i+1]-B[i-1]
  chord_upper=float(np.linalg.norm(chord)+np.linalg.norm(CB,axis=0)@bounds)
  lower=area_lower/chord_upper if chord_upper>1e-12 else 0.
  triple=dict(middle_sample=i,raw_middle_to_chord_px=float(abs(c0)/np.linalg.norm(chord)),
     absolute_area=c0,maximum_area_change_bound=variation,minimum_middle_to_chord_px=lower)
  triples.append(triple)
  # Verify algebra/interval on fixed independent coefficient draws; never choose one for fitting.
  for cc in rng.uniform(-1,1,(4,4))*bounds:
   q=uv[i-1:i+2]+np.einsum('nij,j->ni',B[i-1:i+2],cc)
   actual=float(abs(cross(q[1]-q[0],q[2]-q[0]))/np.linalg.norm(q[2]-q[0]))
   direct=float(cross(q[1]-q[0],q[2]-q[1]));algebra=c0+linear@cc+cc@Q@cc
   assert abs(direct-algebra)<1e-7
   assert actual+1e-9>=lower
   minmargin=min(minmargin,actual-lower);checks+=1
 results.append(dict(id=line['id'],split=line['split'],max_lower_px=max([x['minimum_middle_to_chord_px'] for x in triples],default=0),
    over1px=sum(x['minimum_middle_to_chord_px']>1 for x in triples),triples=triples))
save(O/'interval-bound.json',dict(model=P['model']['name'],bounds=bounds.tolist(),method='Exact quadratic area expansion and triangle-inequality interval over the entire registered coefficient box. No coefficient sweep or optimizer.',
 controls=checks,minimum_control_margin=minmargin,results=results,
 scope='Positive bounds show the allowed D1 smooth mapping cannot make these observed local triples collinear. They do not prove zero real lens distortion, physical straightness, a new parameter set, or human calibration.'))
print(json.dumps([{k:r[k] for k in ['id','split','max_lower_px','over1px']} for r in results]))
