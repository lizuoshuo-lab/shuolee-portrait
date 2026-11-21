from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,'/mnt/d/Projects/Shuolee-smpl-head-align-20261111/src')
import shuolee_portrait.head_alignment as h
base=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official')
assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
orig=h._linear_step
m=h.minimize
details=[]
def linear(start,center,projected,jac,observed,regions,before):
 def diagnostic(fun,x0,**args):
  sol=m(fun,x0,**args)
  entry={'min':float(args['constraints'][0]['fun'](sol.x).min()),'n':float(np.linalg.norm(sol.x)),
         'msg':str(sol.message),'init':float(fun(x0)),'end':float(fun(sol.x))}
  details.append((before,entry))
  return sol
 h.minimize=diagnostic
 try: return orig(start,center,projected,jac,observed,regions,before)
 finally: h.minimize=m
h._linear_step=linear
root=Path('/opt/gs-slam-lab/shuolee-smpl-delivery-round19-016/run')
a=json.loads((root/'gaussian-run/face-supplement/admission.json').read_text())
_,_,x=h.fit_head_alignment(root,assets=assets,official_root=base,auxiliary_frames=a['chosen_frame_ids'])
for q in x['frames']:
 if q['frame'] in ('frame_000051','frame_000055') or q['state']=='committed':
  print(q['frame'],q['state'],[(k,v['rms_px'],q['after'][k]['rms_px']) for k,v in q['before'].items() if k.endswith('_ear')])
  for before,diag in details:
   if before==q['before']: print(diag)
