from pathlib import Path
import sys,json
from time import perf_counter
sys.path.insert(0,'/mnt/d/Projects/Shuolee-smpl-head-align-20261111/src')
from shuolee_portrait.head_alignment import fit_head_alignment
from collections import Counter
base=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official')
assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
for s in ('016','006'):
 root=Path('/opt/gs-slam-lab')/('shuolee-smpl-delivery-round19-'+s)/'run'
 a=json.loads((root/'gaussian-run/face-supplement/admission.json').read_text())
 t=perf_counter()
 b,c,x=fit_head_alignment(root,assets=assets,official_root=base,auxiliary_frames=a['chosen_frame_ids'])
 print(s,round(perf_counter()-t,1),x['changed_frames'])
 for q in x['frames']:
  if q['frame'] in ('frame_000051','frame_000055'):
   print(q['frame'],q['state'],q['before'],q['after'],[(j['fraction'],j['failures']) for j in q['trials']])
 print(Counter(f for q in x['frames'] for j in q['trials'] for f in j['failures']))
