from pathlib import Path
import sys, json
import numpy as np
from PIL import Image
sys.path.insert(0,'/mnt/d/Projects/Shuolee-smpl-head-align-20261111/src')
from shuolee_portrait.face_supplement import _front_depth, _directions
from scipy.spatial.transform import Rotation
for sub in ('016','006'):
 root=Path('/opt/gs-slam-lab')/('shuolee-smpl-delivery-round19-'+sub)/'run'
 sel=json.loads((root/'fit/selection.json').read_text())
 source={x['frame']:x for x in json.loads((root/'data/inputs.json').read_text())['records']}
 archive=dict(np.load(root/'fit/joint-candidates.npz'))
 names=archive['frame_ids'].tolist()
 yaw,pitch,_=_directions(archive,names.index(sel['reference_frame']))
 supplement=json.loads((root/'gaussian-run/face-supplement/admission.json').read_text())['chosen_frame_ids']
 print(sub,'base',len(sel['selected_frames']),'aux',supplement)
 for i,name in enumerate(names):
  if name not in sel['selected_frames']+supplement: continue
  o=archive['observed_native_keypoints'][i]; p=archive['joints_projected_native'][i]; j=archive['joints_camera'][i]
  v=archive['vertices_camera'][i]
  d=_front_depth(p[17:19],v,archive['faces'],archive['K_integer_centers'][i])
  with Image.open(source[name]['mask']) as m:
   a=np.asarray(m.convert('L'))
  coords=np.rint(o[17:19,:2]).astype(int)
  mask=a[coords[:,1],coords[:,0]]
  err=np.linalg.norm(p-o[:,:2],axis=1)
  eye=float(np.sqrt(np.mean(err[86:98]**2)))
  near=np.argmin(j[17:19,2])
  print(name[-3:], 'yaw',round(yaw[i],1),'eye',round(eye,1),'earNear',near,
        'ears err',np.round(err[17:19],1).tolist(),'c',np.round(o[17:19,2],2).tolist(),
        'alpha',mask.tolist(),'dz mm',np.round((j[17:19,2]-d)*1000,1).tolist(),
        'ear-vtx equiv',np.max(np.abs(j[17:19]-v[[616,6]])))
