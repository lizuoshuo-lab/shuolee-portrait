from pathlib import Path
import json
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
REGIONS={'rE':range(86,92),'lE':range(92,98),'nose':range(77,86),'mouth':range(98,118),'rEar':[17],'lEar':[18]}
for sub in ['016','006']:
 r=Path('/opt/gs-slam-lab')/('shuolee-smpl-delivery-round17-'+sub)/'run'
 a=dict(np.load(r/'fit/joint-candidates.npz'))
 sel=json.loads((r/'fit/selection.json').read_text())
 rows={row['frame']:row for row in json.loads((r/'data/inputs.json').read_text())['records']}
 names=a['frame_ids'].tolist(); ref=names.index(sel['reference_frame'])
 root=R.from_rotvec(a['global_orient']).as_matrix()
 head=root.copy()
 for j in [3,6,9,12,15]:
  head=head@R.from_rotvec(a['body_pose_axis_angle'].reshape(-1,21,3)[:,j-1]).as_matrix()
 face_ray = np.einsum("vij,vi->vj",head,-a["joints_camera"][:,0]); print(sub,"poseabs",np.round(np.rad2deg(np.arctan2(face_ray[:,0],face_ray[:,2]))[[0,1,2,3,4,5,6,7,8,ref,-4,-3,-2,-1]],1)); rels=[]
 for M in [root,head]:
  m=M@M[ref].T
  rels.append(np.rad2deg(np.arctan2(m[:,0,2],m[:,2,2])))
 occupied=[]
 for yaw in rels:
  occupied.append(Counter(int(np.round(yaw[names.index(f)]/15))*15 for f in sel['selected_frames']))
 print(sub,'ref',names[ref],'yaw occupied',occupied)
 for i,f in enumerate(names):
  if f in sel['selected_frames']: continue
  row=rows[f]
  o=a['observed_native_keypoints'][i]; p=a['joints_projected_native'][i]; j=a['joints_camera'][i]
  valid=o[67:,2]>.5
  h=np.ptp(o[67:,:2][valid],axis=0).max()
  err=np.linalg.norm(o[:,:2]-p,axis=1)
  with Image.open(row['mask']) as im: mask=np.asarray(im.convert('L'))
  z=[]
  for ids in REGIONS.values():
   ids=np.array(list(ids)); pts=np.rint(o[ids,:2]).astype(int); jv=(o[ids,2]>=.5)
   ina=mask[pts[:,1].clip(0,mask.shape[0]-1),pts[:,0].clip(0,mask.shape[1]-1)]>=254
   jv &= ina
   val=np.sqrt(np.mean(err[ids][jv]**2)) if jv.any() else 0
   z.append((int(jv.sum()),round(val,1),round(float(np.max(err[ids][jv])) if jv.any() else 0,1)))
  print(f[-3:],*[round(yy[i],0) for yy in rels],f'SB{sel["reasons"][f]["source_bin"]}',round(h,0),' '.join('/'.join(map(str,v)) for v in z),f'earDepth{np.round(j[17:19,2]-j[17:19,2].mean(),2)}',sel['reasons'][f]['status'])

