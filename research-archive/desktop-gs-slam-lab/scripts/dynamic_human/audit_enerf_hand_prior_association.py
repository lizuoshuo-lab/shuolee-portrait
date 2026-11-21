"""Independent local-frame transport and native diagnostic-image audit."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from fit_hamer_consensus import SharedHand,basis
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-enerf-hand-prior-control-098';torch.set_num_threads(6)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
z=np.load(O/'association.npz');p=np.load(H/'all_views/parameters.npz');m=SharedHand({k:p[k] for k in ['root_base','pose','translation','betas','log_scale']},device='cpu');maximum=0;normal_maximum=0
assert np.allclose(z['barycentric'].sum(1),1) and z['barycentric'].min()>-1e-8 and z['barycentric'].max()<1+1e-8
assert np.array_equal(z['selected_parent_ids'],z['active_ids'][z['selected_local_ids']])
assert (z['weight']>0).all() and (z['weight']<=1).all() and len(z['weight'])==1937
with torch.no_grad():
 for i in range(0,len(z['camera']),32):
  _,V,_,_=m(torch.from_numpy(basis(z['capture_time'][i:i+32])).float())
  for k,v in enumerate(V.numpy().astype(float)):
   tri=v[m.mano.faces[z['face_ids']]];X=(tri*z['barycentric'][:,:,None]).sum(1);a=tri[:,1]-tri[:,0];a/=np.linalg.norm(a,axis=-1,keepdims=True);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=-1,keepdims=True);b=np.cross(n,a);d=z['reference_local_offset'];T=X+d[:,:1]*a+d[:,1:2]*b+d[:,2:]*n
   error=max(float(abs(T-z['target'][i+k]).max()),float(abs(X-z['surface'][i+k]).max()));maximum=max(maximum,error);normal_maximum=max(normal_maximum,float(abs(n-z['normal'][i+k]).max()))
# Unit normals amplify float32 CPU/GPU vertex differences on small triangles.
# Keep their dimensionless error separate from the actual world-position error.
assert maximum<2e-6 and normal_maximum<2e-4,(maximum,normal_maximum)
photos=0;imagefiles=[]
for c in ['01','02','03','04','05','07','08','09','10','11','13','14','15','16','17']:
 file=O/'association-panels-v2'/f'camera{c}.png';im=np.array(Image.open(file));assert im.shape==(618,480,3)
 for k,t in enumerate([720,726,732]):
  a=np.load(O/'association-panels-v2'/f'camera{c}-frame{t:06d}.npz');r=np.load(S/'native-input'/f'camera{c}-frame{t:06d}.npz');x0,y0,x1,y1=a['bbox'];photo=r['rgb'][y0:y1,x0:x1];y=46+206*k;heat=np.repeat(np.rint(np.clip(a['coverage'],0,1)*255).astype(np.uint8)[:,:,None],3,2)
  assert np.array_equal(im[y:y+160,:160],photo) and np.array_equal(im[y:y+160,160:320],a['parent_render']) and np.array_equal(im[y:y+160,320:],heat);photos+=1
 imagefiles.append(dict(file=str(file.relative_to(O)),sha256=sha(file)))
save(O/'association-audit.json',dict(status='234target_transports_and45native_contexts_verified',target_contexts=len(z['camera']),maximum_transport_difference_world_units=maximum,maximum_unit_normal_difference=normal_maximum,photo_contexts=photos,scope='Numerical consistency of soft associations only; does not validate semantic ownership or hidden fingers. Unitnormal sensitivity from float32 CPU/GPU geometry is reported separately.',goal_complete=False))
save(O/'association-visual-review.json',dict(status='45contexts15cameras_viewed',files=imagefiles,findings=['Soft-prior contribution concentrated on palm/backofhand with little fullforearm/toy coverage; fingertip support remains incomplete.', 'Toy-touching boundary and scattered contributions remain uncertain; only bounded soft regularization allowed, not removal or exact binding.', 'Initial layout had overlap outside the main hand; repairedv2 contains45exactnativephoto/parent-render/weight panels. Originalpanels preserved.'],adopt_as_hard_labels=False,goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print('associationCPUtransport/pixel audit passed',maximum,photos)
