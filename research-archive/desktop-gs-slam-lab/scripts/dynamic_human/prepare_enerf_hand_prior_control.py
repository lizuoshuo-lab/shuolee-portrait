"""Audit original Gaussian/soft MANO surface association before any RGB fitting."""
from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np,torch,trimesh
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from fit_hamer_consensus import SharedHand,basis
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_hand_flow_model import project,splat
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-enerf-hand-prior-control-098';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
if (O/'association-summary.json').exists():
 assert '--repair-layout' in sys.argv
 shutil.copy2(O/'association-summary.json',O/'association-summary-before-layout-repair.json')
torch.set_num_threads(6)
parent=S/'local_bspline/resume.pt';cp=torch.load(parent,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};mask=np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy');ids=np.flatnonzero(mask);assert len(ids)==6330
a=np.load(H/'all_views/parameters.npz');model=SharedHand({k:a[k] for k in ['root_base','pose','translation','betas','log_scale']});model.requires_grad_(False)
with torch.no_grad():J,V,_,_=model(tensor(basis([726])));v=V[0].cpu().numpy().astype(float);j=J[0].cpu().numpy().astype(float)
faces=model.mano.faces;tri=v[faces];points=cp['params']['means'][ids].numpy().astype(float)
# Full triangle search in bounded chunks avoids nearest-vertex/centroid approximation.
closest=[];faceids=[];dist=[]
for start in range(0,len(points),64):
 q=points[start:start+64];tt=np.broadcast_to(tri,(len(q),*tri.shape)).reshape(-1,3,3);qq=np.broadcast_to(q[:,None],(len(q),len(tri),3)).reshape(-1,3)
 z=trimesh.triangles.closest_point(tt,qq).reshape(len(q),len(tri),3);d=np.linalg.norm(z-q[:,None],axis=-1);near=d.argmin(1);closest.append(z[np.arange(len(q)),near]);faceids.append(near);dist.append(d[np.arange(len(q)),near])
closest=np.concatenate(closest);faceids=np.concatenate(faceids);dist=np.concatenate(dist);bary=trimesh.triangles.points_to_barycentric(tri[faceids],closest);assert np.max(abs((tri[faceids]*bary[:,:,None]).sum(1)-closest))<1e-10
scene=float(cp['scene_scale']);sigma=.008*scene;support=np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/camera-support-count.npy')[ids]
# Existing hand+depth support is uncertain. Narrow by mesh distance and exclude wrist end.
wrist_distance=np.linalg.norm(closest-j[0],axis=-1);selected=(dist<2*sigma)&(wrist_distance>.035*np.exp(a['log_scale']))&(support>=3)
weight=np.exp(-.5*(dist/sigma)**2)*np.minimum(support/6,1)*selected
chosen=np.flatnonzero(selected);assert 30<len(chosen)<6330
def frame(tr):
 x=tr[:,1]-tr[:,0];x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12);n=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12);y=np.cross(n,x);return np.stack([x,y,n],-1)
R0=frame(tri[faceids[chosen]]);offset=points[chosen]-closest[chosen];local=np.einsum('nji,nj->ni',R0,offset)
clock=np.load(S/'heldout/capture-affines-estimated.npz');capture=clock['capture_time'];surface=[];target=[];normal=[]
with torch.no_grad():
 for i in range(0,len(capture),32):
  _,vv,_,_=model(tensor(basis(capture[i:i+32])));vv=vv.cpu().numpy().astype(float)
  for mesh in vv:
   tt=mesh[faces[faceids[chosen]]];ss=(tt*bary[chosen,:,None]).sum(1);RR=frame(tt);surface.append(ss);target.append(ss+np.einsum('nij,nj->ni',RR,local));normal.append(RR[:,:,2])
np.savez_compressed(O/'association.npz',active_ids=ids,selected_local_ids=chosen,selected_parent_ids=ids[chosen],weight=weight[chosen].astype(np.float32),reference_positions=points[chosen],reference_surface=closest[chosen],face_ids=faceids[chosen],barycentric=bary[chosen],reference_local_offset=local,camera=clock['camera'],frame=clock['frame'],capture_time=capture,surface=np.array(surface,np.float32),target=np.array(target,np.float32),normal=np.array(normal,np.float32))
save(O/'association-protocol.json',dict(status='fixed_before_support_visualization',reference_frame=726,parent_sha256=sha(parent),MANO_parameters_sha256=sha(H/'all_views/parameters.npz'),selection='Original0846330active subset; at least3 distinct source-mask/depth-supported cameras; full triangle distance<0.016scene_scale; nearest surface more than0.035MANOmeters from wrist. No colors or heldout images used for selection.',weight='exp(-0.5*(distance/(0.008scene_scale))^2)*min(camera_support/6,1); these are heuristic soft weights, not semantic certainty.',target='Frozen reference barycentric point + original local offset transported in oriented triangle frame; same physical time gives same target across cameras. Actualestimatedtimefrom079 used.',surface='Separate signed normal deadzone may be tested; no snapping/removal/replacement of Gaussians.',goal_complete=False))
g=GraphMotion(S/'graph.npz',scene);g.load_state_dict(cp['graph']);g.requires_grad_(False);base=HandJointRenderer(S/'head/capture-affines.npz',g,g,torch.from_numpy(mask).cuda());values=torch.zeros((len(mask),1),device='cuda');values[ids[chosen],0]=tensor(weight[chosen]);boxes={(r['camera'],r['frame']):r['crop'] for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records']};font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);R=O/('association-panels-v2' if '--repair-layout' in sys.argv else 'association-panels');R.mkdir(exist_ok=True);rows=[]
with torch.no_grad():
 for c in ['01','02','03','04','05','07','08','09','10','11','13','14','15','16','17']:
  im=Image.new('RGB',(480,618),'white');draw=ImageDraw.Draw(im)
  for k,t in enumerate([720,726,732]):
   z=np.load(S/'native-input'/f'camera{c}-frame{t:06d}.npz');x0,y0,x1,y1=map(int,boxes[c,t]);K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;W=tensor(z['w2c']);K=tensor(K);means,cov,_,_=base.geometry(p,c,t);part,alpha=splat(values,dict(means=means,cov=cov,W=W,K=K),p);coverage=part[:,:,0].cpu().numpy();raw=z['rgb'][y0:y1,x0:x1];heat=np.repeat(np.rint(np.clip(coverage,0,1)*255).astype(np.uint8)[:,:,None],3,2);rgb,al,_=base.render(p,c,t,W,K,160,160,2);render=np.rint(np.clip((rgb[0]+.5*(1-al[0])).cpu().numpy(),0,1)*255).astype(np.uint8)
   for n,img in enumerate([raw,render,heat]):im.paste(Image.fromarray(img),(160*n,46+k*206))
   draw.text((3,4+k*206),f'{c}/{t} Native / original GS / soft-prior contribution',font=font,fill='black');np.savez_compressed(R/f'camera{c}-frame{t:06d}.npz',coverage=coverage,bbox=np.array([x0,y0,x1,y1]),parent_render=render);rows.append(dict(camera=c,frame=t,maximum_visible_weight=float(coverage.max()),sum_visible_weight=float(coverage.sum())))
  im.save(R/f'camera{c}.png');print('PRIOR_ASSOCIATION',c,flush=True)
summary=dict(status='association_prepared_45native_views_visual_review_pending',total_editable=6330,soft_associated=int(selected.sum()),other_editable=int((~selected).sum()),total_Gaussians=len(mask),scene_scale=scene,distance_quantiles=np.quantile(dist[chosen],[0,.5,.9,1]).tolist(),weight_quantiles=np.quantile(weight[chosen],[0,.5,.9,1]).tolist(),association_sha256=sha(O/'association.npz'),records=rows,meaning='Visible weight from uncertainty-gated existing Gaussian contribution, not semantic GT or new reconstruction.',no_Gaussian_updates=True,goal_complete=False);save(O/'association-summary.json',summary);shutil.copy2(__file__,O/Path(__file__).name);print('ASSOCIATED',selected.sum(),'of',len(ids),flush=True)
