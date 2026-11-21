"""Training-derived hand-local controls; image projections before any new optimization."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer
from enerf_hand_flow_model import geometry_at,project,sample,splat
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';J=L/'outputs/dynamic-human-enerf-hand-joint-084';F=L/'outputs/dynamic-human-enerf-open-hand-flow-083';O=L/'outputs/dynamic-human-enerf-hand-fine-085';O.mkdir(exist_ok=True);R=O/'preparation';R.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
assert not (R/'completion.json').exists();torch.set_num_threads(8)
cp=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);parent_sha=sha(S/'local_bspline/resume.pt');p={k:v.cuda() for k,v in cp['params'].items()};proposal=np.load(J/'spatial-diagnostic/local-node-proposal.npz');ids=proposal['active_Gaussian_ids'];chosen=proposal['parent_Gaussian_ids'];centers=proposal['centers'];assert np.array_equal(cp['params']['means'][chosen].numpy(),centers);assert np.array_equal(ids,np.flatnonzero(np.load(J/'support/active-gaussians.npy')))
points=cp['params']['means'][ids].numpy();dist=np.linalg.norm(points[:,None]-centers[None],axis=-1);near=np.argsort(dist,axis=1)[:,:4];distance=np.take_along_axis(dist,near,axis=1);sigma=np.maximum(distance[:,-1:]*.5,1e-5);weights=np.exp(-.5*(distance/sigma)**2);weights[distance[:,0]<1e-7]=[1,0,0,0];weights=(weights/weights.sum(1,keepdims=True)).astype(np.float32)
nd=np.linalg.norm(centers[:,None]-centers[None],axis=-1);nn=np.argsort(nd,axis=1)[:,1:9];edges=np.c_[np.repeat(np.arange(32),8),nn.ravel()];ew=np.exp(-np.take_along_axis(nd,nn,axis=1)**2/(2*np.maximum(np.take_along_axis(nd,nn,axis=1)[:,-1:],1e-5)**2)).ravel().astype(np.float32)
assert len(edges)==256 and np.max(abs(weights.sum(1)-1))<3e-7
np.savez_compressed(O/'fine-graph.npz',node_points=centers,node_anchor=proposal['parent_body_anchor'],edges=edges,edge_weights=ew,active_Gaussian_ids=ids,point_node_indices=near,point_node_weights=weights,node_coarse_indices=cp['params']['node_indices'][chosen].numpy(),node_coarse_weights=cp['params']['node_weights'][chosen].numpy(),parent_Gaussian_ids=chosen)
save(O/'protocol.json',dict(status='fixed_before_training',parent_sha256=parent_sha,branches=['fine_photo','fine_flow'],steps_each=3000,controls='Compare084joint_photo/joint_flow with same seeds/rates105inputs,plus079parent. Added32node residual graph; original22nodeupdates and6330canonicalGaussianupdates kept.',nodes=32,edges=256,nearest_nodes=4,spatial_weights='SameadaptiveGaussianruleasoriginalgraph;half4thneighbordistance;onehotatnode.',node_trajectories='FixedparentGaussianmateriallocations,advectedbyfrozenparentbodyandcoarsegraph. Residualworldcoordinate rotations/translations compose after original graph, only active6330Gaussians.',time_basis='Same bounded3polynomial+7Bspline; reference726exactidentity. Nodeparameterszero initialized.',fine_regularizer='10*rigidity/scene_scale^2 + .001*rotation_size + .1*translation_size/scene_scale^2, additive to original regularizer.',fine_lr=.0003,loss='Originalphoto+optional.05normalflow/2px; currentvisibilitydetached;no new semantic labels.',evaluation='Original120native contexts,24detail windows,6targethandwindows,26fixedwindowconsecutivecontexts; no checkpointselection.',goal_complete=False))
g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);model=ResidualGraphRenderer(S/'head/capture-affines.npz',g);rows=[r for r in read(F/'hand-masks/completion.json')['records'] if r['camera'] not in ['14','16']];cams=sorted({r['camera'] for r in rows});font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);views={};records=[];hits=np.zeros((13,32),bool)
with torch.no_grad():
 for r in rows:
  c,t=r['camera'],r['frame'];ip=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(ip)==r['input_sha256'];z=np.load(ip);m=np.load(F/'hand-masks'/r['filename']);x0,y0,x1,y1=m['crop'];mask=cv2.erode(m['mask'].astype(np.uint8),np.ones((3,3),np.uint8));K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;W=tensor(z['w2c']);K=tensor(K);means,cov=geometry_at(p,model,c,t);uv,depth=project(means,W,K);source=dict(means=means,cov=cov,W=W,K=K);val,alpha=splat(depth[:,None],source,p);expected=val/alpha.clamp_min(1e-8);xy=uv[chosen];dep=depth[chosen];inside=(xy[:,0]>=0)&(xy[:,0]<159)&(xy[:,1]>=0)&(xy[:,1]<159)&(dep>.1);semantic=sample(tensor(mask)[...,None],xy)[:,0]>.999;visible=abs(dep-sample(expected,xy)[:,0])<.015*cp['scene_scale'];ok=(inside&semantic&visible).cpu().numpy();hits[cams.index(c)]|=ok;xy=xy.cpu().numpy()
  raw=z['rgb'][y0:y1,x0:x1];overlay=Image.fromarray(raw);draw=ImageDraw.Draw(overlay)
  for k,(xx,yy) in enumerate(xy):
   if 0<=xx<160 and 0<=yy<160:
    color=(0,255,100) if ok[k] else (255,145,0);draw.ellipse((xx-2,yy-2,xx+2,yy+2),fill=color,outline='black')
  views[c,t]=[raw,np.array(overlay)];records.append(dict(camera=c,frame=t,inside_and_depth_supported=int(ok.sum()),xy=xy.tolist(),supported=ok.tolist(),input_sha256=r['input_sha256']))
 for c in cams:
  im=Image.new('RGB',(348,602),'white');d=ImageDraw.Draw(im);d.text((8,4),f'Camera{c} | source / proposed controls',font=font,fill='black')
  for k,t in enumerate([728,730,732]):
   for j,arr in enumerate(views[c,t]):im.paste(Image.fromarray(arr),(8+j*172,34+k*188))
   d.text((8,196+k*188),f'Frame{t}; green=mask+depth support',font=font,fill='black')
  im.save(R/f'camera{c}-nodes.png')
assert sha(S/'local_bspline/resume.pt')==parent_sha
save(R/'completion.json',dict(status='32nodes_projected_in39trainingcontexts_visual_review_pending',parent_sha256=parent_sha,graph_sha256=sha(O/'fine-graph.npz'),node_support_camera_count=hits.sum(0).tolist(),all_nodes_two_camera_supported=bool((hits.sum(0)>=2).all()),records=records,qualification='OriginaltrainingderivedGaussianlocations and estimateddepth;not anatomical nodes/groundtruth or a trainedimprovement.',goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name);print('FINE_NODE_PROJECTIONS',hits.sum(0).tolist(),flush=True)

