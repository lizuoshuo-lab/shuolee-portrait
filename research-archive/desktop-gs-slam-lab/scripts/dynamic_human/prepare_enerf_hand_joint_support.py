"""Select local Gaussians from reviewed training-only hand interiors, then inspect actual coverage."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer
from enerf_hand_flow_model import geometry_at,project,sample,splat
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';F=L/'outputs/dynamic-human-enerf-open-hand-flow-083';O=L/'outputs/dynamic-human-enerf-hand-joint-084';O.mkdir(exist_ok=True);R=O/'support';R.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (R/'completion.json').exists();torch.set_num_threads(8)
protocol=dict(status='fixed_before_support_inspection',parent='079/local_bspline/resume.pt',source_masks='083handmasks, reviewed; exclude14and16toycontamination;1pixel erosion',selection='Lightperson Gaussian center insideerodedhandmask;within0.015scene_scale ofparentexpecteddepth;selectedin at least2distincttrainingcameras,atany728/730/732frame.',purpose='Select bounded editable Gaussian subset only; do not change RGB,matte or evaluation targets. No semantic groundtruth assertion.',goal_complete=False);save(R/'protocol.json',protocol)
cp=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);model=ResidualGraphRenderer(S/'head/capture-affines.npz',g)
rows=[r for r in read(F/'hand-masks/completion.json')['records'] if r['camera'] not in ['14','16']];cams=sorted({r['camera'] for r in rows});assert len(rows)==39 and len(cams)==13
hits=torch.zeros((13,len(p['means'])),dtype=torch.bool,device='cuda');cache={};checks=[];font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
with torch.no_grad():
 for r in rows:
  c,t=r['camera'],r['frame'];path=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(path)==r['input_sha256'];z=np.load(path);m=np.load(F/'hand-masks'/r['filename']);mask=cv2.erode(m['mask'].astype(np.uint8),np.ones((3,3),np.uint8));x0,y0,x1,y1=box=m['crop'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;W=tensor(z['w2c']);K=tensor(K);means,cov=geometry_at(p,model,c,t);uv,depth=project(means,W,K);source=dict(means=means,cov=cov,W=W,K=K);v,a=splat(depth[:,None],source,p);expected=v/a.clamp_min(1e-8)
  inside=(uv[:,0]>=0)&(uv[:,0]<159)&(uv[:,1]>=0)&(uv[:,1]<159)&(depth>.1)&(p['anchor']>=6890);ids=torch.where(inside)[0];accepted=(sample(tensor(mask)[...,None],uv[ids])[:,0]>.999)&(abs(depth[ids]-sample(expected,uv[ids])[:,0])<.015*cp['scene_scale']);sel=ids[accepted];hits[cams.index(c),sel]=True
  cache[c,t]=dict(source=source,rgb=z['rgb'][y0:y1,x0:x1],mask=mask,box=box);checks.append(dict(camera=c,frame=t,input_sha256=r['input_sha256'],candidates=len(sel)));print('JOINT_SUPPORT',c,t,len(sel),flush=True)
 active=hits.sum(0)>=2;assert 50<int(active.sum())<len(active)*.08;np.save(R/'active-gaussians.npy',active.cpu().numpy());np.save(R/'camera-support-count.npy',hits.sum(0).cpu().numpy());nodes=torch.zeros(384,dtype=torch.bool,device='cuda');nodes[p['node_indices'][active].flatten()]=True;nodes[g.node_anchor<6890]=False;np.save(R/'active-nodes.npy',nodes.cpu().numpy())
 summaries=[]
 for c in cams:
  im=Image.new('RGB',(680,605),'white');draw=ImageDraw.Draw(im);draw.text((8,4),f'Camera{c}: native / hand selector / parent render / selected contribution',font=font,fill='black')
  for k,t in enumerate([728,730,732]):
   z=cache[c,t];source=z['source'];value,alpha=splat(active.float()[:,None],source,p);raw=z['rgb'];masked=np.rint(raw*z['mask'][...,None]+127.5*(1-z['mask'][...,None])).astype(np.uint8);rgb,_,_=model.render(p,c,t,source['W'],source['K'],160,160,2);pred=u8((rgb[0]+.5*(1-alpha)).cpu().numpy());contrib=u8(value[:,:,0].cpu().numpy());contrib=np.repeat(contrib[...,None],3,2);y=36+k*190
   for j,a in enumerate([raw,masked,pred,contrib]):im.paste(Image.fromarray(a),(8+168*j,y))
   draw.text((8,y+163),f'Frame{t} | white=editable Gaussian contribution; unchanged original inputs',font=font,fill='black');summaries.append(dict(camera=c,frame=t,inside_mean=float(value[:,:,0][tensor(z['mask']).bool()].mean()),outside_mean=float(value[:,:,0][~tensor(z['mask']).bool()].mean())))
  im.save(R/f'camera{c}-coverage.png')
save(R/'completion.json',dict(status='local_support_selected_visual_review_pending',active_Gaussians=int(active.sum()),total_Gaussians=len(active),active_nodes=int(nodes.sum()),parent_sha256=sha(S/'local_bspline/resume.pt'),active_Gaussians_sha256=sha(R/'active-gaussians.npy'),active_nodes_sha256=sha(R/'active-nodes.npy'),scene_scale=cp['scene_scale'],depth_tolerance=.015*cp['scene_scale'],records=checks,contribution=summaries,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name);print('JOINT_SUPPORT_COMPLETE',int(active.sum()),int(nodes.sum()),flush=True)
