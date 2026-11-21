"""Independent SMPL forward and actual-photo projection checks for interval079."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np, torch, smplx
from PIL import Image,ImageDraw,ImageFont
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-second-interval-079';O=R/'motion-audit';O.mkdir(exist_ok=True)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'audit.json').exists();torch.set_num_threads(4)
t=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
model=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10,batch_size=1).cuda().eval()
f=dict(np.load(R/'head/motion-fit.npz'));body=dict(np.load(R/'body/fixed_clock/motion-fit.npz'));ref=dict(np.load(R/'head/reference.npz'));aff=dict(np.load(R/'head/capture-affines.npz'))
for key in ['basepose','basetrans','transcoef','betas','scales','camera_offsets']:assert np.array_equal(f[key],body[key]),key
frozen=np.ones(72,bool);frozen[[36,37,38,45,46,47]]=False;assert np.array_equal(f['posecoef'][:,:,frozen],body['posecoef'][:,:,frozen])
assert int(ref['reference_frame'])==726
def forward(times,params=f):
 tau=(t(times)-726)/6;b=torch.stack([tau**k for k in range(4)],-1);n=len(times)
 pose=t(params['basepose'])[:,None]+torch.einsum('tk,pkd->ptd',b,t(params['posecoef']))*t(params['mask'])
 trans=t(params['basetrans'])[:,None]+torch.einsum('tk,pkd->ptd',b,t(params['transcoef']))
 with torch.no_grad():
  out=model(betas=t(params['betas'])[:,None].expand(-1,n,-1).reshape(-1,10),global_orient=pose[:,:,:3].reshape(-1,3),body_pose=pose[:,:,3:].reshape(-1,69),transl=torch.zeros(2*n,3,device='cuda'))
  v=out.vertices.reshape(2,n,6890,3)*t(params['scales'])[:,None,None,None]+trans[:,:,None,:]
  j=out.joints.reshape(2,n,-1,3)*t(params['scales'])[:,None,None,None]+trans[:,:,None,:]
 return v.permute(1,0,2,3).reshape(n,13780,3).cpu().numpy(),j.permute(1,0,2,3).cpu().numpy()
v0,j0=forward([726]);referror=float(abs(v0[0]-ref['vertices'].reshape(-1,3)).max());assert referror<2e-5
vertices=[];joints=[]
for start in range(0,195,16):
 v,j=forward(aff['capture_time'][start:start+16]);vertices.append(v);joints.append(j)
vertices=np.concatenate(vertices);joints=np.concatenate(joints);verror=float(abs(vertices-aff['vertices']).max());assert verror<2e-5
reconstructed=np.einsum('nvij,vj->nvi',aff['linear'],ref['vertices'].reshape(-1,3))+aff['translation'];aerror=float(abs(reconstructed-vertices).max());assert aerror<2e-5
minimum_singular=1e9;minimum_det=1e9
for a in aff['linear']:
 minimum_singular=min(minimum_singular,float(np.linalg.svd(a,compute_uv=False).min()));minimum_det=min(minimum_det,float(np.linalg.det(a).min()))
assert minimum_singular>.1 and minimum_det>0
np.savez_compressed(O/'independent-joints.npz',camera=aff['camera'],frame=aff['frame'],capture_time=aff['capture_time'],joints=joints)
lookup={(str(c),int(fr)):i for i,(c,fr) in enumerate(zip(aff['camera'],aff['frame']))};obs=read(R/'pose/observations.json')['records'];ids=[16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34];faceids=[24,26,25,28,27];rows=[]
for r in obs:
 ci=lookup[r['camera'],r['frame']];P=np.array(r['K'])@np.array(r['w2c'])[:3]
 for d in r['detections']:
  pi=['person_dark_clothing','person_light_clothing'].index(d['person']);X=joints[ci,pi,ids];q=X@P[:,:3].T+P[:,3];uv=q[:,:2]/q[:,2:]
  for k,j in enumerate(range(5,23)):
   if j in d['disabled_fitting_indices'] or d['scores'][j]<.4:continue
   rows.append(dict(camera=r['camera'],frame=r['frame'],person=pi,index=j,error_px=float(np.linalg.norm(uv[k]-np.array(d['keypoints'][j])))))
def metric(rs):
 e=np.array([r['error_px'] for r in rs]);return dict(count=len(e),median_px=float(np.median(e)),rmse_px=float(np.sqrt(np.mean(e*e))),p90_px=float(np.quantile(e,.9)))
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',13);panels=[]
for c in ['01','02','04','07','13','17']:
 for fr in [720,726,732]:
  z=np.load(R/'native-input'/f'camera{c}-frame{fr:06d}.npz');raw=Image.fromarray(z['rgb']);im=raw.copy();draw=ImageDraw.Draw(im);ci=lookup[c,fr];P=z['K']@z['w2c'][:3]
  for pi,col in enumerate([(255,100,20),(0,160,255)]):
   X=vertices[ci,pi*6890:(pi+1)*6890:12];q=X@P[:,:3].T+P[:,3];uv=q[:,:2]/q[:,2:]
   # Sparse projected mesh samples show the prior surface, not a rendered image.
   for x,y in uv:draw.point((float(x),float(y)),fill=col)
   X=joints[ci,pi,faceids+ids];q=X@P[:,:3].T+P[:,3];uv=q[:,:2]/q[:,2:]
   for idx,(x,y) in enumerate(uv):
    draw.ellipse([x-2,y-2,x+2,y+2],fill=col);draw.text((x+3,y-9),str(idx),font=font,fill=col,stroke_width=1,stroke_fill='white')
  bb=z['bbox'];raw=raw.crop(bb);im=im.crop(bb);sheet=Image.new('RGB',(raw.width*2,raw.height+30),'white');sheet.paste(raw,(0,30));sheet.paste(im,(raw.width,30));ImageDraw.Draw(sheet).text((5,5),f'Camera{c} frame{fr}: original | fitted SMPL projection (not appearance rendering)',font=font,fill='black')
  path=O/f'camera{c}-frame{fr:06d}-smpl.png';sheet.save(path);panels.append(dict(camera=c,frame=fr,path=path.name,sha256=sha(path)))
out=dict(status='195_direct_SMPL_forward_and_affine_reconstructions_verified_visual_review_pending',reference_frame=726,reference_max_world_error=referror,direct_forward_max_world_error=verror,affine_anchor_max_world_error=aerror,minimum_singular=minimum_singular,minimum_determinant=minimum_det,
 body_coefficients_frozen_during_head_fit=True,all_eligible_train=metric([r for r in rows if r['frame']%2==0]),all_eligible_odd=metric([r for r in rows if r['frame']%2]),
 image_panels=panels,source_motion_sha256=sha(R/'head/motion-fit.npz'),source_affines_sha256=sha(R/'head/capture-affines.npz'),
 scope='Body/face prior fitting and geometry. Sparse projected mesh is local diagnostic, not a reconstruction result; no appearance training or clarity acceptance.',goal_complete=False)
save(O/'audit.json',out);shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps({k:v for k,v in out.items() if k!='image_panels'}),flush=True)
