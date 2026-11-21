"""Read-only SMPL/MANO wrist alignment on the frozen training observations.

Neither independently fitted model is ground truth. No source checkpoint changes.
"""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import torch, smplx, cv2
from PIL import Image, ImageDraw
from fit_hamer_consensus import SharedHand, basis, inputs, project

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
S=L/'outputs/dynamic-human-enerf-second-interval-079'
H=L/'outputs/dynamic-human-hand-consensus-097'
O=L/'outputs/dynamic-human-hand-body-alignment-101'
O.mkdir(exist_ok=True); (O/'projection').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4)
assert not (O/'summary.json').exists(), 'Preserve completed read-only diagnostic'
source=[S/'head/motion-fit.npz',S/'head/reference.npz',H/'all_views/parameters.npz']
before={str(p):sha(p) for p in source}
rows,aa,times,xy,weight,spread=inputs()
ts=np.array([times[r['camera'],r['frame']] for r in rows])
assert len(rows)==105 and all(r['frame']%2==0 for r in rows)
assert not set(r['camera'] for r in rows)&{'00','06','12'}
save(O/'protocol.json',dict(status='read_only_alignment_before_any_new_training',contexts=105,
    question='Do the independently fitted body right wrist and MANO root agree in the same world frame and estimated physical time?',
    body_joint_order='SMPL standard: right_elbow19, right_wrist21, right_hand23; MANO wrapper OpenPose wrist0.',
    source_sha256=before,heldout_inputs_used=False,
    caveat='Model-to-model disagreement, not anatomy ground truth. Do not shift MANO toward SMPL or camera without photo evidence.',
    goal_complete=False))
t=lambda x:torch.as_tensor(x,dtype=torch.float32)
f=dict(np.load(S/'head/motion-fit.npz')); ref=dict(np.load(S/'head/reference.npz'))
body=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10).cpu().eval()
body.requires_grad_(False)
hand=SharedHand(dict(np.load(H/'all_views/parameters.npz')),device='cpu').eval();hand.requires_grad_(False)

@torch.no_grad()
def body_query(tt):
    b=t(((np.asarray(tt)-726)/6)[:,None]**np.arange(4));n=len(tt)
    pose=t(f['basepose'])[:,None]+torch.einsum('tk,pkd->ptd',b,t(f['posecoef']))*t(f['mask'])
    tr=t(f['basetrans'])[:,None]+torch.einsum('tk,pkd->ptd',b,t(f['transcoef']))
    rr=body(betas=t(f['betas'])[:,None].expand(-1,n,-1).reshape(-1,10),global_orient=pose[...,:3].reshape(-1,3),body_pose=pose[...,3:].reshape(-1,69))
    joints=rr.joints.reshape(2,n,-1,3)*t(f['scales'])[:,None,None,None]+tr[:,:,None]
    vertices=rr.vertices.reshape(2,n,6890,3)*t(f['scales'])[:,None,None,None]+tr[:,:,None]
    return joints.permute(1,0,2,3).numpy(),vertices.permute(1,0,2,3).numpy()

jj,vv=body_query([726]);errj=float(np.max(abs(jj[0]-ref['joints'])));errv=float(np.max(abs(vv[0]-ref['vertices'])))
assert max(errj,errv)<3e-5,(errj,errv)
parts=[]
for st in range(0,len(ts),12):parts.append(body_query(ts[st:st+12])[0])
BJ=np.concatenate(parts)
with torch.no_grad():HJ,HV,_,_=hand(t(basis(ts)))
HJ=HJ.numpy();HV=HV.numpy()
dist=np.linalg.norm(BJ[:,:,21]-HJ[:,None,0],axis=-1);target=int(np.argmin(np.median(dist,axis=0)))
assert (dist[:,target]<dist[:,1-target]).all(), 'Ambiguous identity across times'
K=np.array([a['K'] for a in aa]);W=np.array([a['w2c'] for a in aa])
huv,hz=project(HJ,K,W);buv,bz=project(BJ[:,target],K,W)
delta=HJ[:,0]-BJ[:,target,21]
per=[];images=[]
for i,r in enumerate(rows):
    c,fr=r['camera'],r['frame']
    per.append(dict(camera=c,frame=fr,capture_time=float(ts[i]),person_index=target,
        wrist_distance_world=float(dist[i,target]),MANO_minus_body_wrist_world=delta[i].tolist(),
        MANO_minus_body_wrist_camera_depth=float(hz[i,0]-bz[i,21]),
        wrist_projection_distance_px=float(np.linalg.norm(huv[i,0]-buv[i,21])),
        body_wrist_to_HaMeR_candidate_px=float(np.linalg.norm(buv[i,21]-xy[i,0])),
        MANO_wrist_to_HaMeR_candidate_px=float(np.linalg.norm(huv[i,0]-xy[i,0]))))
    if c not in ['03','04','13']:continue
    z=np.load(S/'native-input'/f'camera{c}-frame{fr:06d}.npz')
    pts=np.concatenate([huv[i],buv[i,[21,23]],xy[i]])
    x0,y0=np.maximum(np.floor(pts.min(0)-25).astype(int),[0,0]);x1,y1=np.minimum(np.ceil(pts.max(0)+26).astype(int),[1920,1080])
    assert x1-x0<500 and y1-y0<500
    crop=z['rgb'][y0:y1,x0:x1].copy();over=crop.copy();shift=np.array([x0,y0])
    uv=np.rint(huv[i]-shift).astype(int)
    for finger in range(5):
        chain=[0]+list(range(1+4*finger,5+4*finger))
        for a,b in zip(chain[:-1],chain[1:]):cv2.line(over,tuple(uv[a]),tuple(uv[b]),(0,160,230),1,cv2.LINE_AA)
    for a in uv:cv2.circle(over,tuple(a),2,(0,160,230),-1,cv2.LINE_AA)
    bp=np.rint(buv[i,[21,23]]-shift).astype(int)
    cv2.line(over,tuple(bp[0]),tuple(bp[1]),(230,120,0),2,cv2.LINE_AA)
    cv2.drawMarker(over,tuple(bp[0]),(230,120,0),cv2.MARKER_CROSS,11,2)
    cv2.circle(over,tuple(np.rint(xy[i,0]-shift).astype(int)),4,(160,0,180),1,cv2.LINE_AA)
    h,w=crop.shape[:2];im=Image.new('RGB',(2*w,h+48),'white');im.paste(Image.fromarray(crop),(0,48));im.paste(Image.fromarray(over),(w,48));d=ImageDraw.Draw(im)
    d.text((3,3),f'{c}/{fr} | source | projected model comparison',fill='black')
    d.text((3,17),'Blue: MANO; orange: SMPL wrist/hand',fill='black');d.text((3,31),'Purple: HaMeR candidate; none is ground truth',fill='black')
    p=O/'projection'/f'camera{c}-frame{fr:06d}.png';im.save(p);images.append(dict(file=str(p.relative_to(O)),camera=c,frame=fr,bbox=[int(x0),int(y0),int(x1),int(y1)],sha256=sha(p)))
q=lambda a:np.quantile(a,[0,.1,.5,.9,1]).tolist()
np.savez_compressed(O/'geometry-local-only.npz',camera=np.array([r['camera'] for r in rows]),frame=np.array([r['frame'] for r in rows]),capture_time=ts,body_joints=BJ,mano_joints=HJ,mano_vertices=HV)
summary=dict(status='105training_contexts_body_MANO_alignment_measured_no_refit',target_person_index=target,
    target_person_name=['person_dark_clothing','person_light_clothing'][target],
    canonical_reforward_max_error=dict(joints=errj,vertices=errv),
    both_person_wrist_distance_medians=np.median(dist,axis=0).tolist(),
    target_wrist_distance_world_quantiles=q(dist[:,target]),
    target_wrist_projection_distance_px_quantiles=q([r['wrist_projection_distance_px'] for r in per]),
    records=per,projection_panels=images,heldout_inputs_used=False,source_checkpoints_unchanged=all(sha(p)==before[str(p)] for p in source),
    licensed_geometry_kept_local=True,clarity_accepted=False,goal_complete=False)
assert summary['source_checkpoints_unchanged'];save(O/'summary.json',summary);shutil.copy2(__file__,O/Path(__file__).name)
print(json.dumps({k:v for k,v in summary.items() if k not in ['records','projection_panels']}),flush=True)
