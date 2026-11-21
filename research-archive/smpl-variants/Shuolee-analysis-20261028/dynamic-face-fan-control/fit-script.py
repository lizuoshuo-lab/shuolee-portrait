"""Fixed-camera SMPL-X neck, head, jaw and expression control; body remains fixed."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONNOUSERSITE='1')
from pathlib import Path
import argparse,fcntl,hashlib,json,shutil,sys,time
import numpy as np
ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
arguments=argparse.ArgumentParser();arguments.add_argument('--detector',choices=['rtmw','fan'],default='rtmw');args=arguments.parse_args()
OUT=Path('/mnt/d/Projects/Shuolee-analysis-20261028')/('dynamic-face-control' if args.detector=='rtmw' else 'dynamic-face-fan-control');OUT.mkdir(exist_ok=False)
shutil.copyfile(__file__,OUT/'fit-script.py')
(OUT/'progress.json').write_text(json.dumps(dict(state='waiting_for_gpu',detector=args.detector)))
lock=Path('/tmp/shuolee-native122-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
lock.seek(0);lock.truncate();lock.write(json.dumps(dict(pid=os.getpid(),task='SMPL-X dynamic face fitting',output=str(OUT))));lock.flush()
try:
    import torch,smplx
    from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    fit=dict(np.load(ROOT/'fit/fitted.npz'));obs=dict(np.load(ROOT/'data/observations.npz'));weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
    rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records'];n=len(rows)
    g=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    model=smplx.SMPLX('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz',
        num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=n,use_face_contour=False).cuda()
    for p in model.parameters():p.requires_grad_(False)
    pose=g(fit['body_pose']).reshape(n,21,3);rotations=axis_angle_to_matrix(pose)
    beta=g(fit['betas']).expand(n,-1);lh=g(fit['left_hand_pose']);rh=g(fit['right_hand_pose'])
    R=g(fit['root_rotation']);t=g(fit['world_translation']);scale=float(fit['world_scale'])
    E=g(np.array([r['w2c'] for r in rows]));K=g(np.array([r['K'] for r in rows]))
    target_array=obs['keypoints'][:,40:91]
    if args.detector=='fan':
        fan=dict(np.load('/mnt/d/Projects/Shuolee-analysis-20261028/face-local-detector/observations.npz'))
        assert np.array_equal(fan['frame_ids'],fit['frame_ids']);target_array=fan['decoded_face68'][:,17:68]
    targets=g(target_array);w=g(weights['face']);sigma=g(6*(obs['boxes'][:,3]-obs['boxes'][:,1])/1000)[:,None,None]
    tri=model.faces_tensor[model.lmk_faces_idx];bary=model.lmk_bary_coords
    parameters=torch.nn.Parameter(torch.zeros((n,19),device='cuda'))
    lo=g([-.35]*3+[-.5]*3+[0.,-.1,-.1]+[-3.]*10)
    hi=g([.35]*3+[.5]*3+[.5,.1,.1]+[3.]*10)
    def forward():
        pr=rotations.clone()
        pr[:,11]=axis_angle_to_matrix(parameters[:,:3])@rotations[:,11]
        pr[:,14]=axis_angle_to_matrix(parameters[:,3:6])@rotations[:,14]
        body_pose=matrix_to_axis_angle(pr).reshape(n,63)
        body=model(body_pose=body_pose,betas=beta,left_hand_pose=lh,right_hand_pose=rh,
                   jaw_pose=parameters[:,6:9],expression=parameters[:,9:])
        vertices=(body.vertices-body.joints[:,:1])@R.transpose(1,2)*scale+t[:,None]
        face=(vertices[:,tri]*bary[None,:,:,None]).sum(2)
        cam=face@E[:,:3,:3].transpose(1,2)+E[:,None,:3,3];q=cam@K.transpose(1,2)
        uv=q[...,:2]/q[...,2:].clamp_min(.01)-.5
        return vertices,uv,body_pose
    with torch.no_grad():
        initial,initial_uv,_=forward();rebuild=float((initial-g(fit['vertices_world'])).abs().max())
        if rebuild>1e-4:raise RuntimeError(f'SMPL-X neutral reconstruction mismatch {rebuild}')
    optimizer=torch.optim.Adam([parameters],lr=.008);started=time.monotonic();records=[]
    for step in range(801):
        v,uv,bp=forward();e=(uv-targets)/sigma
        face=((torch.sqrt(1+e.square())-1).sum(-1)*w).sum()/w.sum()
        prior=.01*(parameters[:,:6]/.35).square().mean()+.004*(parameters[:,6:9]/.3).square().mean()+.003*parameters[:,9:].square().mean()
        accel=parameters[2:]-2*parameters[1:-1]+parameters[:-2]
        units=g([.03]*6+[.02]*3+[.12]*10)
        smooth=.02*(torch.sqrt(1+(accel/units).square())-1).mean()
        loss=face+prior+smooth
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite dynamic SMPL-X face objective')
        if step%100==0:
            error=torch.linalg.norm(uv-targets,dim=-1)[w>0]
            rec=dict(step=step,loss=float(loss),face=float(face),median_px=float(error.median()),p90_px=float(torch.quantile(error,.9)),seconds=time.monotonic()-started)
            records.append(rec);(OUT/'progress.json').write_text(json.dumps(rec));print(json.dumps(rec),flush=True)
        if step==800:break
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([parameters],10.);optimizer.step()
        with torch.no_grad():parameters.clamp_(lo,hi)
        if step==550:optimizer.param_groups[0]['lr']=.0024
    with torch.no_grad():
        vertices,uv,bp=forward();vertices=vertices.cpu().numpy();uv=uv.cpu().numpy();p=parameters.cpu().numpy();bp=bp.cpu().numpy()
    np.savez_compressed(OUT/'fitted.npz',vertices_world=vertices,reference_vertices=fit['vertices_world'][0],faces=fit['faces'],
        projected_face51=uv,parameters=p,body_pose=bp,jaw_pose=p[:,6:9],expression=p[:,9:],frame_ids=fit['frame_ids'],world_scale=scale)
    # Per-vertex affine from a well-conditioned incident triangle, with unchanged normal thickness.
    faces=fit['faces'];reference=fit['vertices_world'][0].astype(float)
    edge1=reference[faces[:,1]]-reference[faces[:,0]];edge2=reference[faces[:,2]]-reference[faces[:,0]]
    areas=np.linalg.norm(np.cross(edge1,edge2),axis=1)
    choice=np.zeros(len(reference),int);best=np.zeros(len(reference))
    for fi,face in enumerate(faces):
        for vi in face:
            if areas[fi]>best[vi]:choice[vi]=fi;best[vi]=areas[fi]
    chosen=faces[choice];e1=edge1[choice];e2=edge2[choice];normal=np.cross(e1,e2);normal/=np.linalg.norm(normal,axis=1)[:,None]
    thickness=(np.linalg.norm(e1,axis=1)+np.linalg.norm(e2,axis=1))*.5
    base=np.stack([e1,e2,normal*thickness[:,None]],-1);inverse=np.linalg.inv(base)
    a=[]
    for verts in vertices:
        e1=verts[chosen[:,1]]-verts[chosen[:,0]];e2=verts[chosen[:,2]]-verts[chosen[:,0]];normal=np.cross(e1,e2)
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
        a.append(np.stack([e1,e2,normal*thickness[:,None]],-1)@inverse)
    affine=np.asarray(a,dtype=np.float32)
    np.savez_compressed(OUT/'motion.npz',vertices_world=np.concatenate([reference[None],vertices]).astype(np.float32),
        vertex_affine=np.concatenate([np.broadcast_to(np.eye(3),(1,len(reference),3,3)),affine]).astype(np.float32),
        reference_frame=0,training_timestep_offset=1,world_scale=scale)
    err=np.linalg.norm(uv-target_array,axis=-1);valid=weights['face']>0
    before=np.linalg.norm(fit['projected_face51']-target_array,axis=-1)
    stat=lambda e:dict(median=float(np.median(e)),p90=float(np.quantile(e,.9)),maximum=float(np.max(e)))
    report=dict(state='completed',frames=n,steps=800,source_run=str(ROOT),initial_vertices_max_abs=rebuild,detector=args.detector,
                source_sha256=hashlib.sha256((OUT/'fit-script.py').read_bytes()).hexdigest(),
                before=stat(before[valid]),after=stat(err[valid]),history=records,parameters='Per-frame neck, head, jaw and expression; all other body, shape, root and cameras fixed',
                face_visibility_policy='Same original weights; anatomical face visibility not corrected in this control',
                neck_max_degrees=float(np.linalg.norm(p[:,:3],axis=1).max()*180/np.pi),head_max_degrees=float(np.linalg.norm(p[:,3:6],axis=1).max()*180/np.pi),
                jaw_max_radians=float(np.linalg.norm(p[:,6:9],axis=1).max()),expression_max_abs=float(abs(p[:,9:]).max()),
                affine_determinant=stat(np.linalg.det(affine).ravel()),training_started=False,cameras_changed=False,quality_accepted=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
finally:fcntl.flock(lock,fcntl.LOCK_UN)
