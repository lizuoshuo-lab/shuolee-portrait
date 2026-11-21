"""One CPU-only, frozen-camera/body diagnostic of omitted static face variables.

Never writes a production fit, observations, camera or training checkpoint.
The 29 semantic landmarks retain the existing weights, including their known
visibility limitations. Lower training residual is not physical acceptance.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, time, traceback, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import smplx
from PIL import Image, ImageDraw

REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'grayT-facial-articulation-review'
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
MODEL=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
FIT_SOURCE=ROOT.parent/'runtime/src/shuolee_portrait/fitting.py'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):
    a=np.asarray(x).reshape(-1)
    return dict(n=int(a.size),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))
def main():
    plan=read(OUT/'plan.json')
    if (OUT/'started.json').exists():raise RuntimeError('Already started; do not repeat the registered diagnostic.')
    files=[ROOT/'fit/fitted.npz',ROOT/'fit/observation-weights.npz',ROOT/'data/observations.npz',ROOT/'data/inputs.json',MODEL,FIT_SOURCE]
    checks=[dict(path=str(p),sha256=sha(p)) for p in files]
    assert checks==plan['sources'] and sha(Path(__file__))==plan['diagnostic_source_sha256']
    save(OUT/'started.json',dict(updated=now(),pid=os.getpid(),sources=checks,plan_sha256=sha(OUT/'plan.json')))
    start=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    f=np.load(files[0]);w=np.load(files[1]);obs=np.load(files[2]);rows=read(files[3])['records']
    assert list(f['frame_ids'])==list(obs['frame_ids'])==[r['frame'] for r in rows]
    assert np.max(np.abs(f['vertices_world']-f['vertices_world'][0]))==0
    g=lambda x:torch.as_tensor(np.array(x),dtype=torch.float64)
    model=smplx.SMPLX(str(MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False,dtype=torch.float64)
    for p in model.parameters():p.requires_grad_(False)
    assert not model.expression.any() and not model.jaw_pose.any()
    E=g([r['w2c'] for r in rows]);K=g([r['K'] for r in rows])
    target=g(obs['keypoints'][:,40:91]);weights=g(w['face']*w['face_surface_visibility'])
    active=(weights>0).any(0)
    assert torch.where(active)[0].tolist()==list(range(10,19))+list(range(31,51))
    norm=g((obs['boxes'][:,3]-obs['boxes'][:,1])/1000)[:,None,None]
    fixed=dict(body_pose=g(f['body_pose'][:1]),betas=g(f['betas'][:1]),left_hand_pose=g(f['left_hand_pose'][:1]),right_hand_pose=g(f['right_hand_pose'][:1]))
    R=g(f['root_rotation'][:1]);t=g(f['world_translation'][:1]);s=g(f['world_scale'])
    face_ids=model.faces_tensor[model.lmk_faces_idx];bary=model.lmk_bary_coords
    bounds=g(plan['parameter_scales'])
    def project(x):
        cam=x@E[:,:3,:3].transpose(1,2)+E[:,:3,3][:,None]
        q=cam@K.transpose(1,2)
        return q[...,:2]/q[...,2:]-.5,cam[...,2]
    def forward(p):
        v=bounds*torch.tanh(p)
        body=model(**fixed,expression=v[:10][None],jaw_pose=v[10:][None])
        world=(body.vertices-body.joints[:,:1])@R.transpose(1,2)*s+t[:,None]
        face=(world[:,face_ids]*bary[None,:,:,None]).sum(2)
        uv,z=project(face)
        return uv,z,world,v
    def lossfun(uv):
        residual=(uv-target)/(6*norm)
        return ((torch.sqrt(1+residual.square())-1).sum(-1)*weights).sum()/weights.sum()
    p=torch.zeros(13,requires_grad=True)
    uv,z,world,v=forward(p)
    parity=float((uv-g(f['projected_face51'])).abs().max())
    vertex_parity=float((world-g(f['vertices_world'][:1])).abs().max())
    assert parity<plan['parity_tolerance_px'] and bool((z>0).all())
    baseline_uv=uv.detach().numpy();baseline_world=world.detach().numpy();baseline_loss=float(lossfun(uv).detach())
    # Independent central finite differences of the scalar objective at zero.
    loss=lossfun(uv);ad=torch.autograd.grad(loss,p)[0].detach().numpy();fd=[]
    with torch.no_grad():
        for j in range(13):
            d=torch.zeros(13);d[j]=1e-5
            fd.append(float((lossfun(forward(d)[0])-lossfun(forward(-d)[0]))/(2e-5)))
    gradient_delta=float(np.max(np.abs(ad-np.array(fd))))
    assert gradient_delta<1e-5
    controls=dict(updated=now(),CPU_only=True,frame_identity_pass=True,neutral_expression_and_jaw_zero=True,
        baseline_projection_max_abs_px=parity,baseline_vertex_max_abs=vertex_parity,
        loss_gradient_finite_difference_max_abs=gradient_delta,baseline_loss=baseline_loss,all_passed=True)
    save(OUT/'controls.json',controls)
    opt=torch.optim.LBFGS([p],lr=1.,max_iter=100,max_eval=150,tolerance_grad=1e-8,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    history=[]
    def closure():
        if time.monotonic()-start>plan['cpu_seconds']:raise RuntimeError('Registered CPU budget exhausted; no automatic rerun.')
        opt.zero_grad(set_to_none=True);u,zz,_,_=forward(p)
        if not torch.isfinite(u).all() or not (zz>0).all():raise RuntimeError('Invalid projection; no fallback.')
        loss=lossfun(u);loss.backward()
        history.append(dict(evaluation=len(history)+1,loss=float(loss.detach()),seconds=time.monotonic()-start))
        if len(history)%10==0:save(OUT/'progress.json',dict(updated=now(),**history[-1],training_started=False))
        return loss
    opt.step(closure)
    with torch.no_grad():
        uv,z,world,v=forward(p);final_loss=float(lossfun(uv));cand=uv.numpy();params=v.numpy()
        change=np.linalg.norm(cand-baseline_uv,axis=-1);base_err=np.linalg.norm(baseline_uv-target.numpy(),axis=-1);new_err=np.linalg.norm(cand-target.numpy(),axis=-1)
        report={}
        for name,sl in [('all_active',active.numpy()),('nose',np.isin(np.arange(51),np.arange(10,19))),('mouth',np.arange(51)>=31)]:
            valid=(weights.numpy()>0)&sl[None]
            report[name]=dict(baseline_semantic_residual_px=stats(base_err[valid]),candidate_semantic_residual_px=stats(new_err[valid]),projection_displacement_px=stats(change[valid]))
        vertex_delta=np.linalg.norm(world.numpy()-baseline_world,axis=-1)
        dominant=model.lbs_weights.argmax(-1).numpy();bodymask=dominant<12
        boundfraction=np.abs(params/bounds.numpy());hits=np.flatnonzero(boundfraction>.98).tolist()
    assert all(sha(p)==r['sha256'] for p,r in zip(files,checks))
    result=dict(updated=now(),state='completed_visual_review_pending',seconds=time.monotonic()-start,frames=len(rows),controls=controls,
        active_landmarks=29,optimizer_evaluations=len(history),baseline_objective=baseline_loss,candidate_objective=final_loss,
        objective_reduction_fraction=1-final_loss/baseline_loss,metrics=report,
        shared_expression=params[:10].tolist(),shared_jaw_axis_angle_rad=params[10:].tolist(),
        near_bound_parameter_indices=hits,parameter_absolute_bound_fraction=boundfraction.tolist(),
        world_vertex_displacement=stats(vertex_delta),body_lower_joint_region_displacement=stats(vertex_delta[:,bodymask]),
        history=history,sources_unchanged=True,semantic_visibility_not_fixed=True,physical_quality_accepted=False,
        cameras_changed=False,body_pose_changed=False,training_started=False,
        interpretation='Only one shared bounded expression and jaw vary in this diagnosis. All cameras/body/root/shape/observations/weights are frozen. These semantic observations are not independent physical truth. No production candidate is admitted.')
    save(OUT/'result.json',result)
    np.savez_compressed(OUT/'diagnostic-projections.npz',frame_ids=f['frame_ids'],baseline=baseline_uv,candidate=cand,target=target.numpy(),weights=weights.numpy(),expression=params[:10],jaw_pose=params[10:])
    index=[];thumbs=[]
    for frame in plan['review_frames']:
        i=frame-1;row=rows[i];path=Path(row['image']);im=Image.open(path).convert('RGB')
        assert list(im.size)==row['wh']
        # A single original-pixel crop for all four columns; no resizing or warp.
        center=target.numpy()[i,active.numpy()].mean(0);size=320
        x=max(0,min(im.width-size,int(round(center[0]-size/2))));y=max(0,min(im.height-size,int(round(center[1]-size/2))))
        crop=(x,y,x+size,y+size);tiles=[]
        for pts,color in [(None,None),(target.numpy()[i],(0,240,255)),(baseline_uv[i],(255,100,230)),(cand[i],(255,180,0))]:
            tile=im.crop(crop)
            if pts is not None:
                draw=ImageDraw.Draw(tile)
                for a in np.flatnonzero(active.numpy()):
                    xx,yy=pts[a]-[x,y]
                    draw.ellipse((xx-1.2,yy-1.2,xx+1.2,yy+1.2),outline=color,width=1)
            tiles.append(tile)
        sheet=Image.new('RGB',(size*4,size),'white')
        for j,tile in enumerate(tiles):sheet.paste(tile,(j*size,0))
        name=f'frame_{frame:03d}_raw_observed_neutral_shared_expression_native.png';sheet.save(OUT/name)
        index.append(dict(frame=row['frame'],file=name,crop=crop,columns=['raw','semantic_observations_cyan','frozen_neutral_magenta','shared_expression_jaw_orange']))
        thumbs.append(sheet)
    save(OUT/'index.json',dict(images=index))
    combined=Image.new('RGB',(1280,320*len(thumbs)),'white')
    for i,im in enumerate(thumbs):combined.paste(im,(0,320*i))
    combined.save(OUT/'all6_native_articulation_control.png')
    save(OUT/'progress.json',dict(updated=now(),state='complete',evaluations=len(history),training_started=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['history','controls']},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=str(e),traceback=traceback.format_exc(),training_started=False))
        raise
