# Static new-video fitter derived from the recorded local shared-static adapter.
# New inputs, all available masks, mean hand initialization, fixed SfM cameras.
"""All-frame fresh SMPL-X fit with automatic observation weights.

This is a local experimental fitter, not an official SMPLify-X reproduction.
The body is a kinematic prior; clothing observations never define naked joints.
One shared pose and the optimizer start new. No old subject fit loads.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from PIL import Image
from .runtime import c, ASSETS
REPO=ASSETS/"gvhmr"

torch=c.runtime()

def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}

def main(root,steps):
    import smplx
    import nvdiffrast.torch as dr
    from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle
    from smplx.joint_names import JOINT_NAMES
    from shuolee_vposer import load_vposer
    out=root/'fit';out.mkdir(exist_ok=True)
    assert not (out/'fitted.npz').exists(), 'No cross-experiment resume'
    rows=c.read(root/'data/inputs.json')['records'];n=len(rows)
    cfg=c.read(root/'config.json')
    support_mode=cfg.get('landmark_image_support','legacy')
    if support_mode not in ('legacy','mask_margin_4px'):
        raise ValueError('Unknown landmark image support policy.')
    obs=load(root/'data/observations.npz');pred=load(root/'inference/prediction-postprocessed.npz')
    slots=np.array([r['official_prediction_slot'] for r in rows]);assert np.array_equal(obs['frame_ids'],[r['frame'] for r in rows])
    E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);times=np.array([r['time_seconds'] for r in rows])
    vit=torch.load(root/'inference/vitpose-keypoints.pt',weights_only=True,map_location='cpu').numpy()[slots]
    g=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    model=smplx.SMPLX(str(c.MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False).cuda()
    predR=Rotation.from_rotvec(pred['incam_global_orient'][slots]).as_matrix()
    pelvis=pred['incam_joints'][slots,0]
    A=np.concatenate([E[:,:3,:3],-pelvis[:,:,None]],-1).reshape(-1,4);rhs=-E[:,:3,3].reshape(-1)
    sol=least_squares(lambda x:A@x-rhs,np.linalg.lstsq(A,rhs,rcond=None)[0],loss='soft_l1',f_scale=.05)
    s0=float(sol.x[3]);assert np.isfinite(s0) and s0>1e-6, "Unobservable or non-positive body/world scale"
    R0=np.einsum('nji,njk->nik',E[:,:3,:3],predR)
    t0=np.einsum('nji,nj->ni',E[:,:3,:3],s0*pelvis-E[:,:3,3])
    pose0=axis_angle_to_matrix(g(pred['incam_body_pose'][slots]).reshape(n,21,3))
    beta0=g(np.median(pred['incam_betas'][slots],axis=0))[None]
    mapping=torch.load(REPO/'hmr4d/utils/body_model/smplx2smpl_sparse.pt',weights_only=True,map_location='cpu').to_dense()
    reg=g((torch.load(REPO/'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',weights_only=True,map_location='cpu')@mapping).numpy())
    # New-video route: shared mean hand pose initializes finger variables.
    # All 42 freshly detected hand landmarks remain observations in the fit.
    # No HaMeR output, old hand fit or previous subject file is read.
    left0=torch.zeros((n,45),device='cuda')
    right0=torch.zeros((n,45),device='cuda')
    # Scores are arbitrary detector outputs. Relative score and equivariance
    # enter a reliability heuristic; never report these as visibility probability.
    h=obs['boxes'][:,3]-obs['boxes'][:,1];norm=h/1000
    q90=np.maximum(np.quantile(obs['scores'],.9,axis=0),1e-6)
    rel=np.clip(obs['scores']/q90,0,1)**2
    flip=np.exp(-.5*(obs['flip_disagreement']/(.012*h[:,None]))**2)
    rel*=flip
    diff=obs['keypoints'][:,:17]-vit[:,:,:2]
    # Anisotropic uncertainty: disagreement weakens its direction, rather than
    # treating the two correlated learned detectors as two ground truths.
    sigma=(.012*h)[:,None,None,None]
    covariance=sigma**2*np.eye(2)[None,None]+.25*diff[:,:,:,None]*diff[:,:,None,:]
    whiten=g(np.linalg.inv(np.linalg.cholesky(covariance)))
    body_target=g((obs['keypoints'][:,:17]+vit[:,:,:2])*.5)
    vw=np.clip(vit[:,:,2]/np.maximum(np.quantile(vit[:,:,2],.9,axis=0),1e-6),0,1)**2
    body_base=rel[:,:17]*vw
    body_base[:,:5]*=.25
    body_base[:,11:13]*=.2 # COCO hips under clothing have ambiguous semantics.
    camw=np.array([r.get('observation_weight',1. if r['camera_kind']=='calibrated' else .5) for r in rows])
    body_base*=camw[:,None]
    support=np.ones((n,133),np.float32)
    body_support=np.ones((n,17),np.float32)
    if support_mode=='mask_margin_4px':
        from .observation_support import image_support
        for i,r in enumerate(rows):
            assert c.sha(r['mask'])==r['mask_sha256']
            mask=np.asarray(Image.open(r['mask']).convert('L'))>127
            assert list(mask.shape[::-1])==r['wh']
            support[i]=image_support(obs['keypoints'][i],mask)
            body_support[i]=image_support(body_target[i].detach().cpu().numpy(),mask)
        body_base*=body_support
    feet_ids=[JOINT_NAMES.index(x) for x in ['left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']]
    hand_names=['wrist','thumb1','thumb2','thumb3','thumb','index1','index2','index3','index','middle1','middle2','middle3','middle','ring1','ring2','ring3','ring','pinky1','pinky2','pinky3','pinky']
    hand_ids=[JOINT_NAMES.index(side+'_'+name) for side in ['left','right'] for name in hand_names]
    # Only fixed static landmark correspondence: IBUG17..67, no face contour.
    face_ids=model.faces_tensor[model.lmk_faces_idx];face_bary=model.lmk_bary_coords
    face_weight=rel[:,40:91].copy();face_weight[:,:10]=0 # exclude brow/eye and glasses areas
    face_weight[:,19:31]=0
    face_weight*=camw[:,None]
    feet_weight=rel[:,17:23]*camw[:,None];hands_weight=rel[:,91:133]*camw[:,None]
    feet_weight*=support[:,17:23];hands_weight*=support[:,91:133];face_weight*=support[:,40:91]
    bw=g(body_base);fw=g(feet_weight);hw=g(hands_weight);facw=g(face_weight)
    target=g(obs['keypoints']);eg,kg=g(E),g(K);dt=g(np.diff(times));R0g,t0g=g(R0),g(t0)
    param=lambda shape:torch.nn.Parameter(torch.zeros(shape,device='cuda'))
    rv,tv,sv=param((1,3)),param((1,3)),param(())
    shared,temporal,bd=param((1,21,3)),torch.zeros((1,21,3),device='cuda'),param((1,10))
    hd=param((1,2,45))
    vposer,_,vmeta=load_vposer()
    with torch.no_grad():z0=vposer.encode(g(pred['incam_body_pose'][slots])).mean
    latent=torch.nn.Parameter(z0.mean(0,keepdim=True).clone())
    R0g=g(Rotation.from_matrix(R0).mean().as_matrix())[None]
    t0g=g(np.median(t0,axis=0))[None]
    pose0=g(np.stack([Rotation.from_matrix(pose0[:,j].cpu().numpy()).mean().as_matrix() for j in range(21)]))[None]
    left0=left0.median(0,keepdim=True).values;right0=right0.median(0,keepdim=True).values
    torch.manual_seed(6060)
    groups=[{'params':[rv,tv,sv],'lr':.007,'kind':'root'}, {'params':[shared,bd,hd,latent],'lr':.004,'kind':'body'}]
    opt=torch.optim.Adam(groups)
    for group in opt.param_groups:group['initial_lr']=group['lr']
    # Containment on observed masks only: it does not pull bare skin to clothing.
    mask_indices=[];dist=[]
    for i,r in enumerate(rows):
        if r['mask'] is None:continue
        assert c.sha(r['mask'])==r['mask_sha256']
        mask=np.array(Image.open(r['mask']).convert('L'))>127
        df=distance_transform_edt(~mask).astype(np.float32)
        dist.append(np.array(Image.fromarray(df).resize((288,512),Image.Resampling.BILINEAR)));mask_indices.append(i)
    mi=torch.tensor(mask_indices,device='cuda');dist=g(np.stack(dist))[:,None];wh=g(np.array([r['wh'] for r in rows]))
    dominant=model.lbs_weights.argmax(-1)
    containment=((dominant<20)|((dominant>=22)&(dominant<25))).nonzero()[:,0]
    robust=lambda x:torch.sqrt(1+x.square())-1
    def project(x):
        cam=x@eg[:,:3,:3].transpose(1,2)+eg[:,:3,3][:,None];q=cam@kg.transpose(1,2)
        return q[...,:2]/q[...,2:].clamp_min(.01)-.5,cam
    def forward():
        corr=shared+temporal
        pr=axis_angle_to_matrix(corr)@pose0;pose=matrix_to_axis_angle(pr).reshape(1,63)
        beta=beta0+bd;lh=left0+.25*hd[:,0].tanh();rh=right0+.25*hd[:,1].tanh()
        body=model(body_pose=pose,betas=beta,left_hand_pose=lh,right_hand_pose=rh)
        p=body.joints[:,:1];R=axis_angle_to_matrix(rv)@R0g;s=s0*torch.exp(sv);t=t0g+s0*tv
        world=lambda x:((x-p)@R.transpose(1,2)*s+t[:,None]).expand(n,-1,-1)
        v,j=world(body.vertices),world(body.joints);co=world(torch.einsum('jv,nvc->njc',reg,body.vertices))
        uv,cam=project(co);uj,_=project(j)
        face=(v[:,face_ids]*face_bary[None,:,:,None]).sum(2);uf,_=project(face)
        return dict(v=v,j=j,co=co,uv=uv,uj=uj,uf=uf,cam=cam,R=R.expand(n,-1,-1),t=t.expand(n,-1),s=s,body=pose.expand(n,-1),betas=beta,pr=pr.expand(n,-1,-1,-1),lh=lh.expand(n,-1),rh=rh.expand(n,-1),pelvis=p.expand(n,-1,-1),corr=corr)
    # Generic self-occlusion support from fresh current mesh surface, not frame
    # numbers. This is model-derived visibility and is stored for review.
    ctx=dr.RasterizeCudaContext();faces=model.faces_tensor.int()
    regions=[15,15,15,15,15,16,17,18,19,20,21,1,2,4,5,7,8]
    regional=(model.lbs_weights[:,regions]>.20).float()
    @torch.no_grad()
    def visibility(st):
        visible=[]
        for i in range(n):
            v=st['v'][i]@eg[i,:3,:3].T+eg[i,:3,3];z=v[:,2]
            w,himg=wh[i];k=kg[i]
            clip=torch.stack((2*k[0,0]/w*v[:,0]+(2*k[0,2]/w-1)*z,-2*k[1,1]/himg*v[:,1]+(1-2*k[1,2]/himg)*z,1.00002*z-.0200002,z),-1)
            rast,_=dr.rasterize(ctx,clip[None],faces,(512,288))
            seenfaces=torch.unique(rast[0,:,:,3].long());seenfaces=seenfaces[seenfaces>0]-1
            seen=torch.zeros(len(model.v_template),device='cuda');seen[torch.unique(faces[seenfaces].long())]=1
            fraction=(seen[:,None]*regional).sum(0)/regional.sum(0).clamp_min(1)
            visible.append((fraction/.20).clamp(.05,1.))
        return torch.stack(visible)
    vis=visibility(forward())
    def objective(st):
        e=torch.einsum('njab,njb->nja',whiten,st['uv']-body_target)
        w=bw*vis;body=(robust(e).sum(-1)*w).sum()/w.sum().clamp_min(1)
        wl=lambda p,y,w,s:((robust((p-y)/s).sum(-1))*w).sum()/w.sum().clamp_min(1)
        scale=g(norm)[:,None,None]
        feet=wl(st['uj'][:,feet_ids],target[:,17:23],fw,15*scale)
        hand=wl(st['uj'][:,hand_ids],target[:,91:133],hw,8*scale)
        face=wl(st['uf'],target[:,40:91],facw,6*scale)
        vu,_=project(st['v'][:,containment]);grid=2*vu[mi]/(wh[mi,None]-1)-1
        outside=torch.nn.functional.grid_sample(dist,grid[:,:,None],align_corners=True,padding_mode='border')[:,0,:,0]
        contain=robust(torch.relu(outside-5)/15).mean()
        decoded=vposer.decode(latent)['pose_body_matrot'].reshape(1,21,3,3).expand(n,-1,-1,-1)
        vp=(torch.sqrt(1+((st['pr']-decoded).square().sum((-1,-2))/2)/.2**2)-1).mean()+.01*latent.square().mean()
        prior=robust(st['corr']/.25).mean()+.1*robust(rv/.35).mean()
        shape=bd.square().mean()
        x=st['j'][:,:22]/st['s'];vel=(x[1:]-x[:-1])/dt[:,None,None]
        acc=(vel[1:]-vel[:-1])/(.5*(dt[1:]+dt[:-1]))[:,None,None]
        smooth=robust(acc/1.).mean()
        contact=st['s']*0
        positive=100*torch.relu(.05-st['cam'][...,2]).square().mean()
        loss=body+.2*feet+.4*hand+.15*face+2*contain+.15*prior+.10*vp+.04*shape+.02*smooth+0*contact+.03*hd.tanh().square().mean()+positive
        return loss,dict(body=body,feet=feet,hands=hand,face=face,containment=contain,pose_prior=prior,vposer=vp,shape_prior=shape,world_acceleration=smooth,contact=contact)
    protocol=dict(created=c.stamp(),steps=steps,frames=n,all_frames_in_each_update=True,
        calibrated_cameras=sum(r['camera_kind']=='calibrated' for r in rows),
        registered_cameras=sum(r['camera_kind']=='registered_background_pnp' for r in rows),
        full_scene_sfm_cameras=sum(r['camera_kind']=='fresh_full_scene_sfm' for r in rows),
        interpolated_cameras=sum('interpolated' in r['camera_kind'] for r in rows),
        non_original_sfm_observation_weight=1.0,
        weight_policy='Every directly registered new-video frame has camera weight1.0; detector uncertainty is separate.',
        fresh_official_predictions_sha256=c.sha(root/'inference/prediction-postprocessed.npz'),
        observations_sha256=c.sha(root/'data/observations.npz'),input_sha256=c.sha(root/'data/inputs.json'),
        hand_initialization='SMPL-X mean; this-video 2D hand landmarks',source_sha256=c.sha(__file__),template_sha256=c.sha(c.MODEL),
        personalized_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,
        shared_identity=True,per_frame_visibility_hardcoding=False,bare_body_is_not_clothing_surface=True,
        landmark_image_support=support_mode,mask_margin_px=4. if support_mode=='mask_margin_4px' else None,
        landmark_coordinates_changed=False,mask_support_is_not_anatomical_visibility=True,
        deformation='One shared static pose/shape/root/translation for every camera; no per-frame parameters. Old floor term disabled because no fresh floor-world mapping is assumed',vposer=vmeta,
        limitations=['2D learned features and estimated camera poses are uncertain','Single moving camera does not observe simultaneous multiview geometry','This sequence is not an independent robustness benchmark'],geometry_accepted=False)
    c.save(out/'protocol.json',protocol)
    cpu=lambda x:x.detach().cpu().numpy()
    with torch.no_grad():
        initial=forward();np.savez_compressed(out/'initialization.npz',frame_ids=obs['frame_ids'],vertices_world=cpu(initial['v']),joints_world=cpu(initial['j']),body_pose=cpu(initial['body']),betas=cpu(initial['betas']),faces=model.faces)
    history=[];start=time.monotonic()
    for step in range(steps+1):
        for group in opt.param_groups:group['lr']=group['initial_lr']*(1. if group['kind']=='root' or step>=150 else 0.)*(.35 if step>steps*.8 else 1.)
        st=forward()
        if step and step%300==0:vis=visibility(st)
        loss,terms=objective(st);assert torch.isfinite(loss)
        if step%100==0 or step==steps:
            error=torch.linalg.norm(st['uv']-body_target,dim=-1)
            rec=dict(created=c.stamp(),step=step,steps=steps,loss=float(loss.detach()),seconds=time.monotonic()-start,
                weighted_observation_median_px=float(error[(bw*vis)>.25].detach().median()),**{k:float(v.detach()) for k,v in terms.items()})
            history.append(rec);c.save(out/'progress.json',rec);print(json.dumps(rec),flush=True)
        if step==steps:break
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']],10.,error_if_nonfinite=True);opt.step()
    with torch.no_grad():
        st=forward()
        np.savez_compressed(out/'fitted.npz',frame_ids=obs['frame_ids'],vertices_world=cpu(st['v']),joints_world=cpu(st['j']),
            coco17_world=cpu(st['co']),projected_coco17=cpu(st['uv']),projected_joints=cpu(st['uj']),projected_face51=cpu(st['uf']),
            root_rotation=cpu(st['R']),world_translation=cpu(st['t']),world_scale=cpu(st['s']),body_pose=cpu(st['body']),betas=cpu(st['betas']),
            left_hand_pose=cpu(st['lh']),right_hand_pose=cpu(st['rh']),pelvis=cpu(st['pelvis']),flat_hand_mean=False,faces=model.faces)
        np.savez_compressed(out/'observation-weights.npz',body=cpu(bw),visibility=cpu(vis),feet=cpu(fw),hands=cpu(hw),face=cpu(facw),whiten=cpu(whiten),body_target=cpu(body_target))
        if support_mode=='mask_margin_4px':
            np.savez_compressed(out/'image-support.npz',wholebody=support,body_target=body_support)
    c.save(out/'history.json',history)
    c.save(out/'result.json',dict(created=c.stamp(),state='fresh_shared_static_fit_visual_review_pending',body_steps=steps,gaussian_steps=0,
        frames=n,seconds=time.monotonic()-start,fitted_sha256=c.sha(out/'fitted.npz'),geometry_accepted=False,quality_accepted=False))


