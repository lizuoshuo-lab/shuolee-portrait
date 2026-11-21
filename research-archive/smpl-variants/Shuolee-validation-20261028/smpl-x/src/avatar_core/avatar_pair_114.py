"""Explicit static adapters for two different, pinned human repositories.

No legacy learned person is read. This module does not claim that the missing
upstream SMPL-X trainer has been reproduced. Both arms share raw inputs,
projection, foreground supervision and review cameras, not model parameters.
"""
import argparse, ast, json, math, os, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageDraw
import shuolee_common as c
torch=c.runtime()
F=torch.nn.functional
from scene.gaussian_model import GaussianModel
from scene.smplx_gaussian_model import SMPLXGaussianModel
from scene.cameras import CameraProj2
from gaussian_renderer import render
from arguments import OptimizationParams

PAIR=c.BASE/'current/avatar-repository-pair-114'
GA=c.LAB/'repos/GaussianAvatar'
PUBLIC=c.BASE.parent/'viewer/body-avatar114'
PIPE=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
BG=torch.zeros(3,device='cuda')
G=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
SELECT=[0,33,69,99,135,174]

def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}

def uv_samples(vt,ft,size):
    tid=np.full((size,size),-1,np.int32);bc=np.zeros((size,size,3),np.float32)
    for i,t in enumerate(vt[ft]*size-.5):
        lo=np.maximum(np.ceil(t.min(0)).astype(int),0);hi=np.minimum(np.floor(t.max(0)).astype(int),size-1)
        if np.any(lo>hi):continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];pts=np.stack([xx,yy],-1)
        A=np.stack([t[0]-t[2],t[1]-t[2]],1)
        if abs(np.linalg.det(A))<1e-10:continue
        w=(pts-t[2])@np.linalg.inv(A).T;b=np.concatenate([w,1-w.sum(-1,keepdims=True)],-1)
        ok=(b>=-1e-5).all(-1);tid[yy[ok],xx[ok]]=i;bc[yy[ok],xx[ok]]=b[ok]
    return tid,bc

def prepare_views(root,height=1536):
    rows=c.read(root/'data/training-inputs.json')['records'];assert len(rows)==175
    views=[];records=[];errors=[]
    for i,row in enumerate(rows):
        assert row['camera_kind']=='fresh_full_scene_sfm'
        assert c.sha(row['image'])==row['source_sha256'] and c.sha(row['mask'])==row['mask_sha256']
        im=Image.open(row['image']).convert('RGB');ma=Image.open(row['mask']).convert('L')
        assert im.size==ma.size==tuple(row['wh'])
        w=round(im.width*height/im.height);h=height
        K=np.diag([w/im.width,h/im.height,1.])@np.array(row['K']);E=np.array(row['w2c'])
        assert np.linalg.norm(E[:3,:3]@E[:3,:3].T-np.eye(3))<1e-5 and np.linalg.det(E[:3,:3])>.999
        cam=CameraProj2(i,E[:3,:3].T,E[:3,3],K[0,0],K[1,1],'black',w,None,h,row['image'],row['frame'],i,cx=K[0,2],cy=K[1,2],timestep=0)
        cp=np.array([[0.,0.,2.],[.1,.2,3.],[-.4,-.3,4.]])
        world=(cp-E[:3,3])@E[:3,:3];q=cp@K.T;uv=q[:,:2]/q[:,2:]-.5
        ndc=np.c_[world,np.ones(3)]@cam.full_proj_transform.numpy()
        actual=((ndc[:,:2]/ndc[:,3:]+1)*[w,h]-1)/2
        err=float(abs(uv-actual).max());assert err<.001,(i,err);errors.append(err)
        rgb=torch.from_numpy(np.array(im.resize((w,h),Image.Resampling.LANCZOS))).permute(2,0,1)
        alpha=torch.from_numpy(np.array(ma.resize((w,h),Image.Resampling.BILINEAR)))[None]
        views.append((cam,rgb,alpha));records.append(dict(frame=row['frame'],source_frame_index=row['source_frame_index'],K=K.tolist(),w2c=E.tolist(),wh=[w,h],crop=None))
    c.save(root/'gaussian-run/training-cameras.json',dict(records=records,max_projection_difference_px=max(errors),interpolated_cameras=0))
    return rows,views

def target(view):
    cam,rgb,alpha=view
    return cam,rgb.cuda().float()/255,alpha.cuda().float()/255

def foreground_loss(im,gt,alpha):
    valid=alpha>.99
    support=F.max_pool2d(F.pad((~valid).float()[None],(5,5,5,5),value=1),11,stride=1)[0]==0
    a=torch.where(valid,im,0);b=torch.where(valid,gt,0)
    q=torch.arange(11,device=im.device,dtype=im.dtype)-5
    kernel=torch.exp(-q.square()/(2*1.5**2));kernel=kernel/kernel.sum()
    win=(kernel[:,None]*kernel[None,:])[None,None].expand(3,1,11,11).contiguous()
    conv=lambda x:F.conv2d(x[None],win,padding=5,groups=3)[0]
    u,v=conv(a),conv(b);aa=conv(a*a)-u*u;bb=conv(b*b)-v*v;ab=conv(a*b)-u*v
    ss=((2*u*v+.01**2)*(2*ab+.03**2))/((u*u+v*v+.01**2)*(aa+bb+.03**2))
    l1=(im-gt).abs()[:,valid[0]].mean();ssim=ss[:,support[0]].mean()
    assert valid.sum()>1000 and support.sum()>1000
    return .8*l1+.2*(1-ssim),l1,ssim

def neural_geometry(root,count,seed):
    import smplx
    fit=load(root/'fit/fitted.npz');assert np.max(abs(fit['vertices_world']-fit['vertices_world'][0]))==0
    model=smplx.SMPLX(str(c.MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False).cuda()
    kw=dict(betas=G(fit['betas']),left_hand_pose=G(fit['left_hand_pose'][:1]),right_hand_pose=G(fit['right_hand_pose'][:1]),return_full_pose=True)
    with torch.no_grad():
        live=model(body_pose=G(fit['body_pose'][:1]),**kw)
        can_body=torch.zeros((1,63),device='cuda');can_body[0,2]=math.pi/6;can_body[0,5]=-math.pi/6
        can=model(body_pose=can_body,**kw)
    source=GA/'submodules/smplx/lbs.py'
    names={'lbs','vertices2joints','blend_shapes','batch_rodrigues','transform_mat','batch_rigid_transform'}
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    env=dict(torch=torch,np=np,F=F,Tensor=torch.Tensor,Tuple=Tuple,List=List)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    coeff=torch.cat([G(fit['betas']),torch.zeros(1,10,device='cuda')],1)
    dirs=torch.cat([model.shapedirs,model.expr_dirs],-1)
    def lbs(p):return env['lbs'](coeff,p,model.v_template,dirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_affine_mat=True)
    with torch.no_grad():
        vc,_,Ac=lbs(can.full_pose);vl,_,Al=lbs(live.full_pose)
        assert float(abs(vl-live.vertices).max())<1e-5
        relative=(Al@torch.linalg.inv(Ac))[0]
        s=float(fit['world_scale']);R=G(fit['root_rotation'][0]);p=G(fit['pelvis'][0]);t=G(fit['world_translation'][0])
        recon=(vl[0]-p)@R.T*s+t
        error=float(abs(recon-G(fit['vertices_world'][0])).max());assert error<2e-5,error
        with np.load(c.MODEL) as atlas_file:atlas={k:atlas_file[k].copy() for k in ['vt','ft']}
        size=524
        tid,bary=uv_samples(atlas['vt'],atlas['ft'],size)
        valid=np.flatnonzero(tid.ravel()>=0)
        assert len(valid)>=count,(size,len(valid))
        chosen=np.sort(np.random.default_rng(seed).choice(valid,count,replace=False))
        fi=tid.ravel()[chosen];bc=G(bary.reshape(-1,3)[chosen]);faces=model.faces_tensor[torch.from_numpy(fi).cuda().long()]
        weights=(model.lbs_weights[faces]*bc[...,None]).sum(1)
        mats=torch.einsum('nj,jkl->nkl',weights,relative)
        L=(s*R)[None]@mats[:,:3,:3]
        T=(mats[:,:3,3]-p)@R.T*s+t
        base=(G(fit['vertices_world'][0])[faces]*bc[...,None]).sum(1)
        # Static pose-corrective base: inverse LBS places every UV query exactly
        # on this run's posed mesh. No learned per-frame deformation is added.
        query=torch.linalg.solve(L,(base-T)[...,None])[...,0]
        roundtrip=float(abs(torch.einsum('nij,nj->ni',L,query)+T-base).max());assert roundtrip<2e-5
        yy,xx=torch.meshgrid(torch.arange(size,device='cuda'),torch.arange(size,device='cuda'),indexing='ij')
        uv=torch.stack([yy,xx],-1).reshape(1,-1,2).float()/(size-1)
        audit=dict(query_resolution=size,valid_UV_texels=len(valid),sampled_points=count,covered_triangles=len(np.unique(fi)),canonical_leg_angle_deg=30,fit_reconstruction_max_abs=error,UV_LBS_roundtrip_max_abs=roundtrip,static_pose_corrective_base=True,old_fit_read=False)
        c.save(root/'gaussian-run/geometry-audit.json',audit)
        np.savez_compressed(root/'gaussian-run/uv-samples.npz',indices=chosen,triangle_ids=fi,barycentric=bc.cpu().numpy())
    return dict(query=query,L=L,T=T,uv=uv,indices=torch.from_numpy(chosen).cuda(),world_scale=s,audit=audit)

class NeuralArm:
    def __init__(self,root,count,seed):
        self.geometry=neural_geometry(root,count,seed)
        sys.path.append(str(GA))
        from model.network import POP_no_unet
        self.net=POP_no_unet(c_geom=64,geom_layer_type='conv',nf=32,hsize=128).cuda()
        self.features=torch.nn.Parameter(torch.randn(1,64,128,128,device='cuda')*.01)
        self.optimizer=torch.optim.Adam([{'params':self.net.parameters(),'lr':.003},{'params':[self.features],'lr':.0005}])
        self.model=GaussianModel(3);self.model.active_sh_degree=0
        self.model._rotation=torch.zeros(count,4,device='cuda');self.model._rotation[:,0]=1
        self.model._opacity=torch.full((count,1),20.,device='cuda')
        self.model._features_rest=torch.zeros(count,15,3,device='cuda')
    def predict(self,step):
        d=self.geometry;res,scale,color=self.net(None,self.features,d['uv']);idx=d['indices']
        self.offset=res.permute(0,2,1)[0,idx]*.02
        self.raw_scale=scale.permute(0,2,1)[0,idx]*min(max(step,1)/1000.,1.)
        xyz=torch.einsum('nij,nj->ni',d['L'],d['query']+self.offset)+d['T']
        self.model._xyz=xyz.contiguous()
        self.model._scaling=(self.raw_scale.expand(-1,3)*d['world_scale']).clamp_min(1e-8).log().contiguous()
        self.model._features_dc=((color.permute(0,2,1)[0,idx]-.5)/.28209479177387814)[:,None].contiguous()
        return self.model
    def regularizer(self,step):
        epoch=step//175
        return 10*(.85**(epoch//20))*self.offset.square().mean()+.03*self.raw_scale.mean()+self.features.square().mean()
    def schedule(self,step):
        factor=.1**((step>6000)+(step>12000))
        self.optimizer.param_groups[0]['lr']=.003*factor;self.optimizer.param_groups[1]['lr']=.0005*factor
    def checkpoint(self):
        return dict(net=self.net.state_dict(),features=self.features.detach(),optimizer=self.optimizer.state_dict())

def pil(x):return Image.fromarray(np.rint(x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

@torch.no_grad()
def review(root,arm,model,views,rows,step):
    out=root/'gaussian-run';public=PUBLIC/arm;public.mkdir(parents=True,exist_ok=True)
    tiles=[];face_tiles=[];metrics=[]
    observations=load(root/'data/observations.npz')
    for i in SELECT:
        cam,gt,alpha=target(views[i]);im=render(cam,model,PIPE,BG)['render'];a=render(cam,model,PIPE,BG,override_color=torch.ones_like(model.get_xyz))['render'][:1]
        fg=alpha[0]>.99
        row=rows[i];scale=np.array([cam.image_width/row['wh'][0],cam.image_height/row['wh'][1]])
        kp=observations['keypoints'][i,23:91]*scale
        lo=np.floor(kp.min(0)-[32,50]).astype(int);hi=np.ceil(kp.max(0)+[32,30]).astype(int)
        lo=np.maximum(lo,0);hi=np.minimum(hi,[cam.image_width,cam.image_height]);box=tuple(np.r_[lo,hi])
        gtcomp=gt*alpha;diff=(im-gtcomp).abs()
        images=[pil(gtcomp),pil(im),pil(diff)]
        panels=[]
        for image in images:
            panels.append(image.resize((round(image.width*440/image.height),440)))
        tile=Image.new('RGB',(sum(p.width for p in panels),466),'white');draw=ImageDraw.Draw(tile)
        draw.text((6,5),f'{arm} / {row["frame"]} / step {step}: INPUT | RENDER | ABS ERROR',fill='black')
        x=0
        for p in panels:tile.paste(p,(x,26));x+=p.width
        tiles.append(tile)
        heads=[x.crop(box).resize((300,300)) for x in images]
        face=Image.new('RGB',(900,326),'white');ImageDraw.Draw(face).text((6,5),f'{arm} / {row["frame"]} / step {step}',fill='black')
        for j,h in enumerate(heads):face.paste(h,(300*j,26))
        face.save(out/f'face-{i+1:06d}.jpg',quality=94);face_tiles.append(face)
        imfile=out/f'render-{i+1:06d}.png';pil(im).save(imfile)
        metrics.append(dict(frame=row['frame'],foreground_mae=float((im-gt).abs()[:,fg].mean()),outside_alpha=float(a[~(alpha>.01)].mean()),inside_missing_alpha=float((1-a)[alpha>.99].mean()),mask_iou=float(((a>.5)&(alpha>.5)).sum()/((a>.5)|(alpha>.5)).sum().clamp_min(1)),head_box=list(map(int,box))))
    for name,items in [('review',tiles),('heads',face_tiles)]:
        width=max(t.width for t in items);height=max(t.height for t in items)
        sheet=Image.new('RGB',(width*2,height*3),'white')
        for j,tile in enumerate(items):sheet.paste(tile,((j%2)*width,(j//2)*height))
        sheet.save(out/f'{name}-current.jpg',quality=94);sheet.save(public/f'{name}-current.jpg',quality=94)
    result=dict(step=step,points=len(model.get_xyz),records=metrics,trained_views=True,quality_accepted=False)
    c.save(out/'review.json',result);c.save(public/'review.json',result)

def train(root,arm,preflight_only=False):
    out=root/'gaussian-run';out.mkdir(exist_ok=True);(PUBLIC/arm).mkdir(parents=True,exist_ok=True)
    assert not (out/'current.pt').exists(),'Fresh only: refuses previous network or optimizer'
    fit=load(root/'fit/fitted.npz');assert np.max(abs(fit['vertices_world']-fit['vertices_world'][0]))==0
    torch.manual_seed(6060);torch.cuda.manual_seed_all(6060);np.random.seed(6060)
    rows,views=prepare_views(root);source_hash=c.sha(__file__)
    neural=None;options=None
    if arm=='gaussianavatar':
        neural=NeuralArm(root,200000,6060);model=neural.predict(1);optimizer=neural.optimizer
        init=neural.geometry['audit']
    else:
        model=SMPLXGaussianModel(3);model.load_fresh_meshes(root/'fit/fitted.npz');init=model.initialize_surface(200000,6060)
        parser=argparse.ArgumentParser();options=OptimizationParams(parser).extract(parser.parse_args([]))
        options.iterations=18000;options.position_lr_max_steps=18000;options.densify_until_iter=18000
        model.training_setup(options);optimizer=model.optimizer
    config=dict(created=c.stamp(),arm=arm,from_scratch=True,initial_points=200000,steps=18000,seed=6060,source_sha256=source_hash,fit_sha256=c.sha(root/'fit/fitted.npz'),inputs_sha256=c.sha(root/'data/training-inputs.json'),shared_static=True,no_old_checkpoints=True,RGB='0.8 L1 + 0.2 SSIM, complete interior windows only',soft_mask_alpha_weight=.1,initialization=init,LPIPS=False,LPIPS_omission='Shared foreground-only comparison objective; differs from native GaussianAvatar stage1 LPIPS schedule.',quality_accepted=False)
    c.save(out/'config.json',config)
    cam,gt,alpha=target(views[0]);pkg=render(cam,model,PIPE,BG)
    photo,_,_=foreground_loss(pkg['render'],gt,alpha)
    other=gt.clone();other[:,alpha[0]<=.99]=.37
    check=foreground_loss(pkg['render'],other,alpha)[0]
    outside_effect=float(abs(photo-check));assert outside_effect<1e-7
    a=render(cam,model,PIPE,BG,override_color=torch.ones_like(model.get_xyz))['render'][:1]
    loss=photo+.1*(a-alpha).abs().mean();loss.backward()
    params=[p for group in optimizer.param_groups for p in group['params']]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert int(pkg['visibility_filter'].sum())>1000
    optimizer.zero_grad(set_to_none=True)
    # Read back an actual standard world PLY through the native renderer.
    with torch.no_grad():
        model.save_ply(str(out/'preflight.ply')) if neural else model.export_world_ply(out/'preflight.ply',0)
        reloaded=GaussianModel(3);reloaded.load_ply(str(out/'preflight.ply'));reloaded.active_sh_degree=model.active_sh_degree
        reload_image=render(cam,reloaded,PIPE,BG)['render']
        export_error=float(abs(reload_image-pkg['render']).max());assert export_error<.001,export_error
    receipt=dict(created=c.stamp(),state='passed',finite_backward=True,foreground_external_RGB_effect=outside_effect,native_PLY_reload_max_difference=export_error,visible_points=int(pkg['visibility_filter'].sum()),optimizer_updates=0)
    c.save(out/'preflight.json',receipt)
    del pkg,a,loss,photo,check,other,reloaded,reload_image
    torch.cuda.empty_cache()
    if preflight_only:
        print(json.dumps(receipt),flush=True);return
    assert c.read(root/'fit/visual-review.json')['usable_as_coarse_static_initialization']
    assert c.sha(__file__)==source_hash
    state=c.read(c.BASE/'work-status.json');state.update(updated=c.stamp(),local_training_running=True,current_phase='avatar_repository_pair_114')
    state['latest_local_execution']=dict(round='114',arm=arm,state='training',run=str(root),running=True,steps=0,target_steps=18000,from_scratch=True,quality_accepted=False)
    c.save(c.BASE/'work-status.json',state)
    start=time.monotonic();rng=np.random.default_rng(6060);stack=[];counts=np.zeros(175,np.int32)
    extent=float(np.linalg.norm(np.ptp(fit['vertices_world'][0],axis=0)))
    with (out/'loss.jsonl').open('x') as logfile:
        for step in range(1,18001):
            if neural:neural.schedule(step);model=neural.predict(step)
            else:
                model.update_learning_rate(step)
                if step%1000==0:model.oneupSHdegree()
            if not stack:stack=rng.permutation(175).tolist()
            i=stack.pop();counts[i]+=1;cam,gt,alpha=target(views[i])
            pkg=render(cam,model,PIPE,BG);photo,l1,ss=foreground_loss(pkg['render'],gt,alpha)
            aa=render(cam,model,PIPE,BG,override_color=torch.ones_like(model.get_xyz))['render'][:1]
            silhouette=(aa-alpha).abs().mean()
            visible=pkg['visibility_filter']
            if neural:regularizer=neural.regularizer(step)
            else:
                regularizer=torch.relu(model._xyz[visible].norm(dim=-1)-options.threshold_xyz).mean()*options.lambda_xyz+torch.relu(model._scaling[visible].exp()-options.threshold_scale).norm(dim=-1).mean()*options.lambda_scale
            loss=photo+.1*silhouette+regularizer;assert torch.isfinite(loss),step
            loss.backward()
            with torch.no_grad():
                if not neural:
                    model.max_radii2D[visible]=torch.maximum(model.max_radii2D[visible],pkg['radii'][visible])
                    if step<18000:model.add_densification_stats(pkg['viewspace_points'],visible)
                if neural:torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if not neural and step>options.densify_from_iter and step<18000 and step%options.densification_interval==0:
                    model.densify_and_prune(options.densify_grad_threshold,.005,extent,None);optimizer=model.optimizer
                    assert len(model._xyz)<1200000 and int(model.binding_counter.min())>=1
            if step==1 or step%100==0:
                progress=dict(created=c.stamp(),state='training',arm=arm,step=step,target_steps=18000,points=len(model.get_xyz),loss=float(loss),foreground_l1=float(l1),foreground_ssim=float(ss),alpha_mae=float(silhouette),regularizer=float(regularizer),seconds=time.monotonic()-start,views_seen=int((counts>0).sum()),peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,quality_accepted=False)
                c.save(out/'progress.json',progress);c.save(PUBLIC/arm/'progress.json',progress)
                logfile.write(json.dumps(progress)+'\n');logfile.flush();print(json.dumps(progress),flush=True)
            if step in [1000,6000,12000,18000]:
                with torch.no_grad():
                    if neural:neural.net.eval();model=neural.predict(step)
                    temp=out/'current.pt.tmp'
                    state=neural.checkpoint() if neural else dict(capture=model.capture())
                    torch.save(dict(**state,step=step,counts=counts,source_sha256=source_hash,fit_sha256=config['fit_sha256'],from_scratch=True),temp);os.replace(temp,out/'current.pt')
                    if neural:model.save_ply(str(out/'point_cloud.ply'))
                    else:model.export_world_ply(out/'point_cloud.ply',0)
                    review(root,arm,model,views,rows,step)
                    reloaded=GaussianModel(3);reloaded.load_ply(str(out/'point_cloud.ply'));reloaded.active_sh_degree=model.active_sh_degree
                    checks=[]
                    for view_id in SELECT:
                        cam=views[view_id][0];native=render(cam,model,PIPE,BG)['render'];loaded=render(cam,reloaded,PIPE,BG)['render'];delta=abs(native-loaded)
                        checks.append(dict(frame=rows[view_id]['frame'],mean_abs=float(delta.mean()),max_abs=float(delta.max())))
                    assert all(x['mean_abs']<2e-6 and x['max_abs']<.005 for x in checks),checks
                    c.save(out/'native-vs-ply.json',dict(step=step,checks=checks,passed=True,sha256=c.sha(out/'point_cloud.ply')))
                    del reloaded,native,loaded,delta
                    if neural:neural.net.train()
    final=dict(created=c.stamp(),state='completed_pending_same_camera_review',steps=18000,points=len(model.get_xyz),initial_points=200000,from_scratch=True,shared_static=True,per_frame_updates=counts.tolist(),seconds=time.monotonic()-start,ply_sha256=c.sha(out/'point_cloud.ply'),quality_accepted=False)
    c.save(out/'result.json',final);c.save(PUBLIC/arm/'progress.json',final);print(json.dumps(final),flush=True)
    state=c.read(c.BASE/'work-status.json');state.update(updated=c.stamp(),local_training_running=False)
    state['latest_local_execution']=dict(round='114',arm=arm,state=final['state'],run=str(root),running=False,steps=18000,from_scratch=True,quality_accepted=False)
    c.save(c.BASE/'work-status.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['gaussianavatar','smplx'],required=True);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    root=PAIR/args.arm
    try:train(root,args.arm,args.preflight_only)
    except Exception:
        (root/'gaussian-run').mkdir(exist_ok=True)
        (root/'gaussian-run/error.txt').write_text(traceback.format_exc());raise
