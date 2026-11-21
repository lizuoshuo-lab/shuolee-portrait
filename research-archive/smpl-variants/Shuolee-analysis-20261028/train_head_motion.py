"""Matched training control: fixed cameras and measured head/face motion."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONNOUSERSITE='1')
from pathlib import Path
import argparse,fcntl,json,sys,time,traceback
from types import SimpleNamespace
import numpy as np

REPO=Path('/mnt/d/Projects/Shuolee');sys.path.insert(0,str(REPO/'src'))
BASE=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
EVIDENCE=Path('/mnt/d/Projects/Shuolee-analysis-20261028')
arguments=argparse.ArgumentParser();arguments.add_argument('--motion',choices=['head','face','fan'],default='head');arguments.add_argument('--steps',type=int,default=6000)
args=arguments.parse_args()
OUT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029')/dict(head='gray-head-motion',face='gray-dynamic-face',fan='gray-dynamic-face-fan')[args.motion]
assert not OUT.exists(),'Preserve existing controls.'
OUT.mkdir(parents=True)
from shuolee_portrait import runtime as rt
rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',OUT)
rt.save(OUT/'status.json',dict(state='waiting_for_gpu',quality_accepted=False))
lock=Path('/tmp/shuolee-native122-gpu.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX)
lock.seek(0);lock.truncate();lock.write(json.dumps(dict(pid=os.getpid(),run=str(OUT),task='fixed-camera head motion control')));lock.flush()
try:
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from scipy.spatial import cKDTree
    from pytorch3d.transforms import axis_angle_to_matrix
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.general_utils import build_scaling_rotation,strip_symmetric
    from utils.sh_utils import eval_sh
    from arguments import OptimizationParams
    from shuolee_portrait.clothed_surface import make_model
    from shuolee_portrait.training import Views,foreground_loss,image

    rows=rt.read(BASE/'data/training-inputs.json')['records'];cfg=rt.read(BASE/'config.json')
    fit=dict(np.load(BASE/'fit/fitted.npz'));control=dict(np.load(EVIDENCE/'constrained-head-control/neck_head.npz'))
    template=np.load('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
    weights=template['weights'];head_weights=weights[:,[15,22,23,24]].sum(1);neck_weights=weights[:,12]
    tree=cKDTree(fit['vertices_world'][0]);g=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    par=g(control['parameters']);Rn=axis_angle_to_matrix(par[:,:3]);Rh=axis_angle_to_matrix(par[:,3:])
    center=g(control['center']);neck=g(control['neck']);eye=torch.eye(3,device='cuda')
    motion_source=EVIDENCE/'constrained-head-control/neck_head.npz'
    if args.motion!='head':
        motion_source=EVIDENCE/('dynamic-face-control' if args.motion=='face' else 'dynamic-face-fan-control')/'motion.npz';motion=dict(np.load(motion_source))
        vertex_sequence=g(motion['vertices_world']);affine_sequence=g(motion['vertex_affine']);reference_vertices=vertex_sequence[0]
    scale=float(fit['world_scale']);seed=6060
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    model=make_model(BASE)
    parser=argparse.ArgumentParser();options=OptimizationParams(parser).extract(parser.parse_args([]))
    options.iterations=18000;options.position_lr_max_steps=18000
    options.position_lr_init=.00016;options.position_lr_final=.0000016;options.scaling_lr=.005
    options.lambda_xyz=0.;options.lambda_scale=0.;options.densify_until_iter=18000
    model.training_setup(options)
    assert model.binding is None and not model.optimizer.state
    initial_xyz=model._xyz.detach().clone();low=initial_xyz.min(0).values-.1*scale;high=initial_xyz.max(0).values+.1*scale
    extent=float(torch.linalg.norm(initial_xyz.max(0).values-initial_xyz.min(0).values))
    views=Views(rows,cfg['training_height'],cfg['image_cache_frames'])
    pipe=SimpleNamespace(compute_cov3D_python=True,convert_SHs_python=False,debug=False);bg=torch.zeros(3,device='cuda')

    class Deformed:
        def __init__(self,base):self.base=base;self.weights=None;self.last_count=0
        def __getattr__(self,name):return getattr(self.base,name)
        def update_weights(self):
            xyz=self.base._xyz.detach().cpu().numpy()
            if args.motion!='head':
                dist,ids=tree.query(xyz,k=4,workers=4);w=1/(dist**2+(.015*scale)**2);w/=w.sum(1,keepdims=True)
                self.ids=torch.as_tensor(ids,dtype=torch.long,device='cuda');self.weights=g(w)
            else:
                _,ids=tree.query(xyz,workers=4);self.weights=(g(neck_weights[ids])[:,None],g(head_weights[ids])[:,None])
            self.last_count=len(xyz)
        def set_frame(self,index):
            if self.weights is None or self.last_count!=len(self.base._xyz):self.update_weights()
            if args.motion!='head':
                ids,w=self.ids,self.weights;A=affine_sequence[index+1][ids]
                v=vertex_sequence[index+1][ids];ref=reference_vertices[ids]
                self.affine=(A*w[:,:,None,None]).sum(1)
                translation=((v-torch.einsum('nkij,nkj->nki',A,ref))*w[:,:,None]).sum(1)
                self.xyz=torch.einsum('nij,nj->ni',self.affine,self.base._xyz)+translation
                self.index=index;return
            self.index=index;wn,wh=self.weights;x=self.base._xyz;rn,rh=Rn[index],Rh[index]
            self.affine=(1-wn-wh)[...,None]*eye+wn[...,None]*rn+wh[...,None]*(rn@rh)
            pn=(x-neck)@rn.T+neck
            ph=((x-center)@rh.T+center-neck)@rn.T+neck
            self.xyz=(1-wn-wh)*x+wn*pn+wh*ph
        @property
        def get_xyz(self):return self.xyz
        def get_covariance(self,scaling_modifier=1.):
            L=build_scaling_rotation(self.base.get_scaling*scaling_modifier,self.base.get_rotation)
            L=self.affine@L
            return strip_symmetric(L@L.transpose(1,2))
        def colors(self,cam):
            d=self.xyz-cam.camera_center.to(self.xyz.device)[None]
            d=torch.einsum('nji,nj->ni',self.affine,d)
            d=d/d.norm(dim=1,keepdim=True).clamp_min(1e-8)
            sh=self.base.get_features.transpose(1,2)
            return torch.clamp_min(eval_sh(self.base.active_sh_degree,sh,d)+.5,0.)
    moving=Deformed(model);moving.update_weights()

    # Verify the un-deformed wrapper reproduces canonical covariance and rendering.
    cam,gt,alpha=views.get(0)
    original_rn,original_rh=Rn[0].clone(),Rh[0].clone();Rn[0]=eye;Rh[0]=eye
    moving.set_frame(-1 if args.motion!='head' else 0)
    ordinary=SimpleNamespace(compute_cov3D_python=True,convert_SHs_python=False,debug=False)
    with torch.no_grad():
        a=render(cam,model,ordinary,bg)['render'];b=render(cam,moving,pipe,bg,override_color=moving.colors(cam))['render']
        error=float((a-b).abs().max())
        if error>=.001:raise RuntimeError(f'Identity deformation render mismatch: {error}')
    Rn[0]=original_rn;Rh[0]=original_rh
    # Face vertices must use the same complete head rotation as the point-fit control.
    tri=fit['faces'][template['lmk_faces_idx']]
    fwh=(head_weights[tri]*template['lmk_bary_coords']).sum(1)
    rt.save(OUT/'protocol.json',dict(frames=len(rows),target_steps=args.steps,learning_rate_schedule_steps=18000,
        source_base_run=str(BASE),initialization_sha256=rt.sha(BASE/'clothing-init/points.npz'),
        head_motion_sha256=rt.sha(motion_source),motion_source=str(motion_source),motion_kind=args.motion,
        cameras_sha256=rt.sha(BASE/'data/training-inputs.json'),cameras_changed=False,
        observation_sha256=rt.sha(BASE/'data/observations.npz'),from_fresh_gaussian_optimizer=True,
        reused_initial_point_cloud_for_matched_comparison=True,no_trained_gaussians_loaded=True,
        representation='Canonical free Gaussians with measured head/face deformation; no point attraction',
        covariance='Same local deformation affine applied to canonical covariance',
        color_direction='Canonical SH evaluated on normalized transpose-affine view direction',
        identity_render_max_abs=error,face_head_weight_min=float(fwh.min()),
        nearest_template_weights_updated_every_100_steps=True,quality_accepted=False))
    selected=np.unique(np.linspace(0,len(rows)-1,6).astype(int));counts=np.zeros(len(rows),int)
    rng=np.random.default_rng(seed);stack=[];start=time.monotonic();target_steps=args.steps
    log=(OUT/'loss.jsonl').open('x')
    for step in range(1,target_steps+1):
        model.update_learning_rate(step)
        if step%1000==0:model.oneupSHdegree()
        if not stack:stack=rng.permutation(len(rows)).tolist()
        i=stack.pop();counts[i]+=1;cam,gt,alpha=views.get(i)
        if step%100==1:moving.update_weights()
        moving.set_frame(i)
        pkg=render(cam,moving,pipe,bg,override_color=moving.colors(cam));photo,l1,ssim=foreground_loss(pkg['render'],gt,alpha)
        aa=render(cam,moving,pipe,bg,override_color=torch.ones_like(model.get_xyz))['render'][:1]
        silhouette=(aa-alpha).abs().mean();visible=pkg['visibility_filter']
        bounds=(torch.relu(low-model._xyz)+torch.relu(model._xyz-high)).mean()/scale
        large=torch.relu(model.get_scaling/scale-.03).square().mean()
        loss=photo+.1*silhouette+.01*bounds+.1*large
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        with torch.no_grad():
            model.max_radii2D[visible]=torch.maximum(model.max_radii2D[visible],pkg['radii'][visible])
            model.add_densification_stats(pkg['viewspace_points'],visible)
            model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
            if options.densify_from_iter<step<18000 and step%options.densification_interval==0:
                model.densify_and_prune(options.densify_grad_threshold,.005,extent,None)
                if not 1000<len(model._xyz)<650000:raise RuntimeError('Point budget exceeded')
                moving.weights=None
        if step==1 or step%100==0:
            progress=dict(state='training',step=step,target_steps=target_steps,points=len(model._xyz),
                loss=float(loss),foreground_l1=float(l1),foreground_ssim=float(ssim),alpha_mae=float(silhouette),
                seconds=time.monotonic()-start,views_seen=int((counts>0).sum()),quality_accepted=False)
            rt.save(OUT/'status.json',progress);log.write(json.dumps(progress)+'\n');log.flush()
            if step%500==0 or step==1:print(json.dumps(progress),flush=True)
        if step in [1000,6000,12000,target_steps]:
            snap=OUT/f'step-{step:06d}';snap.mkdir()
            model.save_ply(str(snap/'canonical.ply'))
            with torch.no_grad():
                moving.update_weights()
                for j in selected:
                    cam,gt,alpha=views.get(int(j));moving.set_frame(int(j))
                    result=render(cam,moving,pipe,bg,override_color=moving.colors(cam))['render']
                    image(gt).save(snap/f'{rows[j]["frame"]}_input.png')
                    image(gt*alpha).save(snap/f'{rows[j]["frame"]}_masked_input.png')
                    image(result).save(snap/f'{rows[j]["frame"]}_render.png')
                rt.save(snap/'metadata.json',dict(step=step,canonical_ply_sha256=rt.sha(snap/'canonical.ply'),
                    dynamic_views_require_recorded_head_motion=True,quality_accepted=False))
    log.close()
    rt.save(OUT/'result.json',dict(state='completed',steps=target_steps,frames=len(rows),per_frame_updates=counts.tolist(),
                                 canonical_ply=f'step-{target_steps:06d}/canonical.ply',quality_accepted=False))
    rt.save(OUT/'status.json',dict(state='completed',steps=target_steps,quality_accepted=False))
    print('HEAD_MOTION_CONTROL_COMPLETE',flush=True)
except BaseException:
    rt.save(OUT/'status.json',dict(state='failed',error=traceback.format_exc(),quality_accepted=False));raise
finally:
    fcntl.flock(lock,fcntl.LOCK_UN)
