"""Fresh free-world Gaussian optimization through the pinned native renderer.

This is a training component, not a human-data admission entry. The caller must
establish physical camera/material/staticity evidence before any person run.
No checkpoint loader, deformation, body binding, camera optimizer, image resize
or implicit pixel-origin conversion is implemented here.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib, json, time
import numpy as np
import torch
from native_portrait_renderer122 import NativeCameraContract, render_native
from native_scene_package122 import (validate_camera, runtime_record, validate_runtime,
                                    SCHEMA, COORDINATES, RENDERER)
from native_gaussian_ply122 import write_standard_ply


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class NativeTrainingView:
    camera: dict
    rgb: torch.Tensor
    alpha: torch.Tensor
    rgb_valid: torch.Tensor
    observation_valid: torch.Tensor
    w2c: torch.Tensor
    K: torch.Tensor

    @classmethod
    def create(cls, camera, rgb, alpha, rgb_valid, observation_valid, *, device):
        camera=json.loads(json.dumps(camera));validate_camera(camera)
        width,height=camera['width'],camera['height']
        rgb=np.asarray(rgb);alpha=np.asarray(alpha);mask=np.asarray(rgb_valid);valid=np.asarray(observation_valid)
        if rgb.shape!=(height,width,3) or alpha.shape!=(height,width,1) or mask.shape!=(height,width,1) or valid.shape!=mask.shape:
            raise ValueError('Images, masks and camera must have the same exact native dimensions')
        if mask.dtype!=bool or valid.dtype!=bool or np.any(mask&~valid) or not mask.any():
            raise ValueError('Explicit nonempty boolean RGB support contained in observation validity required')
        if not np.isfinite(rgb).all() or not np.isfinite(alpha).all() or np.any((rgb<0)|(rgb>1)) or np.any((alpha<0)|(alpha>1)):
            raise ValueError('Finite unit-range native RGB and alpha required')
        t=lambda x:torch.as_tensor(np.array(x,copy=True),device=device,dtype=torch.float32)
        return cls(camera,t(rgb),t(alpha),torch.as_tensor(mask.copy(),device=device),torch.as_tensor(valid.copy(),device=device),t(camera['w2c'])[None],t(camera['K'])[None])


def foreground_objective(rgb, alpha, view, *, ssim_weight=.2, silhouette_weight=.1):
    """Photometry only on explicit foreground; silhouette only where observable.

    The11x11 SSIM window follows119, now with explicit invalid-occluder support.
    No RGB or alpha value outside its support enters the corresponding loss.
    """
    if rgb.shape!=view.rgb.shape or alpha.shape!=view.alpha.shape:
        raise ValueError('Native renderer output shape changed')
    valid=view.rgb_valid[...,0];observed=view.observation_valid[...,0]
    l1=(rgb[valid]-view.rgb[valid]).abs().mean()
    F=torch.nn.functional
    keep=valid[None,None]
    support=F.max_pool2d(F.pad((~keep).float(),(5,5,5,5),value=1),11,stride=1)==0
    if not support.any(): raise ValueError('No complete foreground SSIM window; do not dilute with background')
    a=torch.where(keep,rgb.permute(2,0,1)[None],0)
    b=torch.where(keep,view.rgb.permute(2,0,1)[None],0)
    axis=torch.arange(11,device=rgb.device,dtype=rgb.dtype)-5
    one=torch.exp(-axis.square()/(2*1.5**2));one=one/one.sum()
    win=(one[:,None]*one[None,:])[None,None].expand(3,1,11,11).contiguous()
    conv=lambda x:F.conv2d(x,win,padding=5,groups=3)
    u,v=conv(a),conv(b);aa,bb,ab=conv(a*a)-u*u,conv(b*b)-v*v,conv(a*b)-u*v
    score=((2*u*v+.01**2)*(2*ab+.03**2))/((u*u+v*v+.01**2)*(aa+bb+.03**2))
    ssim=score[support.expand_as(score)].mean()
    silhouette=(alpha[observed]-view.alpha[observed]).abs().mean()
    loss=(1-ssim_weight)*l1+ssim_weight*(1-ssim)+silhouette_weight*silhouette
    return loss,dict(l1=l1,ssim=ssim,alpha_mae=silhouette)


class FreshNativeGaussians(torch.nn.Module):
    """All centers are directly optimized world XYZ, with full covariance/SH."""
    def __init__(self, initial, *, device):
        super().__init__()
        n=len(initial['means'])
        shapes={'means':(n,3),'quats':(n,4),'scales':(n,3),'opacities':(n,),'sh':(n,16,3)}
        for name,shape in shapes.items():
            a=np.asarray(initial[name])
            if n<=0 or a.shape!=shape or not np.isfinite(a).all(): raise ValueError('Invalid fresh Gaussian '+name)
        if np.any(np.asarray(initial['scales'])<=0) or np.any((np.asarray(initial['opacities'])<=0)|(np.asarray(initial['opacities'])>=1)) or np.any(np.linalg.norm(initial['quats'],axis=-1)<1e-8):
            raise ValueError('Positive scales, interior probabilities and nonzero quaternions required')
        t=lambda x:torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device=device)
        self.means=torch.nn.Parameter(t(initial['means']))
        self.quats=torch.nn.Parameter(t(initial['quats']))
        self.log_scales=torch.nn.Parameter(t(np.log(initial['scales'])))
        op=np.asarray(initial['opacities']);self.opacity_logits=torch.nn.Parameter(t(np.log(op/(1-op))))
        self.sh=torch.nn.Parameter(t(initial['sh']))
        self.initial_count=n

    def parameters_linear(self):
        return dict(means=self.means,quats=torch.nn.functional.normalize(self.quats,dim=-1),
            scales=self.log_scales.exp(),opacities=self.opacity_logits.sigmoid(),sh=self.sh)

    def numpy_linear(self): return {k:v.detach().cpu().numpy().copy() for k,v in self.parameters_linear().items()}

    def optimizer(self, learning_rates):
        names=['means','quats','log_scales','opacity_logits','sh']
        if set(learning_rates)!=set(names) or any(not np.isfinite(x) or x<=0 for x in learning_rates.values()):
            raise ValueError('Explicit positive world-unit learning rates required')
        opt=torch.optim.Adam([dict(params=[getattr(self,k)],lr=learning_rates[k],name=k) for k in names],eps=1e-8)
        if len(opt.state): raise RuntimeError('Fresh optimizer unexpectedly contains state')
        return opt


def render_view(model, view, contract, *, radial=None, sh_degree=3):
    p=model.parameters_linear()
    radial_tensor=None if radial is None else torch.as_tensor(radial,dtype=view.K.dtype,device=view.K.device)[None]
    rgb,alpha,_=render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],view.w2c,view.K,
        view.camera['width'],view.camera['height'],contract=contract,radial_coeffs=radial_tensor,sh_degree=sh_degree)
    return rgb[0],alpha[0]


def export_native_checkpoint(path, model, cameras, contract, *, radial=None, scope):
    """Standard PLY with the existing SHA-bound recorded-camera native package."""
    if scope not in ('synthetic-control-only','person-reconstruction'): raise ValueError('Explicit package scope required')
    contract.validate();path=Path(path);path.mkdir(exist_ok=False)
    for camera in cameras:validate_camera(camera)
    write_standard_ply(path/'point_cloud.ply',**model.numpy_linear())
    write_json(path/'cameras.json',cameras)
    backend=runtime_record();validate_runtime(backend)
    scene=dict(schema=SCHEMA,scope=scope,coordinates=COORDINATES,input_domain=contract.input_domain,
        model=dict(asset=dict(path='point_cloud.ply',sha256=digest(path/'point_cloud.ply')),vertices=model.initial_count,sh_degree=3),
        cameras=dict(path='cameras.json',sha256=digest(path/'cameras.json')),backend=backend,renderer=RENDERER,
        projections={contract.projection:dict(radial_coefficients=radial)})
    write_json(path/'scene.json',scene)
    return path/'scene.json'


def optimize_fresh(model, views, contract, *, steps, learning_rates, seed, output,
                   review_steps=(), review=None, radial=None, checkpoint=None):
    """One fresh optimizer, deterministic complete-view cycles, no resume path.

    Human admission, initialization, adaptive density and full resolution input
    loading are outside this component. No actual person may enter via a boolean
    claiming that this synthetic component test has passed.
    """
    if type(steps)is not int or not 1<=steps<=18000:raise ValueError('Finite step budget required')
    if not views or len({v.camera['id'] for v in views})!=len(views):raise ValueError('Distinct real frame identities required')
    contract.validate()
    review_steps=set(review_steps)
    if steps>=1000 and (1000 not in review_steps or review is None):raise ValueError('Step1000 actual review callback required')
    if review_steps and review is None:raise ValueError('Review cannot silently approve itself')
    if any(type(s)is not int or not 1<=s<=steps for s in review_steps):raise ValueError('Review step outside run')
    output=Path(output);output.mkdir(exist_ok=False)
    before=[(v.w2c.clone(),v.K.clone()) for v in views]
    optimizer=model.optimizer(learning_rates)
    record=dict(scope='training-component-only',initial_points=model.initial_count,optimizer_initial_state_entries=len(optimizer.state),
        old_checkpoint_loaded=False,camera_optimization=False,body_binding=False,dynamic_network=False,
        rgb_supervision='explicit foreground only; full valid11x11windows forSSIM',
        alpha_supervision='observable pixels only; external occlusion must beinvalid',
        model_growth='fixed count in this component; adaptive density is not implemented',
        physical_training_admission=False,review_steps=sorted(review_steps),steps=steps,seed=seed)
    write_json(output/'config.json',record)
    rng=np.random.default_rng(seed);order=[];counts=np.zeros(len(views),int);history=[];start=time.monotonic();stopped=False
    with (output/'loss.jsonl').open('x') as stream:
        for step in range(1,steps+1):
            if not order:order=rng.permutation(len(views)).tolist()
            index=order.pop();counts[index]+=1;v=views[index]
            rgb,alpha=render_view(model,v,contract,radial=radial)
            loss,terms=foreground_objective(rgb,alpha,v)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite native loss')
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Missing/nonfinite gradient:'+name)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            row=dict(step=step,frame_id=v.camera['id'],loss=float(loss),**{k:float(x) for k,x in terms.items()})
            history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if step in review_steps:
                receipt=checkpoint(step,model) if checkpoint else None
                if review(step,receipt) is not True:stopped=True;break
        torch.cuda.synchronize()
    if any(not torch.equal(a,v.w2c) or not torch.equal(b,v.K) for v,(a,b) in zip(views,before)):
        raise RuntimeError('Frozen camera tensor mutated')
    report=dict(state='stopped_at_review' if stopped else 'component_steps_complete',steps=step,planned_steps=steps,
        view_update_counts=counts.tolist(),optimizer_initial_state_entries=0,optimizer_final_state_entries=len(optimizer.state),
        points=model.initial_count,seconds=time.monotonic()-start,renderer=contract.manifest(),
        frozen_camera_tensors_unchanged=True,training_admission=False,person_quality_accepted=False)
    write_json(output/'result.json',report)
    return report,history
