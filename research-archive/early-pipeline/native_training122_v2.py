"""Fixed-camera training component using explicit v2 rendering and export.

Real input admission, fresh body/surface fitting and adaptive density are separate
pipeline stages. Synthetic validation of this component does not admit a person.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from native_training122 import NativeTrainingView,foreground_objective,FreshNativeGaussians as ParameterContainer
from native_portrait_renderer122_v2 import NativeCameraContractV2,render_native
from native_scene_package122_v2 import write_scene,write_json

class FreshNativeGaussians(ParameterContainer):
    def __init__(self,initial,*,device):
        super().__init__(initial,device=device)
        self._fresh_digest=self._parameter_digest()
        self._optimizer_constructed=False

    def _parameter_digest(self):
        h=hashlib.sha256()
        for name,p in self.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def optimizer(self,learning_rates):
        if self._optimizer_constructed or self._parameter_digest()!=self._fresh_digest:raise RuntimeError('Fresh optimizer requires untouched initialization; no second optimizer or cross-run continuation')
        result=super().optimizer(learning_rates);self._optimizer_constructed=True
        return result

def render_view(model,view,contract,*,radial=None,sh_degree=3):
    p=model.parameters_linear()
    rgb,alpha,_=render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],view.w2c,view.K,
        view.camera['width'],view.camera['height'],contract=contract,sh_degree=sh_degree,
        radial_coeffs=None if radial is None else torch.as_tensor(radial,dtype=torch.float32,device=view.K.device)[None])
    return rgb[0],alpha[0]

def export_native_checkpoint(path,model,cameras,contract,*,radial=None,scope):
    return write_scene(path,model.numpy_linear(),cameras,contract=contract,radial=radial,scope=scope)

def optimize_fresh(model,views,contract,*,steps,learning_rates,seed,output,review_steps=(),review=None,radial=None,checkpoint=None):
    if not isinstance(contract,NativeCameraContractV2) or not isinstance(model,FreshNativeGaussians):raise ValueError('V2 model and renderer contract required')
    contract.validate()
    if type(steps)is not int or not 1<=steps<=18000:raise ValueError('Finite step budget required')
    if not views or len({v.camera['id'] for v in views})!=len(views):raise ValueError('Distinct recorded frame identities required')
    review_steps=set(review_steps)
    if steps>=1000 and (1000 not in review_steps or review is None):raise ValueError('Actual step1000 review callback required')
    if review_steps and review is None:raise ValueError('No implicit review approval')
    if any(type(s)is not int or not 1<=s<=steps for s in review_steps):raise ValueError('Review step outside run')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    optimizer=model.optimizer(learning_rates)
    output.mkdir(exist_ok=False)
    before=[(v.w2c.clone(),v.K.clone()) for v in views]
    record=dict(scope='training-component-only',initial_points=model.initial_count,initial_parameter_sha256=model._fresh_digest,
        optimizer_initial_state_entries=len(optimizer.state),old_checkpoint_loaded=False,camera_optimization=False,body_binding=False,dynamic_network=False,
        renderer=contract.manifest(),rgb_supervision='explicit foreground and full valid11x11 SSIM windows',alpha_supervision='observed pixels only',
        model_growth='fixed count; adaptive density not implemented',physical_training_admission=False,review_steps=sorted(review_steps),steps=steps,seed=seed)
    write_json(output/'config.json',record)
    rng=np.random.default_rng(seed);order=[];counts=np.zeros(len(views),int);history=[];start=time.monotonic();stopped=False
    with (output/'loss.jsonl').open('x') as stream:
        for step in range(1,steps+1):
            if not order:order=rng.permutation(len(views)).tolist()
            index=order.pop();counts[index]+=1;v=views[index]
            optimizer.zero_grad(set_to_none=True)
            rgb,alpha=render_view(model,v,contract,radial=radial)
            loss,terms=foreground_objective(rgb,alpha,v)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite native loss')
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Missing/nonfinite gradient: '+name)
            optimizer.step()
            row=dict(step=step,frame_id=v.camera['id'],loss=float(loss.detach()),**{k:float(x.detach()) for k,x in terms.items()})
            history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if step in review_steps:
                receipt=checkpoint(step,model) if checkpoint else None
                if review(step,receipt) is not True:stopped=True;break
    torch.cuda.synchronize()
    if any(not torch.equal(a,v.w2c) or not torch.equal(b,v.K) for v,(a,b) in zip(views,before)):raise RuntimeError('Frozen camera mutated')
    report=dict(state='stopped_at_review' if stopped else 'component_steps_complete',steps=step,planned_steps=steps,
        view_update_counts=counts.tolist(),optimizer_initial_state_entries=0,optimizer_final_state_entries=len(optimizer.state),
        points=model.initial_count,seconds=time.monotonic()-start,renderer=contract.manifest(),frozen_camera_tensors_unchanged=True,
        training_admission=False,person_quality_accepted=False)
    write_json(output/'result.json',report);return report,history
