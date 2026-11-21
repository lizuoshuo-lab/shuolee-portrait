"""Formal fresh optimizer loop using complete frozen output cameras.

Static free centers, full covariance/SH, explicit view sampling, mean loss over
selected views and statistics from every selected view. No adaptive growth,
camera learning, body attraction or optimizer-resume path.
"""
from pathlib import Path
from collections import Counter
import json,time
import numpy as np
import torch
from native_training122 import foreground_objective
from native_training122_v2 import FreshNativeGaussians
from native_training122_v4 import create_view,render_view,assert_frozen_view,export_native_checkpoint
from native_training_policy122 import digest,validate_policy,cycle_schedule,validate_reviews
from native_residual_camera122 import ResidualRenderContract
from native_scene_package122_v2 import write_json

def accumulate_update(model,views,contract,optimizer,*,capture_gradients=False):
    if not isinstance(model,FreshNativeGaussians) or not isinstance(contract,ResidualRenderContract):
        raise ValueError('Fresh free-world model and complete residual contract required')
    if len(views) not in (1,2) or len({v.camera['id'] for v in views})!=len(views):
        raise ValueError('One or two distinct recorded views required')
    optimizer.zero_grad(set_to_none=True)
    n=model.initial_count;device=model.means.device
    counts=torch.zeros(n,dtype=torch.int32,device=device)
    radii=torch.zeros(n,device=device);norm_sum=torch.zeros(n,device=device)
    rows=[]
    for v in views:
        rgb,alpha,meta=render_view(model,v,contract)
        loss,terms=foreground_objective(rgb,alpha,v)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite native loss')
        contribution=[]
        hook=model.means.register_hook(lambda g:contribution.append(g.detach().clone()))
        try:(loss/len(views)).backward()
        finally:hook.remove()
        if len(contribution)!=1:raise RuntimeError('Missing per-view XYZ gradient')
        visible=(meta['radii'][0]>0).all(-1)
        radius=meta['radii'][0].amax(-1).float()
        norm=torch.linalg.vector_norm(contribution[0]*len(views),dim=-1)
        counts+=visible.to(torch.int32)
        radii=torch.maximum(radii,torch.where(visible,radius,0))
        norm_sum+=torch.where(visible,norm,0)
        rows.append(dict(frame_id=v.camera['id'],loss=float(loss.detach()),visible=int(visible.sum()),
            **{k:float(x.detach()) for k,x in terms.items()}))
        del rgb,alpha,meta,loss,terms,contribution
    gradients={}
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Missing/nonfinite gradient '+name)
        if capture_gradients:gradients[name]=p.grad.detach().clone()
    optimizer.step()
    for v in views:assert_frozen_view(v)
    return dict(losses=rows,gradients=gradients,visible_count=counts,max_projected_radius=radii,
        world_gradient_norm_sum=norm_sum,optimizer_steps=1)

def optimize_fresh(model,views,contract,*,steps,learning_rates,seed,output,policy,
                   graph=None,review_steps=(),review=None,checkpoint=None,progress=None):
    validate_policy(policy);validate_reviews(steps,list(review_steps),review,checkpoint)
    if not isinstance(model,FreshNativeGaussians) or not isinstance(contract,ResidualRenderContract):
        raise ValueError('Fresh model and complete residual renderer required')
    contract.validate()
    cameras=[v.camera for v in views]
    for v in views:assert_frozen_view(v)
    fixed_digest=digest(cameras)
    schedule=cycle_schedule(cameras,policy,seed,graph=graph)
    by_id={v.camera['id']:v for v in views}
    out=Path(output)
    if out.exists():raise FileExistsError('Fresh run only; no overwrite or resume')
    optimizer=model.optimizer(learning_rates)
    out.mkdir(parents=True,exist_ok=False)
    record=dict(scope='training-component-only',initial_points=model.initial_count,
        initial_parameter_sha256=model._fresh_digest,optimizer_initial_state_entries=len(optimizer.state),
        policy=policy,schedule=schedule,cameras_sha256=fixed_digest,graph_sha256=digest(graph) if graph else None,
        steps=steps,seed=seed,renderer=contract.manifest(),old_checkpoint_loaded=False,camera_optimization=False,
        body_binding=False,dynamic_network=False,review_steps=list(review_steps),physical_quality_accepted=False,
        statistics='all selected views: unscaled worldXYZ gradient norms,visibility counts,max radii; not screen gradients',
        rgb_supervision='observed foreground and complete11x11 SSIM windows',alpha_supervision='observed pixels only')
    write_json(out/'config.json',record)
    total_counts=torch.zeros(model.initial_count,dtype=torch.int64,device=model.means.device)
    total_norm=torch.zeros(model.initial_count,device=model.means.device)
    max_radius=torch.zeros(model.initial_count,device=model.means.device)
    counts=Counter({i:0 for i in by_id});history=[];snapshots={};start=time.monotonic();stopped=False
    # The supported cycle is fixed. Rotate its edge order per epoch without
    # silently retrying graph search or changing any edge/support evidence.
    with (out/'loss.jsonl').open('x') as stream:
        for step in range(1,steps+1):
            epoch=(step-1)//len(schedule);index=((step-1)%len(schedule)+epoch)%len(schedule)
            batch=schedule[index]
            stats=accumulate_update(model,[by_id[i] for i in batch],contract,optimizer)
            counts.update(batch);total_counts+=stats['visible_count'];total_norm+=stats['world_gradient_norm_sum']
            max_radius=torch.maximum(max_radius,stats['max_projected_radius'])
            row=dict(step=step,frames=batch,loss=float(np.mean([r['loss'] for r in stats['losses']])),per_view=stats['losses'])
            history.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if progress is not None:progress(step,row)
            if step in review_steps:
                receipt=checkpoint(step,model);snapshots[step]=receipt
                if review(step,receipt) is not True:stopped=True;break
        if model.means.is_cuda:torch.cuda.synchronize()
    for v in views:assert_frozen_view(v)
    if digest(cameras)!=fixed_digest:raise RuntimeError('Frozen complete cameras changed')
    if checkpoint is not None and step not in snapshots:snapshots[step]=checkpoint(step,model)
    np.savez(out/'all-view-statistics.npz',visible_count=total_counts.cpu().numpy(),
        max_projected_radius=max_radius.cpu().numpy(),world_gradient_norm_sum=total_norm.cpu().numpy(),
        world_gradient_norm_mean=(total_norm/total_counts.clamp_min(1)).cpu().numpy())
    result=dict(state='stopped_at_review' if stopped else 'steps_complete',steps=step,planned_steps=steps,
        points=model.initial_count,optimizer_initial_state_entries=0,optimizer_final_state_entries=len(optimizer.state),
        view_update_counts=dict(counts),full_cycles=step//len(schedule),seconds=time.monotonic()-start,
        complete_camera_sha256=fixed_digest,policy=policy,statistics='all-view-statistics.npz',
        snapshots=snapshots,person_quality_accepted=False,physical_admission_claimed_by_component=False)
    write_json(out/'result.json',result)
    return result,history
