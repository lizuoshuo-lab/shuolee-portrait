"""Joint bounded pose and surface fit over all training-person observations.

Reuses the completed surface-only018 control from the identical source016.
No heldout RGB selection, no background update, no claim of new data/quality.
"""
from pathlib import Path
import json, logging, math, os, shutil, time, traceback
import torch
from torch import nn
from torch.nn.utils import parametrize
from omegaconf import OmegaConf
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_raw_multiply
import refit_street_human_surface as surface
from resume_street_deformable import cuda_tree, equal_tree, sha

common=surface.common
helper=surface.helper
LAB=helper.LAB
OUT=LAB/'outputs/dynamic-human-street-joint-pose-surface-024'
SOURCE=surface.SOURCE
CONTROL=LAB/'outputs/dynamic-human-street-surface-refit-018/surface-retry-v2/surface_and_appearance'
FIELDS=surface.COLOR+surface.SURFACE
POSES=common.PARAMETERS
UPDATES=100
LIMITS={'instances_trans':.1,'instances_quats':math.radians(10),'smpl_qauts':math.radians(20)}

def status(value,**fields):
    row=dict(status=value,pid=os.getpid(),**fields)
    helper.write_json(OUT/'state.json',row);print(json.dumps(row),flush=True)

class PoseChange(nn.Module):
    def __init__(self,field,original,frames,model_index):
        super().__init__();self.field=field;self.limit=LIMITS[field]
        self.register_buffer('indices',frames*original.shape[1]+model_index)
        self.register_buffer('frame_gaps',frames[1:]-frames[:-1])
        shape=list(original.flatten(0,1)[self.indices].shape);shape[-1]=3
        self.delta=nn.Parameter(torch.zeros(shape,device=original.device,dtype=original.dtype))
    def offset(self):
        return self.limit*self.delta/torch.sqrt(1+self.delta.square().sum(-1,keepdim=True))
    def forward(self,original):
        flat=original.flatten(0,1);selected=flat[self.indices];change=self.offset()
        value=(selected+change if self.field=='instances_trans' else
            quaternion_raw_multiply(axis_angle_to_quaternion(change),selected))
        return flat.index_copy(0,self.indices,value).reshape_as(original)
    def prior(self):
        normalized=self.offset()/self.limit
        gaps=self.frame_gaps.reshape((-1,)+(1,)*(normalized.ndim-1))
        velocity=(normalized[1:]-normalized[:-1])/gaps
        return .001*normalized.square().mean()+.002*velocity.square().mean()

def frozen_check(trainer,saved,point_mask,pose_mask):
    current=trainer.state_dict(only_model=True);count=0
    for name,state in saved['models'].items():
        actual=current['models'][name];assert state.keys()==actual.keys()
        for field,value in state.items():
            other=actual[field]
            if name=='SMPLNodes' and field in FIELDS:
                assert torch.equal(value[~point_mask],other.detach().cpu()[~point_mask]),field
            elif name=='SMPLNodes' and field in POSES:
                assert torch.equal(value[~pose_mask],other.detach().cpu()[~pose_mask]),field
            else: equal_tree(value,other,name+'.'+field);count+=int(isinstance(value,torch.Tensor))
    for key,value in saved.items():
        if key!='models':equal_tree(value,current[key],'trainer.'+key)
    return count

def main():
    assert not (OUT/'protocol.json').exists(),'Preserve all earlier attempts'
    OUT.mkdir(parents=True,exist_ok=True);common.OUT=OUT
    source=SOURCE/'checkpoint-refined.pt'
    control_protocol=json.loads((CONTROL.parent/'protocol.json').read_text())
    control_check=json.loads((CONTROL/'independent-state-audit.json').read_text())
    assert sha(source)==control_protocol['source_checkpoint_sha256']==control_check['source_checkpoint_sha256']
    assert control_check['updates']==UPDATES and control_check['region_backward_evaluations']==10100
    regions=json.loads((SOURCE/'training-rois.json').read_text())['regions']
    rois=json.loads((SOURCE/'person-rois.json').read_text())['regions']
    assert len(regions)==101 and len(rois)==435
    cfg=OmegaConf.load(SOURCE/'config.yaml');OmegaConf.save(cfg,OUT/'config.yaml')
    for name,value in [('training-rois',regions),('person-rois',rois)]:helper.write_json(OUT/(name+'.json'),dict(regions=value))
    shutil.copy2(__file__,OUT/Path(__file__).name)
    protocol=dict(status='sealed_before_joint_refitting',target=79,source_checkpoint_sha256=sha(source),
        script_sha256=sha(Path(__file__)),surface_helper_sha256=sha(Path(surface.__file__)),
        evaluation_helper_sha256=sha(Path(common.__file__)),surface_fields=list(FIELDS),pose_fields=list(POSES),
        control='Completed018 surface_and_appearance: same016 source,101training regions,100Adam updates,10100backwards,unchanged surface parameterization/loss/lr.',
        control_checkpoint_sha256=sha(CONTROL/'checkpoint-refined.pt'),
        optimizer=dict(type='Adam',surface_lr=.03,pose_lr=.01,betas=[.9,.999],eps=1e-8,updates=UPDATES),
        surface_parameterization=control_protocol['parameterization'],surface_objective=control_protocol['objective'],
        pose_parameterization=dict(translation_max_m=.1,root_max_degrees=10,joints_max_degrees=20,
            unit='Additional change relative to selected source016; rotation increments are scalar-first quaternion left products from bounded axis angles.',
            magnitude_prior=.001,normalized_correction_velocity_prior=.002,
            smoothing='Finite differences between consecutive valid training frames divided by frame-index gap; no assumed physical FPS.'),
        frozen='Background,other people,cameras,exposure,voxel skinning,6890targetpoint topology and all stored heldout pose entries.',
        evaluation_pose='Unmodified author interpolation from neighboring training poses at test frames when both neighbors valid; otherwise original fallback.',
        selection='Lowest full training objective among source and100updated states. No heldout selection.',
        evaluation='101training regions;105heldout full images/435boxes with13main-person crops.',
        rationale='Jointly compensate pose misalignment and canonical surface/appearance across all observed training times instead of keeping either block fixed.',
        boundaries=['Exploratory already-inspected split, not an untouched test benchmark.',
            'Stored test poses frozen does not mean test render geometry stays unchanged: author interpolation uses modified neighbors.',
            'Pose block adds parameters and its regularization; equal optimizer-update and observation budgets, not equal computation.',
            'Original AV2 privacy blur and input downsampling remain; no source-face recovery claim.',
            'No person-mask/joint ground truth; boxes contain background and occlusion.',
            'No sharpening, densification or model replacement on training-score evidence.'])
    helper.write_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(6);torch.manual_seed(79)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO);started=time.perf_counter()
    helper.install_capture(helper.ArgoVerseCameraData);helper.install_training_only(helper.DrivingDataset)
    status('loading_shared_dataset')
    dataset=helper.DrivingDataset(OmegaConf.load(common.BASELINE/'config.yaml').data)
    assert all(r['image_id'] in dataset.train_indices and r['frame'] not in dataset.test_timesteps for r in regions)
    trainer=helper.MultiTrainer(**cfg.trainer,num_timesteps=156,model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    trainer.load_state_dict(cuda_tree(saved),load_only_model=True,strict=True)
    equal_tree(saved,trainer.state_dict(only_model=True),'initial_source')
    trainer.completed_updates=30000;trainer.render_each_class=False
    for model in trainer.models.values():
        for p in model.parameters():p.requires_grad_(False)
    node=trainer.models['SMPLNodes']
    initialization=json.loads((surface.ORIGINAL/'initialization.json').read_text())
    model_index=next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance']==79)
    point_mask=node.point_ids[:,0]==model_index;indices=torch.where(point_mask)[0];assert len(indices)==6890
    active=node.instances_fv[:,model_index].clone();active[dataset.test_timesteps]=False
    frames=torch.where(active)[0];assert len(frames)==90
    pose_mask=torch.zeros(node.instances_fv.shape,dtype=torch.bool);pose_mask[frames.cpu(),model_index]=True
    helper.write_json(OUT/'active-parameters.json',dict(model_index=model_index,point_indices=indices.cpu().tolist(),
        training_pose_frames=frames.cpu().tolist(),test_timesteps=[int(t) for t in dataset.test_timesteps]))
    status('scoring_source_training_views')
    before=common.score_training(trainer,dataset,regions,'before')
    assert abs(before['mean_person_psnr_db']-20.404231632252785)<1e-5
    changes={}
    for field in FIELDS+POSES:
        original=getattr(node,field).detach().clone()
        change=(surface.TargetChange(field,original,indices) if field in FIELDS else
            PoseChange(field,original,frames,model_index))
        parametrize.register_parametrization(node,field,change)
        assert torch.equal(original,getattr(node,field)),field
        assert not node.parametrizations[field].original.requires_grad
        changes[field]=change
    helper.write_json(OUT/'initialization-check.json',dict(status='all9zero_delta_fields_exactly_equal_source',
        target_points=6890,training_pose_frames=90,source_training_psnr_db=before['mean_person_psnr_db']))
    optimizer=torch.optim.Adam([dict(params=[changes[f].delta for f in FIELDS],lr=.03),
        dict(params=[changes[f].delta for f in POSES],lr=.01)],betas=(.9,.999),eps=1e-8)
    best=dict(total=float('inf'),iteration=None,deltas=None);rows=[]
    log=(OUT/'training-history.jsonl').open('a',buffering=1)
    training_start=time.perf_counter()
    for iteration in range(UPDATES+1):
        trainer.set_eval();optimizer.zero_grad(set_to_none=True);backward=iteration<UPDATES
        photo=0.;mae=0.;ssim_loss=0.
        with torch.set_grad_enabled(backward):
            for roi in regions:
                info,cam=dataset.full_image_set.get_image(roi['image_id'],1.)
                info,cam=helper.to_cuda(info),helper.to_cuda(cam)
                result=trainer(info,cam,novel_view=True)
                x0,y0,x1,y1=roi['box'];pred=result['rgb'][y0:y1,x0:x1];target=info['pixels'][y0:y1,x0:x1]
                l1=(pred-target).abs().mean()
                ssim=1-trainer.ssim(target.permute(2,0,1)[None],pred.permute(2,0,1)[None])
                loss=(.8*l1+.2*ssim)/len(regions);assert torch.isfinite(loss)
                if backward:loss.backward()
                photo+=float(loss.detach());mae+=float(l1.detach())/len(regions);ssim_loss+=float(ssim.detach())/len(regions)
                del info,cam,result,pred,target,l1,ssim,loss
            surface_prior=sum(changes[f].prior() for f in FIELDS)
            pose_prior=sum(changes[f].prior() for f in POSES)
            prior=surface_prior+pose_prior;assert torch.isfinite(prior)
            if backward:prior.backward()
        gradients={}
        if backward:
            for field,change in changes.items():
                assert change.delta.grad is not None and torch.isfinite(change.delta.grad).all(),field
                gradients[field]=float(change.delta.grad.norm())
            if iteration==0:assert all(v>0 for v in gradients.values()),gradients
        total=photo+float(prior.detach())
        if total<best['total']:
            best.update(total=total,iteration=iteration,deltas={k:v.delta.detach().cpu().clone() for k,v in changes.items()})
        row=dict(iteration=iteration,total=total,photo=photo,l1=mae,ssim_loss=ssim_loss,
            surface_prior=float(surface_prior.detach()),pose_prior=float(pose_prior.detach()),gradient_norms=gradients,
            best_iteration=best['iteration'],seconds=time.perf_counter()-training_start)
        log.write(json.dumps(row)+'\n');rows.append(row)
        if iteration%5==0 or not backward:
            helper.write_json(OUT/'progress.json',row);status('joint_full_training_batch_fit',**row)
        if iteration%10==0 or not backward:
            torch.save(dict(iteration=iteration,best=best,deltas={k:v.delta.detach().cpu() for k,v in changes.items()},
                optimizer=optimizer.state_dict(),source_checkpoint_sha256=protocol['source_checkpoint_sha256']),OUT/'recoverable-deltas.pt')
        if backward:optimizer.step()
    log.close()
    with torch.no_grad():
        for field,change in changes.items():change.delta.copy_(best['deltas'][field].to('cuda'))
    for field in FIELDS+POSES:parametrize.remove_parametrizations(node,field,leave_parametrized=True)
    frozen_count=frozen_check(trainer,saved,point_mask.cpu(),pose_mask)
    torch.save(trainer.state_dict(only_model=True),OUT/'checkpoint-refined.pt')
    helper.write_json(OUT/'training-selection.json',dict(status='selected_using_full_training_objective_only',
        selected_iteration=best['iteration'],selected_total=best['total'],initial_total=rows[0]['total'],
        optimizer_updates=UPDATES,region_backward_evaluations=UPDATES*len(regions),evaluated_training_states=len(rows),
        heldout_selection_used=False,history=rows))
    status('evaluating_selected_candidate')
    after=common.score_training(trainer,dataset,regions,'after')
    metrics=helper.evaluate(trainer,dataset,rois,OUT,'final',complete=True)
    frozen_check(trainer,saved,point_mask.cpu(),pose_mask)
    assert sha(source)==protocol['source_checkpoint_sha256']
    helper.write_json(OUT/'completion.json',dict(status='trained_and_evaluated_independent_audits_pending',
        additional_fullbatch_updates=UPDATES,region_backward_evaluations=UPDATES*len(regions),
        selected_iteration=best['iteration'],training_before=before['mean_person_psnr_db'],training_after=after['mean_person_psnr_db'],
        frozen_tensor_entries=frozen_count,evaluation=metrics,quality_accepted=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started))
    status('completed_independent_audits_pending',training_before=before['mean_person_psnr_db'],training_after=after['mean_person_psnr_db'])

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status('failed_needs_inspection',error=repr(e))
        traceback.print_exc();raise
