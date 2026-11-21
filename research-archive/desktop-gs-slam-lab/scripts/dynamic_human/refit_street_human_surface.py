"""Matched full-batch appearance versus constrained surface/appearance refitting."""
from pathlib import Path
import argparse
import json
import logging
import math
import os
import shutil
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from omegaconf import OmegaConf
from pytorch3d.transforms import axis_angle_to_quaternion,quaternion_raw_multiply
import refine_street_smpl_pose as common
from resume_street_deformable import cuda_tree,equal_tree,sha

helper=common.helper
LAB=helper.LAB
OUT=LAB/'outputs/dynamic-human-street-surface-refit-018'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
ORIGINAL=LAB/'outputs/dynamic-human-street-smpl-joint-011/articulated'
COLOR=('_features_dc','_features_rest')
SURFACE=('_means','_scales','_quats','_opacities')
UPDATES=100


def status(value,**fields):
    row=dict(status=value,pid=os.getpid(),**fields)
    helper.write_json(OUT/'state.json',row);print(json.dumps(row),flush=True)


class TargetChange(nn.Module):
    """Only the6890 target rows are trainable; smooth bounds preserve the start."""
    def __init__(self,field,original,indices):
        super().__init__()
        self.field=field
        self.register_buffer('indices',indices)
        shape=list(original[indices].shape)
        if field=='_quats':shape[-1]=3
        self.delta=nn.Parameter(torch.zeros(shape,device=original.device,dtype=original.dtype))

    def offset(self):
        x=self.delta
        if self.field in ('_means','_quats'):
            limit=.03 if self.field=='_means' else math.radians(20)
            return limit*x/torch.sqrt(1+x.square().sum(-1,keepdim=True))
        if self.field=='_scales':return math.log(2)*torch.tanh(x)
        if self.field=='_opacities':return 4*x
        return (.5 if self.field=='_features_dc' else .1)*x

    def forward(self,original):
        selected=original[self.indices]
        offset=self.offset()
        changed=quaternion_raw_multiply(axis_angle_to_quaternion(offset),selected) if self.field=='_quats' else selected+offset
        return original.index_copy(0,self.indices,changed)

    def prior(self):
        if self.field=='_means':return .0005*(self.offset()/.03).square().sum(-1).mean()
        if self.field=='_quats':return .0002*(self.offset()/math.radians(20)).square().sum(-1).mean()
        if self.field=='_scales':return .0002*(self.offset()/math.log(2)).square().mean()
        return .00001*self.delta.square().mean()


def frozen_check(trainer,saved,point_mask,allowed):
    current=trainer.state_dict(only_model=True)
    count=0
    for name,state in saved['models'].items():
        actual=current['models'][name]
        assert state.keys()==actual.keys()
        for field,value in state.items():
            if name=='SMPLNodes' and field in allowed:
                assert torch.equal(value[~point_mask],actual[field].detach().cpu()[~point_mask]),field
            else:
                equal_tree(value,actual[field],name+'.'+field)
                count+=int(isinstance(value,torch.Tensor))
    for key,value in saved.items():
        if key!='models':equal_tree(value,current[key],'trainer.'+key)
    return count


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--surface-retry',action='store_true');args=parser.parse_args()
    if args.surface_retry:OUT=OUT/'surface-retry-v2'
    assert not (OUT/'protocol.json').exists(),'Preserve earlier attempts'
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    checkpoint=SOURCE/'checkpoint-refined.pt'
    protocol=dict(status='sealed_before_matched_refitting',source_checkpoint_sha256=sha(checkpoint),
        source='Original30k model plus selected training-only pose diagnostic016',script_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(common.__file__)),target=79,branches=dict(appearance_only=list(COLOR),surface_and_appearance=list(COLOR+SURFACE)),
        optimizer=dict(type='Adam',lr=.03,betas=[.9,.999],eps=1e-8,updates=UPDATES),
        budget='Each branch100updates, eachupdate averages all101 identical training regions:10100region backward evaluations; source restored afresh between branches.',
        parameterization=dict(center_displacement_max_m=.03,scale_ratio=[.5,2.],rotation_change_max_degrees=20,
            sh_dc_delta_scale=.5,sh_rest_delta_scale=.1,opacity_logit_delta_scale=4.,
            target_only='Separate trainable deltas for target6890points; all source tensors remain frozen originals during optimization.'),
        objective=dict(rgb_l1=.8,ssim=.2,center_normalized_prior=.0005,scale_normalized_prior=.0002,
            rotation_normalized_prior=.0002,color_and_opacity_delta_prior=.00001),
        selection='Lowest finite full training objective among initial state and all100updated states; no heldout selection.',
        frozen='All poses including heldout, voxel skinning, background,other people,cameras,exposure,point count and topology.',
        renderer_mode='Evaluation mode with gradients enabled: same calibrated rendering, no densification or retained screen-position gradient requirement for color-only optimization.',
        evaluation='All105heldout full images/435person boxes;101training regions; no test RGB loss.',
        boundaries=['Exploratory diagnostic on repeatedly inspected heldout times, not an untouched test benchmark.',
            'Surface branch changes several Gaussian attributes together; not an isolated loss or scale ablation.',
            'Both branches have identical optimization steps and training observations but different parameter counts and runtime.',
            'Projected boxes include background and occluders; no true per-person silhouette or joint labels.',
            'Fixed6890point body topology and the original frozen scene may still limit appearance.',
            'No densification,no sharpening,no generative image repair,no model publication.'])
    if args.surface_retry:
        protocol['branches']={'surface_and_appearance':list(COLOR+SURFACE)}
        protocol['repair']='Original surface branch stopped before updates because quaternion_multiply standardizes sign. Use quaternion_raw_multiply to preserve exact initial tensor values; same physical rotation, source, objective and budget.'
        protocol['completed_control']='Parent appearance_only branch remains the matched control; do not retrain or overwrite it.'
        assert (OUT.parent/'appearance_only/completion.json').exists()
        probe=json.loads((OUT.parent/'parameterization-sign-preserving-smoke.json').read_text())
        assert probe['source_script_sha256']==protocol['script_sha256']
    helper.write_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO)
    started=time.perf_counter()
    helper.install_capture(helper.ArgoVerseCameraData);helper.install_training_only(helper.DrivingDataset)
    status('loading_shared_dataset')
    dataset=helper.DrivingDataset(OmegaConf.load(common.BASELINE/'config.yaml').data)
    cfg=OmegaConf.load(SOURCE/'config.yaml')
    trainer=helper.MultiTrainer(**cfg.trainer,num_timesteps=156,model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    initialization=json.loads((ORIGINAL/'initialization.json').read_text())
    model_index=next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance']==79)
    regions=json.loads((SOURCE/'training-rois.json').read_text())['regions']
    rois=json.loads((SOURCE/'person-rois.json').read_text())['regions']
    assert len(regions)==101 and len(rois)==435
    assert all(r['image_id'] in dataset.train_indices and r['frame'] not in dataset.test_timesteps for r in regions)
    summaries=[]
    for branch,fields in protocol['branches'].items():
        directory=OUT/branch;directory.mkdir()
        common.OUT=directory
        OmegaConf.save(cfg,directory/'config.yaml')
        helper.write_json(directory/'training-rois.json',dict(regions=regions))
        helper.write_json(directory/'person-rois.json',dict(regions=rois))
        status('restoring_source',branch=branch)
        trainer.load_state_dict(cuda_tree(saved),load_only_model=True,strict=True)
        equal_tree(saved,trainer.state_dict(only_model=True),branch+'.source')
        trainer.completed_updates=30000;trainer.render_each_class=False
        for model in trainer.models.values():
            for parameter in model.parameters():parameter.requires_grad_(False)
        node=trainer.models['SMPLNodes']
        mask=(node.point_ids[:,0]==model_index)
        indices=torch.where(mask)[0]
        assert len(indices)==6890
        before=common.score_training(trainer,dataset,regions,'before')
        assert abs(before['mean_person_psnr_db']-20.404231632252785)<1e-5
        changes={}
        for field in fields:
            reference=getattr(node,field).detach().clone()
            change=TargetChange(field,reference,indices)
            parametrize.register_parametrization(node,field,change)
            assert torch.equal(reference,getattr(node,field)),field
            assert not node.parametrizations[field].original.requires_grad
            changes[field]=change
        parameters=[x.delta for x in changes.values()]
        optimizer=torch.optim.Adam(parameters,lr=.03,betas=(.9,.999),eps=1e-8)
        best=dict(total=float('inf'),iteration=None,deltas=None)
        log=(directory/'training-history.jsonl').open('a',buffering=1)
        rows=[]
        branch_start=time.perf_counter()
        for iteration in range(UPDATES+1):
            trainer.set_eval();optimizer.zero_grad(set_to_none=True)
            backward=iteration<UPDATES
            photo=0.;mae=0.;ssim_loss=0.
            with torch.set_grad_enabled(backward):
                for roi in regions:
                    info,cam=dataset.full_image_set.get_image(roi['image_id'],1.)
                    info,cam=helper.to_cuda(info),helper.to_cuda(cam)
                    result=trainer(info,cam,novel_view=True)
                    x0,y0,x1,y1=roi['box']
                    predicted=result['rgb'][y0:y1,x0:x1];target=info['pixels'][y0:y1,x0:x1]
                    l1=(predicted-target).abs().mean()
                    ssim=1-trainer.ssim(target.permute(2,0,1)[None],predicted.permute(2,0,1)[None])
                    loss=(.8*l1+.2*ssim)/len(regions)
                    assert torch.isfinite(loss)
                    if backward:loss.backward()
                    photo+=float(loss.detach());mae+=float(l1.detach())/len(regions);ssim_loss+=float(ssim.detach())/len(regions)
                    del result,info,cam,predicted,target,loss,l1,ssim
                prior=sum(x.prior() for x in changes.values())
                assert torch.isfinite(prior)
                if backward:prior.backward()
            total=photo+float(prior.detach())
            gradients={}
            if backward:
                for field,change in changes.items():
                    assert change.delta.grad is not None and torch.isfinite(change.delta.grad).all(),field
                    gradients[field]=float(change.delta.grad.norm())
                if iteration==0:assert all(value>0 for value in gradients.values()),gradients
            if total<best['total']:
                best.update(total=total,iteration=iteration,deltas={k:x.delta.detach().cpu().clone() for k,x in changes.items()})
            row=dict(iteration=iteration,total=total,photo=photo,l1=mae,ssim_loss=ssim_loss,prior=float(prior.detach()),
                gradient_norms=gradients,best_iteration=best['iteration'],seconds=time.perf_counter()-branch_start)
            log.write(json.dumps(row)+'\n');rows.append(row)
            if iteration%5==0 or iteration==UPDATES:
                helper.write_json(directory/'progress.json',row);status('full_training_batch_refitting',branch=branch,**row)
            if backward:optimizer.step()
        with torch.no_grad():
            for field,change in changes.items():change.delta.copy_(best['deltas'][field].to('cuda'))
        # Materialize the selected values into the original model schema for strict downstream checks.
        for field in fields:parametrize.remove_parametrizations(node,field,leave_parametrized=True)
        frozen_count=frozen_check(trainer,saved,mask.cpu(),fields)
        helper.write_json(directory/'training-selection.json',dict(status='best_full_training_objective_selected',
            selected_iteration=best['iteration'],selected_total=best['total'],initial_total=rows[0]['total'],
            optimizer_updates=UPDATES,region_backward_evaluations=UPDATES*len(regions),evaluated_training_states=len(rows),
            heldout_selection_used=False,history=rows))
        status('evaluating_selected_candidate',branch=branch)
        after=common.score_training(trainer,dataset,regions,'after')
        result=helper.evaluate(trainer,dataset,rois,directory,'final',complete=True)
        frozen_check(trainer,saved,mask.cpu(),fields)
        torch.save(trainer.state_dict(only_model=True),directory/'checkpoint-refined.pt')
        result.update(status='refitting_complete_independent_audits_pending',branch=branch,source_updates=30000,
            source_pose_iterations=120,additional_fullbatch_updates=UPDATES,region_backward_evaluations=UPDATES*len(regions),
            selected_iteration=best['iteration'],training_before=before['mean_person_psnr_db'],training_after=after['mean_person_psnr_db'],
            frozen_tensor_entries=frozen_count,quality_accepted=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.perf_counter()-branch_start)
        helper.write_json(directory/'completion.json',result)
        summaries.append(dict(branch=branch,training_before=result['training_before'],training_after=result['training_after'],selected_iteration=best['iteration']))
        del optimizer,changes,parameters,best
        torch.cuda.empty_cache()
    assert sha(checkpoint)==protocol['source_checkpoint_sha256']
    helper.write_json(OUT/'completion.json',dict(status='both_branches_completed_independent_audits_and_visual_review_pending',
        branches=summaries,seconds=time.perf_counter()-started,quality_accepted=False))
    status('both_branches_completed_independent_audits_pending',branches=summaries)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed_needs_inspection',error=repr(error))
        traceback.print_exc();raise
