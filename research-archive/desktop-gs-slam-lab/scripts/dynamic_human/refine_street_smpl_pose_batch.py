"""Full-training-region L-BFGS pose diagnostic; original30k source preserved."""
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
from torch.nn import functional as F
from omegaconf import OmegaConf
import refine_street_smpl_pose as common
from resume_street_deformable import cuda_tree,equal_tree,sha

helper=common.helper
LAB=helper.LAB
SOURCE=common.SOURCE
OUT=LAB/'outputs/dynamic-human-street-pose-batch-015'
NAMES=common.PARAMETERS


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--long',action='store_true')
    args=parser.parse_args()
    if args.long:OUT=LAB/'outputs/dynamic-human-street-pose-convergence-016'
    assert not (OUT/'protocol.json').exists(),'Preserve earlier attempts'
    OUT.mkdir(parents=True,exist_ok=True)
    common.OUT=OUT
    shutil.copy2(__file__,OUT/Path(__file__).name)
    cfg=OmegaConf.load(SOURCE/'config.yaml')
    OmegaConf.save(cfg,OUT/'config.yaml')
    protocol=dict(status='sealed_before_training_batch_pose_optimization',source_updates=30000,
        source_checkpoint_sha256=sha(SOURCE/'checkpoint-30000.pt'),script_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(common.__file__)),target=79,
        algorithm='Full101training-region averaged RGB L1/SSIM plus pose priors; L-BFGS strong-Wolfe line search.',
        optimizer=dict(lr=.5,max_iter=120 if args.long else 20,max_eval=180 if args.long else 30,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9),
        continuation_reason='Training objective still descending at20iterations, no pose bound reached. Same source and objective, larger convergence budget; no change to camera/Gaussian appearance or geometry.' if args.long else None,
        selection='Best finite in-bounds total training objective observed by closure, including unchanged source as a candidate. No heldout criterion.',
        acceptance_bounds=dict(translation_m=.5,root_degrees=25,joints_degrees=45),
        frozen='All Gaussian attributes, voxel skinning, background, other people, cameras, exposure, heldout pose entries.',
        losses=dict(rgb_l1=.8,ssim=.2,translation_prior=.01,rotation_prior=.01,translation_delta_smooth=.05,rotation_delta_smooth=.01,quaternion_norm_ratio=.01,excess_bounds=10.),
        evaluation='Original105heldout views/435boxes after fixed-budget fit,101training regions before/after.',
        boundaries=['Motivated by worse training fit in stochastic pose diagnostic014; starts from original30k, not failed014 model.',
            'Exploratory one-person diagnostic; full scene remains in compositor. Not equal-budget method benchmark.',
            'Training boxes include background and occluders; no segmentation/joint GT.',
            'Existing Gaussian attributes may compensate wrong pose; pose-only fit has limited scope.',
            'No novel-view/clarity acceptance based on training objective alone.'])
    helper.write_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO)
    started=time.perf_counter()
    helper.install_capture(helper.ArgoVerseCameraData);helper.install_training_only(helper.DrivingDataset)
    common.status('loading_dataset')
    dataset=helper.DrivingDataset(OmegaConf.load(common.BASELINE/'config.yaml').data)
    trainer=helper.MultiTrainer(**cfg.trainer,num_timesteps=156,model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    saved=torch.load(SOURCE/'checkpoint-30000.pt',map_location='cpu',mmap=True,weights_only=False)
    trainer.load_state_dict(cuda_tree(saved),load_only_model=True,strict=True)
    equal_tree(saved,trainer.state_dict(only_model=True),'source_restore')
    trainer.completed_updates=30000;trainer.render_each_class=False
    initialization=json.loads((SOURCE/'initialization.json').read_text())
    model_index=next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance']==79)
    node=trainer.models['SMPLNodes']
    rois=json.loads((SOURCE/'person-rois.json').read_text())['regions']
    regions=common.training_regions(dataset,node,model_index)
    assert regions==json.loads((LAB/'outputs/dynamic-human-street-pose-refinement-014/training-rois.json').read_text())['regions']
    assert len(regions)==101
    helper.write_json(OUT/'training-rois.json',dict(regions=regions))
    helper.write_json(OUT/'person-rois.json',dict(regions=rois))
    for model in trainer.models.values():
        for parameter in model.parameters():parameter.requires_grad_(False)
    active=node.instances_fv[:,model_index].clone();active[dataset.test_timesteps]=False
    assert int(active.sum())==90
    for name in NAMES:getattr(node,name).requires_grad_(True)
    reference={name:getattr(node,name).detach().clone() for name in NAMES}
    pair_mask=active[1:] & active[:-1]
    parameters=[getattr(node,name) for name in NAMES]
    optimizer=torch.optim.LBFGS(parameters,**protocol['optimizer'])
    common.status('scoring_initial_training_regions')
    before=common.score_training(trainer,dataset,regions,'before')
    # Same source, data and render mode must reproduce previous training diagnostics.
    old_before=json.loads((LAB/'outputs/dynamic-human-street-pose-refinement-014/training-evaluation/before/metrics.json').read_text())
    assert abs(before['mean_person_psnr_db']-old_before['mean_person_psnr_db'])<1e-5
    best=dict(total=float('inf'),parameters=None,closure=None)
    history=[]
    def prior_and_bounds():
        delta=node.instances_trans[:,model_index]-reference['instances_trans'][:,model_index]
        total=.01*delta[active].square().mean()+.05*(delta[1:]-delta[:-1])[pair_mask].square().mean()
        distances=delta[active].norm(dim=-1)
        total=total+10*torch.relu(distances-.5).square().mean()
        limits=dict(translation_max_m=float(distances.max()))
        for name,max_degrees in [('instances_quats',25),('smpl_qauts',45)]:
            q=F.normalize(getattr(node,name)[:,model_index],dim=-1)
            q0=F.normalize(reference[name][:,model_index],dim=-1)
            sin2=(1-(q[active]*q0[active]).sum(-1).square()).clamp_min(0)
            total=total+.01*sin2.mean()+10*torch.relu(sin2-math.sin(math.radians(max_degrees)/2)**2).square().mean()
            norm=getattr(node,name)[active,model_index].norm(dim=-1)
            norm0=reference[name][active,model_index].norm(dim=-1)
            total=total+.01*(norm/norm0-1).square().mean()
            # Quaternion correction relative to source; sign-invariant smoothness.
            from pytorch3d.transforms import quaternion_multiply,quaternion_invert
            correction=quaternion_multiply(quaternion_invert(q0),q)
            total=total+.01*(1-(correction[1:]*correction[:-1]).sum(-1).square())[pair_mask].clamp_min(0).mean()
            limits[name+'_max_degrees']=float(torch.acos((q[active]*q0[active]).sum(-1).abs().clamp(max=1)).max()*360/math.pi)
        return total,limits
    log=(OUT/'closure-history.jsonl').open('a',buffering=1)
    def closure():
        trainer.set_train();optimizer.zero_grad(set_to_none=True)
        total_photo=0.;total_l1=0.;total_ssim=0.
        for roi in regions:
            assert roi['frame'] not in dataset.test_timesteps and roi['image_id'] in dataset.train_indices
            info,cam=dataset.full_image_set.get_image(roi['image_id'],1.)
            info,cam=helper.to_cuda(info),helper.to_cuda(cam)
            result=trainer(info,cam,novel_view=True)
            x0,y0,x1,y1=roi['box']
            predicted=result['rgb'][y0:y1,x0:x1];target=info['pixels'][y0:y1,x0:x1]
            l1=(predicted-target).abs().mean()
            ssim=1-trainer.ssim(target.permute(2,0,1)[None],predicted.permute(2,0,1)[None])
            loss=(.8*l1+.2*ssim)/len(regions)
            assert torch.isfinite(loss)
            loss.backward()
            total_photo+=float(loss.detach());total_l1+=float(l1.detach())/len(regions);total_ssim+=float(ssim.detach())/len(regions)
            del result,info,cam,loss,predicted,target,l1,ssim
        prior,limits=prior_and_bounds();assert torch.isfinite(prior)
        prior.backward()
        total=total_photo+float(prior.detach())
        for parameter in parameters:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            mask=torch.zeros(parameter.shape[:2],dtype=torch.bool,device=parameter.device);mask[active,model_index]=True
            parameter.grad[~mask]=0
        accepted=limits['translation_max_m']<=.50001 and limits['instances_quats_max_degrees']<=25.002 and limits['smpl_qauts_max_degrees']<=45.002
        index=len(history)
        if accepted and total<best['total']:
            best.update(total=total,parameters={name:getattr(node,name).detach().cpu().clone() for name in NAMES},closure=index)
        row=dict(closure=index,total=total,photo=total_photo,l1=total_l1,ssim_loss=total_ssim,prior=float(prior.detach()),
            accepted_within_bounds=accepted,best_closure=best['closure'],seconds=time.perf_counter()-started,**limits)
        history.append(row);log.write(json.dumps(row)+'\n');helper.write_json(OUT/'progress.json',row)
        print(json.dumps(row),flush=True)
        return torch.tensor(total,device='cuda')
    common.status('full_training_batch_line_search_running',training_regions=101)
    optimizer.step(closure)
    assert best['parameters'] is not None
    with torch.no_grad():
        for name in NAMES:getattr(node,name).copy_(best['parameters'][name].to('cuda'))
    common.frozen_check(trainer,saved,model_index,active)
    helper.write_json(OUT/'training-selection.json',dict(status='selected_best_in_bounds_full_training_objective',
        chosen_closure=best['closure'],chosen_total=best['total'],initial_total=history[0]['total'],
        closure_evaluations=len(history),training_region_backward_evaluations=len(history)*len(regions),
        optimizer_iterations=int(optimizer.state[parameters[0]]['n_iter']),
        parameters_changed_only_at_target_training_times=True,heldout_selection_used=False,history=history))
    common.status('evaluating_selected_pose_model')
    after=common.score_training(trainer,dataset,regions,'after')
    trainer.completed_updates=30000
    result=helper.evaluate(trainer,dataset,rois,OUT,'final',complete=True)
    common.frozen_check(trainer,saved,model_index,active)
    torch.save(trainer.state_dict(only_model=True),OUT/'checkpoint-refined.pt')
    result.update(status='full_batch_pose_diagnostic_finished_independent_audit_pending',source_updates=30000,
        extra_lbfgs_iterations=int(optimizer.state[parameters[0]]['n_iter']),closure_evaluations=len(history),
        chosen_closure=best['closure'],training_before=before['mean_person_psnr_db'],training_after=after['mean_person_psnr_db'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),quality_accepted=False,seconds=time.perf_counter()-started)
    helper.write_json(OUT/'completion.json',result)
    common.status('completed_independent_image_and_visual_review_pending',training_before=result['training_before'],training_after=result['training_after'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():common.OUT=OUT;common.status('failed_needs_inspection',error=repr(error))
        traceback.print_exc();raise
