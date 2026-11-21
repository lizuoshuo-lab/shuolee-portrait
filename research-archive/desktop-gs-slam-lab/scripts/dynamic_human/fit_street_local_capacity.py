"""Training-only one-view versus two-view bounded surface capacity diagnostic.

Not a replacement scene model, and not a new untouched test benchmark.
"""
from pathlib import Path
import json
import logging
import os
import shutil
import time
import traceback
import torch
from torch.nn.utils import parametrize
from omegaconf import OmegaConf
import refit_street_human_surface as surface
from resume_street_deformable import cuda_tree,equal_tree,sha

common=surface.common
helper=surface.helper
LAB=helper.LAB
OUT=LAB/'outputs/dynamic-human-street-local-capacity-020'
SOURCE=surface.SOURCE
FIELDS=surface.COLOR+surface.SURFACE
BRANCHES={'single_view':[(129,4)],'adjacent_pair':[(129,4),(131,4)]}
UPDATES=1000

def status(value,**fields):
    helper.write_json(OUT/'state.json',dict(status=value,pid=os.getpid(),**fields))
    print(json.dumps(dict(status=value,**fields)),flush=True)

@torch.no_grad()
def snapshot(trainer,dataset,regions,directory,iteration):
    rows=[]
    for roi in regions:
        info,cam=dataset.full_image_set.get_image(roi['image_id'],1.)
        info,cam=helper.to_cuda(info),helper.to_cuda(cam)
        result=trainer(info,cam,novel_view=True)
        pred=result['rgb'].clamp(0,1).cpu().numpy();gt=info['pixels'].cpu().numpy()
        x0,y0,x1,y1=roi['box'];stem=f"frame{roi['frame']:03d}-camera{roi['camera']}"
        helper.save_rgb(directory/f'update{iteration:04d}'/(stem+'-prediction.png'),pred)
        helper.save_rgb(directory/f'update{iteration:04d}'/(stem+'-person79.png'),pred[y0:y1,x0:x1])
        helper.save_rgb(directory/'ground-truth'/(stem+'.png'),gt)
        rows.append(dict(**roi,**helper.score(gt[y0:y1,x0:x1],pred[y0:y1,x0:x1])))
    helper.write_json(directory/f'update{iteration:04d}'/'metrics.json',dict(iteration=iteration,rows=rows,selection_used=False))

def main():
    assert not (OUT/'protocol.json').exists(),'Preserve previous attempts'
    source=SOURCE/'checkpoint-refined.pt'
    source_protocol=json.loads((LAB/'outputs/dynamic-human-street-surface-refit-018/protocol.json').read_text())
    assert sha(source)==source_protocol['source_checkpoint_sha256']
    regions=json.loads((SOURCE/'training-rois.json').read_text())['regions']
    rois=json.loads((SOURCE/'person-rois.json').read_text())['regions']
    subset=[r for r in regions if (r['frame'],r['camera']) in BRANCHES['adjacent_pair']]
    assert len(regions)==101 and len(rois)==435 and len(subset)==2
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    helper.write_json(OUT/'protocol.json',dict(status='sealed_before_capacity_optimization',
        source_checkpoint_sha256=sha(source),script_sha256=sha(Path(__file__)),
        surface_helper_sha256=sha(Path(surface.__file__)),evaluation_helper_sha256=sha(Path(common.__file__)),
        source='Pose-only candidate016; same source and parameterization as matched surface experiment018',
        target=79,fields=list(FIELDS),branches=BRANCHES,training_regions=subset,
        optimizer=dict(type='Adam',lr=.03,betas=[.9,.999],eps=1e-8,updates=UPDATES),
        parameterization=source_protocol['parameterization'],objective=source_protocol['objective'],frozen=source_protocol['frozen'],
        budget='Both branches1000updates; single view1000region backwards, adjacent pair2000region backwards. Equal updates and parameter count, not equal observation/compute budget.',
        selection='Lowest finite branch training objective among initial and1000updated states; no heldout or other-frame selection.',
        evaluation='All101training regions before/after;105heldout full images/435regions after; snapshots of129/131 at0,100,500,1000.',
        rationale='Check whether current bounded shared surface can fit a fixed close training image, then whether adding its adjacent training image makes both fits disagree.',
        boundaries=['Exploratory capacity diagnostic; one-view overfit is not successful multi-view reconstruction.',
            'Only129/4 or129/4+131/4 enter optimization; frame130 and all105heldout images never enter losses.',
            'Source learned from all training times; other101train views are transfer checks, not unseen data.',
            'Compared with018, both observations and update budget differ; not a causal isolation of temporal alignment.',
            'Frozen scene/poses/6890point topology and3cm center bounds may themselves prevent fitting.',
            'Pixel boxes include background and other people; no exact silhouette or cloth geometry GT.',
            'Do not select or publish a replacement model from this diagnostic; no image enhancement.']))
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO)
    started=time.perf_counter()
    helper.install_capture(helper.ArgoVerseCameraData);helper.install_training_only(helper.DrivingDataset)
    status('loading_shared_dataset')
    dataset=helper.DrivingDataset(OmegaConf.load(common.BASELINE/'config.yaml').data)
    assert all(r['image_id'] in dataset.train_indices and r['frame'] not in dataset.test_timesteps for r in subset)
    cfg=OmegaConf.load(SOURCE/'config.yaml')
    trainer=helper.MultiTrainer(**cfg.trainer,num_timesteps=156,model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    initialization=json.loads((surface.ORIGINAL/'initialization.json').read_text())
    model_index=next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance']==79)
    summaries=[]
    for branch,pairs in BRANCHES.items():
        run=OUT/branch;run.mkdir();common.OUT=run
        OmegaConf.save(cfg,run/'config.yaml')
        helper.write_json(run/'training-rois.json',dict(regions=regions))
        helper.write_json(run/'person-rois.json',dict(regions=rois))
        optimized=[r for r in subset if (r['frame'],r['camera']) in pairs]
        helper.write_json(run/'optimized-rois.json',dict(regions=optimized))
        status('restoring_source',branch=branch)
        trainer.load_state_dict(cuda_tree(saved),load_only_model=True,strict=True)
        equal_tree(saved,trainer.state_dict(only_model=True),branch+'.restoration')
        trainer.completed_updates=30000;trainer.render_each_class=False
        for model in trainer.models.values():
            for parameter in model.parameters():parameter.requires_grad_(False)
        node=trainer.models['SMPLNodes'];mask=node.point_ids[:,0]==model_index;indices=torch.where(mask)[0]
        assert len(indices)==6890
        before=common.score_training(trainer,dataset,regions,'before')
        assert abs(before['mean_person_psnr_db']-20.404231632252785)<1e-5
        changes={}
        for field in FIELDS:
            reference=getattr(node,field).detach().clone()
            change=surface.TargetChange(field,reference,indices)
            parametrize.register_parametrization(node,field,change)
            assert torch.equal(reference,getattr(node,field)) and not node.parametrizations[field].original.requires_grad
            changes[field]=change
        optimizer=torch.optim.Adam([x.delta for x in changes.values()],lr=.03,betas=(.9,.999),eps=1e-8)
        rows=[];best=dict(total=float('inf'),iteration=None,deltas=None)
        log=(run/'training-history.jsonl').open('a',buffering=1)
        for iteration in range(UPDATES+1):
            trainer.set_eval();optimizer.zero_grad(set_to_none=True);backward=iteration<UPDATES;photo=0.
            per_view=[]
            with torch.set_grad_enabled(backward):
                for roi in optimized:
                    info,cam=dataset.full_image_set.get_image(roi['image_id'],1.)
                    info,cam=helper.to_cuda(info),helper.to_cuda(cam)
                    result=trainer(info,cam,novel_view=True)
                    x0,y0,x1,y1=roi['box'];pred=result['rgb'][y0:y1,x0:x1];target=info['pixels'][y0:y1,x0:x1]
                    l1=(pred-target).abs().mean()
                    ssim=1-trainer.ssim(target.permute(2,0,1)[None],pred.permute(2,0,1)[None])
                    loss=(.8*l1+.2*ssim)/len(optimized)
                    assert torch.isfinite(loss)
                    if backward:loss.backward()
                    photo+=float(loss.detach())
                    per_view.append(dict(frame=roi['frame'],camera=roi['camera'],l1=float(l1.detach()),ssim_loss=float(ssim.detach())))
                    del result,info,cam,pred,target,l1,ssim,loss
                prior=sum(x.prior() for x in changes.values())
                if backward:prior.backward()
            total=photo+float(prior.detach());assert torch.isfinite(prior)
            gradients={}
            if backward:
                for field,change in changes.items():
                    assert change.delta.grad is not None and torch.isfinite(change.delta.grad).all()
                    gradients[field]=float(change.delta.grad.norm())
                if iteration==0:assert all(x>0 for x in gradients.values())
            if total<best['total']:
                best.update(total=total,iteration=iteration,deltas={k:x.delta.detach().cpu().clone() for k,x in changes.items()})
            row=dict(iteration=iteration,total=total,photo=photo,prior=float(prior.detach()),training_views=per_view,
                gradient_norms=gradients,best_iteration=best['iteration'])
            log.write(json.dumps(row)+'\n');rows.append(row)
            if iteration in (0,100,500,1000):snapshot(trainer,dataset,subset,run/'snapshots',iteration)
            if iteration%50==0:
                helper.write_json(run/'progress.json',row);status('fitting_training_views',branch=branch,**row)
            if backward:optimizer.step()
        log.close()
        with torch.no_grad():
            for field,change in changes.items():change.delta.copy_(best['deltas'][field].to('cuda'))
        for field in FIELDS:parametrize.remove_parametrizations(node,field,leave_parametrized=True)
        frozen=surface.frozen_check(trainer,saved,mask.cpu(),FIELDS)
        helper.write_json(run/'training-selection.json',dict(optimizer_updates=UPDATES,region_backward_evaluations=UPDATES*len(optimized),
            evaluated_training_states=len(rows),selected_iteration=best['iteration'],selected_total=best['total'],
            heldout_selection_used=False,optimized_regions=optimized,history=rows))
        status('evaluating_training_transfer_and_heldout',branch=branch)
        after=common.score_training(trainer,dataset,regions,'after')
        metrics=helper.evaluate(trainer,dataset,rois,run,'final',complete=True)
        surface.frozen_check(trainer,saved,mask.cpu(),FIELDS)
        torch.save(trainer.state_dict(only_model=True),run/'checkpoint-refined.pt')
        result=dict(status='capacity_diagnostic_complete_independent_audits_pending',branch=branch,
            additional_updates=UPDATES,region_backward_evaluations=UPDATES*len(optimized),
            optimized_regions=optimized,selected_iteration=best['iteration'],frozen_tensor_entries=frozen,
            all_training_before=before['mean_person_psnr_db'],all_training_after=after['mean_person_psnr_db'],
            evaluation=metrics,quality_accepted=False)
        helper.write_json(run/'completion.json',result)
        summaries.append({k:result[k] for k in ('branch','additional_updates','region_backward_evaluations','selected_iteration','all_training_before','all_training_after')})
        del optimizer,changes,best
        torch.cuda.empty_cache()
    assert sha(source)==source_protocol['source_checkpoint_sha256']
    helper.write_json(OUT/'completion.json',dict(status='both_capacity_diagnostics_completed_audits_pending',summaries=summaries,
        seconds=time.perf_counter()-started,quality_accepted=False))
    status('completed_independent_audits_pending',summaries=summaries)

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed_needs_inspection',error=repr(error))
        traceback.print_exc();raise
