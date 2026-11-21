"""Render joint024 once; reuse independently verified source/control019 temporal pixels."""
from pathlib import Path
import gc
import json
import os
import shutil
import time
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import train_street_deformable as helper
from resume_street_deformable import cuda_tree,equal_tree,sha

LAB=helper.LAB
OUT=LAB/'outputs/dynamic-human-street-joint-pose-surface-024/temporal'
CACHED=LAB/'outputs/dynamic-human-street-surface-temporal-019'
WINDOWS=LAB/'outputs/dynamic-human-street-temporal-012'
BASELINE=LAB/'outputs/dynamic-human-street-joint-009/resume-10000-expandable'
RUNS={'joint_pose_surface':LAB/'outputs/dynamic-human-street-joint-pose-surface-024'}


def status(value,**fields):
    helper.write_json(OUT/'state.json',dict(status=value,pid=os.getpid(),**fields))
    print(json.dumps(dict(status=value,**fields)),flush=True)


def main():
    assert not (OUT/'protocol.json').exists(),'Preserve earlier attempts'
    for name,run in RUNS.items():
        assert (run/'completion.json').exists() and (run/'independent-state-audit.json').exists()
        if name!='pose_source':assert (run/'independent-image-audit.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    protocol=json.loads((WINDOWS/'protocol.json').read_text())
    protocol['current_comparison']='Same80original views and4windows: newly render joint024; copy source016 and surface018 from audited019 cached outputs; no additional training.'
    protocol['previous_windows_protocol_sha256']=sha(WINDOWS/'protocol.json')
    protocol['boundaries']+=['Source016 has30koriginalupdates+120poseiterations; surface018 and joint024 each add100fullbatchupdates. Equal update/observation count for the two refinements, different parameter count/runtime.',
        'This is a fixed observed-trajectory replay with training and heldout times, not future-motion prediction.',
        'No candidate is accepted as clear moving-human reconstruction merely from a metric improvement.']
    helper.write_json(OUT/'protocol.json',protocol)
    targets=OUT/'ground-truth';targets.mkdir()
    for item in protocol['images']:
        stem=f"frame{item['frame']:03d}-camera{item['camera']}"
        source=WINDOWS/'ground-truth'/(stem+'.png')
        assert sha(source)==item['resized_sha256']
        shutil.copy2(source,targets/source.name)
    helper.write_json(OUT/'render-protocol.json',dict(script_sha256=sha(Path(__file__)),optimized_steps=0,
        checkpoints={name:sha(run/'checkpoint-refined.pt') for name,run in RUNS.items()},
        state_audits={name:sha(run/'independent-state-audit.json') for name,run in RUNS.items()},
        windows_sha256=sha(OUT/'protocol.json'),quality_accepted=False))
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    helper.install_capture(helper.ArgoVerseCameraData);helper.install_training_only(helper.DrivingDataset)
    started=time.perf_counter();status('loading_shared_dataset')
    dataset=helper.DrivingDataset(OmegaConf.load(BASELINE/'config.yaml').data)
    window_map={(r['frame'],r['camera']):w for w in protocol['windows'] for r in w['images']}
    for name,run in RUNS.items():
        directory=OUT/name;directory.mkdir()
        cfg=OmegaConf.load(run/'config.yaml')
        status('restoring_candidate',branch=name)
        trainer=helper.MultiTrainer(**cfg.trainer,num_timesteps=156,model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
            num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
        saved=torch.load(run/'checkpoint-refined.pt',map_location='cpu',mmap=True,weights_only=False)
        trainer.load_state_dict(cuda_tree(saved),load_only_model=True,strict=True)
        equal_tree(saved,trainer.state_dict(only_model=True),name+'.restoration')
        trainer.completed_updates=30000;trainer.render_each_class=False;trainer.set_eval()
        rows=[];temporal=[];previous={};overlap=[]
        with torch.no_grad():
            for item in protocol['images']:
                f,c=item['frame'],item['camera'];stem=f'frame{f:03d}-camera{c}'
                info,cam=dataset.full_image_set.get_image(f*7+c,1.)
                info,cam=helper.to_cuda(info),helper.to_cuda(cam)
                result=trainer(info,cam,novel_view=True)
                predicted=result['rgb'].clamp(0,1).cpu().numpy();target=info['pixels'].cpu().numpy()
                assert np.array_equal(np.uint8(target*255+.5),np.asarray(Image.open(targets/(stem+'.png'))))
                helper.save_rgb(directory/(stem+'-prediction.png'),predicted)
                window=window_map[(f,c)];x0,y0,x1,y1=window['fixed_person_box_union']
                rows.append(dict(frame=f,camera=c,heldout=item['heldout'],window=window['name'],
                    full_image=helper.score(target,predicted),fixed_union_region=helper.score(target[y0:y1,x0:x1],predicted[y0:y1,x0:x1])))
                if item['heldout']:
                    expected=np.asarray(Image.open(run/'evaluation/final'/(stem+'-prediction.png')),dtype=np.int16)
                    diff=np.abs(np.uint8(predicted*255+.5).astype(np.int16)-expected)
                    assert diff.max()<=1
                    overlap.append(dict(frame=f,camera=c,max8bit_difference=int(diff.max())))
                if window['name'] in previous:
                    old_frame,old_target,old_pred=previous[window['name']];assert f==old_frame+1
                    residual=np.abs((predicted-old_pred)-(target-old_target))
                    temporal.append(dict(window=window['name'],previous_frame=old_frame,frame=f,camera=c,
                        heldout_transition=item['heldout'] or old_frame%10==0,
                        full_image_temporal_difference_residual=float(residual.mean()),
                        fixed_union_temporal_difference_residual=float(residual[y0:y1,x0:x1].mean())))
                previous[window['name']]=f,target,predicted
                if len(rows)%20==0:status('rendering_continuous_windows',branch=name,completed_views=len(rows))
                del result,info,cam
        assert len(rows)==80 and len(temporal)==76
        assert len(overlap)==sum(r['heldout'] for r in protocol['images'])
        equal_tree(saved,trainer.state_dict(only_model=True),name+'.no_training_changes')
        helper.write_json(directory/'completion.json',dict(status='all80temporal_views_restored_and_rendered',branch=name,
            optimized_steps=0,rows=rows,temporal_differences=temporal,reload_comparisons=overlap,
            all_model_tensors_unchanged=True,visual_review_pending=True,quality_accepted=False))
        del trainer,saved,previous
        gc.collect();torch.cuda.empty_cache()
    cache_protocol=json.loads((CACHED/'render-protocol.json').read_text())
    current_protocol=json.loads((OUT.parent/'protocol.json').read_text())
    assert cache_protocol['checkpoints']['pose_source']==current_protocol['source_checkpoint_sha256']
    assert cache_protocol['checkpoints']['surface_and_appearance']==current_protocol['control_checkpoint_sha256']
    cached=[]
    for branch in ('pose_source','surface_and_appearance'):
        destination=OUT/branch;destination.mkdir()
        entries=[]
        for item in protocol['images']:
            stem=f"frame{item['frame']:03d}-camera{item['camera']}"
            p=CACHED/branch/(stem+'-prediction.png');q=destination/p.name
            shutil.copy2(p,q);assert sha(p)==sha(q)
            entries.append(dict(file=p.name,sha256=sha(q)))
        shutil.copy2(CACHED/branch/'completion.json',destination/'completion.json')
        cached.append(dict(branch=branch,checkpoint_sha256=cache_protocol['checkpoints'][branch],images=entries))
    helper.write_json(OUT/'cached-controls.json',dict(source_audit_sha256=sha(CACHED/'independent-audit/independent-audit.json'),branches=cached))
    helper.write_json(OUT/'completion.json',dict(status='80new_plus160_cached_temporal_images_ready_for_independent_audit',branches=['pose_source','surface_and_appearance',*RUNS],seconds=time.perf_counter()-started,quality_accepted=False))
    status('completed_independent_video_audit_pending')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed_needs_inspection',error=repr(error))
        raise
