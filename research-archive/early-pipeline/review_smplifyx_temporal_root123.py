"""Incremental CPU-only all-frame review of immutable completed SMPLify-X fits."""
import argparse
import os
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image
import cv2
from review_full_sequence123 import mesh_renderer, joined
from review_smplifyx_orientations123 import load, metrics
from review_smplifyx_root_seed123 import angle
import run_smplifyx_temporal_root123 as run


def main(max_new):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=run.verify_plan()
    out=run.OUT/'review';out.mkdir(exist_ok=True)
    frames_out=out/'frames';frames_out.mkdir(exist_ok=True)
    all_old=load(run.BASE/'fit/fitted.npz')
    ids=list(all_old['frame_ids'].astype(str))
    rows={r['frame']:r for r in run.read(run.BASE/'data/training-inputs.json')['records']}
    record=Path('/mnt/d/00/记录/demo');record.mkdir(exist_ok=True)
    mesh,renderer_source=mesh_renderer();cv2.setNumThreads(2)
    new=[]
    for frame in plan['processing_order']:
        folder=run.OUT/'frames'/frame
        receipt=folder/'result.json';saved=folder/'winner-camera-space.npz'
        reviewed=frames_out/(frame+'.json')
        if reviewed.exists():
            previous=run.read(reviewed)
            assert run.sha(saved)==previous['candidate_sha256'],frame
            continue
        if len(new)>=max_new or not receipt.exists() or not saved.exists():continue
        fit_receipt=run.read(receipt)
        assert fit_receipt['state']=='complete' and fit_receipt['frame_id']==frame
        digest=run.sha(saved)
        baseline=load(run.BASE/'frames'/frame/'winner-camera-space.npz')
        candidate=load(saved)
        row=rows[frame];assert run.sha(row['image'])==row['source_sha256']
        assert str(candidate['frame_id'].item())==frame
        for key in ['faces','betas','K_integer_centers','w2c','camera_translation']:
            assert np.array_equal(candidate[key],baseline[key]),(frame,key)
        target=all_old['observed_native_keypoints'][ids.index(frame)]
        before=metrics(baseline,target);after=metrics(candidate,target)
        raw=Image.open(row['image']).convert('RGB')
        assert list(raw.size)==row['wh']
        overlays=[mesh(raw,np.array([0,0,*raw.size]),data) for data in [baseline,candidate]]
        crops=run.read(run.BASE/'review'/(frame+'_saved-review.json'))
        files=[]
        for region in ['body','face','left_hand','right_hand','feet']:
            crop=next(x['crop_integer_xyxy'] for x in crops['files'] if x['region']==region)
            panels=[raw.crop(crop)]+[x.crop(crop) for x in overlays]
            path=frames_out/f'{frame}_{region}_raw_baseline_temporal.png'
            assert not path.exists()
            joined(panels).save(path)
            with Image.open(path) as im:assert np.array_equal(np.asarray(im)[:,:panels[0].width],np.asarray(panels[0]))
            destination=record/f'SMPL-X_连续朝向_{frame}_{region}_原图_独立初始化_本轮.png'
            assert not destination.exists()
            destination.write_bytes(path.read_bytes())
            files.append(dict(frame=frame,region=region,path=str(path),record_path=str(destination),sha256=run.sha(path),
                crop_integer_xyxy=crop,no_resampling=True,no_text_added=True,panels=['raw','independent_baseline','temporal_body_root']))
        assert run.sha(saved)==digest
        seed=fit_receipt['temporal_seed']['seed']
        seed_jump=None
        if seed:
            s=load(run.OUT/'frames'/seed/'winner-camera-space.npz')
            old_seed=load(run.BASE/'frames'/seed/'winner-camera-space.npz')
            seed_jump=dict(seed=seed,baseline_deg=angle(baseline['global_orient'],old_seed['global_orient']),
                candidate_deg=angle(candidate['global_orient'],s['global_orient']))
        result=dict(frame=frame,source_frame_index=row['source_frame_index'],updated=run.now(),
            candidate_sha256=digest,fit_receipt_sha256=run.sha(receipt),source_image_sha256=row['source_sha256'],
            baseline=before,candidate=after,seed_rotation_change=seed_jump,fit_seconds=fit_receipt['elapsed_seconds'],
            files=files,actual_visual_review=False,quality_accepted=False,
            unchanged_camera_shape_topology=True,source_script_sha256=run.sha(Path(__file__)),renderer_sha256=run.sha(renderer_source))
        run.save(reviewed,result);new.append(frame)
        print('CPU_REVIEW_GENERATED',frame,flush=True)
    results=[run.read(frames_out/(f+'.json')) for f in plan['temporal_roster'] if (frames_out/(f+'.json')).exists()]
    completed=[r['frame'] for r in results]
    pending=[f for f in plan['temporal_roster'] if f not in completed]
    aggregate={}
    for group in ['body17','face51','feet6','left_hand21','right_hand21']:
        old=np.array([r['baseline']['groups'][group]['median'] for r in results])
        new_values=np.array([r['candidate']['groups'][group]['median'] for r in results])
        if len(results):aggregate[group]=dict(baseline_median=float(np.median(old)),candidate_median=float(np.median(new_values)),
            worsened_frames=int(np.sum(new_values>old)),largest_regression=float(np.max(new_values-old)))
    summary=dict(updated=run.now(),state='all59_images_ready_pending_visual_review' if not pending else 'partial_completed_frames_only',
        generated_frames=completed,pending_frames=pending,frames=len(results),total=59,new_this_pass=new,aggregate=aggregate,
        elbows=[dict(frame=r['frame'],left_before=r['baseline']['left_elbow_px'],left_after=r['candidate']['left_elbow_px'],
            right_before=r['baseline']['right_elbow_px'],right_after=r['candidate']['right_elbow_px']) for r in results],
        source_candidate_progress=run.read(run.OUT/'progress.json'),actual_visual_review=False,quality_accepted=False,
        full59_complete=not pending,Gaussian_training_started=False,
        limits=['Partial rows cannot establish full-sequence improvement.','Detector fitting residuals are not independent physical truth.',
            'Original native crops and original observations are retained for every frame. No residual-based removal.',
            'CPU overlay is depth-sorted display, not exact z-buffer or recovered clothing surface.'])
    run.save(out/'summary.json',summary)
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    print(json.dumps({k:summary[k] for k in ['state','frames','total','new_this_pass','aggregate']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-new',type=int,default=10)
    args=p.parse_args();assert 1<=args.max_new<=59;main(args.max_new)
