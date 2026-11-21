"""CPU-only matched body/root seed review; no frame or observation removal."""
import os
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image
import cv2
from review_full_sequence123 import mesh_renderer, joined
from review_smplifyx_orientations123 import load, metrics
import control_smplifyx_root_seed123 as run


def angle(a,b):
    R=cv2.Rodrigues(a.reshape(3))[0]@cv2.Rodrigues(b.reshape(3))[0].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=run.read(run.OUT/'result.json')
    assert result['state']=='complete_pending_review'
    plan=run.verify_plan()
    out=run.OUT/'review';out.mkdir(exist_ok=True)
    assert not (out/'comparison.json').exists()
    # The reused sequence receipt contains its default zero-start description.
    # Correct that metadata after the worker is terminal; never change a fit.
    metadata_amendments=[]
    for arm in run.ARMS:
        for relative in ['input-provenance.json','fit/official-fit-result.json']:
            path=run.OUT/arm/relative
            value=run.read(path)
            source=value if relative=='input-provenance.json' else value['source']
            before=run.sha(path)
            source['base_runner_default_initialization']=source['initialization']
            source['initialization']='Immediate previous baseline-frame VPoser latent'+(' and root orientation' if arm=='body_and_root' else '')+' after unchanged original camera initialization; matched diagnostic only'
            source['old_prior_or_optimized_fit_loaded']=True
            source['explicit_seed_schedule']=plan['temporal_schedule']
            source['initialization_adapter_sha256']=plan['source_hashes'][str(Path(run.__file__).resolve())]
            source['metadata_corrected_after_worker']=run.now()
            run.save(path,value)
            metadata_amendments.append(dict(path=str(path),before_sha256=before,after_sha256=run.sha(path),fit_arrays_changed=False))
    cv2.setNumThreads(2)
    original=load(run.BASE/'fit/fitted.npz')
    ids=list(original['frame_ids'].astype(str))
    rows={r['frame']:r for r in run.read(run.BASE/'data/training-inputs.json')['records']}
    mesh,renderer_source=mesh_renderer()
    record=Path('/mnt/d/00/记录/demo')
    compared=[];images=[]
    for frame in run.FRAMES:
        row=rows[frame]
        assert run.sha(row['image'])==row['source_sha256']
        raw=Image.open(row['image']).convert('RGB')
        targets=original['observed_native_keypoints'][ids.index(frame)]
        data=[load(run.BASE/'frames'/frame/'winner-camera-space.npz')]
        data += [load(run.OUT/arm/'frames'/frame/'winner-camera-space.npz') for arm in run.ARMS]
        seed=next(x['seed'] for x in plan['temporal_schedule'] if x['target']==frame)
        seed_data=load(run.BASE/'frames'/seed/'winner-camera-space.npz')
        for d in data:
            for key in ['faces','betas','K_integer_centers','camera_translation']:
                assert np.array_equal(d[key],data[0][key]),(frame,key)
        met=[metrics(d,targets) for d in data]
        protected={}
        for ref_index,label in [(0,'independent_baseline'),(1,'body_only')]:
            protected[label]={k:dict(delta_px=met[2]['groups'][k]['median']-met[ref_index]['groups'][k]['median'],
                tolerance_px=limit,passed=met[2]['groups'][k]['median']-met[ref_index]['groups'][k]['median']<=limit)
                for k,limit in [('body17',1.),('feet6',1.),('face51',.5)]}
        overlays=[mesh(raw,np.array([0,0,*raw.size]),d) for d in data]
        crop_ref=run.read(run.BASE/'review'/(frame+'_saved-review.json'))
        frame_images=[]
        for region in ['body','face','left_hand','right_hand','feet']:
            crop=next(i['crop_integer_xyxy'] for i in crop_ref['files'] if i['region']==region)
            panels=[raw.crop(crop)]+[p.crop(crop) for p in overlays]
            path=out/f'{frame}_{region}_raw_baseline_body_root.png'
            assert not path.exists()
            joined(panels).save(path)
            with Image.open(path) as check:
                assert np.array_equal(np.asarray(check)[:,:panels[0].width],np.asarray(panels[0]))
            destination=record/f'SMPL-X_邻帧整体朝向_{frame}_{region}_原图_独立初始化_仅身体_身体与整体.png'
            assert not destination.exists()
            destination.write_bytes(path.read_bytes())
            item=dict(frame=frame,region=region,path=str(path),record_path=str(destination),sha256=run.sha(path),
                crop_integer_xyxy=crop,panels=['raw','independent_baseline','predecessor_body','predecessor_body_and_root'],
                no_resampling=True,no_text_added=True)
            frame_images.append(item);images.append(item)
        compared.append(dict(frame=frame,seed=seed,baseline=met[0],body_only=met[1],body_and_root=met[2],
            protections=protected,left_elbow_remaining_ratio_vs_body=met[2]['left_elbow_px']/met[1]['left_elbow_px'],
            root_change_from_predecessor_degrees={name:angle(d['global_orient'],seed_data['global_orient'])
                for name,d in zip(['baseline','body_only','body_and_root'],data)},images=frame_images))
    reduction=all(r['left_elbow_remaining_ratio_vs_body']<=.5 for r in compared if r['frame'] in ('frame_000016','frame_000018'))
    protections=all(p['passed'] for r in compared for ref in r['protections'].values() for p in ref.values())
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    output=dict(state='comparison_complete_visual_review_pending',updated=run.now(),frames=compared,images=images,
        elapsed_seconds=result['elapsed_seconds'],historical_body_only_parity=result['historical_body_only_parity'],
        matched_camera_initialization=result['matched_camera_initialization'],elbow_reduction_passed=reduction,
        protections_passed=protections,numerical_criteria_passed=reduction and protections,
        actual_visual_review=False,quality_accepted=False,full59_admitted=False,Gaussian_training_started=False,
        metadata_amendments=metadata_amendments,
        source_hashes={str(p):run.sha(p) for p in [Path(__file__),renderer_source,run.PLAN,run.OUT/'result.json']},
        scope='Fixed diagnostic subset with old independent predecessor fits, not a full sequential pipeline; detector fitting residuals are not independent physical truth. CPU overlay uses existing depth-sorted mesh display.')
    run.save(out/'comparison.json',output)
    print(json.dumps({k:output[k] for k in ['elapsed_seconds','elbow_reduction_passed','protections_passed','numerical_criteria_passed']}))
    for r in compared:
        print(r['frame'],json.dumps({k:r[k]['left_elbow_px'] for k in ['baseline','body_only','body_and_root']}))


if __name__=='__main__':main()
