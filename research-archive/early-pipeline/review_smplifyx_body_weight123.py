"""Native CPU review of the single registered body-weight diagnostic."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from review_full_sequence123 import mesh_renderer, joined
from review_smplifyx_orientations123 import load, metrics
import control_smplifyx_body_weight123_v2 as repair

control=repair.control

b=control.b


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=b.read(b.OUT/'result.json');assert result['unchanged_camera_shape_topology']
    ref=b.read(control.REFERENCE/'result.json');assert ref['endpoint_parity_passed']
    b.verify_plan();frame=b.FRAME;out=b.OUT/'review';out.mkdir(exist_ok=True)
    assert not (out/'comparison.json').exists()
    base=b.BODY/'full-sequence';current=control.trace.PREVIOUS
    all_before=load(base/'fit/fitted.npz');ids=list(all_before['frame_ids'].astype(str))
    target=all_before['observed_native_keypoints'][ids.index(frame)]
    row=next(x for x in b.read(base/'data/training-inputs.json')['records'] if x['frame']==frame)
    data=[load(base/'frames'/frame/'winner-camera-space.npz'),load(current/'frames'/frame/'winner-camera-space.npz'),load(b.OUT/'traced-winner.npz')]
    for key in ('faces','betas','K_integer_centers','camera_translation'):
        assert all(np.array_equal(item[key],data[0][key]) for item in data)
    names=['independent','warmtail','body_weight_2']
    comparison={name:metrics(item,target) for name,item in zip(names,data)}
    raw=Image.open(row['image']).convert('RGB');assert b.sha(row['image'])==row['source_sha256']
    mesh,renderer_source=mesh_renderer();cv2.setNumThreads(2)
    overlays=[mesh(raw,np.array([0,0,*raw.size]),item) for item in data]
    crops=b.read(base/'review'/(frame+'_saved-review.json'))
    record=Path('/mnt/d/00/记录/demo');record.mkdir(exist_ok=True)
    images=[]
    for region in ('body','face','left_hand','right_hand','feet'):
        crop=next(x['crop_integer_xyxy'] for x in crops['files'] if x['region']==region)
        panels=[raw.crop(crop)]+[image.crop(crop) for image in overlays]
        path=out/(frame+'_'+region+'_raw_independent_warmtail_bodyweight2.png')
        assert not path.exists();joined(panels).save(path)
        with Image.open(path) as check:assert np.array_equal(np.asarray(check)[:,:panels[0].width],np.asarray(panels[0]))
        dest=record/('SMPL-X_身体权重单变量_'+frame+'_'+region+'_原图_独立_原精修_身体权重2.png')
        assert not dest.exists();dest.write_bytes(path.read_bytes())
        images.append(dict(region=region,path=str(path),record_path=str(dest),sha256=b.sha(path),crop_integer_xyxy=crop,no_resampling=True,no_text_added=True))
    protections={group:dict(delta_px=comparison['body_weight_2']['groups'][group]['median']-comparison['independent']['groups'][group]['median'],limit_px=limit)
        for group,limit in [('body17',1.),('feet6',1.),('face51',.5)]}
    for d in protections.values():d['passed']=d['delta_px']<=d['limit_px']
    last=result['evaluations'][-1]
    original_weight_score=last['total']-.75*(last['body_data']+last['feet_data'])
    summary=dict(updated=b.now(),frame=frame,state='native_comparison_generated_pending_review',comparison=comparison,protections=protections,
        numeric_gate_passed=all(v['passed'] for v in protections.values()),images=images,
        fixed_camera_shape_topology=True,original_weight_objective=dict(new_candidate_reweighted_to_original=original_weight_score,
            original_warmtail_total=ref['evaluations'][-1]['total'],changed_weight_total=last['total'],
            algebra='Original-weight total = changed-weight total - 0.75*(changed body_data + changed feet_data). Only these squared observation weights changed.'),
        source_hashes={str(p):b.sha(p) for p in [Path(__file__),renderer_source,b.PLAN,b.OUT/'result.json',b.OUT/'traced-winner.npz']},
        actual_visual_review=False,quality_accepted=False,full59_admitted=False,Gaussian_training_started=False,goal_complete=False)
    b.save(out/'comparison.json',summary)
    print(json.dumps({k:summary[k] for k in ('frame','numeric_gate_passed','protections','original_weight_objective')}),flush=True)


if __name__=='__main__':main()
