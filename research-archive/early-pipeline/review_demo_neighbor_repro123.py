"""Compare both preregistered runs without selecting the better outcome."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import sys
import numpy as np
from PIL import Image
import run_official_gaussians123 as common


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    body = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/fullbody-smplifyx')
    plan = common.read(body/'neighbor-seed-plan.json')
    baseline = body/'full-sequence'
    runs = [body/name for name in plan['reproducibility']['outputs']]
    out = body/'neighbor-seed-repro-review'
    out.mkdir(exist_ok=False)
    helper_path = body/'review/review_anchor_saved.py'
    helper_hash = common.sha(helper_path)
    spec = importlib.util.spec_from_file_location('_existing_saved_review', helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    import cv2
    cv2.setNumThreads(2)
    comparisons, equality, images, timings = [], [], [], []
    for run in runs:
        progress = common.read(run/'progress.json')
        assert progress['state'] == 'complete'
        timings.append(dict(run=run.name, seconds=progress['elapsed_seconds']))
    for name in plan['registered_targets']:
        old = common.read(baseline/'review'/(name+'_saved-review.json'))
        reviews = []
        for run in runs:
            review_dir = run/'review'
            with redirect_stdout(io.StringIO()):
                helper.main(SimpleNamespace(frame_id=name, fit_dir=str(run/'frames'/name), review_dir=str(review_dir)))
            new = common.read(review_dir/(name+'_saved-review.json'))
            reviews.append(new)
            assert old['source_frame'] == new['source_frame']
            groups = {}
            for key, tolerance in (('body17',1.),('feet6',1.),('face51',.5)):
                a = old['new_detector_residuals_px'][key]['active_fit']
                b = new['new_detector_residuals_px'][key]['active_fit']
                assert a['count'] == b['count']
                groups[key] = dict(baseline=a,candidate=b,median_delta_px=b['median']-a['median'],
                    protection_tolerance_px=tolerance,protection_passed=b['median']-a['median']<=tolerance)
            for a,b in zip(old['per_joint'],new['per_joint']):
                assert a['index']==b['index'] and a['confidence']==b['confidence'] and a['active_fit']==b['active_fit']
            elbow0,elbow1=old['per_joint'][6]['error_px'],new['per_joint'][6]['error_px']
            comparisons.append(dict(frame_id=name,run=run.name,groups=groups,
                left_elbow_before_px=elbow0,left_elbow_after_px=elbow1,
                elbow50percent_reduction_passed=elbow1<=elbow0*.5,
                numerical_protection_passed=all(x['protection_passed'] for x in groups.values()),
                per_joint=new['per_joint']))
        with np.load(runs[0]/'frames'/name/'winner-camera-space.npz',allow_pickle=False) as za, np.load(runs[1]/'frames'/name/'winner-camera-space.npz',allow_pickle=False) as zb:
            assert set(za.files)==set(zb.files)
            for key in za.files:
                a,b=za[key],zb[key]
                same=np.array_equal(a,b)
                maximum=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.dtype.kind in 'fiu' and a.size else None
                equality.append(dict(frame_id=name,array=key,shape=list(a.shape),exact_equal=same,max_absolute_difference=maximum))
        for region in ('body','face','feet'):
            cells=[]
            crop=None
            for index,review in enumerate([old]+reviews):
                item=next(x for x in review['files'] if x['region']==region)
                assert common.sha(item['path'])==item['sha256']
                if crop is None:crop=item['crop_integer_xyxy']
                assert crop==item['crop_integer_xyxy']
                x0,y0,x1,y1=crop;w,h=x1-x0,y1-y0
                im=Image.open(item['path']).convert('RGB')
                raw=im.crop((0,0,w,h))
                if not cells:cells.append(raw)
                else:assert np.array_equal(np.asarray(cells[0]),np.asarray(raw))
                cells.append(im.crop((im.width-w,0,im.width,h)))
            merged=Image.new('RGB',(w*len(cells),h))
            for i,cell in enumerate(cells):merged.paste(cell,(i*w,0))
            path=out/(name+'_'+region+'_raw_baseline_run1_run2.png');merged.save(path)
            images.append(dict(frame_id=name,region=region,path=str(path),sha256=common.sha(path),
                crop_integer_xyxy=crop,panel_wh=[w,h],panels=['raw','original','repeat1','repeat2'],resampling=False,text_added=False))
    assert common.sha(helper_path)==helper_hash
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    result=dict(state='comparison_complete_actual_visual_review_pending',updated=common.now(),
        comparisons=comparisons,repeat_array_comparison=equality,
        repeats_exact_equal=all(x['exact_equal'] for x in equality),timings=timings,images=images,
        numerical_criteria_passed=all(x['elbow50percent_reduction_passed'] and x['numerical_protection_passed'] for x in comparisons),
        plan_sha256=common.sha(body/'neighbor-seed-plan.json'),helper_sha256=helper_hash,driver_sha256=common.sha(__file__),
        all_frame_roster_count=len(plan['temporal_schedule']),actually_fitted_unique_frames=len(plan['registered_targets']),
        selected_best_repeat=False,whole_sequence_quality_accepted=False,quality_accepted=False,
        gaussian_training_started=False,actual_visual_review=False,
        limitations=plan['scope_limits']+['Detector residuals are not independent physical ground truth.','Same-device repeatability does not establish cross-device bitwise identity.'])
    common.save(out/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('state','repeats_exact_equal','numerical_criteria_passed','timings','actually_fitted_unique_frames')}),flush=True)


if __name__=='__main__':main()
