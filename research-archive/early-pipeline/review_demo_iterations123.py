"""Same-native-pixel review of the approved original30/60 iteration control."""
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
    candidate, baseline = body/'iteration-control60', body/'full-sequence'
    out = candidate/'review'
    assert not (out/'comparison.json').exists()
    assert common.read(candidate/'progress.json')['state'] == 'complete'
    helper_path = body/'review/review_anchor_saved.py'
    spec = importlib.util.spec_from_file_location('_existing_saved_review', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source_hash = common.sha(helper_path)
    import cv2
    cv2.setNumThreads(2)
    plan = common.read(body/'iteration-control-plan.json')
    assert plan['user_approval']['answer'] == '执行三帧迭代对照（推荐）'
    comparisons, images = [], []
    for name in plan['frames']:
        review_path = out/(name+'_saved-review.json')
        assert not review_path.exists()
        with redirect_stdout(io.StringIO()):
            helper.main(SimpleNamespace(frame_id=name, fit_dir=str(candidate/'frames'/name), review_dir=str(out)))
        old = common.read(baseline/'review'/(name+'_saved-review.json'))
        new = common.read(review_path)
        for field in ('source_frame', 'shared_betas_frozen'):
            assert old[field] == new[field]
        for i, (p, q) in enumerate(zip(old['per_joint'], new['per_joint'])):
            assert p['index'] == q['index'] == i
            assert p['confidence'] == q['confidence'] and p['active_fit'] == q['active_fit']
        groups = {}
        for key, tolerance in (('body17', 1.), ('feet6', 1.), ('face51', .5)):
            a = old['new_detector_residuals_px'][key]['active_fit']
            b = new['new_detector_residuals_px'][key]['active_fit']
            assert a['count'] == b['count']
            groups[key] = dict(baseline=a, candidate=b, median_delta_px=b['median']-a['median'],
                protection_tolerance_px=tolerance, protection_passed=b['median']-a['median'] <= tolerance)
        elbow0, elbow1 = old['per_joint'][6]['error_px'], new['per_joint'][6]['error_px']
        comparisons.append(dict(frame_id=name, groups=groups,
            left_elbow_before_px=elbow0, left_elbow_after_px=elbow1,
            left_elbow_remaining_ratio=elbow1/elbow0,
            baseline_review_sha256=common.sha(baseline/'review'/(name+'_saved-review.json')),
            candidate_review_sha256=common.sha(review_path)))
        for region in ('body', 'face', 'feet'):
            a = next(v for v in old['files'] if v['region'] == region)
            b = next(v for v in new['files'] if v['region'] == region)
            assert a['crop_integer_xyxy'] == b['crop_integer_xyxy']
            x0, y0, x1, y1 = a['crop_integer_xyxy']; w, h = x1-x0, y1-y0
            im0, im1 = Image.open(a['path']).convert('RGB'), Image.open(b['path']).convert('RGB')
            assert common.sha(a['path']) == a['sha256'] and common.sha(b['path']) == b['sha256']
            assert np.array_equal(np.asarray(im0)[:, :w], np.asarray(im1)[:, :w])
            image = Image.new('RGB', (3*w, h))
            image.paste(im0.crop((0, 0, w, h)), (0, 0))
            image.paste(im0.crop((im0.width-w, 0, im0.width, h)), (w, 0))
            image.paste(im1.crop((im1.width-w, 0, im1.width, h)), (2*w, 0))
            path = out/(name+'_'+region+'_raw_iter30_iter60.png')
            assert not path.exists()
            image.save(path)
            images.append(dict(frame_id=name, region=region, path=str(path), sha256=common.sha(path),
                crop_integer_xyxy=a['crop_integer_xyxy'], panel_wh=[w, h],
                panels=['raw', 'original_maxiters30', 'original_maxiters60'],
                resampling=False, text_added=False))
    assert common.sha(helper_path) == source_hash
    assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
    failed = [dict(frame_id=x['frame_id'], group=k, median_delta_px=v['median_delta_px'])
        for x in comparisons for k, v in x['groups'].items() if not v['protection_passed']]
    target = next(v for v in comparisons if v['frame_id'] == 'frame_000018')
    elbow_pass = target['left_elbow_remaining_ratio'] <= .5
    result = dict(state='comparison_complete_actual_visual_review_pending', updated=common.now(),
        comparisons=comparisons, images=images, elbow50percent_reduction_passed=elbow_pass,
        numerical_protection_passed=not failed, protection_failures=failed,
        numerical_criteria_passed=elbow_pass and not failed,
        actual_visual_review=False, quality_accepted=False, full59_restart_admitted=False,
        gaussian_training_started=False, plan_sha256=common.sha(body/'iteration-control-plan.json'),
        helper_sha256=source_hash, driver_sha256=common.sha(__file__),
        scope='Fitted detector residuals, not independent physical ground truth. Fixed native crops and retained all original confidence/active observations; no threshold adjustment.')
    common.save(out/'comparison.json', result)
    print(json.dumps({k: result[k] for k in ('state', 'elbow50percent_reduction_passed',
        'numerical_protection_passed', 'protection_failures')}), flush=True)


if __name__ == '__main__':
    main()
