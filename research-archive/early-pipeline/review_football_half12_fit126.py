"""CPU-only native/synthetic review of the completed football12 actual fit.

Reuse the existing reviewer; only subject paths/names and an empty-hand crop
guard differ. Crop fallbacks are diagnostic and never create fit observations.
"""
import ast
import difflib
import hashlib
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import review_sofa_half8_fit124 as existing
import run_football_half12_fit126 as fit
from run_official_gaussians123 import save, sha, now


def hand_review_crop(arrays, body_box, wh):
    target = np.asarray(arrays['observed_native_keypoints'])[0]
    assert target.shape == (118, 3)
    fingers = target[25:67]
    good = (fingers[:, 2] > 0) & np.isfinite(fingers).all(1)
    points = fingers[good, :2]
    if len(points):
        return existing.crop_box(np.r_[points.min(0), points.max(0)], wh, 55), 'source_finger_2d_crop'
    wrists = target[[4, 7]]
    good = ((wrists[:, 2] > 0) & np.isfinite(wrists).all(1) &
            (wrists[:, 0] >= 0) & (wrists[:, 0] < wh[0]) &
            (wrists[:, 1] >= 0) & (wrists[:, 1] < wh[1]))
    points = wrists[good, :2]
    if len(points):
        return existing.crop_box(np.r_[points.min(0), points.max(0)], wh, 55), 'diagnostic_body_wrist_bbox_fallback_no_finger_observations'
    return list(body_box), 'diagnostic_person_mask_bbox_fallback_no_hand_observations'


original_source = inspect.getsource(existing.main)
source = original_source
changes = [
    ("Path('/mnt/d/00/记录/006')", "Path('/mnt/d/00/记录/011')", 2),
    ('沙发半圈8帧124', '足球人物半圈12帧126', 2),
    ('SOFA_HALF8_NATIVE_REVIEW_READY', 'FOOTBALL_HALF12_NATIVE_REVIEW_READY', 1),
    ("        h=a['observed_native_keypoints'][0,25:67];h=h[h[:,2]>0,:2]",
     '        hand_box,hand_crop_basis=hand_review_crop(a,bb,raw.size)', 1),
    ('hands=crop_box(np.r_[h.min(0),h.max(0)],raw.size,55),', 'hands=hand_box,', 1),
    ('            images[region]=dict(path=str(dest),sha256=sha(dest),native_crop=b,resized=False)',
     "            images[region]=dict(path=str(dest),sha256=sha(dest),native_crop=b,resized=False)\n            if region=='hands':images[region].update(diagnostic_crop_basis=hand_crop_basis,source_observations_modified=False)", 1),
]
for old, new, count in changes:
    assert source.count(old) == count, (old, source.count(old), count)
    source = source.replace(old, new)
assert "for row in p['rows']:" in source
assert 'actual_visual_review=False' in source and 'quality_accepted=False' in source
ast.parse(source)
namespace = dict(existing.main.__globals__, fit=fit, __file__=__file__, hand_review_crop=hand_review_crop)
exec(compile(source, '<same-native-football12-fit-review>', 'exec'), namespace)


def audit():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    sample = np.zeros((1, 118, 3), np.float32)
    sample[0, 25] = [600, 700, .8]
    sample[0, 26] = [650, 740, .7]
    box = [300, 100, 1600, 1500]
    wh = [2704, 1520]
    snapshots = []
    for case in ('fingers', 'empty_fingers_supported_wrists', 'empty_fingers_and_wrists'):
        target = sample.copy()
        if case != 'fingers':
            target[0, 25:67, 2] = 0
        if case == 'empty_fingers_supported_wrists':
            target[0, 4] = [700, 800, .8]
            target[0, 7] = [850, 900, .9]
        before = target.copy()
        crop, basis = hand_review_crop(dict(observed_native_keypoints=target), box, wh)
        assert np.array_equal(target, before)
        if case == 'fingers':
            assert crop == existing.crop_box([600, 700, 650, 740], wh, 55)
        elif case == 'empty_fingers_supported_wrists':
            assert crop == existing.crop_box([700, 800, 850, 900], wh, 55)
        else:
            assert crop == box
        snapshots.append(dict(case=case, crop=crop, diagnostic_crop_basis=basis, input_unchanged=True))
    compile(Path(__file__).read_text(), str(Path(__file__)), 'exec')
    assert not torch.cuda.is_initialized()
    save(Path(__file__).with_name('review_football_half12_fit126_cpu_audit.json'),
        dict(updated=now(), mode='CPU_static_preparation_only', passed=True,
            helper_sha256=sha(__file__), base_reviewer=str(Path(existing.__file__)), base_reviewer_sha256=sha(existing.__file__),
            football_fit_helper_sha256=sha(fit.__file__), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            interface_diff=''.join(difflib.unified_diff(original_source.splitlines(True), source.splitlines(True),
                fromfile='existing-sofa-native-review', tofile='football12-native-review')),
            review_rows_dynamic=True, actual_fit_process_exit_zero_required=True,
            same_native_body_face_hands_feet=True, synthetic_mesh_directions=[0, 90, 180, -90],
            record_directory='/mnt/d/00/记录/011', labels_inside_record_images=False,
            empty_hand_crop_cases=snapshots, source_observations_modified=False,
            renders_executed=False, actual_visual_review=False, quality_accepted=False,
            cuda_initialized=False, optimizer_updates=0, fit_or_Gaussian_helpers_modified=False))
    print('FOOTBALL12_REVIEW_CPU_STATIC_READY', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'render'
    assert mode in ('audit', 'render')
    audit() if mode == 'audit' else namespace['main']()
