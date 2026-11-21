"""Fixed video22 native candidate-pool entry, reusing unchanged core124 code.

audit is CPU-only and does not decode, observe, fit or create experiment output.
decode/observe require the root's visible-region admission. That admission does
not establish actor identity, a training roster or one shared body pose.
"""
import argparse
import inspect
import json
import os
import types
from pathlib import Path

import numpy as np
import prepare_core_subjects124 as core
from run_official_gaussians123 import ASSETS, PROJECT, now, read, save, sha

SUBJECT = 'leafwoman'
VIDEO_SHA256 = 'c5aac80e574dba892137d59186ff17713ab0403ec1a969d788b054a364f68cd0'
NATIVE_WH = (2704, 1520)
DATA = Path('/opt/gs-slam-lab/shuolee-cross-subject-129')
SOURCE = core.REC/'new-person-intake129/video22-arc/arc-intake.json'
SOURCE_REVIEW = SOURCE.with_name('source-availability-review.json')
ADMISSION = SOURCE.with_name('root-source-admission.json')
PINS = {
    Path(core.__file__): 'e388e65e945f56c2fad0ee355fdb31f05ff85878bc5fc8280cf17b49360b14a8',
    Path(__file__).with_name('preprocess_demo_fps3_123.py'): '9d99f91032cd82ad36eac1b13ddab7db3e7b84cd4fa2ca35cd2328eee4566ac6',
    PROJECT/'src/shuolee_portrait/observations.py': 'ea7e6998d92583bd44e0e447ee26f91c806350e58df3d56ff4c9449f5caa35b7',
}
SOURCE_PINS = {
    SOURCE: '3eeb20f2167360c6b37a2bbed88fa870055d56e5cf9b4e451b57c2861875e5fc',
    SOURCE_REVIEW: '1c88bfe2d1b1aab5f952def9d4586a53cc30c988d8ee88ac0888b603599ad972',
}


def paths(subject):
    assert subject == SUBJECT
    return (Path('/mnt/d/00/工作/数据集/video22.mp4'), DATA/SUBJECT,
            core.REC/'leafwoman-native129', '022', VIDEO_SHA256)


def original_entry(mode):
    """Rebind paths only; preserve the original function's exact code object."""
    function = getattr(core, mode)
    scope = dict(core.__dict__, DATA=DATA, paths=paths, __file__=__file__,
                 SUBJECTS={SUBJECT: ('video22.mp4', '022', VIDEO_SHA256)})
    result = types.FunctionType(function.__code__, scope, function.__name__,
                                function.__defaults__, function.__closure__)
    assert result.__code__ is function.__code__
    return result


def verify(require_admission):
    for path, digest in PINS.items():
        assert sha(path) == digest, str(path)
    for path, digest in SOURCE_PINS.items():
        assert sha(path) == digest, str(path)
    source, review = read(SOURCE), read(SOURCE_REVIEW)
    assert source['source_sha256_before'] == source['source_sha256_after'] == VIDEO_SHA256
    assert source['source_wh'] == list(NATIVE_WH)
    assert source['source_stream']['color_transfer'] == 'bt709'
    assert review['actual_six_contact_and_all6_native_crops_viewed'] is True
    assert review['source_video_sha256'] == VIDEO_SHA256
    assert review['physical_person_identity_independently_verified'] is False
    assert review['static_shared_local_pose']['supported_without_further_evidence'] is False
    if require_admission:
        admission = read(ADMISSION)
        assert admission['source_video_sha256'] == VIDEO_SHA256
        for key in ('actual_source_visual_review', 'new_capture_instance_admitted',
                    'visible_region_scope_admitted', 'cropped_or_occluded_regions_may_be_ignored',
                    'source_only_stable_pose_group_review_required_before_fit'):
            assert admission[key] is True, key
        assert admission['physical_person_identity_independently_verified'] is False
        assert admission['training_view_target'] == 12
        return source, review, admission
    return source, review, None


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only audit required.'
    source, review, _ = verify(False)
    video, data, report, record, digest = paths(SUBJECT)
    assert sha(video) == digest
    assert not data.exists() and not report.exists(), 'Inspect an existing intake; never repeat.'
    manifest = read(PROJECT/'configs/server-assets.json')['files']
    assets = {}
    for name in ('detection/yolov8n.pt', 'sam2/sam2.1_hiera_tiny.pt',
                 'rtmw/rtmw-l-384x288.onnx', 'rtmw/normalization.json'):
        entry = next(item for item in manifest if item['path'] == name)
        assert sha(ASSETS/name) == entry['sha256'], name
        assets[name] = entry['sha256']
    # Check the original conditional K algebra without probing/decoding again.
    import cv2
    fs = cv2.FileStorage(str(core.CALIB), cv2.FILE_STORAGE_READ)
    K0 = fs.getNode('camera_matrix').mat()
    wh = (int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real()))
    fs.release()
    assert wh == (4000, 3000) and K0.shape == (3, 3) and np.isfinite(K0).all()
    A = core.integer_resize_matrix(wh, (2704, 2028)); A[1, 2] -= 254
    Ki = A @ K0
    K = Ki.copy(); K[:2, 2] += .5
    entries = {mode: original_entry(mode) for mode in ('decode', 'observe')}
    exact = {mode: entries[mode].__code__ is getattr(core, mode).__code__ for mode in entries}
    assert all(exact.values())
    assert 'torch' not in __import__('sys').modules, 'Audit must not import torch.'
    receipt = dict(updated=now(), state='cpu_static_audit_passed_no_preprocessing',
        helper_sha256=sha(__file__), subject=SUBJECT, record_id=record,
        source_video=str(video), source_video_sha256=digest, native_wh=list(NATIVE_WH),
        output_data_root=str(data), output_report_root=str(report), root_admission_path=str(ADMISSION),
        source_pins={str(p): h for p, h in SOURCE_PINS.items()},
        implementation_pins={str(p): h for p, h in PINS.items()},
        asset_sha256=assets, calibration_sha256=sha(core.CALIB),
        original_function_code_objects_unchanged=exact,
        implementation='Exact core124 decode/observe functions; only paths, subject and __file__ rebound.',
        original_function_source_sha256={mode: __import__('hashlib').sha256(inspect.getsource(getattr(core, mode)).encode()).hexdigest() for mode in entries},
        sample_fps=3, sampling='Original nearest real PTS sampler; source is not decoded/probed again by this audit.',
        K_source=K0.tolist(), source_to_native_integer=A.tolist(), K_integer_centers=Ki.tolist(), K_corner=K.tolist(),
        conditional_K=True, physical_calibration_accepted=False, second_undistortion=False,
        source_scope='New clothing/capture instance; physical actor identity independently unverified.',
        visible_region_scope='Visible upper body and skirt. Cropped/occluded lower body remains unknown.',
        training_view_target=12, training_roster_frozen=False,
        fitting_gate='Actual source-only stable-pose grouping is required before fitting. Raised-hand samples cannot silently join chest-hand shared-pose training.',
        all_frames_share_one_local_pose_accepted=False, stable_pose_groups_created=False,
        source_pixels_or_masks_modified=False, fresh_intake_outputs_created=False,
        intake_repeated=False, ffprobe_or_decode_executed=False, GPU_executed=False,
        detectors_or_masks_executed=False, fitting_executed=False, Gaussian_training_executed=False,
        quality_accepted=False)
    save(Path(__file__).with_suffix('.audit.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def run(mode):
    _, _, admission = verify(True)
    original_entry(mode)(SUBJECT)
    # Additional scope/identity metadata only; the original scientific entry is unchanged.
    plan_path = paths(SUBJECT)[2]/'plan.json'
    plan = read(plan_path)
    plan.update(root_source_admission=str(ADMISSION), root_source_admission_sha256=sha(ADMISSION),
                new_capture_instance_admitted=True, physical_person_identity_independently_verified=False,
                visible_region_scope_admitted=True, cropped_or_occluded_regions_may_be_ignored=True,
                source_only_stable_pose_group_review_required_before_fit=True,
                all_frames_share_one_local_pose_accepted=False, training_roster_frozen=False,
                training_view_target=admission['training_view_target'])
    save(plan_path, plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit', 'decode', 'observe'])
    args = parser.parse_args()
    if args.mode == 'audit':
        audit()
    else:
        run(args.mode)
