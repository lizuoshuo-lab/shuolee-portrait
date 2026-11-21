"""Minimal new-subject CLI over the unchanged successful native intake.

Only audit is CPU/read-only with respect to experiments. Decode and observe
must be invoked explicitly after the root agent's review. No body prior,
fitting, old personalized geometry, Gaussian or optimizer is loaded here.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import prepare_core_subjects124 as core
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS

SUBJECT = 'football'
VIDEO_SHA256 = '4e1c1e52818362dcab22bb3ff9548a2d1a8c8b3f6affb8bd13dacfcc12a706a0'
PINS = {
    Path(__file__).with_name('prepare_core_subjects124.py'): 'e388e65e945f56c2fad0ee355fdb31f05ff85878bc5fc8280cf17b49360b14a8',
    Path(__file__).with_name('preprocess_demo_fps3_123.py'): '9d99f91032cd82ad36eac1b13ddab7db3e7b84cd4fa2ca35cd2328eee4566ac6',
    PROJECT/'src/shuolee_portrait/observations.py': 'ea7e6998d92583bd44e0e447ee26f91c806350e58df3d56ff4c9449f5caa35b7',
}
# Runtime configuration only; do not edit either reusable implementation.
core.DATA = Path('/opt/gs-slam-lab/shuolee-cross-subject-126')
core.SUBJECTS = {SUBJECT: ('video11.mp4', '011', VIDEO_SHA256)}
original_paths = core.paths


def paths(subject):
    video, data, _, record, digest = original_paths(subject)
    return video, data, core.REC/'football-native126', record, digest


core.paths = paths


def audit():
    video, data, report, record, digest = paths(SUBJECT)
    assert sha(video) == digest, 'New-video content changed; review intake first.'
    for path, expected in PINS.items():
        assert sha(path) == expected, str(path)
    assert not data.exists() and not report.exists(), 'Existing native intake: inspect instead of restarting.'
    manifest = read(PROJECT/'configs/server-assets.json')['files']
    assets = {}
    for name in ['detection/yolov8n.pt', 'sam2/sam2.1_hiera_tiny.pt',
                 'rtmw/rtmw-l-384x288.onnx', 'rtmw/normalization.json']:
        expected = next(row for row in manifest if row['path'] == name)
        assert sha(ASSETS/name) == expected['sha256'], name
        assets[name] = expected['sha256']
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_streams', '-show_entries',
        'frame=best_effort_timestamp_time:stream', '-of', 'json', str(video)]))
    stream = probe['streams'][0]
    assert (stream['width'], stream['height']) == (2704, 1520)
    assert not stream.get('tags', {}).get('rotate')
    assert all(float(x.get('rotation', 0)) == 0 for x in stream.get('side_data_list', []))
    assert stream.get('color_transfer') not in ('smpte2084', 'arib-std-b67')
    times = np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    grid, ids = core.sampling_indices(times)
    result = dict(updated=now(), state='cpu_configuration_audit_passed', subject=SUBJECT,
        video=str(video), video_sha256=digest, record_id=record,
        decoded_frame_count=len(times), sample_fps=3, samples=len(ids),
        source_indices=ids.tolist(), source_times_seconds=times[ids].tolist(),
        maximum_sampling_error_seconds=float(np.max(abs(times[ids]-grid))),
        native_wh=[2704, 1520], output_data_root=str(data), output_report_root=str(report),
        source_pixel_format=stream.get('pix_fmt'), decoded_pixel_format='rgb24_8bit',
        source_color_transfer=stream.get('color_transfer'),
        capture_scope='Source-contact review: one side through front to opposite side; back not captured. Not a calibrated180-degree claim.',
        half_circle_training_target=12, full_circle_training_target=24,
        reference_selection='Later deterministic source clarity / open-eye and fit quality; not the first frame by default.',
        hand_observation_policy='Fingers near/in trouser pockets are unknown; uniform hand2D weights0 in later official fitting, body wrists and pose priors retained.',
        implementation='Unchanged prepare_core_subjects124.decode/observe via runtime paths only',
        source_pins={str(p): digest for p, digest in PINS.items()}, asset_sha256=assets,
        detectors='YOLOv8n CUDA + RTMW133 ONNX CPU flip-average + SAM2 tiny CUDA',
        second_undistortion=False, camera_estimation=False, prior_or_GVHMR_called=False,
        conditional_K=True, physical_calibration_accepted=False, geometry_fitted=False, Gaussian_started=False,
        quality_accepted=False, GPU_started=False,
        note='Native candidate pool only. Half-circle12 / full-circle24 roster is selected after source observation review, not here.')
    save(Path(__file__).with_suffix('.audit.json'), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit', 'decode', 'observe'])
    args = parser.parse_args()
    if args.mode == 'audit':
        audit()
    else:
        for path, digest in PINS.items():
            assert sha(path) == digest, str(path)
        getattr(core, args.mode)(SUBJECT)
