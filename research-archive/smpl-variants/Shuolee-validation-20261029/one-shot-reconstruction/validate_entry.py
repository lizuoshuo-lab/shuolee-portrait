from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

project = Path('/mnt/d/Projects/Shuolee')
out = Path('/mnt/d/Projects/Shuolee-validation-20261029/one-shot-reconstruction')
data = Path('/mnt/d/00/工作/数据集')
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()
def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

camera = out / 'camera-unresolved.json'
save(camera, dict(schema='smplx-camera-v1', intrinsics=dict(
    input_domain='unresolved_exported_video', source_model='FISHEYE_8',
    source_calibration=str(data / 'mono_main_calib.yaml'),
    source_calibration_sha256=sha(data / 'mono_main_calib.yaml'),
    provenance='Original sensor YAML present; exported-video rectification K, exact crop and stabilization mapping not established.')))
profile = out / 'local-input-validation-profile.json'
save(profile, dict(schema='smplx-deployment-v1', environment=sys.prefix,
    assets='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets', runs=str(out / 'runs'),
    validation_scope='Local CPU input validation only; not a deployed server profile.', doctor_completed=False))
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(project / 'src'),
           PYTHONNOUSERSITE='1', NO_PROXY='localhost,127.0.0.1,::1')
records = []
for video in ('video3.mp4', 'video4.mp4'):
    run = out / ('actual-' + Path(video).stem)
    command = [sys.executable, '-s', str(project / 'run_smplx.py'), 'train', str(data / video),
        '--camera-config', str(camera), '--deployment', str(profile), '--output', str(run)]
    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=90)
    (out / (Path(video).stem + '-entry.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
    status = json.loads((run / 'status.json').read_text())
    input_info = json.loads((run / 'input.json').read_text())
    records.append(dict(command=command, exit_code=result.returncode, status=status, input=input_info,
                        first_failure_expected=True, actual_training=False))
    assert result.returncode == 1 and status['error_code'] == 'CAMERA_PIXEL_DOMAIN_UNRESOLVED'
    assert input_info['first_frame_decoded'] and not status['training_started']
save(out / 'real-entry-validation.json', dict(updated=datetime.now(timezone.utc).isoformat(),
    python=sys.version, platform=platform.platform(), runs=records,
    expected_failure_behavior_verified=True, camera_mapping_verified=False,
    full_preprocessing_verified=False, real_training_verified=False,
    server_verified=False, quality_accepted=False))
print(json.dumps(dict(runs=len(records), expected_failure_behavior_verified=True,
                     first_error='CAMERA_PIXEL_DOMAIN_UNRESOLVED', real_training_verified=False)))
