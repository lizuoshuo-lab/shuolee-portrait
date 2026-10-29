"""Single video reconstruction entry and recorded failure boundary.

The framewise numerical pipeline is integrated through ``sequence_pipeline``.
Official source, camera and shared-shape inputs are checked before fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import uuid

import numpy as np

from . import runtime as rt
from .video119 import fixed_intrinsics, settings_for, validate_video_stream


class ReconstructionError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def camera_configuration(path, source_wh):
    """Validate the exported video's pixel domain before any expensive work."""
    cfg = rt.read(path)
    if cfg.get('schema') != 'smplx-camera-v1':
        raise ReconstructionError('CAMERA_CONFIG_SCHEMA',
            'Expected smplx-camera-v1 camera configuration for this exported video; '
            'a sensor calibration or historical capture profile is not interchangeable.')
    profile = cfg.get('intrinsics', {})
    if not isinstance(profile, dict) or profile.get('input_domain') != 'rectified_pinhole':
        raise ReconstructionError('CAMERA_PIXEL_DOMAIN_UNRESOLVED',
            'The exported video pixel domain is unresolved. Supply its pinhole K and '
            'rectification/crop provenance, or integrate a distortion-aware camera frontend. '
            'A raw fisheye calibration cannot be treated as an already rectified video.')
    if profile.get('mode') != 'fixed_pinhole':
        raise ReconstructionError('CAMERA_MAPPING_UNSUPPORTED',
            'This entry requires K in the actual exported video dimensions. '
            'Sensor center-crop assumptions are not applied automatically.')
    if profile.get('fixed_processing_across_frames') is not True:
        raise ReconstructionError('CAMERA_PROCESSING_UNRESOLVED',
            'A shared K requires an explicit fixed image-processing mapping across frames; '
            'unknown stabilization or per-frame crops cannot be treated as a fixed camera.')
    distortion = profile.get('distortion')
    if (not isinstance(distortion, list) or not distortion or
            any(type(v) not in (float, int) or not np.isfinite(v) or v != 0 for v in distortion)):
        raise ReconstructionError('CAMERA_DISTORTION_UNRESOLVED',
            'Rectified input must explicitly declare finite zero distortion coefficients.')
    # At this point K is in the SOURCE video domain, before extraction resize.
    fixed_intrinsics(profile, source_wh, source_wh)
    sampling = cfg.get('sampling', {})
    if not isinstance(sampling, dict) or set(sampling) - {
            'frame_stride', 'max_image_size', 'min_frames', 'max_frames'}:
        raise ValueError('sampling accepts frame_stride, max_image_size, min_frames and max_frames only.')
    effective = {**sampling, 'intrinsics': profile}
    settings = settings_for(effective)  # Explicit intrinsics prevents the old default K.
    effective = {**settings, 'intrinsics': profile}
    effective['camera_length_contract'] = cfg.get('camera_length_contract')
    effective['official_smplifyx'] = cfg.get('official_smplifyx')
    return cfg, effective


def inspect_video(video):
    for executable in ('ffprobe', 'ffmpeg'):
        if not shutil.which(executable):
            raise ReconstructionError('VIDEO_TOOL_MISSING', f'Missing {executable}; install FFmpeg.')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_streams', '-of', 'json', str(video)], check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout).get('streams', [])
    if len(streams) != 1:
        raise ValueError('Expected one selected video stream.')
    stream = streams[0]
    validate_video_stream(stream)
    wh = [int(stream['width']), int(stream['height'])]
    # A real frame is decoded; this is an input check, not full preprocessing.
    from io import BytesIO
    from PIL import Image
    sample = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-v', 'error',
        '-noautorotate', '-i', str(video), '-frames:v', '1', '-f', 'image2pipe',
        '-vcodec', 'png', '-pix_fmt', 'rgb24', 'pipe:1'], check=True, capture_output=True)
    with Image.open(BytesIO(sample.stdout)) as image:
        image.load()
        if list(image.size) != wh:
            raise ValueError('Decoded frame dimensions differ from ffprobe.')
    return dict(source_wh=wh, orientation='landscape' if wh[0] > wh[1] else
                'portrait' if wh[1] > wh[0] else 'square',
                first_frame_decoded=True, full_preprocessing_complete=False)


def sequence_pipeline():
    name = 'shuolee_portrait.sequence_pipeline'
    if importlib.util.find_spec(name) is None:
        raise ReconstructionError('PIPELINE_INTEGRATION_INCOMPLETE',
            'This revision has no complete video-to-framewise-SMPL-X training connection. '
            'The missing connection is fresh observations -> official named-frame fit '
            '-> frame-pose bundle -> free-clothing training -> static reference export. '
            'Training has not started. Update to an integrated revision; the static trainer is not used.')
    implementation = importlib.import_module(name)
    if not callable(getattr(implementation, 'reconstruct', None)):
        raise ReconstructionError('PIPELINE_INTERFACE_INVALID',
                                  'sequence_pipeline.reconstruct is missing.')
    return implementation.reconstruct


def train_video(video, deployment, camera_config, *, output=None, reference_frame='frame_000001'):
    profile = rt.read(deployment)
    if profile.get('schema') not in ('shuolee-deployment119-v1', 'smplx-deployment-v1'):
        raise ValueError('Run bash deploy.sh to create the deployment configuration.')
    video = Path(video).expanduser().resolve()
    camera_config = Path(camera_config).expanduser().resolve()
    if output is None:
        label = re.sub(r'[^\w.-]+', '_', video.stem).strip('._')[:48] or 'video'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        output = Path(profile['runs']) / f'{stamp}-{label}-{uuid.uuid4().hex[:8]}'
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    stage = 'input'
    status = dict(state='running', stage=stage, updated=rt.now(), training_started=False,
                  quality_accepted=False, full_preprocessing_complete=False,
                  reference_frame=reference_frame)
    rt.save(root / 'status.json', status)
    print(json.dumps(dict(run=str(root), status=str(root / 'status.json')), ensure_ascii=False), flush=True)
    try:
        if Path(profile['environment']).resolve() != Path(sys.prefix).resolve():
            raise ReconstructionError('ENVIRONMENT_MISMATCH',
                'Use scripts/train.sh with the Python environment recorded by deployment.')
        if not video.is_file():
            raise FileNotFoundError(video)
        if re.fullmatch(r'frame_[0-9]{6}', reference_frame) is None:
            raise ValueError('reference-frame must be an extracted frame ID, for example frame_000001.')
        digest = rt.sha(video)
        video_info = inspect_video(video)
        rt.save(root / 'input.json', dict(video=str(video), video_sha256=digest, **video_info))
        stage = 'camera_input'
        camera_digest = rt.sha(camera_config)
        camera, cfg = camera_configuration(camera_config, video_info['source_wh'])
        from .official_deployment import fitting_options
        cfg['official_smplifyx'] = fitting_options(cfg['official_smplifyx'], profile)
        if rt.sha(camera_config) != camera_digest or rt.sha(video) != digest:
            raise ValueError('Input changed during validation; use immutable inputs and a fresh run.')
        rt.save(root / 'camera.json', camera)
        rt.save(root / 'run.json', dict(schema='smplx-reconstruction-v1', created=rt.now(),
            video=str(video), video_sha256=digest, camera_config_sha256=camera_digest,
            reference_frame=reference_frame, assets=profile['assets'],
            representation='shared_free_clothing_with_framewise_smplx', quality_accepted=False))
        stage = 'pipeline_connection'
        reconstruct = sequence_pipeline()
        # Numerical stages own progress and output validation. A return by itself
        # never turns a run into accepted or completed here.
        return reconstruct(video=video, root=root, assets=Path(profile['assets']),
                           configuration=cfg, reference_frame=reference_frame)
    except BaseException as exc:
        recorded = rt.read(root / 'status.json')
        if recorded.get('stage') != 'input':
            status.update(recorded)
            stage = recorded['stage']
        status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            stage=stage, updated=rt.now(), error_code=getattr(exc, 'code', 'EXECUTION_ERROR'),
            error=str(exc))
        rt.save(root / 'status.json', status)
        (root / 'logs/error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog='smpl-x', description='One video, one reconstruction run.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train', help='Reconstruct one video using its camera configuration.')
    p.add_argument('video', type=Path)
    p.add_argument('--camera-config', type=Path, required=True)
    p.add_argument('--reference-frame', default='frame_000001')
    p.add_argument('--deployment', type=Path, default=Path.home() / 'smpl-x/queue/deployment.json')
    p.add_argument('--output', type=Path)
    p = commands.add_parser('status', help='Read the recorded result or first failure.')
    p.add_argument('--run', type=Path, required=True)
    p = commands.add_parser('configure', help='Validate and save the server environment and storage.')
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--runs', type=Path, default=Path.home() / 'smpl-x/runs')
    p.add_argument('--output', type=Path, default=Path.home() / 'smpl-x/queue/deployment.json')
    p.add_argument('--official-source', type=Path)
    p.add_argument('--profile', choices=['official_smplifyx', 'legacy'], default='official_smplifyx')
    p = commands.add_parser('doctor', help='Check model assets and GPU dependencies.')
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--official-source', type=Path)
    p.add_argument('--deployment', type=Path, default=Path.home() / 'smpl-x/queue/deployment.json')
    p.add_argument('--profile', choices=['official_smplifyx', 'legacy'], default='official_smplifyx')
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            train_video(args.video, args.deployment, args.camera_config,
                        output=args.output, reference_frame=args.reference_frame)
        elif args.command == 'status':
            print(json.dumps(rt.read(args.run / 'status.json'), ensure_ascii=False, indent=2))
        elif args.command == 'configure':
            if args.profile == 'legacy':
                from .route119 import configure_deployment
                configure_deployment(args.assets, args.runs, args.output)
            else:
                from .official_deployment import configure_deployment
                configure_deployment(args.assets, args.runs, args.output, args.official_source)
        else:
            if args.profile == 'legacy':
                from .cli import doctor
                doctor(args.assets)
            else:
                from .official_deployment import doctor
                source = args.official_source
                if source is None and args.deployment.is_file():
                    source = rt.read(args.deployment).get('official_smplifyx', {}).get('source_root')
                if not source:
                    raise ValueError('Run deploy.sh to prepare the official source and deployment profile.')
                print(json.dumps(doctor(args.assets, source), indent=2))
    except Exception as exc:
        print(f'{getattr(exc, "code", "EXECUTION_ERROR")}: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
