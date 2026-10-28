"""SMPL-X video reconstruction: fresh inputs, 18000 training steps, standard PLY.

This entry is separate from the formal reconstruction command. It reproduces
the validated representation and finite schedule; it never admits physical camera or
reconstruction quality. No old fit, Gaussian parameters or optimizer are copied.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

import numpy as np
from PIL import Image

from . import runtime as rt
from .outcomes import failure_feedback


SCHEMA = 'shuolee-119-video-v2'
STAGES = ('prepare', 'prior', 'fit', 'fit-gate', 'surface', 'train', 'export')
DEFAULTS = dict(
    profile='static_single_person', gaussian_representation='static_clothed_world',
    frame_stride=5, max_image_size=1600, min_frames=12, max_frames=300,
    fit_steps=1200, initial_points=200000, iterations=18000, seed=6060,
    training_height=1600, image_cache_frames=8,
    max_fit_median_body_fraction=.03, max_fit_p90_body_fraction=.08,
    pixel_center_convention='renderer_corner_to_detector_integer',
    landmark_image_support='mask_margin_4px',
    surface_spacing_m=.012, surface_padding_m=.4,
    surface_vote_fraction=.97, surface_mask_tolerance_px=2.,
    surface_vote_policy='legacy_all_views_denominator',
    export_frame_mode='scene_up_keep_axis_order',
    quality_accepted=False, physical_calibration_validated=False,
)


def configuration(overrides=None, *, image_height=None):
    """The finite 119 recipe; only the input/front-end budget is configurable."""
    cfg = dict(DEFAULTS, **(overrides or {}))
    if image_height is not None:
        cfg['training_height'] = int(image_height)
    for key in ('frame_stride', 'max_image_size', 'min_frames', 'max_frames',
                'training_height', 'image_cache_frames'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if not 12 <= cfg['min_frames'] <= cfg['max_frames'] <= 300:
        raise ValueError('119 accepts a declared 12..300 frame budget.')
    frozen = ('profile', 'gaussian_representation', 'fit_steps', 'initial_points',
              'iterations', 'seed', 'pixel_center_convention',
              'max_fit_median_body_fraction', 'max_fit_p90_body_fraction',
              'landmark_image_support', 'surface_spacing_m',
              'surface_padding_m', 'surface_vote_fraction',
              'surface_mask_tolerance_px', 'surface_vote_policy',
              'export_frame_mode')
    for key in frozen:
        if cfg[key] != DEFAULTS[key] or type(cfg[key]) is not type(DEFAULTS[key]):
            raise ValueError(f'{key} changes the frozen 119 recipe; version that experiment separately.')
    if cfg.get('anatomical_visibility', 'none') != 'none':
        raise ValueError('119 does not enable the later anatomical-visibility experiment.')
    for key in ('quality_accepted', 'physical_calibration_validated',
                'physical_calibration_accepted', 'formal_training_admitted'):
        if cfg.get(key, False) is not False:
            raise ValueError(f'{key} cannot be pre-approved in a 119 config.')
    return cfg


def inspect_prepared(root, cfg=None):
    """Validate real pixels, masks, frame identity and numerical camera inputs.

    This is a data-consistency check, not independent calibration validation.
    No CUDA, pretrained model or learned subject state is loaded.
    """
    root = Path(root).resolve()
    cfg = configuration(cfg or rt.read(root / 'config.json'))
    inputs = rt.read(root / 'data/inputs.json')
    training = rt.read(root / 'data/training-inputs.json')
    rows = inputs.get('records')
    if not isinstance(rows, list) or not cfg['min_frames'] <= len(rows) <= cfg['max_frames']:
        raise ValueError('Prepared frame count does not match the declared 119 budget.')
    if rows != training.get('records'):
        raise ValueError('119 input and training frame/camera/mask records differ.')
    run_path = root / 'run.json'
    if run_path.is_file():
        video_hash = rt.read(run_path).get('video_sha256')
        for manifest in (inputs, training):
            if manifest.get('video_sha256') is not None and manifest['video_sha256'] != video_hash:
                raise ValueError('Prepared measurements belong to a different uploaded video.')
    names = set()
    previous_time = -np.inf
    previous_source_index = -1
    shape = None
    for i, row in enumerate(rows):
        if not isinstance(row.get('frame'), str) or row['frame'] in names:
            raise ValueError('Prepared frame identities must be unique strings.')
        names.add(row['frame'])
        source_index = row.get('source_frame_index')
        if type(source_index) is not int or source_index <= previous_source_index:
            raise ValueError('Original video frame indices must be nonnegative and strictly increasing.')
        previous_source_index = source_index
        if row.get('official_prediction_slot') != i:
            raise ValueError('Prior slots must match the exact prepared frame order.')
        stamp = row.get('time_seconds')
        if not isinstance(stamp, (float, int)) or not np.isfinite(stamp) or stamp <= previous_time:
            raise ValueError('Prepared video timestamps must be finite and strictly increasing.')
        previous_time = stamp
        wh = row.get('wh')
        if (not isinstance(wh, list) or len(wh) != 2 or
                any(type(x) is not int or x < 2 for x in wh)):
            raise ValueError('Each prepared image needs an integer width and height.')
        if shape is None:
            shape = wh
        if wh != shape:
            raise ValueError('119 uses one common decoded video size.')
        box = np.asarray(row.get('initial_box_xyxy'), dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or
                np.any(box[2:] <= box[:2]) or np.any(box < 0) or
                np.any(box > [wh[0], wh[1], wh[0], wh[1]])):
            raise ValueError('The fresh prior needs a finite, positive full-image person box.')
        for field, digest in (('image', 'source_sha256'), ('mask', 'mask_sha256')):
            path = Path(row[field])
            if not path.is_absolute() or not path.is_file() or rt.sha(path) != row.get(digest):
                raise ValueError(f'{row["frame"]}: missing or changed {field}.')
            with Image.open(path) as image:
                if list(image.size) != wh:
                    raise ValueError(f'{row["frame"]}: {field} size disagrees with its camera.')
        K = np.asarray(row.get('K'), dtype=float)
        E = np.asarray(row.get('w2c'), dtype=float)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or
                not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9):
            raise ValueError('119 requires finite zero-skew pinhole K, including its actual principal point.')
        if (E.shape != (4, 4) or not np.isfinite(E).all() or
                not np.allclose(E[3], [0, 0, 0, 1]) or
                not np.allclose(E[:3, :3] @ E[:3, :3].T, np.eye(3), atol=1e-5) or
                abs(np.linalg.det(E[:3, :3]) - 1) > 1e-5):
            raise ValueError('119 requires proper rigid OpenCV world-to-camera matrices.')
        kind = row.get('camera_kind', '')
        if not kind or 'interpolat' in kind.lower() or row.get('camera_registration', {}).get('interpolation_used') is not False:
            raise ValueError('Prepared cameras must explicitly exclude pose interpolation.')
    with np.load(root / 'data/observations.npz', allow_pickle=False) as obs:
        expected = dict(keypoints=(len(rows), 133, 2), scores=(len(rows), 133),
                        flip_disagreement=(len(rows), 133), boxes=(len(rows), 4))
        if obs['frame_ids'].tolist() != [row['frame'] for row in rows]:
            raise ValueError('Observation frame identities/order do not match cameras.')
        for key, wanted in expected.items():
            if obs[key].shape != wanted or not np.isfinite(obs[key]).all():
                raise ValueError(f'Invalid prepared COCO-133 {key}.')
        if np.any(obs['boxes'][:, 2:] <= obs['boxes'][:, :2]):
            raise ValueError('Prepared person boxes must have positive area.')
        if not np.allclose(obs['boxes'], [row['initial_box_xyxy'] for row in rows]):
            raise ValueError('Observation boxes differ from the prior input boxes.')
        if obs['joint_names'].shape != (133,):
            raise ValueError('Expected the 133 native COCO joint names.')
    return dict(schema=SCHEMA, frames=len(rows), wh=shape,
                inputs_sha256=rt.sha(root / 'data/inputs.json'),
                training_inputs_sha256=rt.sha(root / 'data/training-inputs.json'),
                observations_sha256=rt.sha(root / 'data/observations.npz'),
                pixels_and_masks_verified=True, physical_calibration_accepted=False,
                quality_accepted=False, formal_training_admitted=False)


def initialize_run(root, video, cfg, assets):
    root, video = Path(root).resolve(), Path(video).resolve()
    if root.exists():
        raise FileExistsError(f'Fresh SMPL-X run required; preserve the existing directory: {root}')
    if not video.is_file():
        raise FileNotFoundError(video)
    cfg = configuration(cfg)
    root.mkdir(parents=True)
    try:
        (root / 'data').mkdir()
        (root / 'logs').mkdir()
        rt.save(root / 'config.json', cfg)
        rt.save(root / 'run.json', dict(
            schema=SCHEMA, created=rt.now(), video=str(video), video_sha256=rt.sha(video),
            assets=str(Path(assets).resolve()), from_scratch=True, old_subject_loaded=False,
            old_fitted_person_loaded=False, old_gaussian_loaded=False, old_optimizer_loaded=False,
            execution_kind='finite_119_representation_research', formal_training_admitted=False,
            reproduction_scope='119 clothing representation and schedule; fresh video cameras are a separately recorded frontend',
            historical_camera_reproduction=False,
            physical_calibration_accepted=False, quality_accepted=False,
            representation='shared_static_world_clothed_surface',
            surface_vote_policy='legacy_all_views_denominator'))
        rt.save(root / 'status.json', dict(state='running', stage='prepare', updated=rt.now(),
                quality_accepted=False, formal_training_admitted=False))
    except BaseException as exc:
        rt.save(root / 'status.json', dict(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', stage='initialize', updated=rt.now(),
                error=traceback.format_exc(), feedback=failure_feedback(exc, stage='initialize', interrupted=isinstance(exc, KeyboardInterrupt)),
                quality_accepted=False, formal_training_admitted=False))
        raise
    return root


def recorded_status(root):
    """Read stored evidence, including preparation failures, without CUDA."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    report = dict(run=str(root), process_liveness_checked=False,
                  quality_accepted=False, physical_calibration_accepted=False)
    for key, name in (('run_metadata', 'run.json'), ('status', 'status.json'),
                      ('preparation', 'preparation.json'), ('progress', 'progress.json'),
                      ('video_progress', 'video119-progress.json')):
        if (root / name).is_file():
            report[key] = rt.read(root / name)
    return report


def execute_stage(name, root, assets):
    root = Path(root).resolve()
    run = rt.read(root / 'run.json')
    if run.get('schema') != SCHEMA or run.get('from_scratch') is not True:
        raise ValueError('Not a fresh engineering SMPL-X run.')
    if any(run.get(key) is not False for key in ('old_subject_loaded', 'old_fitted_person_loaded',
            'old_gaussian_loaded', 'old_optimizer_loaded', 'formal_training_admitted', 'quality_accepted')):
        raise ValueError('119 research cannot load old learned state or pre-approve quality.')
    cfg = configuration(rt.read(root / 'config.json'))
    prepared = inspect_prepared(root, cfg)
    if cfg['training_height'] != prepared['wh'][1]:
        raise ValueError('SMPL-X training_height must equal this run\'s decoded image height.')
    rt.configure(assets, root)
    if name == 'prior':
        if (root / 'inference').exists():
            raise FileExistsError('119 prior already exists; stages do not overwrite learned state.')
        from .prior import infer
        infer(root, cfg)
    elif name == 'fit':
        if (root / 'fit').exists():
            raise FileExistsError('119 fit already exists; stages do not overwrite learned state.')
        from .fitting import main
        main(root, cfg['fit_steps'])
    elif name == 'fit-gate':
        if (root / 'fit/geometry-gate.json').exists():
            raise FileExistsError('119 fit-gate already exists; preserve its evidence.')
        from .fit_gate import evaluate
        evaluate(root, cfg)
    elif name == 'surface':
        if (root / 'clothing-init').exists():
            raise FileExistsError('119 initialization already exists.')
        from .clothed_surface import initialize
        initialize(root, cfg)
    elif name == 'train':
        if (root / 'gaussian-run').exists():
            raise FileExistsError('119 Gaussian run already exists; no cross-run resume.')
        from .clothed_training import train_clothed
        train_clothed(root, cfg)
    elif name == 'export':
        if (root / 'export').exists():
            raise FileExistsError('119 export already exists; preserve its evidence.')
        from .export import export
        export(root, cfg)
    else:
        raise ValueError(f'Unknown prepared 119 stage: {name}')


def run_stage(name, root, assets):
    root = Path(root).resolve()
    if not (root / 'run.json').is_file():
        raise FileNotFoundError(root / 'run.json')
    rt.save(root / 'status.json', dict(state='running', stage=name, updated=rt.now(),
            quality_accepted=False, formal_training_admitted=False))
    try:
        execute_stage(name, root, assets)
    except BaseException as exc:
        rt.save(root / 'status.json', dict(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', stage=name, updated=rt.now(),
                error=traceback.format_exc(), feedback=failure_feedback(exc, stage=name, interrupted=isinstance(exc, KeyboardInterrupt)),
                quality_accepted=False, formal_training_admitted=False))
        raise
    result = dict(state='stage_complete', stage=name,
                  updated=rt.now(), quality_accepted=False, formal_training_admitted=False)
    if name == 'train':
        result['training_result'] = rt.read(root / 'gaussian-run/result.json')
        result['state'] = result['training_result']['state']
    elif name == 'export':
        result['result'] = rt.read(root / 'result.json')
        result['state'] = result['result']['state']
    rt.save(root / 'status.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def reconstruct(video, root, assets, cfg=None, *, stop_after='export'):
    """Complete the fixed recipe with automatic numerical checks."""
    if stop_after not in STAGES:
        raise ValueError(f'Unknown stop stage: {stop_after}')
    from .video119 import prepare_video
    from .cli import gpu_lock
    with gpu_lock():
        # Acquire first: a busy task must not leave an unused new run behind.
        root = initialize_run(root, video, configuration(cfg), assets)
        active_stage = 'prepare'
        try:
            prepare_video(Path(video).resolve(), root, rt.read(root / 'config.json'), Path(assets).resolve())
            cfg = configuration(rt.read(root / 'config.json'))
            audit = inspect_prepared(root, cfg)
            cfg = configuration(cfg, image_height=audit['wh'][1])
            rt.save(root / 'config.json', cfg)
            rt.save(root / 'prepared-inputs-audit.json', audit)
            if stop_after == 'prepare':
                rt.save(root / 'status.json', dict(state='prepared_inputs_complete', stage='prepare',
                        updated=rt.now(), quality_accepted=False, formal_training_admitted=False))
                return root
            for name in STAGES[1:]:
                active_stage = name
                run_stage(name, root, assets)
                if name == stop_after:
                    break
                if name == 'train' and rt.read(root / 'gaussian-run/result.json')['steps'] != cfg['iterations']:
                    raise RuntimeError('SMPL-X training was incomplete; no completed 18000-step result is claimed.')
        except BaseException as exc:
            rt.save(root / 'status.json', dict(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', stage=active_stage, updated=rt.now(),
                    error=traceback.format_exc(), feedback=failure_feedback(exc, stage=active_stage, interrupted=isinstance(exc, KeyboardInterrupt)),
                    quality_accepted=False, formal_training_admitted=False))
            raise
    return root


def configure_deployment(assets, runs, output):
    """Bind server-owned assets and run storage after the actual asset/GPU doctor."""
    from .cli import doctor
    assets, runs, output = (Path(p).expanduser().resolve() for p in (assets, runs, output))
    for name, path in (('assets', assets), ('runs', runs), ('deployment profile', output)):
        if path == rt.PROJECT or rt.PROJECT in path.parents:
            raise ValueError(f'{name} must be outside the Git checkout.')
    checks = doctor(assets)
    profile = dict(schema='shuolee-deployment119-v1', created=rt.now(), assets=str(assets),
                   runs=str(runs), environment=str(Path(sys.prefix).resolve()),
                   doctor=checks, viewer_required=False, training_started=False)
    rt.save(output, profile)
    return profile


def train_video(video, deployment, *, output=None):
    """Daily interface: the configured server only needs a new video path."""
    profile = rt.read(deployment)
    if profile.get('schema') != 'shuolee-deployment119-v1':
        raise ValueError('Run setup_server.sh --assets first to configure the SMPL-X deployment.')
    if Path(profile['environment']).resolve() != Path(sys.prefix).resolve():
        raise RuntimeError('Use the Python environment validated by setup_server.sh.')
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if output is None:
        label = re.sub(r'[^\w.-]+', '_', video.stem, flags=re.UNICODE).strip('._')[:48] or 'video'
        stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        output = Path(profile['runs']) / f'{stamp}-{label}-{uuid.uuid4().hex[:8]}'
    output = Path(output).resolve()
    print(json.dumps(dict(run=str(output), video=str(video), viewer_required=False), ensure_ascii=False), flush=True)
    cfg = rt.read(rt.PROJECT / 'configs/video119.json')
    return reconstruct(video, output, profile['assets'], cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='smpl-x', description='SMPL-X video reconstruction: fresh video, 18000 training steps, standard PLY.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train', help='One video through SMPL-X reconstruction and standard export.')
    p.add_argument('video', type=Path)
    p.add_argument('--deployment', type=Path, default=Path.home() / 'smpl-x/queue/deployment.json')
    p.add_argument('--output', type=Path)
    p = commands.add_parser('configure', help='Save server assets and run storage after the GPU/model check.')
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--runs', type=Path, default=Path.home() / 'smpl-x/runs')
    p.add_argument('--output', type=Path, default=Path.home() / 'smpl-x/queue/deployment.json')
    p = commands.add_parser('doctor', help='Check model assets and GPU dependencies.')
    p.add_argument('--assets', type=Path, required=True)
    p = commands.add_parser('reconstruct', help='Full reconstruction with explicit server paths.')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=rt.PROJECT / 'configs/video119.json')
    p.add_argument('--assets', type=Path, required=True)
    p = commands.add_parser('status', help='Read the recorded stage, progress and original error.')
    p.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'train':
        train_video(args.video, args.deployment, output=args.output)
    elif args.command == 'configure':
        configure_deployment(args.assets, args.runs, args.output)
    elif args.command == 'doctor':
        from .cli import doctor
        doctor(args.assets)
    elif args.command == 'status':
        print(json.dumps(recorded_status(args.run), ensure_ascii=False, indent=2))
    else:
        reconstruct(args.video, args.output, args.assets, rt.read(args.config))


if __name__ == '__main__':
    main()
