"""Frozen 119 clothing-surface research, with fresh inputs and explicit reviews.

This entry is separate from the formal reconstruction command. It reproduces
the 119 representation and finite schedule; it never admits physical camera or
reconstruction quality. No old fit, Gaussian parameters or optimizer are copied.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np
from PIL import Image

from . import runtime as rt


SCHEMA = 'shuolee-119-clothed-world-v1'
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
        raise FileExistsError(f'Fresh 119 run required; preserve the existing directory: {root}')
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
    except Exception:
        rt.save(root / 'status.json', dict(state='failed', stage='initialize', updated=rt.now(),
                error=traceback.format_exc(), quality_accepted=False, formal_training_admitted=False))
        raise
    return root


def initialize_prepared(source, root, assets, overrides=None):
    """Import only freshly generated measured inputs, never inference/fit/GS.

    A preparation receipt must identify the same raw video. Receipts and hashes
    are an audit trail, not proof that physical correspondence is correct.
    """
    source = Path(source).resolve()
    receipt = rt.read(source / 'preparation.json')
    if (receipt.get('from_this_video') is not True or
            receipt.get('old_subject_state_loaded') is not False or
            not re.fullmatch('[0-9a-f]{64}', receipt.get('video_sha256', ''))):
        raise ValueError('Prepared input needs a fresh-video preparation receipt; historical 118 caches are not an input interface.')
    video = Path(receipt['video'])
    if not video.is_absolute() or not video.is_file() or rt.sha(video) != receipt['video_sha256']:
        raise ValueError('Prepared input is not bound to the available original video.')
    source_cfg = rt.read(source / 'config.json')
    cfg = configuration(dict(source_cfg, **(overrides or {})))
    report = inspect_prepared(source, cfg)
    cfg = configuration(cfg, image_height=report['wh'][1])
    root = initialize_run(root, video, cfg, assets)
    try:
        for name in ('inputs.json', 'training-inputs.json', 'observations.npz'):
            shutil.copy2(source / 'data' / name, root / 'data' / name)
        for name in ('intrinsic-contract.json', 'preparation.json'):
            if (source / name).is_file():
                shutil.copy2(source / name, root / name)
        rt.save(root / 'prepared-inputs-audit.json', dict(
            **inspect_prepared(root, cfg), source=str(source),
            copied_only_measured_inputs=True, old_fit_or_gaussian_or_optimizer_copied=False))
    except Exception:
        rt.save(root / 'status.json', dict(state='failed', stage='import-prepared', updated=rt.now(),
                error=traceback.format_exc(), quality_accepted=False, formal_training_admitted=False))
        raise
    return root


def require_review(root, name, required, artifact, digest_key):
    review = rt.read(root / 'reviews' / name)
    if any(review.get(key) is not True for key in required) or review.get(digest_key) != rt.sha(root / artifact):
        raise RuntimeError(f'Actual review missing, rejected or stale: reviews/{name}')
    return review


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


def record_review(root, kind, decision, note):
    """Record the user's actual visual judgment, bound to current artifacts.

    This command cannot establish physical calibration or quality acceptance.
    It never starts a stage or generates a positive decision automatically.
    """
    root = Path(root).resolve()
    run = rt.read(root / 'run.json')
    if run.get('schema') != SCHEMA:
        raise ValueError('Reviews require an engineering 119 run.')
    if decision not in ('approve', 'reject') or not isinstance(note, str) or not note.strip():
        raise ValueError('Record approve or reject with a nonempty note after actual visual review.')
    approved = decision == 'approve'
    entry = dict(created=rt.now(), decision=decision, note=note.strip(),
                 judgment_source='explicit_user_visual_review_record',
                 quality_accepted=False, physical_calibration_accepted=False,
                 formal_training_admitted=False)
    if kind == 'fit':
        artifact = root / 'fit/fitted.npz'
        gate = rt.read(root / 'fit/geometry-gate.json')
        if gate.get('fit_sha256') != rt.sha(artifact) or gate.get('coarse_initialization_gate_passed') is not True:
            raise ValueError('Fit gate is failed or stale; a review cannot override a numerical failure.')
        if not list((root / 'reviews/fit').glob('*_mesh_overlay.png')):
            raise FileNotFoundError('Inspect the actual images in reviews/fit before recording the fit review.')
        filename = 'fit-decision.json'
        entry.update(fit_sha256=rt.sha(artifact), actual_same_camera_fit_reviewed=True,
                     allow_surface_initialization=approved)
    elif kind == 'surface':
        artifact = root / 'clothing-init/points.npz'
        audit = rt.read(root / 'clothing-init/audit.json')
        if audit.get('point_sha256') != rt.sha(artifact):
            raise ValueError('Initialization audit does not match the actual points.')
        if not (root / 'clothing-init/initialization.ply').is_file() or not list((root / 'clothing-init').glob('*_initialization.png')):
            raise FileNotFoundError('Inspect clothing-init/initialization.ply and its actual renders before recording the surface review.')
        filename = 'surface-decision.json'
        entry.update(points_sha256=rt.sha(artifact), actual_initialization_review=True,
                     allow_finite_research_training=approved)
    elif kind == 'checkpoint':
        snapshot = root / 'gaussian-run/step-001000'
        artifact = snapshot / 'world.ply'
        parity = rt.read(snapshot / 'native-ply-parity.json')
        if parity.get('passed') is not True or parity.get('sha256') != rt.sha(artifact):
            raise ValueError('The actual 1000-step PLY has no matching native-parity check.')
        if not (snapshot / 'center-distribution.json').is_file() or not list(snapshot.glob('*_render.png')):
            raise FileNotFoundError('Inspect the 1000-step renders, point centers and PLY before recording continuation.')
        filename = 'gaussian-step-001000-decision.json'
        entry.update(snapshot_sha256=rt.sha(artifact), actual_same_camera_review=True,
                     actual_center_distribution_review=True, continue_same_run=approved)
    else:
        raise ValueError(f'Unknown actual-review type: {kind}')
    target = root / 'reviews' / filename
    if target.is_file():
        with (target.parent / 'review-history.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps(dict(previous_file=filename, previous=rt.read(target)), ensure_ascii=False) + '\n')
    rt.save(target, entry)
    return dict(review=str(target), **entry)


def checkpoint_review(root, step, snapshot, *, timeout_seconds=1800):
    if step != 1000:
        return True
    decision = root / 'reviews/gaussian-step-001000-decision.json'
    digest = rt.sha(snapshot / 'world.ply')
    rt.save(root / 'status.json', dict(state='actual_review_pending', stage='train', step=step,
            optimizer_waiting=True, snapshot_sha256=digest, quality_accepted=False,
            formal_training_admitted=False, updated=rt.now()))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if decision.is_file():
            d = rt.read(decision)
            if (d.get('snapshot_sha256') != digest or d.get('actual_same_camera_review') is not True or
                    d.get('actual_center_distribution_review') is not True):
                raise RuntimeError('1000-step decision does not match the actual reviewed snapshot.')
            return d.get('continue_same_run') is True
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    return False


def execute_stage(name, root, assets):
    root = Path(root).resolve()
    run = rt.read(root / 'run.json')
    if run.get('schema') != SCHEMA or run.get('from_scratch') is not True:
        raise ValueError('Not a fresh engineering 119 run.')
    if any(run.get(key) is not False for key in ('old_subject_loaded', 'old_fitted_person_loaded',
            'old_gaussian_loaded', 'old_optimizer_loaded', 'formal_training_admitted', 'quality_accepted')):
        raise ValueError('119 research cannot load old learned state or pre-approve quality.')
    cfg = configuration(rt.read(root / 'config.json'))
    prepared = inspect_prepared(root, cfg)
    if cfg['training_height'] != prepared['wh'][1]:
        raise ValueError('119 training_height must equal this run\'s decoded image height.')
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
        require_review(root, 'fit-decision.json',
            ('actual_same_camera_fit_reviewed', 'allow_surface_initialization'),
            'fit/fitted.npz', 'fit_sha256')
        from .clothed_surface import initialize
        initialize(root, cfg)
    elif name == 'train':
        if (root / 'gaussian-run').exists():
            raise FileExistsError('119 Gaussian run already exists; no cross-run resume.')
        require_review(root, 'surface-decision.json',
            ('actual_initialization_review', 'allow_finite_research_training'),
            'clothing-init/points.npz', 'points_sha256')
        from .clothed_training import train_clothed
        train_clothed(root, cfg, checkpoint_review=lambda step, snapshot: checkpoint_review(root, step, snapshot))
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
    except Exception:
        rt.save(root / 'status.json', dict(state='failed', stage=name, updated=rt.now(),
                error=traceback.format_exc(), quality_accepted=False, formal_training_admitted=False))
        raise
    result = dict(state='stage_complete_pending_actual_review', stage=name,
                  updated=rt.now(), quality_accepted=False, formal_training_admitted=False)
    if name == 'train':
        result['training_result'] = rt.read(root / 'gaussian-run/result.json')
        result['state'] = result['training_result']['state']
    rt.save(root / 'status.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def reconstruct(video, root, assets, cfg=None, *, stop_after='fit-gate'):
    """Fresh video front end plus 119 stages; review gates cannot be skipped."""
    try:
        from .video119 import prepare_video
    except ImportError as exc:
        raise RuntimeError('119 video preparation is not installed. Only explicit prepared inputs are supported; video-to-PLY is unavailable.') from exc
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
                    raise RuntimeError('119 stopped after actual review; no completed 18000-step result is claimed.')
        except Exception:
            rt.save(root / 'status.json', dict(state='failed', stage=active_stage, updated=rt.now(),
                    error=traceback.format_exc(), quality_accepted=False, formal_training_admitted=False))
            raise
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Finite 119 research only. It retains the historical FOV vote limitation and never grants formal alignment/quality acceptance.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('reconstruct', help='Fresh video preparation and 119 research stages; default stops after coarse fit diagnostics.')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--stop-after', choices=STAGES, default='fit-gate')
    p = commands.add_parser('init-prepared', help='Import measured inputs with a fresh-video receipt; never imports fit/GS state.')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--assets', type=Path, required=True)
    p = commands.add_parser('inspect', help='CPU-only pixel/camera/COCO-133 consistency audit.')
    p.add_argument('--run', type=Path, required=True)
    p = commands.add_parser('status', help='Read stored run/preparation/progress evidence, including failures; does not infer PID liveness.')
    p.add_argument('--run', type=Path, required=True)
    p = commands.add_parser('review', help='After viewing the actual review images/PLY, record approve/reject; this does not validate calibration or quality.')
    p.add_argument('kind', choices=['fit', 'surface', 'checkpoint'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--decision', choices=['approve', 'reject'], required=True)
    p.add_argument('--note', required=True, help='State what was actually seen in review images/viewer. No automatic approval is generated.')
    p = commands.add_parser('stage', help='Run one reviewed 119 stage without changing existing outputs.')
    p.add_argument('stage', choices=STAGES[1:])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'inspect':
        print(json.dumps(inspect_prepared(args.run), ensure_ascii=False, indent=2))
    elif args.command == 'status':
        print(json.dumps(recorded_status(args.run), ensure_ascii=False, indent=2))
    elif args.command == 'review':
        print(json.dumps(record_review(args.run, args.kind, args.decision, args.note), ensure_ascii=False, indent=2))
    elif args.command == 'init-prepared':
        initialize_prepared(args.prepared, args.output, args.assets, rt.read(args.config) if args.config else None)
    elif args.command == 'reconstruct':
        reconstruct(args.video, args.output, args.assets, rt.read(args.config) if args.config else None, stop_after=args.stop_after)
    else:
        from .cli import gpu_lock
        with gpu_lock():
            run_stage(args.stage, args.run, args.assets)


if __name__ == '__main__':
    main()
