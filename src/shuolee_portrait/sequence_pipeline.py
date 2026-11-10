"""Fresh video to official named fits, existing Gaussian training and generic PLY."""
from pathlib import Path

import numpy as np

from . import runtime as rt
from .camera_world_units import validate_camera_length_contract

OBSERVATION_ASSETS = ('detection/yolov8n.pt', 'sam2/sam2.1_hiera_tiny.pt',
                      'rtmw/rtmw-l-384x288.onnx', 'rtmw/normalization.json')
GAUSSIAN_CONFIG = dict(initial_points=200000, iterations=18000, training_height=None,
                       seed=6060, image_cache_frames=8, export_frame_mode='scene_up_keep_axis_order')


def prepare_native(video, root, cfg, assets):
    """Stable camera-relative route, using explicit native K and stock observations."""
    from .observations import observe
    from .preprocessing import extract_dual_resolution
    from .video119 import fixed_intrinsics

    contract = extract_dual_resolution(video, root, cfg, native_only=True)
    decoded = rt.read(root/'decoded.json')
    wh = contract['source_wh']
    Ki, _ = fixed_intrinsics(cfg['intrinsics'], wh, wh)
    K = Ki.copy()
    K[:2, 2] += .5
    identity = np.eye(4).tolist()
    rows = [dict(row, K=K.tolist(), K_integer_centers=Ki.tolist(), w2c=identity,
                 mask=None, mask_sha256=None, official_prediction_slot=i, observation_weight=1.,
                 camera_kind='camera_relative_fixedK_body_motion',
                 camera_registration=dict(static_trajectory_claim=False, interpolation_used=False))
            for i, row in enumerate(decoded['records'])]
    manifest = dict(records=rows, video_sha256=decoded['video_sha256'], from_scratch=True,
        pixel_domain='Already rectified native MP4 pixels', sample_fps=3, conditional_K=True,
        camera_source='Identity per-view camera in native SMPLify-X coordinates, no static trajectory claim',
        formal_calibration_accepted=False, physical_calibration_accepted=False)
    rt.save(root/'data/inputs.json', manifest)
    rt.configure(assets, root)
    rt.verify_assets(OBSERVATION_ASSETS)
    observe(root, cfg)


def deliver_official(root, cfg, progress, reference_frame, selection=None, *, fit_reused=False):
    from .export import export
    from .reconstruction import ReconstructionError
    from .training import train_official

    progress('gaussian_inputs', fitting_complete=True, reference_frame=reference_frame)
    trained = train_official(root, cfg)
    progress('export', training_complete=True)
    export(root, cfg)
    result = rt.read(root/'result.json')
    ply = root/'export/point_cloud.ply'
    if (result.get('state') != 'completed' or result.get('format_passed') is not True or
            result.get('reference_frame_id') != reference_frame or
            not ply.is_file() or result.get('sha256') != rt.sha(ply)):
        raise ReconstructionError('EXPORT_RESULT_INCOMPLETE', 'No verified named-reference PLY export.')
    if selection is not None:
        from .bound_review_panels import write_panels
        from .training import replay_official_bound

        progress('bound_review', static_export_passed=True, static_export_completed_at=rt.now())
        replay = replay_official_bound(root, replay=trained)
        replay_path = root/'gaussian-run'/result['final_snapshot']/'posed-replay/result.json'
        panels = write_panels(root, replay_path)
        panel_path = root/'audit/bound-panels/result.json'
        selected = set(selection['selected_frames'])
        if (set(x['frame'] for x in replay['records']) != selected or
                len(replay['records']) != len(selected) or not panel_path.is_file() or
                panels != rt.read(panel_path)):
            raise ReconstructionError('BOUND_REVIEW_INCOMPLETE', 'Posed replay must cover the exact selected frame roster.')
        result.update(bound_replay=str(replay_path.relative_to(root)),
            bound_replay_sha256=rt.sha(replay_path),
            bound_panels=str(panel_path.relative_to(root)), bound_panels_sha256=rt.sha(panel_path))
    if fit_reused:
        result.update(from_scratch=False, gaussian_from_scratch=True, fit_reused=True, fit_calls=0, cold_video_clock=False)
    rt.save(root/'result.json', result)
    progress('complete', state='completed', format_passed=True, quality_accepted=False,
             ply=str(ply), sha256=result['sha256'])
    return result


def train_fitted(root):
    """Run the normal SH0 path on a staged, previously verified official fit."""
    from .cli import gpu_lock
    from .training import verify_official_fit

    root = Path(root).resolve(strict=True)
    cfg = rt.read(root/'config.json')
    metadata = rt.read(root/'run.json')
    if cfg.get('gaussian_profile') != 'smpl_sh0':
        raise ValueError('train-fitted requires the public smpl_sh0 profile.')

    def progress(stage, **values):
        path = root/'status.json'
        status = rt.read(path) if path.exists() else dict(state='running', fit_reused=True, fit_calls=0)
        status.update(stage=stage, updated=rt.now(), **values)
        rt.save(path, status)

    with gpu_lock():
        if any((root/name).exists() for name in ('status.json', 'gaussian-run', 'export', 'audit', 'result.json')):
            raise FileExistsError('train-fitted requires fresh Gaussian outputs.')
        rt.configure(Path(metadata['assets']), root)
        verify_official_fit(root)
        inputs = rt.read(root/'data/training-inputs.json')['records']
        if any(rt.sha(Path(row['mask'])) != row['mask_sha256'] for row in inputs):
            raise ValueError('A selected native mask differs from the fitted inputs.')
        selection = rt.read(root/'fit/official-fit-result.json')['selection']
        try:
            return deliver_official(root, cfg, progress, selection['reference_frame'], selection, fit_reused=True)
        except Exception as error:
            state = rt.read(root/'status.json')
            progress(state['stage'], state='failed', error=str(error))
            raise


def reconstruct(*, video, root, assets, configuration, reference_frame):
    from .reconstruction import ReconstructionError
    from .official_smplifyx import adapted_entry_source
    from .official_sequence import fit_named_sequence, read_shape_policy
    from .cli import gpu_lock
    from .video119 import prepare_video
    from .training import smpl_sh0_effective_config

    root, assets = Path(root), Path(assets)
    def progress(stage, **values):
        status = rt.read(root / 'status.json')
        status.update(stage=stage, updated=rt.now(), **values)
        rt.save(root / 'status.json', status)

    progress('official_inputs')
    options = configuration.get('official_smplifyx')
    if not isinstance(options, dict) or not options.get('source_root'):
        raise ReconstructionError('OFFICIAL_SMPLIFYX_SOURCE_REQUIRED',
            'Provide the pinned official SMPLify-X source_root and explicit shape_policy in official_smplifyx.')
    if set(options) != {'source_root', 'shape_policy'}:
        raise ReconstructionError('SHARED_SHAPE_POLICY_REQUIRED',
            'official_smplifyx requires source_root and shape_policy only.')
    read_shape_policy(options['shape_policy'])
    adapted_entry_source(options['source_root'])
    cfg = dict(configuration, **GAUSSIAN_CONFIG,
               pixel_center_convention='renderer_corner_to_detector_integer')
    stable = 'capture_layout' in cfg
    if stable:
        cfg.update(smpl_sh0_effective_config())
        cfg['export_frame_mode'] = 'reference_camera_up'
        length = dict(translation_unit='meters', meters_per_camera_unit=1., basis='model_conditioned',
                      evidence='Per-frame camera identity; SMPLify-X body translation is in model metres.')
        cfg['camera_length_contract'] = length
    else:
        length = validate_camera_length_contract(configuration.get('camera_length_contract'))
        if length['basis'] == 'synthetic_known':
            raise ReconstructionError('CAMERA_SCALE_UNRESOLVED', 'A synthetic scale cannot describe a real video.')
    rt.save(root / 'config.json', cfg)
    (root / 'data').mkdir(exist_ok=False)
    with gpu_lock():
        progress('prepare')
        # Only segmentation/detection/keypoint assets are required here; no image-to-3D prior.
        if stable:
            prepare_native(video, root, cfg, assets)
        else:
            prepare_video(video, root, cfg, assets, required_assets=OBSERVATION_ASSETS)
        rt.configure(assets, root)
        model_assets = [r['path'] for r in rt.read(rt.PROJECT/'configs/server-assets.json')['files']
                        if r['path'].startswith(('smplx/', 'vposer/'))]
        rt.verify_assets(model_assets)
        progress('official_fit', full_preprocessing_complete=True)
        stable_options = {'capture_layout': cfg['capture_layout']} if stable else {}
        fit_result = fit_named_sequence(frames_path=root/'data/inputs.json', observations_path=root/'data/observations.npz',
            root=root, official_root=options['source_root'], assets=assets,
            camera_length_contract=dict(length), reference_frame=reference_frame,
            shape_policy=options['shape_policy'], **stable_options)
        if stable:
            reference_frame = fit_result['selection']['reference_frame']
        selection = fit_result['selection'] if stable else None
        return deliver_official(root, cfg, progress, reference_frame, selection)
