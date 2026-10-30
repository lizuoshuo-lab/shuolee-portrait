"""Fresh video to official named fits, existing Gaussian training and generic PLY."""
from pathlib import Path

from . import runtime as rt
from .camera_world_units import validate_camera_length_contract

OBSERVATION_ASSETS = ('detection/yolov8n.pt', 'sam2/sam2.1_hiera_tiny.pt',
                      'rtmw/rtmw-l-384x288.onnx', 'rtmw/normalization.json')
GAUSSIAN_CONFIG = dict(initial_points=200000, iterations=18000, training_height=None,
                       seed=6060, image_cache_frames=8, export_frame_mode='scene_up_keep_axis_order')


def reconstruct(*, video, root, assets, configuration, reference_frame):
    from .reconstruction import ReconstructionError
    from .official_smplifyx import adapted_entry_source
    from .official_sequence import fit_named_sequence, read_shape_policy
    from .cli import gpu_lock
    from .video119 import prepare_video
    from .training import train_official
    from .export import export

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
    length = validate_camera_length_contract(configuration.get('camera_length_contract'))
    if length['basis'] == 'synthetic_known':
        raise ReconstructionError('CAMERA_SCALE_UNRESOLVED', 'A synthetic scale cannot describe a real video.')
    cfg = dict(configuration, **GAUSSIAN_CONFIG,
               pixel_center_convention='renderer_corner_to_detector_integer')
    rt.save(root / 'config.json', cfg)
    (root / 'data').mkdir(exist_ok=False)
    with gpu_lock():
        progress('prepare')
        # Only segmentation/detection/keypoint assets are required here; no image-to-3D prior.
        prepare_video(video, root, cfg, assets, required_assets=OBSERVATION_ASSETS)
        rt.configure(assets, root)
        model_assets = [r['path'] for r in rt.read(rt.PROJECT/'configs/server-assets.json')['files']
                        if r['path'].startswith(('smplx/', 'vposer/'))]
        rt.verify_assets(model_assets)
        progress('official_fit', full_preprocessing_complete=True)
        fit_named_sequence(frames_path=root/'data/inputs.json', observations_path=root/'data/observations.npz',
            root=root, official_root=options['source_root'], assets=assets,
            camera_length_contract=dict(length), reference_frame=reference_frame,
            shape_policy=options['shape_policy'])
        progress('gaussian_inputs', fitting_complete=True)
        train_official(root, cfg)
        progress('export', training_complete=True)
        export(root, cfg)
        result = rt.read(root/'result.json')
        ply = root/'export/point_cloud.ply'
        if (result.get('state') != 'completed' or result.get('format_passed') is not True or
                result.get('reference_frame_id') != reference_frame or
                not ply.is_file() or result.get('sha256') != rt.sha(ply)):
            raise ReconstructionError('EXPORT_RESULT_INCOMPLETE', 'No verified named-reference PLY export.')
        progress('complete', state='completed', format_passed=True, quality_accepted=False,
                 ply=str(ply), sha256=result['sha256'])
        return result
