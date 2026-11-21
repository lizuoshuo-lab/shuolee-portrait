"""Fresh-video producers joined to the shared-identity sequence fitter.

The first unavailable downstream connection is recorded explicitly. Existing
static fitting/training is never selected as a fallback.
"""
from pathlib import Path

from . import runtime as rt
from .camera_world_units import validate_camera_length_contract


def bound_file(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=rt.sha(path))


def write_sequence_inputs(root, configuration, reference_frame):
    """Bind only fresh inputs produced inside this run, without person caches."""
    from .sequence_input_adapter import load_sequence_inputs, SCHEMA
    root = Path(root).resolve()
    run = rt.read(root / 'run.json')
    manifest = dict(schema=SCHEMA, purpose='real_research_adaptation',
        pixel_domain='rectified_pinhole', pixel_convention='integer_centers',
        root_policy='provided_initialization', initialization_kind='fresh_same_video_gvhmr_incam',
        initialization_evidence='Original GVHMR incam prediction generated from this run video; no postprocessing or fitted subject state.',
        video_sha256=run['video_sha256'], reference_frame_id=reference_frame,
        camera_length_contract=dict(validate_camera_length_contract(configuration['camera_length_contract'])),
        frames=bound_file(root / 'data/inputs.json'),
        observations=bound_file(root / 'data/observations.npz'),
        prior=bound_file(root / 'inference/prediction-original.npz'),
        prior_provenance=bound_file(root / 'inference/original-receipt.json'),
        crop_mapping=bound_file(root / 'inference/crop-mapping.json'))
    path = root / 'sequence-inputs.json'
    if path.exists():
        raise FileExistsError(path)
    rt.save(path, manifest)
    bundle = load_sequence_inputs(path, decode_images=True)
    rt.save(root / 'sequence-inputs-check.json', bundle.report())
    return path


def reconstruct(*, video, root, assets, configuration, reference_frame):
    """Do not select the experimental solver as the product default."""
    from .reconstruction import ReconstructionError
    status = rt.read(root / 'status.json')
    status.update(stage='official_smplifyx_connection', updated=rt.now())
    rt.save(root / 'status.json', status)
    raise ReconstructionError('OFFICIAL_SMPLIFYX_CONNECTION_PENDING',
        'The official SMPLify-X input/output connection is being integrated. '
        'No experimental full-sequence solver or historical static trainer has been started.')


def _experimental_shared_identity_reconstruct(*, video, root, assets, configuration, reference_frame):
    """Retained experimental work; intentionally unreachable from the product CLI."""
    from .reconstruction import ReconstructionError
    from .cli import gpu_lock
    cfg = dict(configuration, pixel_center_convention='renderer_corner_to_detector_integer')

    def progress(stage, **values):
        status = rt.read(root / 'status.json')
        status.update(stage=stage, updated=rt.now(), **values)
        rt.save(root / 'status.json', status)

    progress('camera_scale')
    try:
        length = validate_camera_length_contract(cfg.get('camera_length_contract'))
        if length['basis'] == 'synthetic_known':
            raise ValueError('Synthetic scale cannot describe a real video.')
    except ValueError as exc:
        raise ReconstructionError('CAMERA_SCALE_UNRESOLVED',
            'A common camera-to-model length conversion is required before sequence fitting. '
            'This frontend does not estimate it; arbitrary SfM translations are not metres. ' + str(exc)) from exc
    rt.save(root / 'config.json', cfg)
    for name in ('data',):
        (root / name).mkdir()
    with gpu_lock():
        progress('prepare')
        from .video119 import prepare_video
        prepare_video(video, root, cfg, assets)
        run = rt.read(root / 'run.json')
        run['decoded_manifest'] = bound_file(root / 'decoded.json')
        rt.save(root / 'run.json', run)
        rows = rt.read(root / 'data/inputs.json')['records']
        if reference_frame not in {r['frame'] for r in rows}:
            raise ReconstructionError('REFERENCE_FRAME_UNREGISTERED',
                'The requested reference frame is not in the registered camera roster; choose an actual registered frame.')
        progress('prior', full_preprocessing_complete=True)
        rt.configure(assets, root)
        from .prior import infer
        infer(root, cfg, postprocess=False)
        progress('sequence_inputs')
        inputs = write_sequence_inputs(root, cfg, reference_frame)
        progress('fit')
        from .sequence_fit import fit_sequence
        fit_sequence(inputs, root / 'fit', assets=assets)
        progress('assemble')
        from .sequence_bundle import assemble_final_bundle
        bundle, _ = assemble_final_bundle(root / 'fit', root / 'frame-pose',
            source_run=root / 'run.json', sequence_id=root.name)
        bundle.assert_unchanged()
        progress('visibility_inputs')
        raise ReconstructionError('VALIDITY_PRODUCER_UNAVAILABLE',
            'The final framewise fit bundle is available. Automatic per-frame external-occlusion '
            'validity inputs are not implemented yet; unknown/occluded regions cannot be labeled '
            'as visible background. The native training/renderer/export components do not replace '
            'this missing producer. Gaussian training has not started.')
