"""Development-release training admission, separate from the coarse fit check.

Independent camera, material and staticity validators are not implemented yet.
Consequently this module cannot admit training in this release. The generated
JSON is a diagnostic output, never a user-editable permission to start training.
Do not replace missing measurements with a configurable accepted=True flag.
"""
from pathlib import Path

from . import runtime as rt


class AlignmentNotAdmitted(RuntimeError):
    """The input has no independently validated static reconstruction geometry."""


def inspect_alignment(root):
    """Read current run artifacts without modifying the run or loading a model.

    Hashes bind the diagnostic to these files only. They do not verify the
    physical camera calibration or every image named inside a manifest.
    """
    root = Path(root)
    artifacts = {}
    for name in ('run.json', 'config.json', 'data/training-inputs.json',
                 'fit/geometry-gate.json', 'fit/fitted.npz',
                 'inference/crop-mapping.json'):
        path = root / name
        artifacts[name] = dict(present=path.is_file(), sha256=rt.sha(path) if path.is_file() else None)
    fit_gate = rt.read(root / 'fit/geometry-gate.json') if artifacts['fit/geometry-gate.json']['present'] else {}
    run = rt.read(root / 'run.json') if artifacts['run.json']['present'] else {}
    rows = rt.read(root / 'data/training-inputs.json').get('records', []) if artifacts['data/training-inputs.json']['present'] else []
    fit_hash = artifacts['fit/fitted.npz']['sha256']
    return dict(
        schema='shuolee-training-readiness-v1', updated=rt.now(),
        state='alignment_not_admitted', training_admitted=False,
        independent_alignment_validation_implemented=False,
        release_training_available=False, quality_accepted=False,
        current_video_sha256=run.get('video_sha256'), frames=len(rows),
        run_artifacts=artifacts,
        coarse_fit=dict(
            passed=fit_gate.get('coarse_initialization_gate_passed') is True,
            matches_current_fit=bool(fit_hash and fit_gate.get('fit_sha256') == fit_hash),
            independent_geometry_evidence=False),
        diagnostic_components=dict(fixed_camera_point_projection=dict(
            implemented=True, automatic_material_matching=False,
            physical_validation_complete=False,
            entry='scripts/evaluate_fixed_tracks.py')),
        required_checks=[
            dict(id='static_scene_camera', state='not_implemented',
                 evidence='Independent static-scene tracks or capture calibration with held validation, coverage and uncertainty.'),
            dict(id='cross_view_material_identity', state='not_implemented',
                 evidence='Source-verified shared physical points across views, including face and clothing; detector labels alone are insufficient.'),
            dict(id='subject_staticity', state='not_implemented',
                 evidence='Subject residual motion separated from camera uncertainty, with static-scene negative controls.')],
        reason='Coarse body/SfM fitting residuals do not establish the three independent alignment conditions. This development release has no validator that can admit formal Gaussian training.',
        next_action='Use bounded preprocessing/fit diagnostics. Implement and validate the independent alignment conditions before enabling formal training. No manual approval flag is supported.',
        scope='Artifact identity and missing-validation report only; not a new camera experiment, physical calibration result or training-quality verdict.')


def require_alignment(root):
    """Stop before Gaussian allocation; stale or hand-written pass files cannot authorize it."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = inspect_alignment(root)
    report = root / 'alignment/training-readiness.json'
    rt.save(report, result)
    raise AlignmentNotAdmitted(
        'Gaussian training is unavailable in this development release: independent '
        'camera, material-correspondence and staticity validation is not implemented. '
        f'Coarse fit acceptance is insufficient. See {report}. '
        'No Gaussian model or optimizer was initialized.')
