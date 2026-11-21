"""Replay immutable formal296 poses against the terminal 006 Area model."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


LAB = Path('/opt/gs-slam-lab')
BASE = LAB/'shuolee-smpl-placement-006/run'
AREA = LAB/'shuolee-smpl-alpha006-area/run'
POSE_ARCHIVE = LAB/'shuolee-smpl-placement-006-formal296'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-foreground-alpha-area')
COVERAGE_LOCK = Path('/mnt/d/Projects/Shuolee-smpl-view-coverage/.gpu-task.lock')
ARCHIVE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/'
               'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
EVALUATOR = ARCHIVE/'sofa-placement006-formal296/evaluate.py'
METRICS = ARCHIVE/'sofa-release-formal295/review.py'
BENCHMARK_CONTACTS = ARCHIVE/'sofa-cold-placement269-formal-heldout-fit/contacts.json'
BENCHMARK = ARCHIVE/'sofa-cold-placement269-formal-render-replay/result.json'
HELD = {'frame_000008': 56, 'frame_000022': 168, 'frame_000037': 288, 'frame_000050': 392}
AREA_COMMIT = 'e2b90ed849c4b1d83bc481ad8e194a3b4dea6948'
AREA_PLY = '89fbd5b44f23e0a28e3a90dff3e721a61faae6f8bbddc206c2fa1cf7bc31c2d7'
BASE_PLY = 'cc1f5e36aff8211e33df0ca123c117bfc3972d7b26b70542c815a9fb39e8cdc1'
FIXED_PINS = {
    EVALUATOR: '2983d35dcee61366d8c46aec1c053fc4b3407b90f2f09619338d7eb7e720be6f',
    METRICS: '72fb6738b6b6807dc0fe6d8124be41791eb31cdd3fc478a4349a139809a29f8a',
    BENCHMARK_CONTACTS: '12f36ec276d12cf3940587298cad073a8da20a9a2de48699fd6e40dbe9e39643',
    BENCHMARK: '83cd174bfd609bc64b929293f6d5c5db8ff02df79a1b8431d04a02fb0d4a5b12',
    POSE_ARCHIVE/'evaluation-fitted.npz': '1a7ec6326fefe955c1273a90e2f965525643bad4fe32182605c467d13d3b6c78',
    POSE_ARCHIVE/'fit-result.json': '9d85b3ae3936b03f87391a7a80b6d17fbcf91a3a99f48fb0aa8bb84c81364146',
    POSE_ARCHIVE/'inputs.json': '4c700fec55869ae776dd23c763b636ea62d0f19fc40132d63553863ba7804290',
    POSE_ARCHIVE/'renders/result.json': '09a2858aa7bed966dac5f30683362f618cf2321a2eacf6de2571ca766901e707',
    AREA/'export/point_cloud.ply': AREA_PLY,
    BASE/'export/point_cloud.ply': BASE_PLY,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_pins(pins):
    for path, expected in pins.items():
        need(sha(path) == expected, f'Pinned input changed: {path}')


def load_replay():
    source = EVALUATOR.read_text(encoding='utf-8-sig').split('def render_holdouts(', 1)[1].split('\ndef main():', 1)[0]
    source = 'def render_holdouts(' + source
    replacements = (
        ('output/"evaluation-fitted.npz"', 'state.pose_archive/"evaluation-fitted.npz"', 1),
        ('output/"fit-result.json"', 'state.pose_archive/"fit-result.json"', 2),
        ('output/"inputs.json"', 'state.pose_archive/"inputs.json"', 2),
        ('== state.pins, "Evaluation fitting is stale."', '== state.pose_pins, "Evaluation fitting is stale."', 1),
        ('                with Image.open(row["image"])',
         '                prior = state.baseline[frame]\n'
         '                need(record_posed == prior["posed_camera"] and record_fixed == prior["frozen_camera"],\n'
         '                     f"Baseline/Area native evaluation cameras differ: {frame}")\n'
         '                with Image.open(row["image"])', 1),
        ('                    values[label] = metrics(',
         '                    need(torch.isfinite(color).all() and torch.isfinite(alpha).all(), "Nonfinite native render.")\n'
         '                    values[label] = metrics(', 1),
        ('                    archived269_metrics=archived["metrics"]))',
         '                    baseline006_metrics=prior["metrics"], archived269_metrics=archived["metrics"]))', 1),
        ('            scope="Four frozen Gaussian holdouts; evaluation poses use heldout 2D observations and are fitted independently. "\n'
         '                  "Scores use exact archived 269 masks, not ground truth. Cross-run scores also reflect different evaluation fits "\n'
         '                  "and training masks; report pose-screen failures individually. PNG columns are source, posed and fixed reference.")',
         '            area_model_sha256=state.result["sha256"], baseline006_model_sha256=BASE_PLY,\n'
         '            evaluation_fitted_sha256=fit["fitted_sha256"],\n'
         '            input_pins_sha256=state.pins, evaluation_refits=0,\n'
         '            scope="Same four saved formal296 poses, native cameras, RGB and archived269 scoring masks for both 006 models. "\n'
         '                  "The Area model uses its own trained shared mesh offset. Scores are mask-relative and pose-screen failures remain flagged. "\n'
         '                  "The baseline-to-Area contrast includes the foreground/alpha recipe changes. PNG columns are source, posed and fixed reference.")', 1),
    )
    for old, new, count in replacements:
        need(source.count(old) == count, f'Archived renderer adapter boundary differs: {old}')
        source = source.replace(old, new)
    namespace = dict(RUN=AREA, HOLDOUTS=HELD, BASE_PLY=BASE_PLY, need=need, sha=sha,
                     read=read, check_pins=check_pins, Path=Path, TemporaryDirectory=TemporaryDirectory,
                     SimpleNamespace=SimpleNamespace, np=np)
    from PIL import Image
    namespace['Image'] = Image
    exec(compile(source, str(EVALUATOR), 'exec'), namespace)
    return namespace['render_holdouts']


def preflight():
    check_pins(FIXED_PINS)
    launch, run, base_run = read(AREA.parent/'preregistration.json'), read(AREA/'run.json'), read(BASE/'run.json')
    need(launch['commit'] == AREA_COMMIT and launch['baseline_run'] == run['fitted_inputs_reused_from'] == str(BASE)
         and launch['baseline_model_sha256'] == BASE_PLY and run['video_sha256'] == base_run['video_sha256']
         and run['camera_config_sha256'] == base_run['camera_config_sha256'], 'Area source and baseline fit identities differ.')
    pins = dict(FIXED_PINS)
    for name, expected in launch['reused_inputs_sha256'].items():
        pins[AREA/name] = pins[BASE/name] = expected
    for name in ('export/cameras.json', 'gaussian-run/training-cameras.json'):
        pins[AREA/name] = pins[BASE/name] = sha(BASE/name)
    selection = read(BASE/'fit/selection.json')
    need(selection['retained_holdouts'] == list(HELD) and not set(HELD) & set(selection['selected_frames']),
         'A formal holdout is not protected by the common training selection.')
    shared = read(BASE/'fit/shared-body.json')
    anchors = {frame for group in shared['anchor_groups'].values() for frame in group}
    fitted = {path.name for path in (BASE/'fit/frames').iterdir() if path.is_dir()}
    need(not set(HELD) & (anchors | fitted), 'A formal holdout entered training pose estimation.')
    beta = np.asarray(shared['betas'], dtype=np.float32)

    area_result, baseline_result = read(AREA/'result.json'), read(BASE/'result.json')
    for root, result, expected in ((AREA, area_result, AREA_PLY), (BASE, baseline_result, BASE_PLY)):
        terminal = read(root/'gaussian-run/result.json')
        need(read(root/'status.json')['state'] == result['state'] == terminal['state'] == 'completed'
             and result['format_passed'] is True and result['sha256'] == expected
             and result['steps'] == terminal['steps'] == 10000 and result['sh_degree'] == terminal['sh_degree'] == 0
             and result['reference_frame_id'] == selection['reference_frame']
             and result['scene_sha256'] == sha(root/'export/scene.zip')
             and result['bound_replay_sha256'] == sha(root/result['bound_replay'])
             and result['bound_panels_sha256'] == sha(root/result['bound_panels']), 'Missing verified terminal delivery.')
        for name in ('status.json', 'run.json', 'result.json', 'gaussian-run/result.json', 'export/validation.json',
                     'export/scene.zip', result['bound_replay'], result['bound_panels']):
            pins[root/name] = sha(root/name)
    base_basis, area_basis = (read(root/'export/validation.json') for root in (BASE, AREA))
    need(all(base_basis[key] == area_basis[key] for key in ('rotation', 'origin_world', 'scale', 'reference_frame_id')),
         'Area and baseline static exports use different gauges.')
    snapshot = AREA/'gaussian-run/step-010000'
    bound = read(snapshot/'bound-replay.json')
    asset = Path(run['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    need(bound['source_files_sha256']['src/shuolee_portrait/training.py'] == launch['source_file_sha256'],
         'Area training source differs from its pinned profile.')
    pins.update({PROJECT/name: expected for name, expected in bound['source_files_sha256'].items()})
    pins.update({asset: bound['asset_sha256'], snapshot/'bound-local.ply': bound['bound_ply_sha256']})
    for name, path in dict(fitted=AREA/'fit/fitted.npz', training_rows=AREA/'data/training-inputs.json',
                           reference_mesh=snapshot/'reference-mesh.npz', world_ply=snapshot/'world.ply',
                           native_ply_parity=snapshot/'native-ply-parity.json').items():
        pins[path] = bound['inputs_sha256'][name]

    pose_receipt = read(POSE_ARCHIVE/'inputs.json')
    pose_pins = pose_receipt['input_sha256']
    pins.update({Path(name): expected for name, expected in pose_pins.items()})
    fit = read(POSE_ARCHIVE/'fit-result.json')
    baseline = read(POSE_ARCHIVE/'renders/result.json')
    need(pose_receipt['evaluation_only'] and pose_receipt['training_or_model_updates'] == 0
         and fit['evaluation_only'] and fit['gaussian_optimizer_updates'] == 0
         and fit['protected_frames'] == pose_receipt['protected_frames'] == baseline['protected_frames'] == HELD
         and baseline['fit_result_sha256'] == FIXED_PINS[POSE_ARCHIVE/'fit-result.json']
         and fit['input_receipt_sha256'] == FIXED_PINS[POSE_ARCHIVE/'inputs.json']
         and fit['fitted_sha256'] == FIXED_PINS[POSE_ARCHIVE/'evaluation-fitted.npz']
         and pose_receipt['frozen_q90'] == read(BASE/'fit/official-fit-result.json')['source']['confidence']['per_joint_q90'],
         'Formal296 evaluation pose inputs or training confidence changed.')
    helper = runpy.run_path(str(METRICS))
    native_rows = {row['frame']: row for row in read(BASE/'data/inputs.json')['records']}
    rows = [native_rows[frame] for frame in HELD]
    contacts, old_records = read(BENCHMARK_CONTACTS)['records'], read(BENCHMARK)['records']
    with np.load(POSE_ARCHIVE/'evaluation-fitted.npz', allow_pickle=False) as poses:
        need(poses['frame_ids'].tolist() == list(HELD) and poses['source_frames'].tolist() == list(HELD.values())
             and np.array_equal(poses['shared_betas'], beta)
             and np.array_equal(poses['K_integer_centers'], [row['K_integer_centers'] for row in rows])
             and np.array_equal(poses['w2c'], [row['w2c'] for row in rows])
             and poses['image_sha256'].tolist() == [row['source_sha256'] for row in rows]
             and poses['mask_sha256'].tolist() == [row['mask_sha256'] for row in rows],
             'Saved evaluation pose arrays do not match the protected native rows and shared shape.')
    baseline_rows = {}
    for row, contact, older, prior, diagnostic in zip(rows, contacts, old_records, baseline['records'], fit['diagnostics'], strict=True):
        frame = row['frame']
        need(frame == contact['frame'] == older['frame'] == prior['frame'] == diagnostic['frame']
             and row['source_frame_index'] == contact['source_frame_index'] == prior['source_frame_index'] == HELD[frame]
             and row['source_sha256'] == contact['source_sha256'] == prior['native_source_sha256']
             and row['mask_sha256'] == prior['fit_mask_sha256']
             and contact['mask_sha256'] == older['mask_sha256'] == prior['scoring_mask_sha256']
             and prior['pose_screen_flags'] == diagnostic['pose_screen_flags']
             and all(row[key] == contact[key] for key in ('wh', 'K_integer_centers', 'K', 'w2c')),
             f'Archived benchmark or fitted native frame differs: {frame}')
        pins[Path(row['image'])] = row['source_sha256']
        pins[Path(row['mask'])] = row['mask_sha256']
        pins[helper['relocate'](contact['mask'])] = contact['mask_sha256']
        pins[POSE_ARCHIVE/'frames'/frame/'result.json'] = diagnostic['report_sha256']
        pins[POSE_ARCHIVE/'frames'/frame/'winner-camera-space.npz'] = diagnostic['winner_sha256']
        baseline_rows[frame] = prior
    pins[AREA.parent/'preregistration.json'] = sha(AREA.parent/'preregistration.json')
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    check_pins(pins)
    return SimpleNamespace(run=run, result=area_result, snapshot=snapshot, asset=asset, subset={'rows':rows},
                           helper=helper, old_rows=contacts, old_records=old_records, beta=beta,
                           pose_archive=POSE_ARCHIVE, pose_pins=pose_pins, target_hash=fit['shared_target_sha256'],
                           baseline=baseline_rows, pins={str(path): expected for path, expected in pins.items()})


def no_live_training():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        need(not (b'train' in argv and any(arg.endswith(b'/run_smplx.py') for arg in argv)),
             'A reconstruction command is still live. Wait for its terminal state.')


def check_cameras(state):
    from shuolee_portrait.training import camera

    rigid, basis = state.helper['rigid'], state.helper['basis']
    with np.load(AREA/'fit/fitted.npz', allow_pickle=False) as fitted:
        slot = fitted['frame_ids'].tolist().index(state.result['reference_frame_id'])
        inverse_reference = np.linalg.inv(rigid(fitted['root_rotation'][slot], fitted['world_translation'][slot]))
        first = len(fitted['frame_ids'])
    inverse_basis = np.linalg.inv(basis(read(AREA/'export/validation.json')))
    with np.load(POSE_ARCHIVE/'evaluation-fitted.npz', allow_pickle=False) as poses:
        for i, row in enumerate(state.subset['rows']):
            pelvis = rigid(poses['root_rotation'][i], poses['world_translation'][i])
            fixed_row = dict(row, w2c=(np.asarray(row['w2c']) @ pelvis @ inverse_reference @ inverse_basis).tolist())
            prior = state.baseline[row['frame']]
            need(camera(row, i, pose_index=first+i)[1] == prior['posed_camera']
                 and camera(fixed_row, i)[1] == prior['frozen_camera'],
                 f"Baseline/Area evaluation camera differs: {row['frame']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'render'))
    parser.add_argument('--output', type=Path, default=LAB/'shuolee-smpl-alpha006-area-formal-paired')
    args = parser.parse_args()
    output = args.output.resolve()
    need(not output.exists() and not any(output.is_relative_to(path) for path in (AREA.parent, BASE.parent, POSE_ARCHIVE, PROJECT)),
         'Use a fresh output outside both model and evaluation archives.')
    state = preflight()
    replay = load_replay()
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    need(rt.PROJECT == PROJECT, 'The process imported a different Gaussian renderer worktree.')
    state.seq = SimpleNamespace(rt=rt)
    check_cameras(state)
    if args.mode == 'check':
        print(json.dumps(dict(state='frozen_area_and_evaluation_inputs_verified', protected_frames=HELD,
                              input_pins=len(state.pins), exact_baseline_camera_pairs=len(HELD),
                              evaluation_refits=0, gpu_calls=0)))
        return
    with COVERAGE_LOCK.open('rb') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_live_training()
        output.mkdir(parents=True, exist_ok=False)
        print(replay(state, output)['state'])


if __name__ == '__main__':
    main()
