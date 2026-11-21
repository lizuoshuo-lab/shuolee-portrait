"""Run a paired R21 experiment with the exact R20 fitted inputs."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

import launch_smpl_delivery_round20 as previous


cold = previous.cold
need, read, pin, sha, write = previous.need, previous.read, previous.pin, previous.sha, previous.write
PYTHON, OUT = previous.PYTHON, previous.OUT
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-support-20261111')
PARENT = 'ad52fbee2d78527e022ef145b20a497c4c4e2fa1'
DIRECTOR = previous.DIRECTOR_ROOT/'round20_director_review_20261111.json'
DIRECTOR_SHA = '0222df3ae07a44c10507db9695dd285cce52bbe6ad545a1a9452c14adff2e835'
DELIVERY = OUT/'smpl-delivery-round20/receipt.json'
DELIVERY_SHA = 'be40feef7d37dc34de765746ac8f44ab4a1a02e83fd553a3cd228525def52b9d'
POLICY_PATH = OUT.parent/'shuolee-coordination/point-cloud-upper-lower-view-policy-20261111.json'
POLICY_SHA = '8cec8754264c5a258bb5e42e3d1eb0d8235eebac35d11b02d416366abca87c34'
FROZEN_VIEW_POLICY_SHA = '9a5896fde8f9cec7a2158b94cce622740828c26722d94ea6215fba42667c9e4a'
SCHEMA = 'shuolee-round21-frozen-fit-surface-duration-v1'
MIDPOINT = 10000
STEPS = 18000
CUDA_PROOF = OUT/'smpl-delivery-round21/input-audit/cuda-surface-attachment-proof.json'
CUDA_FIXTURE = Path(__file__).with_name('probe_r21_surface_attachment_cuda.py')
ARCHIVED_SCALE_PROOF_SHA = '6371495cd08546cfdac0ba16d04025cc8b60fd83e8948175218aebb5f761e83b'
VIEW_DIR = CUDA_PROOF.parent
VIEW_SHA = {'016':'b3e7f1d775d9be5a493b199b0e3b8b6997fcdb4a8af3545b5c6f5748e4ca9f0d',
            '006':'c860d552162f96bb5cc335d489e744975aef1bb616881be459147fd860876505'}
MODULE_PREFIX = 'src/shuolee_portrait/'
ALLOWED = {MODULE_PREFIX+name for name in ('head_support.py','head_support_views.py','head_support_pitch.py',
           'training.py','render_support.py','size_child_optimization.py','sequence_pipeline.py','duration_checkpoint.py',
           'export.py','surface_attachment.py','clothing_surface_guide.py')}
DURATION_POLICY = dict(checkpoints=[MIDPOINT,STEPS],schedule_steps=STEPS,
                       independent_postprocessing=True,single_main_training=True)


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round21-{subject}'


def _input_bytes_path(path,digest):
    if path == POLICY_PATH and digest == POLICY_SHA:
        frozen = VIEW_DIR/'frozen-r21-surface-duration-policy.json'
        if frozen.is_file():
            return frozen
    return path


def _pin_input(path,digest):
    pin(_input_bytes_path(path,digest),digest)


def source_inventory(subject):
    """Recover the signed pose protocol, all original bytes and the paired R20 endpoints."""
    root = previous.root_for(subject)
    run = root/'run'
    # This archived inventory binds the preimages used by the independent R20
    # fit verifier. Keep it separate from the active corrected bytes staged here.
    archived = read(root/'protected-inputs.json')
    plan = read(root/'preregistration.json')
    files = dict(archived['files'])

    def record(path,digest=None):
        path = Path(path)
        if digest is None:
            digest = sha(path)
        else:
            _pin_input(path,digest)
        need(files.setdefault(str(path),digest) == digest,f'Conflicting archive identity for {path}.')
        return dict(path=str(path),sha256=digest)

    record(DIRECTOR,DIRECTOR_SHA)
    record(DELIVERY,DELIVERY_SHA)
    record(POLICY_PATH,POLICY_SHA)
    director, delivery = read(DIRECTOR),read(DELIVERY)
    policy = read(_input_bytes_path(POLICY_PATH,POLICY_SHA))
    need(director['round'] == delivery['round'] == 20 and policy['round'] == 21
         and director['commit'] == delivery['commit'] == plan['candidate_commit'] == PARENT
         and director['receipt_sha256'] == DELIVERY_SHA and delivery['quality_accepted'] is False
         and delivery['stable_release_commit'] == previous.STABLE_TAG_COMMIT
         and policy['training_duration_comparison']['main_training_checkpoints'] == [MIDPOINT,STEPS]
         and isinstance(policy['surface_alignment_steering']['face_priority'],list)
         and isinstance(policy['surface_alignment_steering']['clothing_priority'],list),
         'The R20 review and current R21 quality directive do not bind the same frozen baseline.')
    for name,kind in (('quality-findings.json','quality_findings'),('review-receipt.json','review_receipt')):
        record(DELIVERY.parent/name,delivery[kind]['sha256'])
    review = delivery['subjects'][subject]
    record(DELIVERY.parent/subject/'receipt.json',review['delivery']['sha256'])
    terminal_path = root/'wrapper-terminal.json'
    terminal_record = record(terminal_path,review['wrapper']['sha256'])
    terminal = read(terminal_path)
    record(root/'protected-inputs.json',terminal['protected_input_inventory_sha256'])
    record(root/'preregistration.json',terminal['preregistration_sha256'])
    record(root/'launch.json',terminal['launch_sha256'])
    record(root/'entry.log',terminal['entry_log_sha256'])
    record(root/'cpu-preparation.json',plan['cpu_preparation_sha256'])
    record(run/'result.json',review['result_sha256'])
    for path,digest in archived['files'].items():
        pin(Path(path),digest)

    head = previous.verify_head_fit(run,archived)
    main = archived['selected_frames']
    aux = plan['cpu_fit_preflight']['chosen_face_frames']
    held = archived['retained_holdouts']
    protection = terminal['protected_face']
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == PARENT
         and terminal['result_sha256'] == review['result_sha256']
         and terminal['fit_sha256'] == head['fit_sha256'] == plan['head_fit']['fit_sha256']
         and terminal['head_fit'] == plan['head_fit'] == head
         and terminal['same_round19_training_selection'] is True and terminal['verified_pose_count'] == 28
         and plan['selected_frames'] == main and set(review['selected_frames']) == set(main)
         and terminal['face_supplement']['frames'] == aux == archived['archived_face_supplement']['frames']
         and terminal['heldout_replay']['frames'] == review['retained_holdouts'] == held
         and len(set(main+aux+held)) == 32 and (len(main),len(aux),len(held)) == (24,4,4)
         and protection['state'] == 'rolled_back' and protection['attempted_updates'] == 336
         and protection['committed_updates'] == 0,
         'The R20 terminal, corrected fit, paired roster or real protected rollback differs.')

    staged = {name:record(run/name,head['active_files_sha256'].get(name,entry['sha256']))
              for name,entry in archived['staged_files'].items()}
    staged.update({name:record(run/name,digest) for name,digest in head['active_files_sha256'].items()})
    for name in previous.MUTABLE_STAGED:
        source = 'fit/head-alignment/source/'+Path(name).name
        staged[source] = record(run/source,archived['staged_files'][name]['sha256'])
    for row in head['holdout_plan']['records']:
        name = row['official_fit_path']
        staged[name] = record(run/name,row['official_fit_sha256'])
    need(staged['fit/head-alignment/result.json']['sha256'] == review['head_fit']['sha256']
         and staged['fit/head-alignment/holdout-plan.json']['sha256'] == review['heldout_plan']['sha256'],
         'The frozen head alignment or evaluation pose plan differs from the R20 delivery.')

    profiles = {name:record(run/name) for name in archived['profiles']}
    record(run/'gaussian-run'/protection['file'],protection['sha256'])
    main_endpoint = run/'gaussian-run'/protection['main_endpoint']
    record(main_endpoint,protection['main_endpoint_sha256'])
    endpoint = read(main_endpoint)
    for name,digest in endpoint['files_sha256'].items():
        record(run/'gaussian-run'/endpoint['snapshot']/name,digest)
    for step,parity in zip((1000,previous.TRAINING_STEPS),terminal['native_ply_parity'],strict=True):
        need(previous.verify_parity(run,step,archived['reference_frame']) == parity,
             'The paired R20 native PLY comparison is no longer verifiable.')
        record(run/parity['file'],parity['sha256'])
    for kind in ('admission','training','replay','poses'):
        name = f'{kind}.npz' if kind == 'poses' else f'{kind}.json'
        record(run/'gaussian-run/face-supplement'/name,terminal['face_supplement'][f'{kind}_sha256'])
    heldout = terminal['heldout_replay']
    record(run/heldout['file'],heldout['sha256'])
    heldout_dir = (run/heldout['file']).parent
    record(heldout_dir/'meshes.npz',heldout['mesh_sha256'])
    for row in heldout['records']:
        for item in row['files'].values():
            record(heldout_dir/item['file'],item['sha256'])
    result = read(run/'result.json')
    record(run/result['ply'],result['sha256'])
    record(run/result['scene'],result['scene_sha256'])
    record(run/result['camera'],review['candidate_camera']['sha256'])
    need(result['sha256'] == review['candidate_ply']['sha256'], 'The paired static PLY differs from delivery.')
    cloth_source = run/'gaussian-run/step-010000/reference-mesh.npz'
    cloth_baseline = record(cloth_source,read(cloth_source.parent/'native-ply-parity.json')['mesh_sha256'])
    need(cloth_baseline['sha256'] == endpoint['files_sha256']['reference-mesh.npz'],
         'The reviewed R20 garment guide differs between its main and final outputs.')
    name = 'fit/r21-baseline-clothing-mesh.npz'
    staged[name] = cloth_baseline
    frozen_surface = dict(file=name,sha256=cloth_baseline['sha256'])
    return dict(archived,schema=SCHEMA,baseline_commit=PARENT,baseline_run=str(run),staged_files=staged,
                profiles=profiles,files=files,archived_round20_terminal=terminal_record,
                archived_round20_inventory=dict(path=str(root/'protected-inputs.json'),
                    sha256=terminal['protected_input_inventory_sha256']),
                archived_round20_plan=dict(path=str(root/'preregistration.json'),sha256=terminal['preregistration_sha256']),
                archived_face_supplement=terminal['face_supplement'],archived_face_cpu_preflight=plan['cpu_fit_preflight'],
                fit_sha256=head['fit_sha256'],head_fit=head,frozen_surface=frozen_surface)


def verify_frozen_fit(run,inventory):
    """Recheck the active and archived fitted arrays without writing or recomputing poses."""
    for name,item in inventory['staged_files'].items():
        pin(run/name,item['sha256'])
    archived = inventory['archived_round20_inventory']
    path = Path(archived['path'])
    pin(path,archived['sha256'])
    need(previous.verify_head_fit(run,read(path)) == inventory['head_fit'],
         'R21 altered the signed R20 fit, evaluated pose plan or frozen source inputs.')
    return inventory['head_fit']


def check_code(commit,baseline_plan):
    pointer = (PROJECT/'.git').read_text().strip()
    need(pointer.startswith('gitdir: '),'The candidate must be a linked worktree.')
    git_dir = subprocess.check_output(['/usr/bin/wslpath','-u',pointer.removeprefix('gitdir: ')],text=True).strip()
    git = ['/usr/bin/git','--no-optional-locks','-c','core.fileMode=false','--git-dir',git_dir,
           '--work-tree',str(PROJECT)]
    output = lambda *args: subprocess.check_output([*git,*args],cwd=PROJECT,text=True).strip()
    need(output('rev-parse','HEAD') == commit and not output('status','--porcelain','--untracked-files=all'),
         'The candidate is not the exact requested clean worktree.')
    need(output('rev-parse','refs/tags/smpl-stable-20261111^{commit}') == previous.STABLE_TAG_COMMIT,
         'The fixed local stable Git release tag changed.')
    subprocess.run([*git,'merge-base','--is-ancestor',PARENT,commit],cwd=PROJECT,check=True)
    changed = set(output('diff','--name-only',PARENT,commit).splitlines())
    focused_tests = {name for name in changed if Path(name).parent == Path('tests')
                     and Path(name).name.startswith(('test_head_support','test_duration_checkpoint',
                         'test_surface_attachment','test_clothing_surface')) and Path(name).suffix == '.py'}
    required = {MODULE_PREFIX+name for name in ('surface_attachment.py','clothing_surface_guide.py',
                'duration_checkpoint.py','training.py','export.py','sequence_pipeline.py')}
    need(required <= changed <= ALLOWED | focused_tests,
         f'Unexpected R21 changes to the stable pipeline: {sorted(changed)}')
    for name,digest in baseline_plan['code_sha256'].items():
        if name not in changed:
            pin(PROJECT/name,digest)
    files = set(baseline_plan['code_sha256']) | changed
    return {name:sha(PROJECT/name) for name in sorted(files)}


def _setup_archive(subject):
    return root_for(subject).with_name(root_for(subject).name+'-before-frozen-surface')


def _archived_setup(subject):
    root = _setup_archive(subject)
    if not root.is_dir():
        return None
    plan = read(root/'preregistration.json')
    wrapper = VIEW_DIR/'launch-wrapper-before-frozen-surface.py'
    pin(wrapper,plan['wrapper_sha256'])
    names = ['preregistration.json','protected-inputs.json']
    if subject == '016':
        names.extend(('wrapper-terminal.json','entry.log','launch.json','run/status.json'))
    return dict(root=str(root),reason='Frozen reviewed R20 garment guide was absent before Gaussian initialization.',
                optimizer_updates=0,wrapper_sha256=plan['wrapper_sha256'],
                files_sha256={name:sha(root/name) for name in names})


def archive_incomplete_setup(subject):
    root = root_for(subject)
    destination = _setup_archive(subject)
    need(root.resolve(strict=True) == root and destination.parent.resolve(strict=True) == cold.LAB
         and destination.parent == root.parent and not destination.exists()
         and not (root/'run/gaussian-run').exists(),
         'Only the exact unused R21 setup directory can be preserved for retry.')
    plan = read(root/'preregistration.json')
    pin(VIEW_DIR/'launch-wrapper-before-frozen-surface.py',plan['wrapper_sha256'])
    if subject == '016':
        terminal,status = (read(root/name) for name in ('wrapper-terminal.json','run/status.json'))
        need(terminal['state'] == status['state'] == 'failed' and status['stage'] == 'gaussian_inputs'
             and not status.get('training_started') and terminal['returncode'] != 0
             and status['error'] == 'The paired head-support experiment requires a frozen fitted surface.',
             'Only the proven zero-update setup failure may be archived for this recovery.')
        pin(root/'entry.log',terminal['entry_log_sha256'])
    else:
        need(not any((root/name).exists() for name in ('launch.json','entry.log','wrapper-terminal.json','run/status.json')),
             'The unlaunched companion already began execution.')
    root.rename(destination)
    print(dict(state='preserved_pretraining_attempt',subject=subject,archive=_archived_setup(subject)),flush=True)


def _cuda_control(path,digest,code,runtime):
    need(re.fullmatch('[0-9a-f]{64}',digest) is not None,'A SHA-256 for the actual R21 CUDA control is required.')
    pin(path,digest)
    proof = read(path)
    sources = proof['source_sha256']
    expected = {MODULE_PREFIX+name for name in ('surface_attachment.py','duration_checkpoint.py',
                                                'clothing_surface_guide.py')}
    raster, = [value for name,value in runtime['file_sha256'].items()
               if 'diff_gaussian_rasterization' in name and Path(name).name.startswith('_C')]
    need(proof['state'] == 'passed' and proof['schema'] == 'shuolee-r21-cuda-surface-proof/1' and set(sources) == expected
         and all(code[name] == value for name,value in sources.items())
         and proof['fixture_sha256'] == sha(CUDA_FIXTURE)
         and proof['runtime']['raster_extension_sha256'] == raster
         and [row['subject'] for row in proof['subjects']] == ['016','006']
         and set(proof['cloth_proofs']) == {'016','006'},
         'The CUDA control does not bind the actual paired R21 surface and state implementations.')
    for row in proof['subjects']:
        subject = row['subject']
        source = previous.root_for(subject)/'run'
        archive = source/'gaussian-run/step-010000'
        pin(archive/'bound-local.ply',row['frozen_archive_sha256'])
        checkpoint = row['checkpoint']
        grad = row['absolute_grad_sum']
        frames = [item['frame'] for item in read(source/'data/training-inputs.json')['records']]
        need(row['diagnostic_native_views'] == len(frames) == 24 and row['native_pose_sample'] in frames
             and row['support']['supported_views'] > 0 and row['support']['repeated_skin_triangles'] > 0
             and all(map(math.isfinite,grad.values())) and grad['center'] > 0 and max(grad['scale'],grad['rotation']) > 0
             and checkpoint['schema'] == 'shuolee-duration-checkpoint/1'
             and checkpoint['weights_only_load'] is True and checkpoint['cuda_cpu_and_numpy_python_rng_exact'] is True
             and 0 <= checkpoint['next_update_max_abs_parameter_difference'] < 1e-6
             and checkpoint['adam_steps_after_next']['xyz'] == 2 and checkpoint['scheduled_xyz_lr'] > 0,
             f'CUDA did not prove real native attachment gradients and restorable Adam/RNG for {subject}.')
        cloth = path.parent/f'cuda-clothing-guide-{subject}.json'
        pin(cloth,proof['cloth_proofs'][subject])
        garment = read(cloth)
        need(garment['subject'] == subject and garment['implementation_sha256'] == code[MODULE_PREFIX+'clothing_surface_guide.py']
             and garment['optimizer_scope'] == 'shared garment offsets only' and garment['replay_error_m'] < 2e-6
             and _clothing_trial_selection(garment,frames),
             f'CUDA clothing control did not verify a guarded shared garment candidate for {subject}.')
        pin(archive/'reference-mesh.npz',garment['baseline_mesh_sha256'])
    retired = path.parent/'cuda-head-support-proof.json'
    pin(retired,ARCHIVED_SCALE_PROOF_SHA)
    need(read(retired)['parents_committed'] == 0,'Retired size-control evidence differs.')
    return proof


def _pin_view_protocol(path,digest,subject,inventory):
    import numpy as np

    need(path == VIEW_DIR/f'head-support-views-{subject}.json' and digest == VIEW_SHA[subject],
         'Use the immutable 50-camera review manifest for this subject.')
    pin(path,digest)
    protocol = read(path)
    views = protocol['views']
    reference = inventory['reference_frame']
    with np.load(Path(inventory['staged_files']['fit/fitted.npz']['path']),allow_pickle=False) as data:
        index = data['frame_ids'].tolist().index(reference)
        native_center_K = data['K_integer_centers'][index]
    native_corner_K = native_center_K.copy()
    native_corner_K[:2,2] += .5
    expected = {(name,azimuth,pitch) for name in ('normal','close') for azimuth in (-90,-45,0,45,90)
                for pitch in (-30,-15,0,15,30)}
    combinations = {(row['distance_class'],row['azimuth_deg'],row['pitch_deg']) for row in views.values()}
    need(protocol['schema'] == 'shuolee-head-support-static-views/1' and protocol['subject'] == subject
         and protocol['reference_frame'] == reference and protocol['pixel_convention'] == 'pixel_corners'
         and np.array_equal(protocol['native_K'],native_corner_K) and len(views) == 50 and combinations == expected,
         'The pinned static review does not preserve native K, the R20 reference, and the complete 50-camera grid.')
    for name,row in views.items():
        camera = row['world_camera']
        matrix = np.asarray(camera['w2c'])
        R = matrix[:3,:3]
        need(row['real_rgb_ground_truth_available'] is False and camera['frame'] == reference
             and camera['wh'] == protocol['native_wh'] and np.array_equal(camera['K'],native_corner_K)
             and np.array_equal(camera['K_integer_centers'],native_center_K)
             and matrix.shape == (4,4) and np.isfinite(matrix).all() and np.array_equal(matrix[3],[0,0,0,1])
             and np.allclose(R@R.T,np.eye(3),atol=1e-6,rtol=0) and np.isclose(np.linalg.det(R),1.,atol=1e-6)
             and np.isclose(row['distance_m'],protocol[row['distance_class']+'_distance_m']),
             f'Fixed synthetic camera {name} changed dimensions, calibration or world geometry.')
    anchors = protocol['proof']
    need(anchors['round20_receipt']['sha256'] == DELIVERY_SHA
         and anchors['round20_director']['sha256'] == DIRECTOR_SHA
         and anchors['user_view_priority']['sha256'] == FROZEN_VIEW_POLICY_SHA
         and anchors['reference_fitted']['sha256'] == inventory['fit_sha256'],
         'The camera review was derived from a different frozen fit or delivery.')
    for name,item in anchors.items():
        recorded = VIEW_DIR/'frozen-user-view-priority.json' if name == 'user_view_priority' else Path(item['file'])
        pin(recorded,item['sha256'])
    return protocol


def plan_for(subject,commit,cuda_proof_sha,views,views_sha):
    inventory = source_inventory(subject)
    old_plan = read(Path(inventory['archived_round20_plan']['path']))
    code = check_code(commit,old_plan)
    runtime = cold.cpu_runtime(cold.clean_env())
    need(runtime == old_plan['runtime'],'Training runtime differs from R20.')
    _cuda_control(CUDA_PROOF,cuda_proof_sha,code,runtime)
    _pin_view_protocol(views,views_sha,subject,inventory)
    assets = old_plan['asset_sha256']
    for name,digest in assets.items():
        pin(Path(name),digest)
    need(min(shutil.disk_usage(cold.LAB).free,shutil.disk_usage('/mnt/c').free) > 8*1024**3,
         'Insufficient disk space for the paired reconstruction and replay records.')
    run = root_for(subject)/'run'
    command = [str(PYTHON),'-B','-u','-m','shuolee_portrait.cli','train-fitted','--run',str(run)]
    return dict(schema=SCHEMA,subject=subject,candidate_commit=commit,baseline_commit=PARENT,
                quality_baseline_commit=PARENT,stable_git_release_commit=previous.STABLE_TAG_COMMIT,
                baseline_run=inventory['baseline_run'],baseline_result_sha256=inventory['profiles']['result.json']['sha256'],
                output=str(run),director_review_sha256=DIRECTOR_SHA,baseline_delivery_sha256=DELIVERY_SHA,
                quality_policy_sha256=POLICY_SHA,cuda_fixture_sha256=cuda_proof_sha,
                archived_size_control_sha256=ARCHIVED_SCALE_PROOF_SHA,
                pitch_camera_manifest=dict(path=str(views),sha256=views_sha),wrapper_sha256=sha(Path(__file__)),
                reused_verifier_sha256=sha(Path(previous.__file__)),runtime=runtime,code_sha256=code,asset_sha256=assets,
                selected_frames=inventory['selected_frames'],retained_holdouts=inventory['retained_holdouts'],
                candidate_count=52,full_body_views=24,face_only_views=inventory['archived_face_supplement']['frames'],
                commands=[command],gaussian_steps=STEPS,comparison_steps=[MIDPOINT,STEPS],position_lr_max_steps=STEPS,
                main_face_aux_updates=0,color_policy=previous.POLICY,
                head_fit=inventory['head_fit'],frozen_surface=inventory['frozen_surface'],
                archived_setup=_archived_setup(subject),max_wall_seconds=previous.MAX_WALL_SECONDS),inventory


def stage(run,inventory,view_record):
    previous.stage(run.parent,inventory)
    config = run/'config.json'
    raw = read(config)
    need('frozen_surface' not in raw,'The old config unexpectedly includes a different frozen surface.')
    temporary = config.with_suffix('.r21.tmp')
    write(temporary,dict(raw,frozen_surface=inventory['frozen_surface']))
    temporary.replace(config)
    views = run/'data/head-support-views.json'
    with Path(view_record['path']).open('rb') as source, views.open('xb') as target:
        shutil.copyfileobj(source,target)
    pin(views,view_record['sha256'])
    verify_frozen_fit(run,inventory)


def cpu_preflight(run,inventory):
    """Use the current named-fit loader and preserve the R20 sampler/face admission."""
    script = """
from collections import Counter
from hashlib import sha256
import json,sys
from pathlib import Path
import numpy as np
import torch
from shuolee_portrait import runtime as rt
from shuolee_portrait.training import verify_official_fit, smpl_sh0_effective_config, _frozen_smpl_surface, named_pose_layout
from shuolee_portrait.sh0_main import frozen_surface
from shuolee_portrait.stable_selection import source_bin_training_order
from shuolee_portrait.face_supplement import prepare_face_supplements

run = Path(sys.argv[1])
rt.configure(Path(rt.read(run/'run.json')['assets']),run)
torch.set_num_threads(4)
fit = verify_official_fit(run)
rows = rt.read(run/'data/training-inputs.json')['records']
with np.load(run/'fit/fitted.npz',allow_pickle=False) as archive:
    fitted = dict(archive)
layout = named_pose_layout(fitted,rows)
cfg = rt.read(run/'config.json')
base,linear,clothing,protected,_ = _frozen_smpl_surface(fitted,rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
mesh,offset = frozen_surface(run,cfg['frozen_surface'],fitted,layout,clothing)
slot = layout['reference_pose_index']
replayed = base[slot]+(linear[slot] @ torch.from_numpy(offset)[...,None]).squeeze(-1)
with np.load(mesh,allow_pickle=False) as archive:
    max_error = float((replayed-torch.from_numpy(archive['vertices_world'])).abs().max())
if max_error >= 2e-6 or not np.array_equal(replayed.numpy()[protected],base[slot].numpy()[protected]):
    raise ValueError('The frozen R20 garment guide does not replay against this exact fitted identity.')
supplement,admission = prepare_face_supplements(run,rows)
selection = rt.read(run/'fit/official-fit-result.json')['selection']
effective = smpl_sh0_effective_config()
bins,order = source_bin_training_order(rows,selection,seed=effective['seed'])
draws = bytes(next(order) for _ in range(18000))
def receipt(sequence):
    by_frame = Counter(sequence)
    by_bin = Counter(bins[index] for index in sequence)
    return dict(expected_sampling_order_sha256=sha256(sequence).hexdigest(),
                expected_per_frame_updates=[by_frame[index] for index in range(len(rows))],
                expected_source_bin_draws=by_bin)
if torch.cuda.is_initialized():
    raise RuntimeError('CPU staging unexpectedly initialized CUDA.')
print(json.dumps(dict(fit_sha256=fit['fit_sha256'],training_inputs_sha256=fit['training_inputs_sha256'],
    frames=len(fit['official_frame_results']),cuda_initialized=False,
    frozen_surface_sha256=rt.sha(mesh),frozen_surface_replay_max_abs_m=max_error,
    **receipt(draws[:10000]),continuous_18000=receipt(draws),
    face_admission_payload_sha256=sha256(json.dumps(admission,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest(),
    face_candidate_count=len(admission['fit_candidate_frame_ids']),face_admission_considered=len(admission['candidates']),
    chosen_face_frames=[item.row['frame'] for item in supplement])))
"""
    env = dict(cold.clean_env(),CUDA_VISIBLE_DEVICES='')
    actual = json.loads(subprocess.check_output([str(PYTHON),'-B','-c',script,str(run)],
                                              cwd=PROJECT/'src',env=env,text=True))
    old = inventory['archived_face_cpu_preflight']
    need(all(actual[key] == value for key,value in old.items())
         and sum(actual['continuous_18000']['expected_per_frame_updates']) == STEPS
         and actual['frozen_surface_sha256'] == inventory['frozen_surface']['sha256']
         and 0 <= actual['frozen_surface_replay_max_abs_m'] < 2e-6,
         'The current entry changed the R20 fit, original 10k sampler prefix or face admission.')
    return actual


def register(subject,commit,cuda_proof_sha,views,views_sha):
    plan,inventory = plan_for(subject,commit,cuda_proof_sha,views,views_sha)
    root = root_for(subject)
    root.mkdir(exist_ok=True)
    need(not any(root.iterdir()),'Registration requires an untouched R21 run root.')
    run = root/'run'
    stage(run,inventory,plan['pitch_camera_manifest'])
    cpu = cpu_preflight(run,inventory)
    protected = root/'protected-inputs.json'
    write(protected,inventory)
    write(root/'preregistration.json',dict(plan,registered_at=cold.now(),cpu_fit_preflight=cpu,
          protected_input_inventory_sha256=sha(protected)))
    print(dict(state='registered',subject=subject,files=len(inventory['files']),fit_sha256=inventory['fit_sha256'],
               views_sha256=views_sha,inventory_sha256=sha(protected)),flush=True)


def registered(subject,commit,cuda_proof_sha,views,views_sha):
    root = root_for(subject)
    plan = read(root/'preregistration.json')
    protected = root/'protected-inputs.json'
    pin(protected,plan['protected_input_inventory_sha256'])
    inventory = read(protected)
    current,frozen = plan_for(subject,commit,cuda_proof_sha,views,views_sha)
    need(inventory == frozen and all(plan[key] == value for key,value in current.items() if key != 'wrapper_sha256'),
         'Formal inputs differ from the candidate, pose and view preregistration.')
    if plan['wrapper_sha256'] != current['wrapper_sha256']:
        pin(VIEW_DIR/'launch-wrapper-registered-r21.py',plan['wrapper_sha256'])
        pin(VIEW_DIR/'frozen-r21-surface-duration-policy.json',POLICY_SHA)
    verify_frozen_fit(root/'run',inventory)
    pin(root/'run/data/head-support-views.json',views_sha)
    need(read(root/'run/config.json')['frozen_surface'] == inventory['frozen_surface'],
         'The formal config no longer uses the registered R20 garment guide.')
    return plan,inventory


def verify_protected_stage(run,integration,trained,effective,inventory,face_frames,step):
    out = run/'gaussian-run'
    stage = out/'protected-face'
    final = out/f'step-{step:06d}'
    main = out/f'step-{step:06d}-main'
    desc = integration['protected_face']
    audit_path, endpoint_path = (stage/name for name in ('optimization.json','main-endpoint.json'))
    need(desc == trained['protected_face'] and desc['file'] == 'protected-face/optimization.json'
         and desc['main_endpoint'] == 'protected-face/main-endpoint.json'
         and desc['main_snapshot'] == trained['main_snapshot'] == main.name,
         'The normal training result does not bind the saved main and protected stages.')
    pin(audit_path,desc['sha256'])
    pin(endpoint_path,desc['main_endpoint_sha256'])
    proof, endpoint = read(audit_path),read(endpoint_path)
    need(endpoint['state'] == 'saved' and endpoint['snapshot'] == main.name
         and endpoint['main_optimizer_updates'] == step and endpoint['supplemental_updates'] == 0
         and set(endpoint['files_sha256']) == {'world.ply','reference-render.png','reference-mesh.npz',
               'native-ply-parity.json','bound-local.ply','bound-replay.json'},
         'Missing intact same-duration original-only endpoint.')
    for name,digest in endpoint['files_sha256'].items():
        pin(main/name,digest)
    parity = previous._snapshot_parity(main,inventory['reference_frame'],run)
    main_bound, final_bound = (read(snapshot/'bound-replay.json') for snapshot in (main,final))
    for snapshot,replay in ((main,main_bound),(final,final_bound)):
        pin(snapshot/'bound-local.ply',replay['bound_ply_sha256'])
        bound_inputs = {'fitted':run/'fit/fitted.npz','training_rows':run/'data/training-inputs.json',
                        'reference_mesh':snapshot/'reference-mesh.npz','world_ply':snapshot/'world.ply',
                        'native_ply_parity':snapshot/'native-ply-parity.json'}
        need(set(replay['inputs_sha256']) == set(bound_inputs),'Incomplete bound snapshot input manifest.')
        for key,path in bound_inputs.items():
            pin(path,replay['inputs_sha256'][key])
        need(replay['points'] == trained['points'] and replay['reference_frame_id'] == inventory['reference_frame']
             and replay['reference_pose_index'] == 0 and replay['sh_degree'] == 0
             and [row['frame'] for row in replay['frames']] == inventory['selected_frames'],
             'Main or final bound replay does not preserve the 24 original named poses.')
    need(main_bound['frames'] == final_bound['frames'] and all(main_bound['inputs_sha256'][key] ==
         final_bound['inputs_sha256'][key] for key in ('fitted','training_rows')),
         'Main and final bound snapshots used different original pose/camera inputs.')
    if not face_frames:
        need(proof['state'] == desc['state'] == 'skipped' and proof['reason'] == 'no_admitted_auxiliary_views'
             and all(desc[key] == proof[key] == 0 for key in ('attempted_updates','committed_updates'))
             and desc['committed_auxiliary_frame_ids'] == [], 'An empty auxiliary roster performed updates.')
        return dict(desc,main_native_ply_parity=parity,baseline_views=24,unprotected_control_updates=0)
    committed = proof['state'] == 'committed'
    identity = previous.verify_mesh_and_bound_identity(run,main,final,('f_dc','opacity'),committed)
    need(proof['schema'] == 'shuolee-protected-face/1' and proof['state'] in ('committed','rolled_back')
         and all(desc[key] == proof[key] for key in ('state','attempted_updates','committed_updates','seconds'))
         and proof['main_optimizer_updates'] == 0 and proof['head_rows'] == identity['head_rows']
         and proof['rng_unchanged'] is True and proof['unprotected']['same_endpoint'] is True
         and proof['unprotected']['rng_unchanged'] is True
         and {key:value for key,value in proof['endpoint'].items() if key != 'mutable_head'}
             == {key:value for key,value in proof['live_final'].items() if key != 'mutable_head'}
         and proof['endpoint']['points'] == trained['points'],
         'Protected refinement lost the fixed topology, optimizer identity or common RNG endpoint.')
    roster = (list(zip(inventory['selected_frames'],['original']*24,effective['training_source_bins'],strict=True))
              +[(name,'supplemental',i) for i,name in enumerate(face_frames)])
    guard = effective['unified_main']['world_scale']['render_support']
    baseline, final_metrics = proof['baseline'],proof['final']
    previous._verify_fixed_measurement(baseline,baseline,roster,guard,require_pass=True)
    previous._verify_fixed_measurement(proof['unprotected']['measurement'],baseline,roster,guard)
    previous._verify_fixed_measurement(final_metrics,baseline,roster,guard,require_pass=True)
    sweep = previous._verify_protected_sweeps(proof,roster)
    need(desc['committed_auxiliary_frame_ids'] == [frame for frame in face_frames if sweep[frame] > 0],
         'Committed auxiliary IDs differ from the exact chosen Adam updates.')
    gains = proof['final_gains']
    need(math.isclose(gains['joint'],baseline['objective']-final_metrics['objective'],abs_tol=1e-9)
         and math.isclose(gains['supplemental'],baseline['supplemental_objective']-final_metrics['supplemental_objective'],abs_tol=1e-9),
         'Reported final gains do not correspond to the same main endpoint.')
    if committed:
        need(proof['selected'] is not None and proof['committed_updates'] > 0
             and gains['joint'] >= proof['config']['minimum_gain'] == .0001
             and gains['supplemental'] >= proof['config']['minimum_gain']
             and proof['live_final']['mutable_head'] != proof['endpoint']['mutable_head'],
             'Protected commit has no verified nonzero joint and supplemental gains.')
        state = previous._verify_committed_state(stage,proof,main,final)
    else:
        need(proof['committed_updates'] == 0 and proof['optimizer_state'] is None
             and proof['endpoint'] == proof['live_final'] and not (stage/'protected-optimizer-state.pt').exists()
             and math.isclose(gains['joint'],0.,abs_tol=1e-7) and math.isclose(gains['supplemental'],0.,abs_tol=1e-7),
             'A rejected protected stage did not fully roll back to the original saved state.')
        state = None
    return dict(desc,main_native_ply_parity=parity,identity=identity,optimizer_state=state,final_gains=gains,
                committed_frame_draws={frame:sweep[frame] for frame in face_frames},
                baseline_views=len(roster),unprotected_control_updates=proof['unprotected']['optimizer_updates'])


def _branch_effective(run,inventory,step):
    effective = read(run/'gaussian-run/config.json')
    baseline = read(Path(inventory['profiles']['gaussian-run/config.json']['path']))
    unchanged = set(baseline)-{'iterations','optimization','snapshot_steps','frozen_surface_sha256'}
    need(all(effective[key] == baseline[key] for key in unchanged)
         and effective['iterations'] == effective['optimization']['iterations'] == step
         and effective['training_duration_policy'] == DURATION_POLICY
         and effective['frozen_surface_sha256'] == inventory['frozen_surface']['sha256']
         and read(run/'config.json')['frozen_surface'] == inventory['frozen_surface']
         and effective['optimization']['position_lr_max_steps'] == STEPS
         and {key:value for key,value in effective['optimization'].items() if key != 'iterations'}
             == {key:value for key,value in baseline['optimization'].items() if key != 'iterations'}
         and effective['opacity_reset_steps'] == baseline['opacity_reset_steps'],
         'The continuous duration comparison changed the 18k LR or existing reset and density schedules.')
    return effective


def _raw_duration_state(run,endpoint,step,plan,inventory,optimization,bins):
    import torch

    raw = run/'gaussian-run'/f'raw-step-{step:06d}'
    need(endpoint['step'] == step and endpoint['raw'] == str(raw.relative_to(run)),
         'A duration branch does not bind its raw endpoint directory.')
    state_json = raw/'state.json'
    pin(state_json,endpoint['raw_receipt_sha256'])
    meta = read(state_json)
    state_path = raw/'state.pt'
    need(meta['state_sha256'] == endpoint['raw_state_sha256'],'Raw optimizer state hashes disagree.')
    pin(state_path,meta['state_sha256'])
    archive = torch.load(state_path,map_location='cpu',weights_only=True)
    need(set(archive) == {'training','state','sampled_rows','optimization'} and archive['optimization'] == optimization
         and archive['training'] == {key:value for key,value in meta.items() if key not in ('files_sha256','state_sha256')},
         'The raw checkpoint scheduler or metadata differs from the actual saved state.')
    state = archive['state']
    need(meta['schema'] == 'shuolee-duration-endpoint-v1' and state['schema'] == 'shuolee-duration-checkpoint/1'
         and meta['step'] == meta['main_optimizer_updates'] == meta['scheduler_step'] == step
         and meta['schedule_steps'] == STEPS and meta['seed'] == 0
         and meta['reference_frame_id'] == inventory['reference_frame']
         and meta['source_sampler'] == 'seeded_equal_occupied_source_bin_cycles'
         and meta['source_draw_encoding'] == 'one unsigned byte per selected row index',
         'The saved state is not the original-only optimizer and scheduler at the exact requested update.')
    indices = archive['sampled_rows']
    need(indices.dtype == torch.uint8 and tuple(indices.shape) == (step,) and int(indices.max()) < len(bins),
         'The real sampled-row trace is missing or contains a non-original view.')
    draws = bytes(indices.tolist())
    per_frame = Counter(draws)
    per_bin = Counter(str(bins[index]) for index in draws)
    cpu = plan['cpu_fit_preflight']
    expected = cpu if step == MIDPOINT else cpu['continuous_18000']
    digest = hashlib.sha256(draws).hexdigest()
    prefix = hashlib.sha256(draws[:MIDPOINT]).hexdigest()
    need(meta['source_draw_sha256'] == digest == expected['expected_sampling_order_sha256']
         and meta['source_draw_prefix_10000_sha256'] == prefix == cpu['expected_sampling_order_sha256']
         and meta['per_frame_updates'] == [per_frame[index] for index in range(len(bins))]
         == expected['expected_per_frame_updates'] and meta['source_bin_draws'] == dict(per_bin)
         == expected['expected_source_bin_draws'],
         'Actual frame draws are not the preregistered uninterrupted original 10k/18k sample order.')

    model,parameters,optimizer = (state[key] for key in ('model','parameters','optimizer'))
    groups = {'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest','opacity':'_opacity',
              'scaling':'_scaling','rotation':'_rotation'}
    need(set(optimizer['groups']) == set(optimizer['states']) == set(groups)
         and set(parameters) == set(groups.values()) | {'static_offset'} and state['fixed']['num_timesteps'] == 24
         and state['fixed']['max_sh_degree'] == 0 and state['color_projection']['optimizer_steps']['value'] == step,
         'The raw endpoint did not preserve named Adam groups, SH0 or color-guard continuity.')
    binding = model['binding']['tensor'].to(torch.int64)
    owners = model['binding_counter']['tensor'].to(torch.int64)
    stable = model['densification_ids']['tensor'].to(torch.int64)
    points = meta['points']
    need(len(binding) == len(stable) == points and int(binding.min()) >= 0 and int(binding.max()) < len(owners)
         and bool((stable[1:] > stable[:-1]).all()) and bool((owners > 0).all())
         and torch.equal(torch.bincount(binding,minlength=len(owners)),owners),
         'The saved raw optimizer rows no longer match the one-or-more-per-triangle binding.')
    for group,name in groups.items():
        value = parameters[name]['value']['tensor']
        need(len(value) == points and bool(torch.isfinite(value).all()),
             f'The saved parameter {name} is nonfinite or its rows do not match the raw state.')
        saved = optimizer['states'][group]
        adam_steps = int(saved['step']['tensor']) if saved and 'step' in saved else 0
        lr = optimizer['groups'][group]['lr']['value']
        need(adam_steps == meta['optimizer_group_steps'][group] and lr == meta['optimizer_group_lrs'][group]
             and math.isfinite(lr) and lr >= 0,
             f'The raw {group} Adam clock or learning rate differs from its serialized state.')
        if saved is not None:
            for key in ('exp_avg','exp_avg_sq'):
                moment = saved[key]['tensor']
                need(moment.shape == value.shape and bool(torch.isfinite(moment).all()),
                     f'The raw {group} {key} is nonfinite or misaligned.')
    need(all(meta['optimizer_group_steps'][name] == step for name in ('xyz','f_dc','opacity','scaling','rotation'))
         and parameters['static_offset']['requires_grad'] is False,
         'At least one active Adam group restarted, or the frozen reference mesh was optimized during the raw prefix.')
    scale = model['spatial_lr_scale']['value']
    t = step/STEPS
    expected_lr = math.exp((1-t)*math.log(optimization['position_lr_init']*scale)
                           +t*math.log(optimization['position_lr_final']*scale))
    need(math.isclose(meta['optimizer_group_lrs']['xyz'],expected_lr,rel_tol=1e-12),
         'The raw XYZ learning rate does not follow the single registered 18k schedule.')
    offset = parameters['static_offset']['value']['tensor']
    need(tuple(offset.shape) == (10475,3) and bool(torch.isfinite(offset).all()),
         'The raw checkpoint lacks its frozen shared clothing offset.')
    fixed = dict(binding_sha256=hashlib.sha256(binding.numpy().tobytes()).hexdigest(),
                 stable_id_sha256=hashlib.sha256(stable.numpy().tobytes()).hexdigest(),
                 frozen_offset_sha256=hashlib.sha256(offset.numpy().tobytes()).hexdigest())
    need(set(meta['files_sha256']) == {'world.ply','reference-mesh.npz','native-ply-parity.json',
                                      'reference-render.png','attachment.json'},
         'Raw endpoint manifest lacks the original world geometry or parity evidence.')
    for name,file_digest in meta['files_sha256'].items():
        pin(raw/name,file_digest)
    parity = previous._snapshot_parity(raw,inventory['reference_frame'],run)
    portable_path = raw/'portable-result.json'
    need(endpoint['raw_portable_result'] == str(portable_path.relative_to(run)),
         'The raw portable export did not originate from the same-duration state.')
    pin(portable_path,endpoint['raw_portable_result_sha256'])
    portable = read(portable_path)
    need(portable['state'] == 'completed' and portable['step'] == step and portable['format_passed'] is True
         and portable['points'] == points and portable['source_world_sha256'] == meta['files_sha256']['world.ply'],
         'The portable raw duration snapshot is not the saved unmodified endpoint.')
    for name,digest_key in (('ply','sha256'),('camera','camera_sha256'),('scene','scene_sha256')):
        path = (raw/portable[name]).resolve(strict=True)
        need(path.is_relative_to(raw),'A raw portable artifact escapes its endpoint.')
        pin(path,portable[digest_key])
    pin(run/endpoint['result'],endpoint['result_sha256'])
    return dict(raw=str(raw.relative_to(run)),state_sha256=meta['state_sha256'],points=points,step=step,
        files_sha256=meta['files_sha256'],scheduler_step=meta['scheduler_step'],schedule_steps=meta['schedule_steps'],
        optimizer_group_steps=meta['optimizer_group_steps'],optimizer_group_lrs=meta['optimizer_group_lrs'],
        source_draw_sha256=meta['source_draw_sha256'],source_draw_prefix_10000_sha256=prefix,
        postwarmup_source_draws=[draws[2400:].count(index) for index in range(len(bins))],
        raw_portable_result_sha256=endpoint['raw_portable_result_sha256'],native_ply_parity=parity,
        main_seconds=meta['main_seconds'],**fixed),draws


def verify_continuity(run,plan,inventory):
    duration = run/'gaussian-run/duration-comparison.json'
    report = read(duration)
    result = read(run/'result.json')['duration_comparison']
    need(report['schema'] == 'shuolee-duration-comparison-v1'
         and all(report[key] == value for key,value in DURATION_POLICY.items())
         and all(result[key] == value for key,value in report.items())
         and result['file'] == str(duration.relative_to(run)) and result['sha256'] == sha(duration)
         and result['secondary']['result'] == 'duration-10000/result.json' and result['secondary']['steps'] == MIDPOINT,
         'The normal entry did not bind the two independent exports to a single training schedule.')
    pin(run/result['secondary']['result'],result['secondary']['result_sha256'])
    effective = _branch_effective(run,inventory,STEPS)
    endpoints = [_raw_duration_state(run,item,step,plan,inventory,effective['optimization'],
                 effective['training_source_bins']) for step,item in zip((MIDPOINT,STEPS),report['endpoints'],strict=True)]
    (first,first_draws),(last,last_draws) = endpoints
    unchanged = ('points','binding_sha256','stable_id_sha256','frozen_offset_sha256')
    need(first_draws == last_draws[:MIDPOINT] and first['source_draw_sha256'] == report['source_draw_prefix_10000_sha256']
         == last['source_draw_prefix_10000_sha256'] and all(first[key] == last[key] for key in unchanged)
         and 0 < last['optimizer_group_lrs']['xyz'] < first['optimizer_group_lrs']['xyz']
         and first['main_seconds'] < last['main_seconds'],
         'The second duration changed the frozen main topology, sampler prefix, or uninterrupted LR/Adam state.')
    loss = run/'gaussian-run/loss.jsonl'
    need(report['loss_log'] == str(loss.relative_to(run)),'Unexpected actual main training loss trace.')
    pin(loss,report['loss_log_sha256'])
    records = loss.read_text().splitlines(keepends=True)
    samples = [json.loads(line) for line in records]
    expected_steps = [1,*range(100,STEPS+1,100)]
    branch_loss = run/'duration-10000/gaussian-run/loss.jsonl'
    need([item['step'] for item in samples] == expected_steps
         and all(item['target_steps'] == STEPS and item['stage'] == 'training' for item in samples)
         and all(math.isfinite(item['loss']) for item in samples)
         and branch_loss.read_text() == ''.join(records[:1+MIDPOINT//100]),
         'The 10k and 18k loss logs are not from the same uninterrupted main process.')
    need((run/last['raw']/'state.json').stat().st_mtime_ns <= (run/'duration-10000/gaussian-run/integration.json').stat().st_mtime_ns,
         'Postprocessing of the midpoint began before the uninterrupted 18k state was saved.')
    return dict(file=str(duration.relative_to(run)),sha256=sha(duration),loss_log=report['loss_log'],
                loss_log_sha256=report['loss_log_sha256'],raw_endpoints=[first,last],single_main_training=True,
                endpoint_postprocess_started_after_18000=True,raw_10000_is_prefix=True)


def _clothing_trial_selection(audit,frames):
    policy = audit['policy']
    trials = audit.get('optimizer_trials',[])
    state = audit['state']
    accepted = state == 'accepted_visible_surface'
    need(audit['schema'] == 'shuolee-shared-clothing-guide-v1' and audit['posture_and_identity_frozen'] is True
         and audit['protected_vertices_bit_exact'] is True and policy['cycles'] == 12
         and policy['source_bins_min'] == 2 and policy['max_change_m'] == .018
         and state in ('accepted_visible_surface','rejected_keep_original_surface','insufficient_rgb_or_multiview_support'),
         'Shared clothing evidence does not preserve the fixed body and bounded multiview policy.')
    support = audit['support']
    enough = (support['supported_faces'] > 0 and support['native_rgb_tracked_observations'] >= policy['minimum_tracks']
              and support['rgb_tracked_views'] >= policy['minimum_track_views'])
    if not enough:
        need(state == 'insufficient_rgb_or_multiview_support' and not trials
             and audit['adam_updates'] == audit['selected_cycle'] == audit['maximum_change_m'] == 0,
             'Insufficient RGB or multiview support did not preserve the baseline surface.')
        return False
    need(len(trials) == policy['cycles'] and audit['adam_updates'] == len(frames)*len(trials)
         and [item['cycle'] for item in trials] == list(range(1,1+len(trials)))
         and [row['frame'] for row in audit['before']] == [row['frame'] for row in audit['after']] == frames,
         'The shared clothing optimizer did not assess all original views on each registered cycle.')
    for row in trials:
        mesh = row['mesh']
        values = [row[key] for key in ('edge_gain_px','max_edge_regression_px',
                  'track_mean_regression_px','max_track_regression_px')]
        need(all(map(math.isfinite,values)),'Clothing trial has nonfinite silhouette or tracked RGB evidence.')
        verdict = (mesh['finite'] and mesh['normal_dot_min'] >= policy['minimum_normal_dot']
            and mesh['area_ratio_min'] >= policy['minimum_area_ratio']
            and mesh['area_ratio_max'] <= policy['maximum_area_ratio']
            and values[0] >= policy['silhouette_gain_px'] and values[1] <= policy['silhouette_view_regression_px']
            and values[2] <= policy['track_mean_regression_px'] and values[3] <= policy['track_view_regression_px'])
        need(row['topology_line_fraction'] in policy['feasible_line_fractions'] and row['accepted'] is verdict,
             'The shared clothing candidate disagrees with its mesh, silhouette or RGB gates.')
    best = max((row for row in trials if row['accepted']),key=lambda item:item['edge_gain_px'],default=None)
    need(accepted is (best is not None) and audit['selected_cycle'] == (best['cycle'] if best else 0)
         and audit['selected_topology_line_fraction'] == (best['topology_line_fraction'] if best else 0.)
         and (accepted or audit['after'] == audit['before']),
         'Selected clothing geometry is not the best measured feasible candidate.')
    if accepted:
        comparisons = zip(audit['before'],audit['after'],strict=True)
        edge_changes,track_changes = [],[]
        for before,after in comparisons:
            if before['boundary_faces']:
                edge_changes.append(after['edge_error_native_px_equivalent']-before['edge_error_native_px_equivalent'])
            if before['tracked_observations']:
                track_changes.append(after['rgb_track_robust_mean_px']-before['rgb_track_robust_mean_px'])
        need(edge_changes and track_changes and all(math.isclose(best[key],value,abs_tol=1e-8)
             for key,value in (('edge_gain_px',-sum(edge_changes)/len(edge_changes)),
                 ('max_edge_regression_px',max(edge_changes)),('track_mean_regression_px',sum(track_changes)/len(track_changes)),
                 ('max_track_regression_px',max(track_changes)))),
             'Accepted clothing gains do not reproduce from the actual native source reports.')
    return accepted


def _clothing_offset_identity(run,raw,terminal,audit,archive=None):
    import numpy as np

    config = read(run/'config.json')
    frozen = config['frozen_surface']
    source = run/frozen['file']
    pin(source,frozen['sha256'])
    with np.load(source,allow_pickle=False) as f, np.load(raw/'reference-mesh.npz',allow_pickle=False) as r, \
            np.load(terminal/'reference-mesh.npz',allow_pickle=False) as t:
        base,offset,mesh = f['static_offset'],r['static_offset'],r['vertices_world']
        faces = r['faces']
        need(r.files == t.files and all(np.array_equal(r[key],t[key]) for key in r.files)
             and np.array_equal(faces,f['faces']) and np.isfinite(mesh).all(),
             'The postprocess changed its frozen endpoint mesh or topology.')
    need(offset.dtype == base.dtype == np.float32 and offset.shape == base.shape == (10475,3),
         'Shared clothing offset is not one explicit fixed rest-space vertex field.')
    assets = Path(read(run/'run.json')['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    with np.load(assets,allow_pickle=False) as data:
        dominant = data['weights'].argmax(-1)
        protected = np.isin(dominant,[12,15,22,23,24])
        landmarks = np.concatenate([data['lmk_faces_idx'].reshape(-1),data['dynamic_lmk_faces_idx'].reshape(-1)])
        need(np.array_equal(faces,data['f']),'Clothing endpoint has different official SMPL-X topology.')
    protected[faces[landmarks].reshape(-1)] = True
    clothing = np.isin(dominant,[0,1,2,3,4,5,6,9,13,14,16,17,18,19]) & ~protected
    delta = np.linalg.norm(offset-base,axis=1)
    changed = delta > 0
    accepted = audit['state'] == 'accepted_visible_surface'
    need(np.isfinite(delta).all() and not np.any(changed & ~clothing)
         and int(changed.sum()) == audit.get('changed_vertices',0)
         and math.isclose(float(delta.max()),audit['maximum_change_m'],abs_tol=2e-8)
         and (np.any(changed) == accepted) and float(delta.max()) <= audit['policy']['max_change_m']*
             audit.get('selected_topology_line_fraction',0.)+2e-8,
         'The actual shared offset changed protected geometry or exceeded its measured displacement.')
    if archive is not None:
        with np.load(archive,allow_pickle=False) as data:
            # The stage should provide one selected face mask, shared by the 24 poses.
            support = data['supported_faces']
            need(support.dtype == np.bool_ and support.shape == (len(faces),)
                 and not np.any(support & ~clothing[faces].all(1))
                 and int(support.sum()) == (audit['support']['supported_faces'] if accepted else 0)
                 and hashlib.sha256(np.flatnonzero(support).astype(np.int64).tobytes()).hexdigest()
                     == audit['supported_face_ids_sha256'],
                 'A nonclothing triangle received shared garment support.')
            if 'original_offset' in data.files:
                need(np.array_equal(data['original_offset'],base),'Clothing offset archive changed the baseline field.')
            need(np.array_equal(data['final_offset'],offset),'Clothing offset archive differs from the raw model.')
            if 'eligible_vertices' in data.files:
                eligible = data['eligible_vertices']
                need(eligible.dtype == np.bool_ and eligible.shape == changed.shape
                     and np.array_equal(eligible,clothing) and not np.any(changed & ~eligible),
                     'A changed vertex is outside the official garment-capable region.')
    return dict(changed_vertices=int(changed.sum()),maximum_change_m=float(delta.max()),
                protected_vertices_bit_exact=True,offset_sha256=hashlib.sha256(offset.tobytes()).hexdigest())


def _geometry_measurements(rows,scope):
    """Verify report units and quantiles while preserving unsupported material."""
    for row in rows:
        points = row['supported_rows'] if scope == 'skin' else row['active_rows']
        flattened = row.get('flattened_rows',0)
        need(type(points) is int and type(flattened) is int and 0 <= flattened <= points,
             f'The {scope} covariance measurement has an invalid active point count.')
        for name in ('abs_normal_center_mm','signed_normal_center_mm','normal_sigma_mm','flattened_thin_axis_angle_deg'):
            data = row.get(name)
            expected = flattened > 0 if name == 'flattened_thin_axis_angle_deg' else points > 0
            need((data is not None) is expected,f'The {scope} {name} measurement substitutes unsupported geometry.')
            if data is not None:
                need(set(data) == {'p50','p95'} and all(map(math.isfinite,data.values()))
                     and data['p50'] <= data['p95']
                     and (data['p50'] >= 0 or name == 'signed_normal_center_mm')
                     and (data['p95'] <= 90.001 or name != 'flattened_thin_axis_angle_deg'),
                     f'The {scope} {name} values do not represent native-scale finite covariance evidence.')


def verify_surface_attachment(run,plan,inventory,integration,trained,effective,step,raw,raw_info):
    out = run/'gaussian-run'
    terminal = out/f'step-{step:06d}'
    descriptor = integration['surface_attachment']
    need(descriptor['file'] == 'surface-attachment.json','Unexpected surface-attachment artifact.')
    path = out/descriptor['file']
    pin(path,descriptor['sha256'])
    report = read(path)
    raw_path = raw/'attachment.json'
    pin(raw_path,raw_info['files_sha256']['attachment.json'])
    initial_path = out/'surface-attachment-initial.json'
    evidence = read(raw_path)
    initial = read(initial_path)
    policy = effective['surface_attachment_policy']
    need(report['schema'] == 'shuolee-surface-attachment-v1' and report['step'] == step
         and report['policy'] == policy and policy['face_weight'] == .02
         and policy['postprocess'] == 'protected DC and opacity only'
         and report['training'] == evidence and initial == evidence['initial']
         and initial['step'] == effective['unified_main']['warmup_steps'] == 2400
         and report['reference_mesh_sha256'] == sha(terminal/'reference-mesh.npz')
         and report['postprocessing_changes_geometry'] is False and 'head_support' not in integration,
         'The actual result lacks independent during-training surface attachment and appearance-only postprocessing.')
    frames = inventory['selected_frames']
    rows = read(run/'data/training-inputs.json')['records']
    proof = evidence['support']
    config = proof['config']
    views = proof['views']
    need(proof['schema'] == 'shuolee-visible-skin-surface-attachment/1'
         and [row['frame'] for row in views] == frames
         and [row['angular_bin'] for row in views] == effective['training_source_bins']
         and [row['loss_calls'] for row in views] == raw_info['postwarmup_source_draws']
         and sum(row['loss_calls'] for row in views) == step-2400
         and proof['supported_views'] == sum(row['repeated_skin_triangles'] > 0 for row in views)
         and config['min_independent_source_bins'] == 2,
         'Visible skin attachment did not use the exact original-view draw sequence after warmup.')
    for source,row in zip(rows,views,strict=True):
        need(0 <= row['supported_calls'] <= row['loss_calls']
             and 0 <= row['flattened_row_draws'] <= row['supported_row_draws']
             and 0 <= row['repeated_skin_triangles'] <= row['visible_skin_triangles']
             and row['repeated_skin_triangles'] <= proof['repeated_skin_triangles']
             and (not row['supported_calls'] or (row['status'] == 'supported' and row['repeated_skin_triangles'] > 0)),
             f'The skin attachment has unsupported optimization work for {row["frame"]}.')
        if row['status'] == 'supported':
            x0,y0,x1,y1 = row['box_xyxy']
            w,h = source['wh']
            need(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
                 and config['min_material_pixels'] <= row['skin_pixels'] <= (x1-x0)*(y1-y0)
                 and re.fullmatch('[0-9a-f]{64}',row['native_material_mask_sha256']) is not None,
                 'The native skin support mask has no verifiable native crop and identity.')
        need(all(math.isfinite(row[key]) and row[key] >= 0 for key in ('mean_loss_on_supported_calls',
                  'mean_center_penalty','mean_thickness_penalty','mean_alignment_penalty')),
             'The skin attachment has a nonfinite or invalid active penalty.')
    for entry in (initial,evidence,report):
        skin = entry['skin']['views']
        need([row['frame'] for row in skin] == frames,'Native skin geometry measurements omit a protected source view.')
        _geometry_measurements(skin,'skin')
        _geometry_measurements([entry['clothing']],'clothing')

    garment = report['clothing_guide']
    need(garment['file'] == 'clothing-surface.json','Unexpected clothing guide artifact.')
    cloth_path = out/garment['file']
    pin(cloth_path,garment['sha256'])
    audit = read(cloth_path)
    inputs = audit['inputs']
    frozen = read(run/'config.json')['frozen_surface']
    need(inputs['frames'] == frames and inputs['image_sha256'] == [row['source_sha256'] for row in rows]
         and inputs['mask_sha256'] == [row['mask_sha256'] for row in rows]
         and audit['policy'] == policy['clothing_guide'] and audit['changed_pose_parameters'] == 0
         and audit['original_mesh_sha256'] == frozen['sha256']
         and audit['support_file'] == 'shared-clothing-support.npz',
         'Garment geometry did not consume the frozen source RGB/masks, shared shape or bounded policy.')
    accepted = _clothing_trial_selection(audit,frames)
    offset_archive = out/audit['support_file']
    pin(offset_archive,audit['support_file_sha256'])
    if accepted:
        active = out/'shared-clothing-offset.npz'
        need(audit['active_clothing_surface'] == active.name,'The kept cloth geometry has no active mesh.')
        pin(active,audit['active_mesh_sha256'])
        import numpy as np
        with np.load(active,allow_pickle=False) as a, np.load(raw/'reference-mesh.npz',allow_pickle=False) as b:
            need(a.files == b.files and all(np.array_equal(a[key],b[key]) for key in a.files),
                 'The kept shared guide differs from the trained endpoint mesh.')
    else:
        need(audit['active_clothing_surface'] is None and audit['active_mesh_sha256'] == frozen['sha256'],
             'A rejected shared cloth candidate was activated.')
    identity = _clothing_offset_identity(run,raw,terminal,audit,offset_archive)
    need(identity['offset_sha256'] == raw_info['frozen_offset_sha256'],
         'The actual serialized mesh did not use the raw optimizer’s frozen clothing guide.')
    return dict(file=str(path.relative_to(run)),sha256=descriptor['sha256'],clothing_guide_sha256=garment['sha256'],
        clothing_guide_state=audit['state'],clothing_support_sha256=audit['support_file_sha256'],
        clothing_adam_updates=audit['adam_updates'],clothing_selected_cycle=audit['selected_cycle'],
        clothing_shared_offset=identity,skin_supported_source_views=proof['supported_views'],
        skin_supported_calls=sum(row['supported_calls'] for row in views),postprocessing_changes_geometry=False)


def verify_branch(run,plan,inventory,step,raw,raw_info):
    """Verify one branch against its uninterrupted raw endpoint, independently of visual quality."""
    effective = _branch_effective(run,inventory,step)
    result = read(run/'result.json')
    trained = read(run/'gaussian-run/result.json')
    integration = read(run/'gaussian-run/integration.json')
    old_train = read(Path(inventory['profiles']['gaussian-run/result.json']['path']))
    old_integration = read(Path(inventory['profiles']['gaussian-run/integration.json']['path']))
    sampler = plan['cpu_fit_preflight']
    if step == STEPS:
        sampler = sampler['continuous_18000']
    fixed = ('policy','initial_points','local_parameter_sha256','optimizer_initial_state_entries',
             'no_prior_gaussian_loaded','schedule','stock_densifications',
             'raw_native_rgb_target','prior_render_used_as_target')
    need(all(integration[key] == old_integration[key] for key in fixed)
         and integration['optimizer_updates'] == step and trained['pose_layout'] == old_train['pose_layout']
         and integration['frozen_surface_sha256'] == inventory['frozen_surface']['sha256']
         and trained['per_frame_updates'] == sampler['expected_per_frame_updates']
         and trained['source_bin_draws'] == sampler['expected_source_bin_draws']
         and result['unified_main_training'] == integration,
         'The branch differs from its verified original-only sample sequence or optimizer duration.')
    snapshot_name = f'step-{step:06d}'
    need(result['state'] == 'completed' and result['format_passed'] and result['gaussian_from_scratch']
         and result['fit_reused'] and result['fit_calls'] == 0 and result['sh_degree'] == 0 and result['steps'] == step
         and result['reference_frame_id'] == inventory['reference_frame'] and result['final_snapshot'] == snapshot_name
         and result['frames'] == len(inventory['selected_frames']),
         'The branch did not produce a complete fitted-sequence export.')
    for key,digest in (('ply','sha256'),('scene','scene_sha256'),('bound_replay','bound_replay_sha256'),
                       ('bound_panels','bound_panels_sha256')):
        target = (run/result[key]).resolve(strict=True)
        need(target.is_relative_to(run),'Export escaped its registered run root.')
        pin(target,result[digest])
    with zipfile.ZipFile(run/result['scene']) as bundle:
        with bundle.open('cameras.json') as cameras, bundle.open(f'point_cloud/iteration_{step}/point_cloud.ply') as ply:
            need(cold.stream_sha(cameras) == sha(run/result['camera']) and cold.stream_sha(ply) == result['sha256'],
                 'The scene package differs from the exported PLY and camera.')
    replay,panels = read(run/result['bound_replay']),read(run/result['bound_panels'])
    need([row['frame'] for row in replay['records']] == [row['frame'] for row in panels['panels']]
         == inventory['selected_frames'] and replay['retained_holdouts'] == inventory['retained_holdouts'],
         'Bound replay changed the frozen original pose roster.')

    bound = run/'gaussian-run'/snapshot_name/'bound-local.ply'
    color_path = run/'gaussian-run/color-constraint.json'
    color_audit = read(color_path)
    color_desc = integration[previous.POLICY_KEY]
    pin(color_path,color_desc['sha256'])
    need(color_desc['file'] == color_path.name and color_desc['terminal'] == color_audit['terminal']
         and color_audit['policy'] == previous.POLICY and color_audit['optimizer_steps'] == step
         and abs(color_audit['color_basis_c0']-previous.COLOR_C0) < 1e-12,
         'The original SH0 LDR parameter guard did not cover the actual training.')
    for phase,count in (('initial',20908),('terminal',result['points'])):
        stats = color_audit[phase]
        need(stats['points'] == count and stats['finite'] and stats['channels_outside_bounds'] == 0
             and -.0000001 <= stats['rgb_min'] <= stats['rgb_max'] <= 1.0000001,
             f'The {phase} SH0 coefficients exceed the LDR bounds.')
    late = previous.check_snapshot(run,step)
    need(late['stats'] == color_audit['terminal'],'The terminal snapshot has different color bounds.')
    parity = previous.verify_parity(run,step,inventory['reference_frame'])
    stats = dict(bound_local=previous.color_stats(bound),standard_world=previous.color_stats(run/result['ply']))
    need(all(row['points'] == result['points'] for row in stats.values()),'Terminal point counts disagree.')
    color = dict(file=str(color_path.relative_to(run)),sha256=sha(color_path),policy=previous.POLICY,
                 ply_stats=stats,snapshots=[late])
    binding = previous.check_binding(bound,run/'fit/fitted.npz',result['points'])
    world = integration['world_scale']
    pin(run/'gaussian-run'/world['file'],world['sha256'])
    measure = world['terminal']['all_poses_max']['all']
    need(world['terminal']['step'] == step and measure['points'] == result['points'] and measure['over_limit'] == 0,
         'The actual terminal topology exceeds the all-pose world covariance limits.')
    protection = verify_protected_stage(run,integration,trained,effective,inventory,
                                        plan['cpu_fit_preflight']['chosen_face_frames'],step)
    original_steps = previous.TRAINING_STEPS
    previous.TRAINING_STEPS = step
    try:
        face = previous.verify_face_supplement(run,integration,inventory,bound,plan['cpu_fit_preflight'],protection)
    finally:
        previous.TRAINING_STEPS = original_steps
    heldout = previous.verify_heldout_replay(run,result,plan,inventory,bound)
    for name,digest in raw_info['files_sha256'].items():
        if name != 'attachment.json':
            pin(run/'gaussian-run'/trained['main_snapshot']/name,digest)
    surface = verify_surface_attachment(run,plan,inventory,integration,trained,effective,step,raw,raw_info)
    return dict(root=str(run),step=step,result_sha256=sha(run/'result.json'),points=result['points'],
                world_scale_sha256=world['sha256'],color_constraint=color,binding=binding,native_ply_parity=parity,
                face_supplement=face,heldout_replay=heldout,protected_face=protection,surface_attachment=surface)


def verify_outputs(run,plan,inventory):
    for path,digest in inventory['files'].items():
        _pin_input(Path(path),digest)
    for path,digest in plan['code_sha256'].items():
        pin(PROJECT/path,digest)
    head = verify_frozen_fit(run,inventory)
    pin(run/'data/head-support-views.json',plan['pitch_camera_manifest']['sha256'])
    for name in ('gaussian-run/training-cameras.json','gaussian-run/initialization.json'):
        pin(run/name,inventory['profiles'][name]['sha256'])
    status = read(run/'status.json')
    need(status['state'] == 'completed' and status['static_export_passed'],'The continuous paired run did not finish.')
    continuity = verify_continuity(run,plan,inventory)
    raw = {step:run/'gaussian-run'/f'raw-step-{step:06d}' for step in (MIDPOINT,STEPS)}
    raw_midpoint,raw_terminal = continuity['raw_endpoints']
    intermediate = verify_branch(run/'duration-10000',plan,inventory,MIDPOINT,raw[MIDPOINT],raw_midpoint)
    final = verify_branch(run,plan,inventory,STEPS,raw[STEPS],raw_terminal)
    need(all(intermediate['surface_attachment'][key] == final['surface_attachment'][key] for key in (
         'clothing_guide_sha256','clothing_support_sha256','clothing_shared_offset')),
         'The two durations used different garment guidance or shared vertex offsets.')
    integration = read(run/'gaussian-run/integration.json')
    effective = _branch_effective(run,inventory,STEPS)
    legacy = previous.prior.prior.prior
    original_steps = legacy.TRAINING_STEPS
    legacy.TRAINING_STEPS = STEPS
    try:
        densification = legacy.verify_densification(run,integration,effective,inventory['selected_frames'],final['points'])
    finally:
        legacy.TRAINING_STEPS = original_steps
    for payload in (read(run/densification['file']),read(run/'gaussian-run/world-scale.json'),integration['coverage_events']):
        need(not previous._has_main_auxiliary(payload),'Auxiliary frames influenced original main topology decisions.')
    early = previous.check_snapshot(run,1000)
    early_parity = previous.verify_parity(run,1000,inventory['reference_frame'])
    selection = read(run/'fit/selection.json')
    return dict(final,state='completed',reference_frame=inventory['reference_frame'],selected_frames=selection['selected_frames'],
        retained_holdouts=selection['retained_holdouts'],coverage=selection['coverage'],
        fit_method='reused_exact_round20_corrected_fit',fit_sha256=head['fit_sha256'],head_fit=head,
        training_inputs_sha256=inventory['staged_files']['data/training-inputs.json']['sha256'],
        fresh_gaussian=True,original_video_to_export=False,native_inputs_reused=True,
        same_round20_fit_and_training_selection=True,verified_pose_count=24+len(final['face_supplement']['frames']),
        duration_comparison=dict(continuity=continuity,intermediate=intermediate,terminal=final),
        densification_audit=densification,early_color=early,
        native_ply_parity=[early_parity,intermediate['native_ply_parity'],final['native_ply_parity']])


def finish(root,plan,inventory,returncode,elapsed):
    run = root/'run'
    receipt = dict(state='failed' if returncode else 'incomplete',subject=plan['subject'],returncode=returncode,
        stage='train_fitted',candidate_commit=plan['candidate_commit'],baseline_commit=PARENT,
        quality_baseline_commit=PARENT,quality_accepted=False,command_to_end_seconds=elapsed,
        preregistration_sha256=sha(root/'preregistration.json'),launch_sha256=sha(root/'launch.json'),
        protected_input_inventory_sha256=plan['protected_input_inventory_sha256'],entry_log_sha256=sha(root/'entry.log'))
    if returncode == 0:
        try:
            receipt.update(verify_outputs(run,plan,inventory))
        except Exception as error:
            receipt['validation_error'] = f'{type(error).__name__}: {error}'
    if receipt['state'] != 'completed':
        out = run/'gaussian-run'
        receipt['preserved_snapshots'] = [previous.snapshot_failure(path,run) for path in sorted(
            out.glob('step-*/native-ply-parity.json'))]
        receipt['preserved_duration_evidence'] = [dict(file=str(path.relative_to(run)),sha256=sha(path))
            for path in sorted(out.glob('raw-step-*/state.json'))]
        receipt['retry_performed'] = False
    write(root/'wrapper-terminal.json',receipt)
    print(dict(state=receipt['state'],subject=plan['subject'],validation_error=receipt.get('validation_error')),flush=True)
    return 0 if receipt['state']=='completed' else returncode or 1


def verify_completed_policy(subject,commit,cuda_proof_sha,views,views_sha):
    root = root_for(subject)
    plan,inventory = registered(subject,commit,cuda_proof_sha,views,views_sha)
    original_path = root/'wrapper-terminal.json'
    original = read(original_path)
    expected = f'ValueError: Unexpected SHA-256 for {POLICY_PATH}.'
    need(original['state'] == 'incomplete' and original['returncode'] == 0
         and original.get('validation_error') == expected
         and read(root/'run/status.json')['state'] == 'completed',
         'Policy replay may verify only a complete production run with the exact archived-path failure.')
    pin(root/'entry.log',original['entry_log_sha256'])
    receipt = {key:value for key,value in original.items() if key != 'validation_error'}
    receipt.update(verify_outputs(root/'run',plan,inventory))
    receipt['verification_amendment'] = dict(reason='Live steering changed after registration. Reused exact registered policy bytes.',
        original_terminal_sha256=sha(original_path),original_validation_error=expected,
        registered_wrapper_sha256=plan['wrapper_sha256'],verification_wrapper_sha256=sha(Path(__file__)),
        registered_policy_sha256=POLICY_SHA,frozen_policy=str(_input_bytes_path(POLICY_PATH,POLICY_SHA)),
        training_repeated=False)
    path = root/'wrapper-terminal-frozen-policy-verified.json'
    write(path,receipt)
    print(dict(state='completed',subject=subject,read_only_verification=True,file=str(path),sha256=sha(path)),flush=True)


def _completion_receipt(root):
    original = root/'wrapper-terminal.json'
    receipt = read(original)
    if receipt['state'] == 'completed':
        return receipt
    path = root/'wrapper-terminal-frozen-policy-verified.json'
    checked = read(path)
    amendment = checked['verification_amendment']
    pin(original,amendment['original_terminal_sha256'])
    need(checked['state'] == 'completed' and amendment['training_repeated'] is False,
         'Independent terminal verification did not complete.')
    return checked


def launch(subject,commit,cuda_proof_sha,views,views_sha):
    root = root_for(subject)
    plan,inventory = registered(subject,commit,cuda_proof_sha,views,views_sha)
    with cold.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subject == '006':
            first = root_for('016')
            receipt = _completion_receipt(first)
            need(receipt['state'] == 'completed' and receipt['candidate_commit'] == commit,
                 'First complete and validate the registered 016 run.')
            pin(first/'run/result.json',receipt['result_sha256'])
        need(not any((root/name).exists() for name in ('launch.json','entry.log','wrapper-terminal.json'))
             and not any((root/'run'/name).exists() for name in ('status.json','gaussian-run','export','result.json')),
             'Formal R21 launch must use a fresh staged run.')
        write(root/'launch.json',dict(subject=subject,candidate_commit=commit,commands=plan['commands'],
            cwd=str(PROJECT/'src'),started=cold.now(),cuda_fixture_sha256=cuda_proof_sha,
            registered_wrapper_sha256=plan['wrapper_sha256'],execution_wrapper_sha256=sha(Path(__file__)),
            registered_policy_path=str(_input_bytes_path(POLICY_PATH,POLICY_SHA)),
            preregistration_sha256=sha(root/'preregistration.json')))
        start = time.monotonic()
        with (root/'entry.log').open('xb') as log:
            command, = plan['commands']
            returncode = subprocess.run(['/usr/bin/timeout','--kill-after=15s',f'{plan["max_wall_seconds"]}s',*command],
                env=cold.clean_env(),cwd=PROJECT/'src',stdout=log,stderr=subprocess.STDOUT).returncode
        return finish(root,plan,inventory,returncode,time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check-inputs','check','archive-incomplete-setup','register',
                                          'launch','check-early','verify-completed-policy'))
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('--commit')
    parser.add_argument('--cuda-proof-sha256')
    parser.add_argument('--views',type=Path)
    parser.add_argument('--views-sha256')
    args = parser.parse_args()
    if args.action == 'archive-incomplete-setup':
        archive_incomplete_setup(args.subject)
        return
    if args.action == 'check-inputs':
        inventory = source_inventory(args.subject)
        print(dict(state='source_preflight_passed',subject=args.subject,files=len(inventory['files']),
                   staged_files=len(inventory['staged_files']),fit_sha256=inventory['fit_sha256'],
                   body=len(inventory['selected_frames']),face=len(inventory['archived_face_supplement']['frames']),
                   heldout=len(inventory['retained_holdouts'])),flush=True)
        return
    if args.action == 'check-early':
        root = root_for(args.subject)
        plan = read(root/'preregistration.json')
        protected = root/'protected-inputs.json'
        pin(protected,plan['protected_input_inventory_sha256'])
        run = root/'run'
        print(dict(color=previous.check_snapshot(run,1000),native_ply_parity=previous.verify_parity(
                   run,1000,read(protected)['reference_frame'])),flush=True)
        return
    if not all((args.commit,args.cuda_proof_sha256,args.views,args.views_sha256)):
        parser.error('--commit, --cuda-proof-sha256, --views and --views-sha256 are required.')
    values = args.subject,args.commit,args.cuda_proof_sha256,args.views.resolve(strict=True),args.views_sha256
    if args.action == 'launch':
        raise SystemExit(launch(*values))
    if args.action == 'verify-completed-policy':
        verify_completed_policy(*values)
    elif args.action == 'register':
        register(*values)
    else:
        plan,inventory = plan_for(*values)
        print(dict(state='preflight_passed',subject=args.subject,files=len(inventory['files']),
                   candidate=plan['candidate_commit'],view_manifest=plan['pitch_camera_manifest']),flush=True)


if __name__ == '__main__':
    main()
