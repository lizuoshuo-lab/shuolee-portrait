"""Register and launch isolated 016 then 006 through the stock video entry."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys
import time
import zipfile


LAB = Path('/opt/gs-slam-lab')
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-unified-20261110')
ENV = LAB/'shuolee-portrait-clean-001/env'
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
SOURCE = Path('/mnt/d/00/工作/数据集')
OFFICIAL = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001'
                '/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official')
LEASE = Path('/tmp/shuolee-native122-gpu.lock')
PREVIOUS = {
    '016': dict(parent='shuolee-smpl-bound-016', video='video16.mp4',
        video_sha256='3b7fa0403e3e9d0d63e7a4f40bb9f07bbd9c1c05b1d75e483b97be0ee5398d69',
        camera_sha256='b27cc605d0466cbe2f2ee0c34ad1b0b9ecdfd4265aa263af85e4553b905c589b',
        source_holdouts=[80,216,312,409],
        named_holdouts=['frame_000011','frame_000028','frame_000040','frame_000052']),
    '006': dict(parent='shuolee-smpl-sixworker-opacity-006', video='video6.mp4',
        video_sha256='cb11ad325834b399c655371dc5d161e1801a3719b0892da959dc1e57e764572f',
        camera_sha256='c42ef6148efc4a5d4d35629e15dbada73c226b040fff5a87d1a4e968f1b9a4b7',
        source_holdouts=[56,168,288,392],
        named_holdouts=['frame_000008','frame_000022','frame_000037','frame_000050']),
}
SOURCE_FILES = (
    'run_smplx.py', 'scripts/train.sh', 'configs/server-assets.json',
    'src/shuolee_portrait/runtime.py', 'src/shuolee_portrait/cli.py',
    'src/shuolee_portrait/reconstruction.py', 'src/shuolee_portrait/sequence_pipeline.py',
    'src/shuolee_portrait/preprocessing.py', 'src/shuolee_portrait/observations.py',
    'src/shuolee_portrait/official_deployment.py', 'src/shuolee_portrait/official_smplifyx.py',
    'src/shuolee_portrait/official_sequence.py', 'src/shuolee_portrait/smplifyx_runtime.py',
    'src/shuolee_portrait/stable_selection.py', 'src/shuolee_portrait/training.py',
    'src/shuolee_portrait/export.py', 'src/shuolee_portrait/bound_replay.py',
    'src/shuolee_portrait/bound_review_panels.py', 'src/avatar_core/scene/gaussian_model.py',
    'src/avatar_core/scene/smplx_gaussian_model.py', 'src/avatar_core/gaussian_renderer/__init__.py',
)
BASE = '76ba8a0'
OBSERVATION_FILES = ('detection/yolov8n.pt','sam2/sam2.1_hiera_tiny.pt',
                     'rtmw/normalization.json','rtmw/rtmw-l-384x288.onnx')
MAX_WALL_S = 3600


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as file:
        return stream_sha(file)


def stream_sha(file):
    hash_value = hashlib.sha256()
    for chunk in iter(lambda:file.read(4*1024*1024),b''):
        hash_value.update(chunk)
    return hash_value.hexdigest()


def write(path, payload):
    with path.open('x', encoding='utf-8') as file:
        json.dump(payload,file,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        file.write('\n')


def pin(path, expected):
    need(sha(path)==expected,f'Unexpected SHA-256 for {path}.')


def now():
    return datetime.now(timezone.utc).isoformat()


def root_for(subject):
    return LAB/f'shuolee-smpl-unified-{subject}'


def clean_env():
    """Allowlist only runtime settings needed by this pinned environment."""
    return dict(PATH=f'{ENV}/bin:/usr/local/bin:/usr/bin:/bin',
                HOME=pwd.getpwuid(os.getuid()).pw_dir, LANG='C.UTF-8', LC_ALL='C.UTF-8',
                CUDA_VISIBLE_DEVICES='0', PYTHONNOUSERSITE='1',
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')


def git_state(commit):
    pointer=(PROJECT/'.git').read_text(encoding='utf-8').strip()
    need(pointer.startswith('gitdir: '),'Expected the detached unified Windows worktree.')
    git_dir=subprocess.check_output(['/usr/bin/wslpath','-u',pointer.removeprefix('gitdir: ')],text=True).strip()
    git=['/usr/bin/git','-c','core.fileMode=false','--git-dir',git_dir,'--work-tree',str(PROJECT)]
    head=subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()
    need(head==commit and re.fullmatch('[0-9a-f]{40}',head) is not None,
         f'Unified checkout is {head}; expected {commit}.')
    subprocess.run([*git,'merge-base','--is-ancestor',BASE,head],check=True)
    state=subprocess.check_output([*git,'status','--porcelain','--untracked-files=all'],
                                  cwd=PROJECT,text=True)
    need(not state.strip(),f'Unified checkout is dirty: {state}')
    return head


def cpu_runtime(env):
    check = (
        "import importlib.util,json,site,sys; "
        "from pathlib import Path; "
        "print(json.dumps(dict(prefix=sys.prefix,python=sys.version,isolated=sys.flags.isolated,"
        "user_site=site.ENABLE_USER_SITE,torch=importlib.util.find_spec('torch').origin,"
        "rasterizer=importlib.util.find_spec('diff_gaussian_rasterization').origin)))"
    )
    identity=json.loads(subprocess.check_output([str(ENV/'bin/python'),'-I','-B','-c',check],env=env,text=True))
    need(Path(identity['prefix']).resolve()==ENV.resolve() and identity['isolated']==1
         and identity['user_site'] is False,'Python did not use the isolated deployment profile.')
    pins={}
    for key in ('torch','rasterizer'):
        p=Path(identity[key]).resolve()
        need(p.is_relative_to(ENV.resolve()),f'{key} resolves outside the dedicated environment.')
        pins[str(p)]=sha(p)
    binary,=Path(identity['rasterizer']).parent.glob('_C*.so')
    pins[str(binary)]=sha(binary)
    interpreter=(ENV/'bin/python').resolve()
    pins[str(interpreter)]=sha(interpreter)
    pins[str(ENV/'pyvenv.cfg')]=sha(ENV/'pyvenv.cfg')
    for tool in ('ffmpeg','ffprobe'):
        path=Path(shutil.which(tool,path=env['PATH']) or '')
        need(path==Path('/usr/bin')/tool,f'Unexpected {tool} runtime location.')
        pins[str(path)]=sha(path)
    identity['file_sha256']=pins
    return identity


def official_pins():
    # Read the stock pinned dictionary without importing or configuring any model.
    import ast
    entry=PROJECT/'src/shuolee_portrait/official_smplifyx.py'
    tree=ast.parse(entry.read_text(encoding='utf-8-sig'))
    record,=(x for x in tree.body if isinstance(x,ast.Assign) and
             any(isinstance(n,ast.Name) and n.id=='PINNED' for n in x.targets))
    official={str(OFFICIAL/name):value for name,value in ast.literal_eval(record.value).items()}
    manifest=read(PROJECT/'configs/server-assets.json')['files']
    needed={item['path']:item for item in manifest if item['path'] in OBSERVATION_FILES
            or item['path'].startswith(('smplx/','vposer/'))}
    need(set(OBSERVATION_FILES)<=set(needed),'Missing required local assets in the pinned manifest.')
    for name,item in needed.items():
        path=ASSETS/name
        need(path.stat().st_size==item['bytes'],f'Asset size differs: {name}.')
        official[str(path)]=item['sha256']
    for p,h in official.items():
        pin(Path(p),h)
    return official


def deployment(root):
    return dict(schema='smplx-deployment-v1',profile='official_smplifyx',environment=str(ENV),
                assets=str(ASSETS),runs=str(root),official_smplifyx=dict(source_root=str(OFFICIAL)))


def command(subject):
    root=root_for(subject)
    return [str(ENV/'bin/python'),'-I','-B','-u',str(PROJECT/'run_smplx.py'),'train',
        str(SOURCE/PREVIOUS[subject]['video']),'--deployment',str(root/'deployment.json'),
        '--camera-config',str(root/'camera.json'),'--output',str(root/'run')]


def check(subject, commit):
    need(sys.platform=='linux' and Path(sys.prefix).resolve()==ENV.resolve(),
         'Run the wrapper using the pinned WSL deployment Python.')
    git_state(commit)
    p=PREVIOUS[subject]
    root=root_for(subject)
    video=SOURCE/p['video']
    camera=LAB/p['parent']/'camera.json'
    pin(video,p['video_sha256'])
    pin(camera,p['camera_sha256'])
    profile=read(camera)
    layout=profile['capture_layout']
    need(layout==dict(arc='half',video_sha256=p['video_sha256'],front_source_ranges=[[0,441]],
                      rear_source_ranges=[],held_source_indices=p['source_holdouts'])
         and profile['sampling']['sample_fps']==3
         and profile['intrinsics']['image_wh']==[2704,1520]
         and profile['intrinsics']['input_domain']=='rectified_pinhole'
         and profile['intrinsics']['pixel_convention']=='integer_centers'
         and profile['official_smplifyx']=={'shape_policy':{'mode':'fit_reference'}},
         'Capture layout, camera domain or shared-shape policy differs.')
    need(min(shutil.disk_usage(LAB).free,shutil.disk_usage('/mnt/c').free)>8*1024**3,
         'Less than 8 GiB available for the fresh local run.')
    runtime=cpu_runtime(clean_env())
    return dict(schema='shuolee-unified-cold-run-v1',subject=subject,candidate_commit=commit,
        output=str(root/'run'),prior_run=str(LAB/p['parent']/'run'),video=str(video),
        source_sha256=p['video_sha256'],camera_source=str(camera),camera_config_sha256=p['camera_sha256'],
        calibration='Same conditional native full K and source half arc; no physical calibration claim.',
        expected_native_samples=56,expected_fit_candidates=52,expected_shape_anchors=24,
        expected_holdouts=p['named_holdouts'],expected_held_source_indices=p['source_holdouts'],
        camera_domain='native rectified 2704x1520, corner render / integer detector, model-relative identity',
        code_sha256={p:sha(PROJECT/p) for p in SOURCE_FILES},
        wrapper_sha256=sha(Path(__file__).resolve()),official_and_asset_sha256=official_pins(),
        runtime=runtime,entry_command=command(subject),
        entry_environment=clean_env(),
        original_video_to_export=True,new_preprocessing=True,independent_shape=True,
        fresh_gaussian_optimizer=True,previous_fit_or_checkpoint_loaded=False,
        gaussian_step_budget=10000,position_lr_schedule_steps=18000,snapshot_steps=[1000,10000],
        opacity_reset_steps=[1800,3600,5400,7200],
        cpu_workers=6,ffprobe_decoder_threads=4,initial_points=20908,sh_degree=0,
        preset_reference=None,quality_accepted=False,max_wall_seconds=MAX_WALL_S)


def register(subject, commit):
    evidence=check(subject,commit)
    root=root_for(subject)
    root.mkdir(exist_ok=False)
    shutil.copyfile(evidence['camera_source'],root/'camera.json')
    pin(root/'camera.json',evidence['camera_config_sha256'])
    write(root/'deployment.json',deployment(root))
    write(root/'preregistration.json',dict(evidence,registered_at=now(),
          deployment_config_sha256=sha(root/'deployment.json')))
    print(json.dumps(dict(state='registered',path=str(root/'preregistration.json'),
                          sha256=sha(root/'preregistration.json'))),flush=True)


def verify_registered(subject,commit):
    root=root_for(subject)
    plan=read(root/'preregistration.json')
    expected=check(subject,commit)
    need({k:v for k,v in plan.items() if k not in ('registered_at','deployment_config_sha256')}==expected,
         'Registered code, source, command or runtime changed.')
    pin(root/'camera.json',plan['camera_config_sha256'])
    pin(root/'deployment.json',plan['deployment_config_sha256'])
    need(read(root/'deployment.json')==deployment(root),'Unexpected deployment profile.')
    return plan


def finish(subject, root, plan, stamp, elapsed, rc):
    run=root/'run'
    status_path=run/'status.json'
    status=read(status_path) if status_path.exists() else {}
    receipt=dict(state='failed' if rc else 'incomplete',subject=subject,returncode=rc,
                 command_started_at=stamp,command_to_end_seconds=elapsed,process_status=status,
                 preregistration_sha256=sha(root/'preregistration.json'))
    if rc==0:
        try:
            result=read(run/'result.json')
            selection=read(run/'fit/selection.json')
            cfg=read(run/'config.json')
            trained=read(run/'gaussian-run/result.json')
            gaussian=read(run/'gaussian-run/config.json')
            fit=read(run/'fit/official-fit-result.json')
            shared=read(run/'fit/shared-body.json')
            run_input=read(run/'run.json')
            need(status.get('state')=='completed' and status.get('static_export_passed') is True
                 and result.get('state')=='completed' and result.get('format_passed') is True
                 and result.get('quality_accepted') is False and result.get('from_scratch') is True
                 and result['steps']==plan['gaussian_step_budget'] and result['sh_degree']==0
                 and fit['fitted_candidate_frames']==plan['expected_fit_candidates']
                 and sum(len(rows) for rows in shared['anchor_groups'].values())==plan['expected_shape_anchors']
                 and len(read(run/'data/inputs.json')['records'])==plan['expected_native_samples']
                 and run_input['video_sha256']==plan['source_sha256']
                 and run_input['camera_config_sha256']==plan['camera_config_sha256'],
                 'Terminal entry, provenance, or cold-run contract differs.')
            need(trained['optimizer_updates']==plan['gaussian_step_budget'] and trained['from_scratch'] is True
                 and trained['schedule_steps']==plan['position_lr_schedule_steps']
                 and gaussian['optimizer_initial_state_entries']==0
                 and gaussian['snapshot_steps']==plan['snapshot_steps']
                 and gaussian['opacity_reset_steps']==plan['opacity_reset_steps']
                 and gaussian['initial_points']==plan['initial_points']
                 and gaussian['sampling_policy']=='seeded_equal_occupied_source_bin_cycles',
                 'Gaussian source-bin, optimizer, split initialization or opacity schedule differs.')
            ids=selection['selected_frames']
            need(selection['retained_holdouts']==plan['expected_holdouts'] and ids
                 and not set(ids)&set(selection['retained_holdouts']) and selection['arc']=='half'
                 and selection['coverage']['rear']['target']==0 and len(ids)==result['frames']
                 and selection['reference_frame']==result['reference_frame_id'],
                 'Selected or heldout roster differs from the registered cold run.')
            for key,digest_key in (('ply','sha256'),('scene','scene_sha256'),
                                   ('bound_replay','bound_replay_sha256'),('bound_panels','bound_panels_sha256')):
                path=(run/result[key]).resolve()
                need(path.is_relative_to(run.resolve()),f'Artifact {key} is outside the run.')
                pin(path,result[digest_key])
            with zipfile.ZipFile(run/result['scene']) as bundle:
                with bundle.open('cameras.json') as camera, bundle.open(
                        f'point_cloud/iteration_{result["steps"]}/point_cloud.ply') as ply:
                    need(stream_sha(camera)==sha(run/result['camera']) and stream_sha(ply)==result['sha256'],
                         'Packaged camera or PLY differs from the delivered artifacts.')
            replay=read(run/result['bound_replay'])
            panels=read(run/result['bound_panels'])
            need({r['frame'] for r in replay['records']}=={r['frame'] for r in panels['panels']}==set(ids),
                 'Native replay or panel roster differs.')
            for p in plan['code_sha256']:
                pin(PROJECT/p,plan['code_sha256'][p])
            static_at=datetime.fromisoformat(status['static_export_completed_at'])
            wall_static=(static_at-datetime.fromisoformat(stamp)).total_seconds()
            need(0<wall_static<=elapsed+1.,'Invalid static export timing boundary.')
            receipt.update(state='completed',reference_frame=result['reference_frame_id'],
                points=result['points'],result_sha256=sha(run/'result.json'),selected_frames=ids,
                coverage=selection['coverage'],full_bin_coverage=selection['full_bin_coverage'],
                static_export_seconds=wall_static,static_export_within_15_minutes=wall_static<=900.,
                panels_complete_seconds=elapsed,optimization=cfg['optimization'],
                formal_training_admitted=result.get('formal_training_admitted',False),quality_accepted=False)
        except Exception as exc:
            receipt['validation_error']=str(exc)
    write(root/'wrapper-terminal.json',receipt)
    print(json.dumps(dict(state=receipt['state'],terminal=str(root/'wrapper-terminal.json'),
                          sha256=sha(root/'wrapper-terminal.json'))),flush=True)
    return 0 if receipt['state']=='completed' else (rc or 1)


def launch(subject, commit):
    root=root_for(subject)
    plan=verify_registered(subject,commit)
    with LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subject=='006':
            predecessor=root_for('016')
            previous=read(predecessor/'wrapper-terminal.json')
            need(previous['state']=='completed' and previous['subject']=='016'
                 and read(predecessor/'preregistration.json')['candidate_commit']==commit,
                 '016 must complete verified delivery first on the same unified commit.')
            pin(predecessor/'run/result.json',previous['result_sha256'])
        need(not any((root/p).exists() for p in ('run','launch.json','wrapper-terminal.json','entry.log')),
             'Fresh output required. Existing outputs are preserved.')
        stamp=now()
        write(root/'launch.json',dict(subject=subject,started=stamp,pid=os.getpid(),
              command=command(subject),preregistration_sha256=sha(root/'preregistration.json')))
        print(json.dumps(dict(state='launching',command=command(subject),log=str(root/'entry.log'))),flush=True)
        started=time.monotonic()
        with (root/'entry.log').open('xb') as log:
            args=['/usr/bin/timeout','--kill-after=15s',f'{MAX_WALL_S-15}s',*command(subject)]
            rc=subprocess.run(args,env=clean_env(),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT).returncode
        return finish(subject,root,plan,stamp,time.monotonic()-started,rc)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','register','launch'))
    parser.add_argument('subject',choices=PREVIOUS)
    parser.add_argument('--commit',required=True,help='Full verified unified Git HEAD.')
    args=parser.parse_args()
    if args.action=='check':
        result=check(args.subject,args.commit)
        print(json.dumps(dict(state='CPU_preflight_passed',no_gpu_calls=True,
            subject=args.subject,output=result['output'],entry_command=result['entry_command'],
            verified_local_assets=len(result['official_and_asset_sha256']))),flush=True)
        return
    if args.action=='register':
        register(args.subject,args.commit)
        return
    raise SystemExit(launch(args.subject,args.commit))


if __name__=='__main__':
    main()
