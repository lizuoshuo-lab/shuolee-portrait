"""User-authorized local sofa-style surface training; new roots and optimizers.

This is a finite research delivery, not physical subpixel calibration acceptance.
The verified sofa119 representation/optimizer is reused without editing it.
After the observed SMPL-X failures, any future entry also requires a separately
reviewed fine-alignment contract. Historical coarse permissions do not satisfy it.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
from local_surface123_alignment_gate import require_alignment_admission, require_camera_admission

HERE = Path(__file__).resolve().parent
TOP = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = TOP/'shuolee/current/local-clothed-surface-123'
ROOT = Path('/opt/gs-slam-lab/shuolee-local-surface-123')
ORIGINAL = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
PROJECT = ROOT/'runtime'
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
SOURCE = Path('/mnt/d/00/工作/数据集')
COORD = HERE.parent/'shuolee-coordination'
PARENT = TOP/'shuolee/current/robust-portrait-122'

def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def status(subject, stage=None, **kw):
    if stage is None:stage=kw.pop('state')
    d=dict(updated=now(),subject=subject,stage=stage,pid=os.getpid(),command=sys.argv,
           quality_accepted=False,physical_calibration_accepted=False,**kw)
    save(REC/f'{subject}-state.json',d)
    q=read(REC/'queue-state.json');q.update(updated=now(),active_subject=subject,
        current_stage=stage,worker_pid=os.getpid(),worker_command=sys.argv)
    for row in q['subjects']:
        if row['id']==subject:row.update(stage=stage,updated=d['updated'])
    save(REC/'queue-state.json',q)
    print(json.dumps(d,ensure_ascii=False),flush=True)

SUBJECTS = [
 ('winterhat','戴帽蓝衣人物新版','视频-777d211114ab4f39a1b5612466dda6d5.mp4'),
 ('whiteT','室外白T恤人物','视频-dc1571c16c894d9989695137a386fb1e.mp4'),
 ('grayT','室外灰T恤人物','视频-f5420380646f495f872c9ce9bcce5448.mp4'),
 ('pattern','室内花纹上衣人物','视频-fc2a55a8ce9d4263b1ebcb7915a6b18c.mp4'),
 ('field','田地深色上衣人物','cb3f4555-8b01.mp4'),
 ('grass','草地白衣人物','2f1aac19-8ae1_1791971027322.mp4'),
 ('indoor','室内白衣人物','df776c03-91a4_1792572380952.mp4'),
]

def register():
    if (REC/'protocol.json').exists():
        raise RuntimeError('Round already registered; inspect it, do not replace.')
    import cv2
    REC.mkdir(parents=True);ROOT.mkdir(exist_ok=False)
    PROJECT.mkdir()
    for folder in ['src','configs']:
        shutil.copytree(ORIGINAL/folder,PROJECT/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    shutil.copy2(ORIGINAL/'run.py',PROJECT/'run.py')
    (PROJECT/'third_party').symlink_to(ORIGINAL/'third_party',target_is_directory=True)
    copied=[p for folder in ['src','configs'] for p in (PROJECT/folder).rglob('*') if p.is_file()]
    copied.append(PROJECT/'run.py')
    save(REC/'runtime-source-manifest.json',dict(created=now(),source=str(ORIGINAL),
        snapshot=str(PROJECT),files=[dict(path=str(p.relative_to(PROJECT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(copied)],
        third_party_read_only_source=str(ORIGINAL/'third_party'),no_weights_copied=True))
    fs=cv2.FileStorage(str(SOURCE/'mono_main_calib.yaml'),cv2.FILE_STORAGE_READ)
    calibration=dict(image_wh=[int(fs.getNode('image_width').real()),int(fs.getNode('image_height').real())],
        K=fs.getNode('camera_matrix').mat().tolist(),sha256=sha(SOURCE/'mono_main_calib.yaml'))
    fs.release()
    rows=[]
    for ident,label,name in SUBJECTS:
        p=SOURCE/name
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_streams','-of','json',str(p)]))['streams'][0]
        rows.append(dict(id=ident,label=label,video=name,video_sha256=sha(p),
            source_wh=[stream['width'],stream['height']],source_frames=int(stream['nb_frames']),
            selected_frames=len(range(0,int(stream['nb_frames']),5)),run_root=str(ROOT/ident),stage='queued'))
    # Copy only the already-read camera function into this round. No historical
    # runner, inputs, optimized poses, fits or checkpoints enter the new run.
    src=HERE/'capture118_all_subjects.py';s=src.read_text(encoding='utf-8')
    body=s[s.index('def camera(root):'):s.index('\ndef run(subject_id, inputs):')]
    pre='from pathlib import Path\nimport hashlib,json,os,time\nimport numpy as np\nfrom local_surface123 import PROJECT,REC,status as state,now\nimport sys\nsys.path.insert(0,str(PROJECT/"src"))\nfrom shuolee_portrait import runtime as rt\n'
    target=HERE/'local_surface123_camera.py'
    if target.exists():raise RuntimeError('Camera implementation already exists; inspect, not overwrite.')
    target.write_text(pre+'\n'+body+'\n',encoding='utf-8',newline='\n')
    plan=dict(created=now(),authorization='先本地把数据集训练好，放到查看器，方法按照沙发的表面自由点',
        execution='Local research reconstruction now takes priority over unexecuted R7 engineering; quality target is not waived.',
        reference='Existing sofa119 18k result preserved; use its unbound static clothed-world representation/optimizer, no sofa rerun.',
        subjects=rows,calibration=calibration,
        input_contract=dict(frame_stride=5,selected_frames_predeclared=True,all_selected_frames_must_register=True,
            feature_domain='Native decoded MP4, no downsampling for SIFT and no second undistortion',
            K='First4000x3000 YAML matrix, explicit centered aspect crop and half-pixel size transform; fixed conditional K',
            no_COLMAP=True,no_interpolated_poses=True,old_pose_or_3D_initialization=False,
            unsupported_residual_distortion_not_silently_applied=True,training_max_side=1600),
        method=dict(gaussian_representation='static_clothed_world',initial_points=200000,iterations=18000,
            fit_steps=1200,seed=6060,surface_spacing_m=.012,surface_padding_m=.40,
            surface_vote_fraction=.97,surface_mask_tolerance_native_px=2,
            binding=False,body_attraction=False,per_triangle_quota=False,
            fresh_fit=True,empty_optimizer=True,foreground_only_RGB=True),
        review=dict(fit='Actual same-camera coarse mesh placement',surface='Actual silhouette surface and center distribution',
            step1000='Actual same-camera face/body/centers before continuation',
            final='Native PLY parity, full covariance+SH,world+Y cameras,36novelviews,actual OpenFile Gaussian/points'),
        budgets=dict(camera_candidates_per_subject=1,camera_seconds=1800,fit_steps=1200,
            new_training_runs_per_subject=1,max_steps=18000,review_wait_seconds=7200,max_points=650000),
        scientific_limits=['This user-authorized finite training does not certify K, staticity, face correspondences or subpixel accuracy.',
            'Any actual unregistered camera, nonfinite fit, empty/clipped surface or catastrophic mask problem stops that subject; do not relax thresholds automatically.',
            'All errors and held observations preserved; held appearance tracks are not independently verified physical ground truth.',
            'Occluded feet and unobserved backs are unsupported, not reliable empty-space evidence.'],
        source_camera=dict(parent=str(src),parent_sha256=sha(src),round_source=str(target),sha256=sha(target)),
        quality_accepted=False,physical_calibration_accepted=False,formal_training_admitted=False)
    save(REC/'protocol.json',plan)
    save(REC/'queue-state.json',dict(updated=now(),state='active',active_subject=None,current_stage='registered',
        subjects=[{k:r[k] for k in ['id','label','stage','selected_frames','run_root']} for r in rows],
        sofa_reference='new-capture-119-sofa-clothed',quality_accepted=False,training_started=False))
    shutil.copy2(HERE/'local_surface123.py',REC/'orchestrator-source.py')
    save(REC/'evidence-ledger.json',dict(updated=now(),entries=[dict(kind='user_steering',
        text=plan['authorization'],date=now(),runtime_snapshot=str(PROJECT),no_training_yet=True)]))
    q=read(PARENT/'queue-state.json')
    q.update(updated=now(),phase='LOCAL_SURFACE123_registered',next='Execute fresh local sofa-style free-surface queue; R7 draft test deferred by current user priority.',
        local_surface_round=str(REC),training_admitted=False,training_started=False,
        last_goal_turn=dict(classification='progress',current='User requests actual local dataset training with sofa free-surface method',publication_complete=False))
    save(PARENT/'queue-state.json',q)
    save(PARENT/'native-residual-pipeline/progress.json',dict(updated=now(),state='deferred_by_user_priority',
        reason=plan['authorization'],test_executed=False,source_drafts_only=True,goal_remains_active=True))
    p=read(PARENT/'protocol.json');p['latest_user_steering']=dict(updated=now(),statement=plan['authorization'],
        plan=str(REC/'protocol.json'),finite_research_training_authorized=True,quality_acceptance_unchanged=True)
    save(PARENT/'protocol.json',p)
    note='\n\n## 2026-10-28 本地表面自由点训练123\n用户明确优先本地训练和查看器交付。沿用沙发119的静态穿衣表面自由点路线；沙发模型保留，其余7段按冻结队列逐个从头拟合、20万点、空优化器，1000步实际审阅后最多18k。每5帧取1帧，所有预登记训练帧必须真实注册。内参使用第一组4000×3000经居中裁剪/尺寸变换的条件K，原生MP4检测，不再次去畸变。有限研究训练与物理亚像素质量验收分别记录。未执行的R7组件测试让位于本次实训，目标保持继续。\n'
    with (PARENT/'long-term-plan.md').open('a',encoding='utf-8') as f:f.write(note)
    save(COORD/'local-surface123-status.json',dict(updated=now(),priority='local dataset training',
        plan=str(REC/'protocol.json'),mode='goal_not_timer',company_server_used=False,
        research_training_authorized=True,physical_quality_accepted=False))
    print(json.dumps(dict(registered=str(REC),subjects=[(r['id'],r['selected_frames']) for r in rows]),ensure_ascii=False),flush=True)

@contextmanager
def gpu_slot():
    import fcntl
    common=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(common,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=ORIGINAL/'.gpu-task.lock'
    fd=None
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        os.write(fd,json.dumps(dict(pid=os.getpid(),command=sys.argv,round=123)).encode());os.close(fd);fd=None
        yield
    finally:
        if fd is not None:os.close(fd)
        if lock.exists() and read(lock).get('pid')==os.getpid():lock.unlink()
        fcntl.flock(common,fcntl.LOCK_UN);common.close()

def prepare(subject):
    # Fail before creating a run or decoding when the reviewed observation
    # contract is absent. The old implicit SIFT route is no longer a fallback.
    from local_surface123_frozen_observations import load_subject_input,solver_tracks
    frozen=load_subject_input(subject)
    solver_tracks(frozen)
    import numpy as np
    from shuolee_portrait import runtime as rt
    root=ROOT/subject['id'];root.mkdir(exist_ok=False)
    for folder in ['data','camera','logs','reviews']:(root/folder).mkdir()
    video=frozen.source_video
    if sha(video)!=subject['video_sha256']:raise RuntimeError('Video content changed after registration.')
    cfg=read(PROJECT/'configs/server.json')
    cfg.update(frame_stride=5,max_image_size=max(subject['source_wh']),min_frames=12,max_frames=300,
        training_height=round(subject['source_wh'][1]*min(1.,1600/max(subject['source_wh']))),
        seed=6060,initial_points=200000,iterations=18000,fit_steps=1200,
        pixel_center_convention='renderer_corner_to_detector_integer',landmark_image_support='mask_margin_4px',
        gaussian_representation='static_clothed_world',surface_spacing_m=.012,surface_padding_m=.40,
        surface_vote_fraction=.97,surface_mask_tolerance_px=2.,export_frame_mode='scene_up_keep_axis_order',
        camera_estimator='Frozen reviewed native fixed-K OpenCV essential/PnP/BA research',physical_calibration_validated=False)
    save(root/'config.json',cfg)
    save(root/'run.json',dict(created=now(),video=str(video),video_sha256=subject['video_sha256'],subject=subject,
        old_subject_loaded=False,old_fitted_person_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False))
    status(subject['id'],'copying_frozen_native_frames',run_root=str(root),training_running=False)
    (root/'raw').mkdir()
    for row,frame in zip(frozen.decoded['records'],frozen.frames):
        target=root/'raw'/row['name']
        shutil.copy2(frame['image'],target)
        if sha(target)!=frame['image_sha256']:raise RuntimeError('Frozen native image copy differs.')
    bindings={k:p for k,p,h in frozen.bound_files}
    shutil.copy2(bindings['decoded'],root/'decoded.json')
    save(root/'frozen-observation-input.json',frozen.audit())
    decoded=read(root/'decoded.json')['records'];w,h=decoded[0]['wh']
    if len(decoded)!=subject['selected_frames'] or [w,h]!=subject['source_wh']:
        raise RuntimeError('Native frame count/dimensions differ from preregistration.')
    factory=read(REC/'protocol.json')['calibration'];fw,fh=factory['image_wh']
    aspect=w/h
    cw,ch=(fw,fw/aspect) if aspect>=fw/fh else (fh*aspect,fh)
    x0,y0=(fw-cw)/2,(fh-ch)/2
    A=np.array([[w/cw,0,w/cw*(.5-x0)-.5],[0,h/ch,h/ch*(.5-y0)-.5],[0,0,1.]])
    K=A@np.asarray(factory['K'])
    save(root/'intrinsic-contract.json',dict(factory_sha256=factory['sha256'],factory_K=factory['K'],
        crop_xywh=[x0,y0,cw,ch],crop_is_assumption=True,raw_K_integer=K.tolist(),pixel_map=A.tolist(),
        wh=[w,h],source='Native processed MP4 pixels, no lens remap',D8_applied=False,
        calibration_applicability_confirmed=False,principal_center_convention='integer; renderer receives+0.5'))
    save(root/'frame-plan.json',dict(native_video_frames=subject['source_frames'],selected_frames=len(decoded),
        frame_stride=5,selected_source_indices=[r['source_frame_index'] for r in decoded],
        decoded_sha256=sha(root/'decoded.json'),no_interpolation=True))
    from local_surface123_camera_reviewed import camera
    camera(root,frozen)
    return root

def stage(root, name, timeout=1800):
    command=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage',name,'--output',str(root),'--assets',str(ASSETS)]
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
    with gpu_slot(),(root/'logs'/f'{name}.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status(root.name,name+'_running',child_pid=proc.pid,child_command=command,training_running=False)
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
            raise RuntimeError(name+' time budget exceeded; child stopped and logs preserved.')
        if rc:raise RuntimeError(f'{name} failed with{rc}; {root}/logs/{name}.log')

def await_review(root, phase, payload, decision_key):
    status(root.name,phase+'_review_pending',review=payload,training_running=phase.startswith('gaussian'),
           optimizer_waiting=phase.startswith('gaussian'))
    path=root/'reviews'/f'{phase}-decision.json'
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        if path.exists():
            d=read(path)
            if any(d.get(k)!=v for k,v in payload.items()):raise RuntimeError('Review hashes/identity do not match current stage.')
            if not d.get('actual_visual_review'):raise RuntimeError('Actual visual review receipt missing.')
            return d.get(decision_key) is True
        time.sleep(5)
    return False

def run_subject(subject):
    from shuolee_portrait import runtime as rt
    root=prepare(subject)
    for name in ['observations','prior']:stage(root,name)
    require_camera_admission(root)
    for name in ['fit','fit-gate']:stage(root,name)
    if not await_review(root,'fit',dict(fit_sha256=sha(root/'fit/fitted.npz')),'allow_surface_initialization'):
        raise RuntimeError('Stopped after fit review or review deadline; no Gaussian training.')
    require_alignment_admission(root)
    rt.configure(ASSETS,root);cfg=read(root/'config.json')
    with gpu_slot():
        from shuolee_portrait.clothed_surface import initialize
        status(root.name,'fresh_clothing_surface',training_running=False);initialize(root,cfg)
    if not await_review(root,'surface',dict(points_sha256=sha(root/'clothing-init/points.npz')),'allow_finite_research_training'):
        raise RuntimeError('Stopped after actual surface review or review deadline.')
    require_alignment_admission(root)
    def checkpoint(step,snapshot):
        if step!=1000:return True
        require_alignment_admission(root)
        return await_review(root,'gaussian-step-001000',dict(snapshot_sha256=sha(snapshot/'world.ply')),'continue_same_run')
    with gpu_slot():
        from shuolee_portrait.clothed_training import train_clothed
        status(root.name,'training',training_running=True,step=0)
        q=read(REC/'queue-state.json');q['training_started']=True;save(REC/'queue-state.json',q)
        train_clothed(root,cfg,checkpoint_review=checkpoint)
        from shuolee_portrait.export import export
        status(root.name,'exporting',training_running=False);export(root,cfg)
    status(root.name,'exported_actual_quality_UI_review_pending',training_running=False,
           result=read(root/'result.json'))

def run(selected=None):
    import fcntl
    lock=open(ROOT/'worker.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(json.dumps(dict(pid=os.getpid(),command=sys.argv)));lock.flush()
    sys.path.insert(0,str(PROJECT/'src'))
    snapshot=read(REC/'runtime-source-manifest.json')
    for r in snapshot['files']:
        if sha(PROJECT/r['path'])!=r['sha256']:raise RuntimeError('Frozen runtime source changed: '+r['path'])
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    p=read(REC/'protocol.json')
    if sha(p['source_camera']['round_source'])!=p['source_camera']['sha256']:raise RuntimeError('Camera source changed.')
    execution=read(REC/'camera-observation-entry/code-manifest.json')
    for item in execution['files']:
        if sha(HERE/item['name'])!=item['sha256']:raise RuntimeError('Reviewed entry source changed: '+item['name'])
    # No stage continuation, checkpoint loading or automatic retrial. Completed
    # preparation is never overwritten and each new subject root is exclusive.
    for subject in p['subjects']:
        if selected and subject['id']!=selected:continue
        if (ROOT/subject['id']).exists():
            raise RuntimeError('Existing subject root: inspect current worker, never restart it.')
        try:run_subject(subject)
        except Exception as exc:
            evidence=dict(created=now(),subject=subject['id'],error=repr(exc),traceback=traceback.format_exc(),
                preserved_root=str(ROOT/subject['id']),automatic_retry=False)
            save(REC/f'{subject["id"]}-failure.json',evidence)
            status(subject['id'],'failed_preserved',training_running=False,error=repr(exc))
            print(evidence['traceback'],flush=True)
        finally:
            import gc
            gc.collect()
            if 'torch' in sys.modules:sys.modules['torch'].cuda.empty_cache()
    q=read(REC/'queue-state.json');q.update(updated=now(),state='worker_finished_review_pending',
        active_subject=None,current_stage='review_results',worker_pid=None)
    save(REC/'queue-state.json',q)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--subject')
    a=p.parse_args()
    if a.register:register()
    elif a.run:run(a.subject)
    else:p.error('Choose --register or --run')
