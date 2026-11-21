from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess

P=Path('D:/00/工作/人体重建')
BASE=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab')
R=BASE/'shuolee-portrait-server-check-009'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not R.exists(), 'Fresh output required'
assert not (P/'.gpu-task.lock').exists(), 'Another task may be running'
assert not subprocess.check_output(['git','-C',str(P),'status','--porcelain']).strip()
commit=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()
assert commit=='e7c8be57bf4df50d87c05a1dbbc6f4cdd4c13744'
files=list((P/'src/shuolee_portrait').glob('*.py'))
files+=[P/p for p in ['run.py','configs/smoke.json','configs/server-assets.json','upstream.lock.json','requirements.txt','constraints-linux-cu128.txt']]
files+=list((P/'src/avatar_core').glob('*.py'))
manifest={p.relative_to(P).as_posix():sha(p) for p in files}
now=datetime.now(timezone.utc).isoformat()
plan=dict(created=now,purpose='Bounded fresh integration regression after prior crop pixel-coordinate correction.',
    source_commit=commit,source=str(P),source_hashes=manifest,
    environment='/opt/gs-slam-lab/shuolee-portrait-clean-001/env',
    output='/opt/gs-slam-lab/shuolee-portrait-server-check-009',config='configs/smoke.json',stop_after='fit-gate',
    hypotheses=['Corrected crop code can produce self-consistent forward/inverse maps for all current detections and complete the raw-video-to-body-fit path.'],
    scope='Engineering integration only. Fresh SfM is nondeterministic, so 008/009 fitting metrics do not isolate a causal improvement from the crop correction.',
    raw_video='/mnt/d/00/工作/minisplatting2/HOVER_VERSA_0102.mp4',
    raw_video_sha256=read(BASE/'shuolee-portrait-server-check-008/run.json')['video_sha256'],
    existing_reference_video=True,new_person_test=False,from_scratch=True,
    old_camera_or_subject_or_gaussians_loaded=False,gaussian_training_allowed=False,
    formal_full_frame_experiment=False,frame_stride=8,fit_steps=1200,
    wall_time_limit_minutes=20,retries_allowed=0,
    stop_conditions=['Any stage failure preserves logs without resuming.','Stop at fit-gate regardless of coarse result.','No formal Gaussian training or geometry quality claim.'])
save(P/'provenance/server-interface-check-009-plan.json',plan)
goal=read(P/'provenance/long-goal-state.json')
goal.update(updated=now,active_engineering_check='009',no_active_training=True,
    next_actions=['Complete one bounded post-crop-fix fresh integration run through fit-gate.','Review actual inputs/mesh overlays and per-frame crop maps.','Independent alignment evidence remains required before formal training.'])
save(P/'provenance/long-goal-state.json',goal)
print(json.dumps({'prepared':True,'commit':commit,'source_files':len(manifest),'output':plan['output'],'formal_training':False}))
