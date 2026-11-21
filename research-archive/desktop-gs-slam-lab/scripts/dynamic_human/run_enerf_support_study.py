"""Sequential four-condition motion support experiment on the same local inputs."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,subprocess,hashlib

LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-support-030';out.mkdir(exist_ok=True)
source=LAB/'outputs/dynamic-human-enerf-local-time-029/offset-index'
manifest=source/'appearance-manifest.json';initial=source/'first-frame/resume.pt'
script=LAB/'scripts/dynamic_human/train_enerf_motion.py';state_path=out/'pipeline-state.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,v):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8');os.replace(tmp,path)
if state_path.exists():
    old=json.loads(state_path.read_text());pid=old.get('parent_pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists() and str(Path(__file__).resolve()).encode() in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError('Previous pipeline is live; inspect its existing handle')
    if old['status']=='all_training_complete_evaluation_pending':raise RuntimeError('Training already complete')
protocol=dict(status='frozen_before_four_condition_training',purpose='Test whether fixed per-Gaussian scale/opacity prevents continuous hand/toy surface support',conditions=['frozen','scales','opacities','both'],shared_initial_checkpoint_sha256=sha(initial),manifest_sha256=sha(manifest),trainer_sha256=sha(script),frames=list(range(34,59)),train_cameras=15,train_observations=375,appearance_heldout_cameras=['00','06','12'],appearance_heldout_images=75,per_model_budget='Shared6000first-frame fit +24x600motion updates =20400. No fresh per-frame reinitialization.',unchanged=['Initial parameters, input observations, seed74, observation order and random samples','Mean/quaternion/SH learning rates','Rigidity, rotation, isometry, color continuity and image losses','Gaussian count and identity, neighbor graph, no splitting or pruning','Native resolution and crop32pixel margin, fixed cameras'],interventions=dict(scales='Train log standard deviations at LR0.0005, prior0.01mean squared temporal log-scale change, initial-relative +/-ln2 bounds',opacities='Train opacity logits at LR0.002, prior0.01mean squared temporal sigmoid-opacity change, initial-relative +/-2 bounds',both='Both interventions together'),baseline='Fresh frozen run of same current code. First-frame35 reproduction demonstrated CUDA numerical nondeterminism, so old run is not substituted as the only control.',evaluation='All4models use same75appearance-heldout photographs; full canvas, matte foreground, each person, upper-body boxes, consecutive image change residuals and complete videos. Fixed visual samples34,46,58 and previously problematic50, plus worst regression.',limits=['Earlier calibration images from all18cameras; heldout is appearance-only','No temporal holdout in this developmental window','No source background model, physical surface correspondences or metric scale verified','Both changes add priors/bounds; this tests the specified constrained adaptation, not arbitrary unconstrained freedom','Fixed one-seed experiment; small score changes need repeat validation, not significance claims'])
if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
else:save(out/'protocol.json',protocol)
history=[]
for variant in protocol['conditions']:
    group=out/variant;marker=group/'motion/completion.json'
    if marker.exists():
        done=json.loads(marker.read_text());assert done['updates']==14400 and done['frames']==protocol['frames'];history.append(dict(variant=variant,status='previously_completed',completion_sha256=sha(marker)));continue
    cmd=[sys.executable,'-u',str(script),'--out',str(group),'--manifest',str(manifest),'--initial-checkpoint',str(initial),'--root','/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor1_4','--support',variant]
    log=out/f'{variant}.log';start=time.time()
    with log.open('a',buffering=1) as stream:
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
        save(state_path,dict(status='running',parent_pid=os.getpid(),child_pid=child.pid,variant=variant,command=cmd,log=str(log),started_unix=start,history=history))
        print('START',variant,'pid',child.pid,flush=True);code=child.wait()
    record=dict(variant=variant,exit_code=code,elapsed_seconds=time.time()-start,marker_exists=marker.exists());history.append(record)
    if code!=0 or not marker.exists():save(state_path,dict(status='failed',parent_pid=os.getpid(),variant=variant,history=history));raise RuntimeError(record)
    print('DONE',variant,round(time.time()-start,1),flush=True)
save(state_path,dict(status='all_training_complete_evaluation_pending',parent_pid=os.getpid(),history=history,completed_utc=datetime.now(timezone.utc).isoformat()))
print('ALL_TRAINING_COMPLETE',flush=True)

