"""One fresh fixed-K research run, with a mandatory 1000-step visual decision.

Formal alignment admission remains rejected; this never alters that gate.
No old Gaussian, optimizer, or fitted person from a different run is loaded.
"""
from pathlib import Path
import json,sys,os,time,hashlib
import numpy as np
from datetime import datetime,timezone
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.cli import gpu_lock


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
    rt.save(REC/'queue-state.json',dict(updated=now(),pid=os.getpid(),run_root=str(ROOT),
        formal_training_admitted=False,quality_accepted=False,old_probe_steps=1000,**kw))


def review(step,snapshot):
    if step!=1000:return True
    decision=ROOT/'reviews/gaussian-step-001000-decision.json'
    state(state='fixedK_research_waiting_1000_visual_review',stage='review',training_running=True,
        optimizer_waiting=True,steps=step,step_budget=18000,snapshot=str(snapshot),decision_file=str(decision))
    print('WAITING_ACTUAL_1000_REVIEW '+str(decision),flush=True)
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        if decision.is_file():
            d=rt.read(decision)
            if d.get('snapshot_sha256')!=sha(snapshot/'world.ply') or not d.get('reviewed_same_view_images'):
                raise RuntimeError('Review does not match this exact fresh checkpoint.')
            if d.get('formal_calibration_accepted') is not False:
                raise RuntimeError('Research review cannot grant calibration acceptance.')
            allowed=d.get('continue_same_run') is True
            state(state='fixedK_research_training' if allowed else 'fixedK_research_stopped_after_review',
                stage='gaussian',training_running=allowed,steps=step,step_budget=18000,decision=d)
            return allowed
        time.sleep(5)
    rt.save(decision,dict(created=now(),snapshot_sha256=sha(snapshot/'world.ply'),
        reviewed_same_view_images=False,continue_same_run=False,formal_calibration_accepted=False,
        reason='Bounded visual-review wait expired; no automatic promotion or training continuation.'))
    return False


with gpu_lock():
    assert not (ROOT/'gaussian-run').exists(),'Never overwrite or resume an existing Gaussian run.'
    v=rt.read(REC/'native-fit-visual-review.json')
    assert v['selected_arm']=='reference' and v['reviewed_frames']==['frame_000001','frame_000018','frame_000036','frame_000053','frame_000071','frame_000089']
    assert v['fit_sha256']==sha(ROOT/'fit/fitted.npz') and v['formal_geometry_accepted'] is False
    plan=rt.read(REC/'fixedk-research-training-plan.json')
    assert plan['kind']=='finite_research_training_with_early_review' and plan['max_steps']==18000
    assert plan['fit_sha256']==v['fit_sha256'] and plan['camera_sha256']==sha(ROOT/'data/training-inputs.json')
    assert plan['training_source_sha256']==sha(PROJECT/'src/shuolee_portrait/training.py')
    cfg=rt.read(ROOT/'config.json');cfg.update(training_height=899,initial_points=200000,iterations=18000)
    rows=rt.read(ROOT/'data/training-inputs.json')['records']
    assert len(rows)==89 and all(r['wh']==[1600,899] for r in rows)
    rt.save(ROOT/'research-training-plan.json',plan)
    state(state='fixedK_research_training',stage='gaussian',training_running=True,steps=0,step_budget=18000)
    rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',ROOT)
    from shuolee_portrait.training import _train_static
    try:
        _train_static(ROOT,cfg,execution_kind='finite_unvalidated_calibration_research',checkpoint_review=review)
        result=rt.read(ROOT/'gaussian-run/result.json')
        rt.save(REC/'fixedk-research-training-result.json',result)
        state(state='fixedK_research_training_finished_pending_export_ui',stage='actual_review_and_export',
            training_running=False,steps=result['steps'],step_budget=18000,result=result)
    except BaseException as exc:
        state(state='fixedK_research_training_failed',stage='gaussian',training_running=False,error=repr(exc))
        raise
