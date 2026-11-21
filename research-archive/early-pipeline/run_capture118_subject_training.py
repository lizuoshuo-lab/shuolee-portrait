"""Fresh fit or reviewed finite static research training for one registered subject."""
import argparse,os,sys,time,subprocess,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture118_all_subjects import ROOT,REC,PROJECT,ASSETS,rt,state
from shuolee_portrait.cli import gpu_lock
ap=argparse.ArgumentParser();ap.add_argument('--subject',required=True,choices=['grass','sofa','portrait','indoor118']);ap.add_argument('--stage',required=True,choices=['fit','train']);args=ap.parse_args()
root=ROOT/args.subject
def review(step,snapshot):
 if step!=1000:return True
 decision=root/'reviews/gaussian-step-001000-decision.json'
 state(args.subject,state='gaussian_review_pending',run_root=str(root),training_running=True,optimizer_waiting=True,step=step,decision=str(decision),snapshot_sha256=rt.sha(snapshot/'world.ply'))
 deadline=time.monotonic()+900
 while time.monotonic()<deadline:
  if decision.exists():
   d=rt.read(decision)
   if d.get('snapshot_sha256')!=rt.sha(snapshot/'world.ply') or d.get('actual_same_camera_review') is not True or d.get('physical_calibration_accepted') is not False:
    raise RuntimeError('Checkpoint review missing or does not bind the actual research snapshot.')
   allowed=d.get('continue_same_run') is True
   state(args.subject,state='gaussian_training' if allowed else 'stopped_after_review',run_root=str(root),training_running=allowed,step=step)
   return allowed
  time.sleep(5)
 return False

with gpu_lock():
 try:
  cfg=rt.read(root/'config.json')
  assert cfg['initial_points']==200000 and cfg['iterations']==18000 and cfg['fit_steps']==1200
  if args.stage=='fit':
   decision=rt.read(root/'reviews/input-decision.json')
   assert decision['actual_raw_masks_and_tracks_reviewed'] and decision['allow_fresh_fit']
   assert decision['camera_sha256']==rt.sha(root/'camera/candidate-cameras.json')
   assert decision['observations_sha256']==rt.sha(root/'data/observations.npz')
   assert not (root/'fit').exists(),'Never reuse or overwrite an earlier personalized fit.'
   for stage in ['fit','fit-gate']:
    command=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage',stage,'--output',str(root),'--assets',ASSETS]
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
    with (root/'logs'/f'{stage}.log').open('x') as log:
     proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     state(args.subject,state='fit_running',run_root=str(root),training_running=False,child_pid=proc.pid,child_command=command)
     try:rc=proc.wait(timeout=600)
     except subprocess.TimeoutExpired:
      os.killpg(proc.pid,signal.SIGTERM)
      try:proc.wait(timeout=15)
      except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
      raise RuntimeError('Fresh fit budget exhausted.')
     if rc:raise RuntimeError(stage+' exit '+str(rc))
   state(args.subject,state='fit_review_pending',run_root=str(root),training_running=False,fit_sha256=rt.sha(root/'fit/fitted.npz'))
  else:
   assert not (root/'gaussian-run').exists(),'No cross-experiment continuation or overwrite.'
   decision=rt.read(root/'reviews/fit-decision.json')
   assert decision['fit_sha256']==rt.sha(root/'fit/fitted.npz') and decision['allow_finite_research_training'] is True
   assert decision['actual_same_camera_fit_reviewed'] and decision['physical_calibration_accepted'] is False
   plan=dict(created=rt.now(),subject=args.subject,max_steps=18000,initial_points=200000,from_scratch=True,
    old_personalized_fit_loaded=False,old_gaussians_loaded=False,old_optimizer_loaded=False,
    fit_sha256=rt.sha(root/'fit/fitted.npz'),camera_sha256=rt.sha(root/'data/training-inputs.json'),
    training_source_sha256=rt.sha(PROJECT/'src/shuolee_portrait/training.py'),
    execution_kind='finite_unvalidated_calibration_research',formal_training_admitted=False,
    checkpoint_review='Actual1000-step image review, no approval within900s stops the run.',quality_accepted=False)
   rt.save(root/'research-training-plan.json',plan)
   rt.configure(ASSETS,root)
   from shuolee_portrait.training import _train_static
   state(args.subject,state='gaussian_training',run_root=str(root),training_running=True,step=0)
   _train_static(root,cfg,execution_kind='finite_unvalidated_calibration_research',checkpoint_review=review)
   result=rt.read(root/'gaussian-run/result.json')
   state(args.subject,state='gaussian_finished_export_review_pending',run_root=str(root),training_running=False,result=result)
 except BaseException as exc:
  state(args.subject,state=args.stage+'_failed_preserved_for_review',run_root=str(root),training_running=False,error=repr(exc))
  raise
