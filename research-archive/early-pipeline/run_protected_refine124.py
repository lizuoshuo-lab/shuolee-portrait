"""One bounded protected-torso/arms candidate; reuse the existing fit runner."""
from pathlib import Path
import inspect,sys,fcntl
import smplify_protected_refine124 as adapter
import run_selected_refine124 as previous
from run_official_gaussians123 import read,save,sha,now

OUT=previous.shared.ROOT/'grass-protected-refine124'
BASE=previous.BASE
H=Path(__file__).resolve().parent
scope=previous.__dict__.copy()
scope.update(OUT=OUT,adapter=adapter)
for name in ('verify','preflight','run'):
    exec(compile(inspect.getsource(getattr(previous,name)),'<reused-bounded-runner-'+name+'>','exec'),scope)


def prepare():
    assert not OUT.exists()
    closed=read(previous.OUT/'conclusion.json')
    assert closed['state']=='completed_rejected_depth_regression'
    plan=read(BASE/'plan.json')
    sources=[Path(__file__),Path(adapter.__file__),Path(previous.__file__),
        Path(previous.adapter.__file__),Path(previous.entry.__file__),Path(previous.shared.__file__),
        H/'control_smplifyx_root_seed123.py',H/'smplify_temporal124.py',
        BASE/'plan.json',BASE/'selected-fit.npz',BASE/'conclusion.json',
        previous.OUT/'conclusion.json',previous.shared.SRC/'data/inputs.json',
        previous.shared.SRC/'data/observations.npz',previous.shared.SRC/'selection.json']
    sources += [BASE/'frames'/f/'winner-camera-space.npz' for f in plan['frames']]
    sources += [previous.shared.OFFICIAL/k for k in previous.entry.PINNED]
    code,delta=adapter.source(previous.shared.OFFICIAL)
    OUT.mkdir();(OUT/'interface.diff').write_text(delta,encoding='utf-8')
    save(OUT/'plan.json',dict(updated=now(),state='registered',frames=plan['frames'],rows=plan['rows'],
        hypothesis='The rejected per-view control lost the shared arm/torso geometry. Preserve that learned local configuration and allow only peripheral body rotations to adapt to small observed movements.',
        variable='VPoser output mask preserves9 named spine/collar/shoulder/elbow rotations at the frozen multi-view estimate. No other loss or schedule change from the rejected refinement.',
        initialization='Every frame restarts from its same coherent8-view shared-pose result, never the rejected refined output.',
        protected_joint_ids=list(adapter.PROTECTED_IDS),protected_joint_names=list(adapter.PROTECTED_NAMES),
        free_body_joint_ids=list(adapter.FREE_IDS),
        unchanged=['full K','shared betas','observations and confidences','original stage3/4,30iterations,LBFGS','VPoser/L2/bending priors','shared pose penalty30','root/placement/face/finger freedom','existing3percent screening'],
        risks=['The shared torso/arm assumption may reject true movement.','Changing root or peripheral joints can still worsen projection or anatomy.','A preserved bad common pose would remain bad; invariance is not physical accuracy.'],
        budget=dict(candidates=1,frames=8,max_wall_seconds=600,original_stage_indices=[3,4],maxiters=30),
        required_preflight=['all8 zero-step complete seed replays','protected output zero gradient and invariance under changed latent','free output nonzero finite gradient','actual shoulder/elbow/wrist FK unchanged by free latent probe'],
        admission='Review all8 selected candidate overlays and side geometry; apply the unchanged quality-first budget selector. No automatic GS and no all94frame gate.',
        no_hand_chest_contact_target=True,no_best_parts_stitching=True,new_optimizer=False,
        source_hashes={str(p):sha(p) for p in sources},quality_accepted=False,gaussian_admitted=False))
    print('PROTECTED_REFINEMENT_REGISTERED',flush=True)


def preflight():
    scope['preflight']()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    body,vposer=previous.shared.make_models(1)
    checks=[]
    for frame in read(OUT/'plan.json')['frames']:
        seed=previous.load(BASE/'frames'/frame/'winner-camera-space.npz')
        checks.append(dict(frame=frame,**adapter.verify_decoder(body,vposer,seed)))
    p=read(OUT/'preflight.json');p.update(protected_checks=checks,protected_checks_passed=True)
    save(OUT/'preflight.json',p)
    print('EIGHT_PROTECTED_JOINT_GPU_CHECKS_PASSED',flush=True)


def run():
    assert read(OUT/'preflight.json')['protected_checks_passed']
    scope['run']()


if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'run':run}[sys.argv[1]]()
