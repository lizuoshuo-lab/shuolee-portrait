"""One matched 18k degree-zero run after the upstream SH0 warm-up check."""
import argparse,ast,copy,shutil,traceback
from pathlib import Path
import train_smplify_eight_dc123 as pilot
from run_official_gaussians123 import read,save,sha,now
OUT=pilot.OUT.parent/'body-gaussian-smplify-eight-dc18000'
LIMIT=18000

def source():
    fn,_,s=pilot.source()
    old="            if step == 1000 and not getattr(self,'geometry_failed',False):\n                decision ="
    assert s.count(old)==1
    s=s.replace(old,"            if False:\n                decision =",1)
    s=s.replace('User-requested SH degree0 single-variable1000-step control; inspect before any continuation',
                'Single matched18000-step degree0 run; original SH warm-up separately checked; no automatic follow-on')
    ns=dict(fn.__globals__);ns.update(OUT=OUT,LIMIT=LIMIT,SNAPSHOTS=[1000,10000,LIMIT])
    exec(compile(s,'<upstream-SH0-matched18000>','exec'),ns)
    return ns['train'],s

def prepare():
    assert not OUT.exists()
    p=copy.deepcopy(read(pilot.OUT/'plan.json'))
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    done=read(pilot.OUT/'gaussian-run/result.json')
    assert done['snapshot_step']==1000 and done['state']=='bounded_control_completed'
    gate=read(pilot.OUT/'gaussian-run/geometry-step-001000.json')
    assert gate['new_relative_normal_reversals']==gate['protected_face_motion_m']==0
    orbit=read(pilot.OUT/'orbit-comparison/result.json')
    assert orbit['alpha_ablation_identical'] and orbit['stats'][2]['directional_colour_range_rgb_mean']>0
    assert orbit['stats'][3]['directional_colour_range_rgb_mean']==0
    OUT.mkdir()
    for name in ['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json',
                 'smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json']:
        t=OUT/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pilot.OUT/name,t)
        assert sha(t)==sha(pilot.OUT/name)
    p.update(updated=now(),state='prepared',comparison_budget=LIMIT,snapshots=[1000,10000,LIMIT],
        baseline_snapshot_step=18000,runtime_budget_seconds=1800,
        prerequisite='1000-step upstream warm-up uses DC in both runs and is only an integration/stability check. Same18k geometry SH3/DC inference ablation confirms directional variation without alpha changes.',
        only_parameter_change='Original model sh_degree3->0. Same8 inputs, SMPLify meshes, normalized clothing-offset Laplacian, cameras, Gaussian and vertex LRs, losses and original density schedule.',
        from_scratch=True,optimizer_state_reused=False,old_gaussian_checkpoint_loaded=False,
        actual_optimizer_updates_at_saved_step='Original save occurs before optimizer update: step18000 snapshot has17999 updates.',
        review_gate='All59 geometry checks every100 steps. At18k native body/face/hands/feet and fixed36 novel cameras; no auto acceptance or automatic next training.',
        admission='Same-step reference10000/18000. Colour directionality must be zero by representation; spatial speckle, pose defects, unseen surfaces and face errors are separate quality failures.',
        supersedes='Training-only bounded SH0 warm-up; starts fresh, does not resume its checkpoint',
        training_function='Pinned upstream train.py::training unchanged; local bounded runner and original SH-degree option',
        quality_accepted=False,model_promoted=False,end_to_end_15min_accepted=False)
    p['source_hashes'][str(Path(__file__).resolve())]=sha(__file__)
    save(OUT/'plan.json',p)
    _,s=source();ast.parse(s);(OUT/'adapter.py').write_text(s)
    save(OUT/'progress.json',dict(updated=now(),state='prepared',pid=None,step=0,quality_accepted=False))
    print('MATCHED18K_DC_ONLY_PREPARED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','train']);p.add_argument('--pid',type=int);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='run':
            control=pilot.baseline.eight.direct.control;control.OUT=OUT;control.__file__=__file__;control.run(a.pid)
        else:source()[0]()
    except BaseException as e:
        if OUT.exists():save(OUT/(a.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
