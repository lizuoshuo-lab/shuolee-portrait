"""Reuse existing immutable-pair native orbit review for matched SH3/DC18k."""
import argparse,inspect
from pathlib import Path
import review_body_root_sh_orbit123 as base

BASE_ROOT=base.ROOT
base.ROOT=BASE_ROOT.parent/'body-gaussian-smplify-eight-laplacian'
base.CONTROL=BASE_ROOT.parent/'body-gaussian-smplify-eight-dc18000'
base.OUT=base.CONTROL/'gaussian-run/step-018000/orbit-comparison'
base.__file__=__file__
prepare_source=inspect.getsource(base.prepare).replace('step-010000','step-018000').replace('step=10000','step=18000').replace('Matched radiance-coordinate comparison; original baseline on left, root-SH control on right.','Matched18000-step SH-degree-only comparison; original SH3 on left, newly trained DC-only on right.')
replay_source=inspect.getsource(base.replay).replace('step-010000','step-018000')
replay_source=replay_source.replace('for run in (ROOT,CONTROL):','for run,degree in ((ROOT,3),(CONTROL,0)):').replace('model=GaussianModel(3);','model=GaussianModel(degree);').replace('model.active_sh_degree=3;','model.active_sh_degree=degree;')
replay_source=replay_source.replace('MATCHED_ROOT_SH_ORBIT_READY','MATCHED18K_SH0_ORBIT_READY')
ns=base.__dict__
exec(compile(prepare_source+'\n'+replay_source,'<reuse-native-orbit-SH-degree-control>','exec'),ns)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','replay']);p.add_argument('--pid',type=int);a=p.parse_args()
    if a.mode=='prepare':
        base.prepare()
        plan=base.rt.read(base.OUT/'plan.json')
        original=Path(inspect.getfile(base.run));plan['files'][str(original)]=base.rt.sha(original)
        plan.update(renderer_degrees=[3,0],matching_step=18000,training_parameter_change='max_sh_degree3->0',synthetic_views_not_ground_truth=True)
        base.rt.save(base.OUT/'plan.json',plan)
    elif a.mode=='run':base.run(a.pid)
    else:base.replay(a.pid)
