"""Reuse the native fixed-camera orbit for the fitted-input-only SH0 control."""
import argparse
import inspect
from pathlib import Path
import review_body_root_sh_orbit123 as base

ROOT = base.ROOT.parent
base.ROOT = ROOT/'body-gaussian-smplify-eight-dc18000'
base.CONTROL = ROOT/'body-gaussian-smplify-eight-dc-reviewedfit'
base.OUT = base.CONTROL/'gaussian-run/step-018000/orbit-comparison'
base.__file__ = __file__

prepare_source = inspect.getsource(base.prepare).replace('step-010000', 'step-018000').replace('step=10000', 'step=18000')
prepare_source = prepare_source.replace(
    'Matched radiance-coordinate comparison; original baseline on left, root-SH control on right.',
    'Matched18000-step SH0 comparison: early fitted-input baseline on left, reviewed fitted-input candidate on right. Both trained independently from scratch.')
replay_source = inspect.getsource(base.replay).replace('step-010000', 'step-018000')
replay_source = replay_source.replace('GaussianModel(3)', 'GaussianModel(0)').replace('model.active_sh_degree=3', 'model.active_sh_degree=0')
replay_source = replay_source.replace('MATCHED_ROOT_SH_ORBIT_READY', 'MATCHED_REVIEWED_FIT_ORBIT_READY')
exec(compile(prepare_source+'\n'+replay_source, '<reuse-native-orbit-reviewed-fit>', 'exec'), base.__dict__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'replay'])
    parser.add_argument('--pid', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        for root in (base.ROOT, base.CONTROL):
            assert base.rt.read(root/'gaussian-run/step-018000/result.json')['active_sh_degree'] == 0
        base.prepare()
        plan = base.rt.read(base.OUT/'plan.json')
        original = Path(inspect.getfile(base.run))
        plan['files'][str(original)] = base.rt.sha(original)
        plan.update(renderer_degrees=[0,0], matching_step=18000,
                    training_parameter_change='reviewed_SMPLify_X_input_source_only',
                    synthetic_views_not_ground_truth=True)
        base.rt.save(base.OUT/'plan.json', plan)
    elif args.mode == 'run':
        base.run(args.pid)
    else:
        base.replay(args.pid)
