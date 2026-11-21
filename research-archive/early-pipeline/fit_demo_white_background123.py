"""Prepared one-setting control using the unchanged selected-reference runner.

Execution is gated on a recorded human decision. This does not invent a new
loss, optimizer, stage schedule, reference frame, or camera.
"""
import argparse,dataclasses
from pathlib import Path
from run_official_gaussians123 import read,save,sha,now
import fit_selected_vhap_demo123 as original

PLAN=original.ROOT/'background-white-control-plan.json'
DECISION=original.ROOT/'background-white-control-decision.json'
OUT=original.ROOT/'vhap-selected-reference-white'

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare-only',action='store_true');args=args.parse_args()
    baseline=original.ROOT/'vhap-selected-reference'
    if args.prepare_only:
        assert read(baseline/'visual-review.json')['state']=='reviewed_failed'
        save(PLAN,dict(updated=now(),state='prepared_pending_user_decision',
            baseline=str(baseline),output=str(OUT),changed_setting={'render.background_train':['target','white']},
            unchanged=['selected slot32/source256','all59 frames','native2704x1520','conditional K',
                'reference 2500 updates, sequential50/frame, global30 epochs','upstream optimizer/loss/regularizers',
                'shared shape/canonical offset/texture','hand repair deferred'],
            fresh_initialization=True,baseline_parameters_not_loaded=True,max_updates=7220,
            expected_fit_time_seconds=read(baseline/'result.json')['elapsed_seconds'],
            stop_on=['nonfinite parameters','existing two-hour process guard','completed7220 updates'],
            acceptance='Actual all59 body/face review, compare mask coverage and offset regions, no new severe folding; do not auto-start Gaussians.',
            risks=['Mask errors, held device, hat and footwear can pull the surface incorrectly',
                'Exposing uncovered clothing error may increase surface expansion or compensate wrong body orientation',
                'This control may not solve the pose ambiguity'],
            wrapper_sha256=sha(__file__),original_runner_sha256=sha(Path(original.__file__)),
            adapter_sha256=sha(Path(original.__file__).with_name('vhap_demo_adapter123.py'))))
        print('prepared, no fitting started');return
    p=read(PLAN);decision=read(DECISION)
    assert decision.get('user_authorized') is True and decision.get('changed_setting')==p['changed_setting']
    assert sha(__file__)==p['wrapper_sha256']
    assert sha(Path(original.__file__))==p['original_runner_sha256']
    assert sha(Path(original.__file__).with_name('vhap_demo_adapter123.py'))==p['adapter_sha256']
    make=original.make_tracker
    def adapted(output):
        tracker,rows=make(output)
        before=dataclasses.asdict(tracker.cfg)
        assert tracker.cfg.render.background_train=='target'
        tracker.cfg.render.background_train='white'
        after=dataclasses.asdict(tracker.cfg)
        before['render']['background_train']='white'
        assert before==after,'Unexpected extra config change.'
        save(output/'single-variable-control.json',dict(plan=p,decision=decision,original_config_except_background_unchanged=True,
            wrapper_sha256=sha(__file__),optimizer_updates_so_far=0))
        return tracker,rows
    original.OUT=OUT;original.make_tracker=adapted
    original.main()

if __name__=='__main__':main()
