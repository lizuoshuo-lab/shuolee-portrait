"""One from-scratch Gaussian control of the original 600k -> 18k clock.

The exact target cache is a separately measured equivalent input interface.
No pose solver, pixel, mask, loss, LR amplitude or densifier is replaced.
"""
import copy
import difflib
from pathlib import Path
import shutil
import sys
import cross_subject_sh0124 as gs
from cache_training_targets124 import adapt

SUBJECT='winterhat-clock18k-cache124'
BASE=gs.ROOT/'winterhat-selected-sh0-pipeline124'
OUT=gs.ROOT/SUBJECT
OVERRIDES=dict(iterations=18000,position_lr_max_steps=18000,
    densify_from_iter=300,densification_interval=60,
    densify_until_iter=18000,opacity_reset_interval=1800)


def prepare():
    assert not OUT.exists(),'Preserve registered experiments'
    for name in ['target-cache-control124','cached-native-step-control124']:
        result=gs.read(gs.ROOT/name/'result.json')
        assert result['all_source_hashes_unchanged']
        assert not result['quality_accepted']
    baseline=gs.read(BASE/'plan.json')
    for path,digest in baseline['source_hashes'].items():assert gs.sha(path)==digest,path
    assert gs.read(BASE/'process-exit.json')['exit_code']==0
    assert gs.read(BASE/'gaussian-run/step-018000/visual-review.json')['actual_visual_review']
    source=(BASE/'adapter.py').read_text()
    cached=adapt(source)
    marker="    opt.lambda_laplacian=plan['shared_laplacian_weight']\n"
    assert cached.count(marker)==1
    changed=cached.replace(marker,marker+"    for name, value in plan['schedule_overrides'].items():\n        setattr(opt, name, value)\n")
    changed=changed.replace('Cross-subject frozen-SH0 configuration; inspect1000, bounded18000; cached subject fit',
        'Original600k clock scaled to18k; exact GPU target cache; inspect1000, bounded18000; cached subject fit')
    changed=changed.replace('different subject inputs; same SH0 training configuration',
        'Same winterhat inputs, original training-clock scaled600k-to18k and separately verified exact target cache')
    changed=changed.replace('Gaussian options and SH0 unchanged; only reviewed fitted-input source changed',
        'Only six original time-valued options changed as one clock profile; exact target cache enabled; loss and LR amplitudes unchanged')
    (OUT/'fit').mkdir(parents=True);(OUT/'data').mkdir()
    for name in ['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json',
                 'smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json']:
        shutil.copyfile(BASE/name,OUT/name)
        assert gs.sha(BASE/name)==gs.sha(OUT/name)
    (OUT/'adapter.py').write_text(changed)
    (OUT/'target-cache-interface.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),cached.splitlines(True))))
    (OUT/'clock-profile.diff').write_text(''.join(difflib.unified_diff(cached.splitlines(True),changed.splitlines(True))))
    p=copy.deepcopy(baseline)
    p.update(updated=gs.now(),state='registered_ready',subject=SUBJECT,label='冬帽人物 · 原版时序换算18k＋GPU缓存对照',
        only_parameter_change='Original time-valued parameters scaled by18000/600000; exact target storage cache independently validated.',
        schedule_overrides=OVERRIDES,source_fit=str(BASE/'fit/fitted.npz'),baseline_run=str(BASE),
        from_scratch=True,gaussian_parameters_reused=False,optimizer_state_reused=False,
        source_fitted_pose_reused_for_control=True,new_pose_fitting=False,original_densifier_unchanged=True,
        learning_rate_amplitudes_unchanged=True,losses_unchanged=True,native_pixels_masks_and_K_unchanged=True,
        cache_parity_proof=str(gs.ROOT/'cached-native-step-control124/result.json'),
        goal_points_approx=200000,goal_point_band=[170000,230000],max_points_safety_stop=300000,
        point_budget_behavior='Original safety stop at300k, no post-hoc deletion or invented geometry. Target band is a goal, not guaranteed.',
        expected_densify_calls_through1000=11,expected_densify_calls_before18000=294,
        comparison_budget=18000,snapshots=[1000,10000,18000],record_id='001',
        review_gate='1000 actual native all8 plus geometry and point counts; continue same live optimizer only without new severe degeneration. End by18000 or original guard. No retries or threshold sweep.',
        risks=['More early splits and resets may cause fragments or temporary opacity loss.',
               'Full position-LR decay now fits18k; this changes the whole clock profile and is not isolated to one schedule field.',
               'Runtime/memory may rise as points grow;200k and cold-video15min remain unverified.',
               'Existing initialization and unseen-region uncertainties remain.'],
        quality_accepted=False,end_to_end_15min_accepted=False,model_promoted=False)
    p['options'].update(OVERRIDES)
    sources=[Path(__file__),gs.HERE/'cache_training_targets124.py',gs.HERE/'native_target_cache124.py',
        BASE/'plan.json',BASE/'adapter.py',gs.ROOT/'target-cache-control124/result.json',gs.ROOT/'cached-native-step-control124/result.json']
    sources += [OUT/name for name in ['adapter.py','fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json','smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json']]
    p['source_hashes'].update({str(f):gs.sha(f) for f in sources})
    gs.save(OUT/'plan.json',p);gs.save(OUT/'progress.json',dict(updated=gs.now(),state='registered_ready',pid=None,step=0))
    print('WINTERHAT_CLOCK18K_READY',OVERRIDES,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='train':gs.train(SUBJECT)
    else:raise ValueError('prepare or train required')
