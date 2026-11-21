"""One original-clock-to18k control, on the reviewed indoor ROI2x input."""
import ast
import copy
import difflib
from pathlib import Path
import shutil
import sys
import traceback
import cross_subject_sh0124 as gs

SUBJECT='indoor-zoom-clock18k124'
BASE=gs.ROOT/'indoor-person-zoom2x124'
OUT=gs.ROOT/SUBJECT
OVERRIDES=dict(iterations=18000,position_lr_max_steps=18000,densify_from_iter=300,
    densification_interval=60,densify_until_iter=18000,opacity_reset_interval=1800)

def prepare():
    assert not OUT.exists(),'No duplicate candidate'
    baseline=gs.read(BASE/'plan.json')
    for path,digest in baseline['source_hashes'].items():assert gs.sha(path)==digest,path
    assert gs.read(BASE/'gaussian-run/step-001000/visual-review.json')['actual_visual_review']
    audit=gs.read(gs.ROOT/'indoor-surface-support-audit124/result.json')
    assert audit['source_hashes_unchanged'] and audit['optimizer_updates']==0
    source=(BASE/'adapter.py').read_text()
    marker="    opt.lambda_laplacian=plan['shared_laplacian_weight']\n"
    assert source.count(marker)==1
    changed=source.replace(marker,marker+"    for name,value in plan['schedule_overrides'].items():\n        setattr(opt,name,value)\n")
    edits={
        'Person ROI plus2x sampling control; hard stop1000; source fit reused only for input-scale comparison':
            'Same reviewed indoor ROI2x; original600k clock scaled18k;1000review and hard18k budget',
        'Only deterministic person ROI plus2x input sampling and corresponding fullK; original SH0 optimizer and density unchanged':
            'Only six original time-valued options scaled600k-to18k from reviewed ROI2x control; LR amplitudes and losses unchanged',
        'Indoor ROI-plus2x inputs with preserved known/unknown labels; original SH0 optimizer and density schedule':
            'Same indoor ROI2x and known labels; original density algorithm with time-valued options scaled18k',
    }
    for old,new in edits.items():assert changed.count(old)==1,old;changed=changed.replace(old,new)
    ast.parse(changed)
    for folder in ('fit','data'):(OUT/folder).mkdir(parents=True)
    names=['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json',
        'smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json',
        'source-native-inputs.json','pixel-mapping.json','loss_utils_validity.py']
    for name in names:
        shutil.copyfile(BASE/name,OUT/name);assert gs.sha(BASE/name)==gs.sha(OUT/name)
    (OUT/'adapter.py').write_text(changed)
    (OUT/'clock-profile.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),fromfile='frozen-roi2x',tofile='original-clock18k')))
    inputs=gs.read(BASE/'validity-review.json');inputs.update(updated=gs.now(),
        inherited_from=str(BASE/'validity-review.json'),same_six_pixels_masks_validity_geometry=True,
        scope='Identical already reviewed inputs; timing options only change. No new physical calibration claim.')
    gs.save(OUT/'validity-review.json',inputs)
    preflight=gs.read(BASE/'validity-cuda-preflight.json');preflight.update(inherited_from=str(BASE/'validity-cuda-preflight.json'),
        timing_only_candidate=True,new_zero_update_test_claimed=False)
    gs.save(OUT/'validity-cuda-preflight.json',preflight)
    p=copy.deepcopy(baseline)
    p.update(updated=gs.now(),state='registered_ready',subject=SUBJECT,
        label='室内人物 · 人物放大＋原版时序换算18k',baseline_run=str(BASE),
        user_authorization='Continue active reconstruction goal; user delegated necessary minimal training decisions and requested enlarged-person training.',
        hypothesis='Original600k clock gives only3 density events before18k; scaled clock may improve observed shirt/arm surface support and novel-view continuity.',
        causal_scope='Six related time-valued options are one clock profile. Input ROI2x is held fixed relative to the previous1000-step control.18k comparison to whole-frame baseline is a combined input-and-clock effect.',
        fixed='Same6 frames and derived pixels/masks/unknown labels/fullK, same source fit/reference105, SH0, original loss definitions and weights, LR amplitudes, initial20908 points, RGB cache.',
        only_parameter_change='Original clock600000->18000; no density thresholds, pose solver or loss changes.',
        schedule_overrides=OVERRIDES,comparison_budget=18000,snapshots=[1000,10000,18000],
        from_scratch=True,optimizer_state_reused=False,gaussian_parameters_reused=False,
        source_fitted_pose_reused_for_control=True,goal_points_approx=200000,
        goal_point_band=[170000,230000],max_points_safety_stop=300000,
        expected_densify_calls_through1000=11,expected_densify_calls_before18000=294,
        criteria='Review all6 at1000; continue same live optimizer only without new severe visible body/face deterioration. Final same6 plus36 novel directions and paired export. Hidden legs excluded.',
        risks=['Earlier/more frequent split/prune and opacity resets may create fragments or transient opacity loss.',
               'Earlier LR decay changes convergence along with density clock; do not attribute outcome to density alone.',
               'Original mask and pose errors remain; higher point count does not prove better geometry.'],
        quality_accepted=False,end_to_end_15min_accepted=False,model_promoted=False)
    p['options'].update(OVERRIDES)
    paths=[Path(__file__),OUT/'adapter.py',BASE/'plan.json',gs.ROOT/'indoor-surface-support-audit124/result.json']
    paths += [OUT/n for n in names]+[OUT/'validity-review.json',OUT/'validity-cuda-preflight.json']
    p['source_hashes'].update({str(path):gs.sha(path) for path in paths})
    gs.save(OUT/'plan.json',p);gs.save(OUT/'progress.json',dict(updated=gs.now(),state='registered_ready',pid=None,step=0,budget=18000))
    print('INDOOR_ZOOM_CLOCK_READY',OVERRIDES,flush=True)

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='train':
            assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
            gs.train(SUBJECT)
        else:raise ValueError('prepare or train')
    except BaseException as e:
        if OUT.exists():gs.save(OUT/(sys.argv[1]+'-failure.json'),dict(updated=gs.now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
