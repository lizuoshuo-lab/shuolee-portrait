"""One1000-step matched-background control; original loop and loss retained."""
import ast
import copy
import difflib
from pathlib import Path
import shutil
import sys
import traceback
import cross_subject_sh0124 as gs

SUBJECT='indoor-paired-background124'
BASE=gs.ROOT/'indoor-zoom-clock18k124'
OUT=gs.ROOT/SUBJECT


def prepare():
    assert not OUT.exists()
    baseline=gs.read(BASE/'plan.json')
    for p,h in baseline['source_hashes'].items():assert gs.sha(p)==h,p
    assert gs.read(gs.ROOT/'indoor-dark-fragments124/visual-review.json')['actual_visual_review']
    source=(BASE/'adapter.py').read_text();changed=source
    replacements={
      'def train():\n':'def train():\n    LIMIT = 1000\n    SNAPSHOTS = [1000]\n',
      '    namespace=dict(torch=torch, F=F, DataLoader=loader, prepare_output_and_logger=lambda d:None,':
        "    from paired_training_background124 import PairedTrainingBackground\n    paired_background = PairedTrainingBackground(render, masks, [r['frame'] for r in rows], target_cache)\n    namespace=dict(torch=torch, F=F, DataLoader=loader, prepare_output_and_logger=lambda d:None,",
      '        render=render, l1_loss=l1_loss, ssim=ssim, training_report=report,':
        '        render=paired_background, l1_loss=l1_loss, ssim=ssim, training_report=report,',
      "            saved.append(step)\n":
        "            saved.append(step)\n            save(folder/'background-receipt.json', paired_background.receipt())\n            assert paired_background.calls == step\n",
      'Same reviewed indoor ROI2x; original600k clock scaled18k;1000review and hard18k budget':
        'Same indoor ROI2x and18k clock; only matched stochastic background; hard stop1000',
      'Only six original time-valued options scaled600k-to18k from reviewed ROI2x control; LR amplitudes and losses unchanged':
        'Only paired stochastic training background relative to indoor zoom clock; original RGB loss and all options unchanged',
      'Same indoor ROI2x and known labels; original density algorithm with time-valued options scaled18k':
        'Same indoor ROI2x and18k clock; training renderer and target share a seeded stochastic background',
    }
    for old,new in replacements.items():assert changed.count(old)==1,old;changed=changed.replace(old,new)
    ast.parse(changed)
    names=['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json',
           'smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json',
           'source-native-inputs.json','pixel-mapping.json','loss_utils_validity.py',
           'validity-review.json','validity-cuda-preflight.json']
    for folder in ('fit','data'):(OUT/folder).mkdir(parents=True)
    for name in names:
        shutil.copyfile(BASE/name,OUT/name);assert gs.sha(BASE/name)==gs.sha(OUT/name)
    (OUT/'adapter.py').write_text(changed)
    (OUT/'paired-background.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),fromfile='frozen-zoom-clock',tofile='paired-background')))
    p=copy.deepcopy(baseline)
    p.update(updated=gs.now(),state='registered_preflight_pending',subject=SUBJECT,label='室内人物 · 配对随机背景1000步',
        baseline_run=str(BASE),comparison_budget=1000,snapshots=[1000],
        hypothesis='A fixed black composite lets opaque black Gaussians hide projected silhouette mismatch. Matched stochastic backgrounds may reduce this camouflage without changing pose or adding a new loss.',
        only_parameter_change='Training background constant black -> seeded uniformRGB; target composite uses precisely the same sampled background.',
        causal_scope='One paired background augmentation; keep derived cameras,6poses,2x inputs,masks/validity,SH0,18kclock,loss and LR amplitudes unchanged. Random-background contrasts alter gradients, a hypothesized effect rather than a bug fix.',
        fixed='Same immutable input geometry,20908 initial Gaussians,empty optimizer,global RNGseed0,reference105 and6sourceviews; same original training() function.',
        source_fitted_pose_reused_for_control=True,from_scratch=True,optimizer_state_reused=False,gaussian_parameters_reused=False,
        image_loss='Original0.8L1+0.2DSSIM,known pixels only; random backgrounds paired between render and masked target; unknown pixels excluded with full SSIM neighborhood support.',
        criteria='Hard stop1000. Actualall6same-camera and four frozen novel cameras. Reject if new body/face collapse, relative mesh folds,unknown supervision leakage or export parity failure. No automatic18k extension or random seed/weight scan.',
        expected_densify_calls_through1000=11,
        risks=['Boundary contrast may amplify mask or pose errors and create floats or slower convergence.',
               'Fixed black camouflage may be only one failure mode; true clothing geometry and unknown views may remain incorrect.'],
        quality_accepted=False,end_to_end_15min_accepted=False,model_promoted=False)
    paths=[Path(__file__),Path(__file__).with_name('paired_training_background124.py'),OUT/'adapter.py',
           BASE/'gaussian-run/step-001000/result.json',gs.ROOT/'indoor-directional-support124/result.json',
           gs.ROOT/'indoor-dark-fragments124/result.json',gs.ROOT/'indoor-dark-fragments124/visual-review.json']
    paths += [OUT/name for name in names]
    p['source_hashes'].update({str(path):gs.sha(path) for path in paths})
    gs.save(OUT/'plan.json',p)
    gs.save(OUT/'progress.json',dict(updated=gs.now(),state='registered_preflight_pending',step=0,pid=None,budget=1000))
    print('PAIRED_BACKGROUND_CONTROL_PREPARED',flush=True)


if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='train':
            assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
            assert gs.read(OUT/'background-preflight.json')['passed']
            gs.train(SUBJECT)
        else:raise ValueError('prepare or train')
    except BaseException as e:
        if OUT.exists():gs.save(OUT/(sys.argv[1]+'-failure.json'),dict(updated=gs.now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
