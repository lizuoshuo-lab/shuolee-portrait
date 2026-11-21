"""One bounded offset-LR control after the frozen same-state gradient audit."""
import argparse
import ast
import copy
import difflib
from pathlib import Path
import shutil
import traceback
import train_smplify_eight_reviewedfit123 as previous
from run_official_gaussians123 import read,save,sha,now

OUT=previous.OUT.parent/'body-gaussian-smplify-eight-offsetlr-control'
DIAG=previous.OUT/'frozen-rgb-gradient-diagnostic'
LIMIT=1000


def source():
    fn,_,old=previous.source();s=old
    assert s.count("'lr': 1.e-6")==1
    s=s.replace("'lr': 1.e-6","'lr': 1.e-5")
    s=s.replace("g['lr'] == 1.e-6","g['lr'] == 1.e-5")
    s=s.replace('lr=1.e-6','lr=1.e-5')
    s=s.replace('shared_static_offset_lr=1.e-6','shared_static_offset_lr=1.e-5')
    s=s.replace("    assert torch.equal(model.static_offset.detach().cpu(),", "    offset_receipt['optimizer_lr']=1e-5\n    assert torch.equal(model.static_offset.detach().cpu(),")
    s=s.replace('Shared clothing-only canonical offset at1e-6 from zero','Shared clothing-only canonical offset at1e-5 from zero')
    s=s.replace('Gaussian options and SH0 unchanged; only reviewed fitted-input source changed','Shared clothing static_offset LR1e-6 to1e-5; all other settings and inputs unchanged')
    s=s.replace('reviewed fitted-input archive relative to matched eight-frame SH0 run','shared static_offset LR1e-6 to1e-5 relative to reviewed-input SH0')
    s=s.replace('Reviewed-SMPLify-fit input-only SH0 candidate; inspect at1000, bounded18000','One offset-LR control, stop at1000 or first geometry failure; no automatic continuation')
    ns=dict(fn.__globals__);ns.update(OUT=OUT,LIMIT=LIMIT,SNAPSHOTS=[1000])
    exec(compile(s,'<offset-LR-only1000-control>','exec'),ns)
    return ns['train'],old,s


def prepare():
    assert not OUT.exists(),'Never repeat or resume this control.'
    evidence=read(DIAG/'result.json');review=read(DIAG/'visual-review.json')
    assert evidence['parameter_tensors_bitwise_unchanged'] and len(evidence['gradients'])==8
    assert review['actual_visual_review'] and review['reject_render_only_clamp']
    assert all(r['components']['photometric']['parameters']['mesh_offset_m']['norm']>0 for r in evidence['gradients'])
    p=copy.deepcopy(read(previous.OUT/'plan.json'))
    for path,h in p['source_hashes'].items():assert sha(path)==h,path
    OUT.mkdir()
    for name in ['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json','selection.json','selection-rule.json','smplify-gaussian-interface-receipt.json']:
        dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(previous.OUT/name,dest)
        assert sha(dest)==sha(previous.OUT/name)
    p.update(updated=now(),state='registered_before_training',baseline_run=str(previous.OUT),
        hypothesis='RGB gradients reach clothing mesh but saved-state predicted steps are sub-micrometre. Test whether a10x offset-group LR makes useful mesh progress without fold/face damage, holding the objective and Gaussian LR fixed.',
        only_parameter_change='Shared clothing static_offset Adam LR1e-6 ->1e-5. Keep original upstream training function, same fixed8 roster, reviewed SMPLify-X input, SH0, full-image RGB, xyz/scale, shared Laplacian1e6, density schedule and all Gaussian LRs.',
        source_fit_quality_accepted=False,shared_static_offset_lr=1e-5,baseline_static_offset_lr=1e-6,
        source_pose_frames_changed=[],selected_training_frames_changed=[],
        comparison_budget=1000,snapshots=[1000],runtime_budget_seconds=360,
        review_gate='Stop at1000 regardless of result; compare same-step all59 native source/mesh/Gaussian and masks. Face gradients/motion must remain0, relative normal reversals0, area ratio>.1, displacement<20mm. Save the first failed state.',
        no_automatic_new_experiment=True,no_coefficient_sweep=True,automatic_continuation=False,
        no_gaussian_center_teacher=True,no_RGB_range_clamp=True,
        from_scratch=True,initial_points=20908,optimizer_state_reused=False,old_gaussian_checkpoint_loaded=False,
        risk='Larger displacements may fold thin triangles or increase heldout pose errors; gradient existence and larger movement do not prove clothing fit.',
        known_limitation='This is a single optimization-scale diagnostic, not proof of surface observability, full-body quality or15min end-to-end. No interpolation or view filtering.',
        quality_accepted=False,model_promoted=False,end_to_end_15min_accepted=False)
    for path in [Path(__file__).resolve(),DIAG/'result.json',DIAG/'visual-review.json']:
        p['source_hashes'][str(path)]=sha(path)
    save(OUT/'plan.json',p)
    _,old,new=source();ast.parse(new)
    (OUT/'adapter.py').write_text(new)
    (OUT/'offset-lr-only.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))))
    save(OUT/'progress.json',dict(updated=now(),state='prepared',step=0,quality_accepted=False))
    print('OFFSET_LR_CONTROL_PREPARED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','train']);p.add_argument('--pid',type=int);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='train':source()[0]()
        else:
            control=previous.pilot.baseline.eight.direct.control
            control.OUT=OUT;control.__file__=__file__;control.run(a.pid)
    except BaseException as e:
        if OUT.exists():save(OUT/(a.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
