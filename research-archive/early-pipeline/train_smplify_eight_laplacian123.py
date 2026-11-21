"""Minimal shared-offset adapter for the existing upstream Laplacian hook.

The eight-image roster, optimizer, cameras, poses and facial exclusion remain
fixed. No implementation change to the live upstream trainer or original repo.
"""
from pathlib import Path
import argparse, ast, copy, difflib, shutil, traceback
import train_smplify_eight123 as eight
from probe_smplify_eight_laplacian123 import ROOT as OUT, LAP, WEIGHT
from run_official_gaussians123 import read, save, sha, now

LIMIT=18000

def source():
    fn,_,old=eight.source();s=old
    def replace(a,b):
        nonlocal s
        assert s.count(a)==1,a;s=s.replace(a,b,1)
    replace("    assert vars(opt) == plan['options']", """    original_options=dict(vars(opt))
    assert original_options==plan['upstream_options']
    opt.lambda_laplacian=plan['shared_laplacian_weight']
    assert vars(opt)==plan['options']""")
    replace('        shared_offset_vertices = clothing_vertices', """        shared_offset_vertices = clothing_vertices
        def compute_laplacian_loss(self):
            from facial_geometry_boundary123 import clothing_only_offset
            safe=clothing_only_offset(self.static_offset,self.surface_clothing,self.surface_protected)
            return torch.sparse.mm(self.shared_laplacian,safe).square().sum(-1).mean()""")
    replace('    initial_offset = model.static_offset.detach().clone()', """    initial_offset = model.static_offset.detach().clone()
    with np.load(plan['laplacian_cache'],allow_pickle=False) as z: lap=dict(z)
    assert np.array_equal(lap['faces'],data['faces'])
    model.shared_laplacian=torch.sparse_coo_tensor(torch.tensor(lap['indices'],device='cuda'),
        torch.tensor(lap['values'],device='cuda'),tuple(lap['shape'])).coalesce()
    assert float(model.compute_laplacian_loss())==0""")
    replace('            if step == 1000:\n                suspended_baseline()', "            if step == LIMIT and not getattr(self,'geometry_failed',False):\n                suspended_baseline()")
    replace("stop_reason='Preregistered1000-step eight-image roster comparison'", "stop_reason='Preregistered eight-frame shared-Laplacian bounded candidate; no further automatic run'")
    replace('            if False:\n                decision =', "            if step == 1000 and not getattr(self,'geometry_failed',False):\n                decision =")
    replace("                while not decision.exists(): time.sleep(2)", """                deadline=time.monotonic()+600
                while not decision.exists():
                    assert time.monotonic()<deadline,'Review timeout; checkpoint retained'
                    time.sleep(2)""")
    replace("            assert gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0", """            save(run/f'geometry-step-{step:06d}.json',dict(updated=now(),step=step,**gate))
            failed=(gate['protected_face_motion_m']!=0 or gate['protected_face_gradient']!=0 or
                gate['new_relative_normal_reversals']!=0 or gate['minimum_area_ratio']<=.1 or gate['max_added_offset_m']>=.02)
            if failed:
                # Keep the first actual failed state before stopping; never replay for artifacts.
                scene.geometry_failed=True
                if step not in saved: scene.save(step)
                save(run/'result.json',dict(updated=now(),state='geometry_stopped',step=step,
                    geometry=gate,seconds=time.monotonic()-started,quality_accepted=False))
                status('geometry_stopped',step,geometry=gate)
                raise ControlComplete()
            assert gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0""")
    replace('                original_optimization_defaults_equal=True,', "                original_optimization_defaults_equal=False, only_original_loss_option_changed='lambda_laplacian0->1e6',")
    replace('        unmodified_training_function=True, unchanged_optimization_defaults=True,', "        unmodified_training_function=True, unchanged_optimization_defaults=False, only_option_changed='lambda_laplacian',")
    ns=dict(fn.__globals__);ns.update(OUT=OUT,LIMIT=LIMIT,SNAPSHOTS=[900,1000,10000,LIMIT])
    exec(compile(s,'<shared-clothing-offset-original-laplacian-hook>','exec'),ns)
    return ns['train'],old,s

def prepare():
    assert not (OUT/'plan.json').exists()
    pre=read(OUT/'precheck-result.json');assert pre['state']=='cpu_contract_passed'
    p=copy.deepcopy(read(eight.OUT/'plan.json'))
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    for name in ['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json',
                 'smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json']:
        target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(eight.OUT/name,target)
        assert sha(target)==sha(eight.OUT/name)
    p['upstream_options']=copy.deepcopy(p['options'])
    p['options']['lambda_laplacian']=WEIGHT
    p.update(updated=now(),state='prepared',hypothesis='Original normalized shared-displacement smoothness may prevent localized thin-triangle reversals with the same8 RGB frames. It must not be accepted solely by suppressing mesh movement.',
        baseline_run=str(eight.OUT),shared_laplacian_weight=WEIGHT,laplacian_cache=str(LAP),
        weight_source='Original VHAP normalized-offset default1e6; original GA lambda_laplacian0. No coefficient search.',
        only_parameter_change='Enable upstream Laplacian loss hook for the existing shared clothing-only static_offset. Gaussian LR/density/RGB losses and8-frame selection unchanged.',
        implementation='Original GA training function unchanged. Local Body.compute_laplacian_loss maps the previously verified original sparse formula to shared static offset; original GA method only covered dynamic_offset.',
        comparison_budget=LIMIT,snapshots=[900,1000,10000,LIMIT],runtime_budget_seconds=2400,
        review_gate='At1000 inspect all59 same-camera body/face/hands/feet and selected8 vs excluded51 separately. Continue this same candidate only if no new gross distortion, face motion or topology failure; quality acceptance remains separate.',
        stop='Same all59 relative-normal/area/20mm/face-zero guards; save failed state before stopping. Do not relax gates or retry.',
        no_coefficient_sweep=True,auto_retry=False,from_scratch=True,optimizer_state_reused=False,
        admission='No visual model promotion without final quality/package/UI checks. No end-to-end15min claim from this Gaussian-stage timing.',
        quality_accepted=False,end_to_end_15min_accepted=False)
    for f in [Path(__file__).resolve(),LAP,OUT/'precheck-result.json']:
        p['source_hashes'][str(f)]=sha(f)
    save(OUT/'plan.json',p)
    _,old,new=source();ast.parse(new)
    (OUT/'shared-laplacian-adapter.py').write_text(new)
    (OUT/'single-change.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))))
    save(OUT/'progress.json',dict(updated=now(),state='prepared',pid=None,quality_accepted=False))
    print('EIGHT_SHARED_LAPLACIAN_PREPARED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','train']);ap.add_argument('--pid',type=int);args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='run':
            control=eight.direct.control;control.OUT=OUT;control.__file__=__file__;control.run(args.pid)
        else:source()[0]()
    except BaseException as e:
        if OUT.exists():save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
