"""User-requested view-independent colour control using upstream SH degree0.

Only the original model SH-degree option changes. No renderer clamp, coefficient
rewrite of an old PLY, or change to geometry/loss/camera/training-frame settings.
"""
from pathlib import Path
import argparse,ast,copy,difflib,shutil,traceback
import train_smplify_eight_laplacian123 as baseline
from run_official_gaussians123 import read,save,sha,now

OUT=baseline.OUT.parent/'body-gaussian-smplify-eight-dc'
LIMIT=1000

def source():
    fn,_,old=baseline.source();s=old
    def replace(a,b):
        nonlocal s
        assert s.count(a)==1,a;s=s.replace(a,b,1)
    replace('    model = Body(3)','    model = Body(0)')
    replace('            plain = official.GaussianModel(3);','            plain = official.GaussianModel(0);')
    replace('    dataset=SimpleNamespace(bind_to_mesh=True, sh_degree=3,','    dataset=SimpleNamespace(bind_to_mesh=True, sh_degree=0,')
    replace("only_original_loss_option_changed='lambda_laplacian0->1e6'", "only_original_loss_option_changed='shared Laplacian unchanged from reference; model SH degree3->0'")
    replace("only_option_changed='lambda_laplacian'", "only_option_changed='model SH degree3->0 relative to matched eight-frame Laplacian run'")
    replace("stop_reason='Preregistered eight-frame shared-Laplacian bounded candidate; no further automatic run'", "stop_reason='User-requested SH degree0 single-variable1000-step control; inspect before any continuation'")
    replace("        assert torch.isfinite(losses['total'])", """        assert model.max_sh_degree==0 and model.active_sh_degree==0
        assert model._features_rest.numel()==0
        assert torch.isfinite(losses['total'])""")
    ns=dict(fn.__globals__);ns.update(OUT=OUT,LIMIT=LIMIT,SNAPSHOTS=[LIMIT])
    exec(compile(s,'<original-SH-degree-zero-only-control>','exec'),ns)
    return ns['train'],old,s

def prepare():
    assert not OUT.exists(),'Do not silently restart the colour control.'
    p=copy.deepcopy(read(baseline.OUT/'plan.json'))
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    OUT.mkdir()
    for name in ['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json',
                 'smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json']:
        target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(baseline.OUT/name,target)
        assert sha(target)==sha(baseline.OUT/name)
    p.update(updated=now(),state='prepared_waiting_for_current_GPU_candidate',
        user_authorization='限制球谐函数的颜色，现在各向异性颜色还是太强',
        hypothesis='Removing all directional SH terms prevents learned view-dependent colour compensation. It does not repair wrong poses, missing geometry or unobserved surfaces.',
        baseline_run=str(baseline.OUT),baseline_snapshot_step=1000,model_sh_degree=0,baseline_model_sh_degree=3,
        only_parameter_change='Use original model sh_degree=0 instead of3. Same8 RGB images, SMPLify initialization, Gaussian/vertex LRs, normalized shared Laplacian, mask, camera, pose and density schedule.',
        comparison_budget=LIMIT,snapshots=[LIMIT],runtime_budget_seconds=300,
        risk='True view-dependent reflectance cannot be represented; geometry may still be wrong. Colour consistency is not surface-quality acceptance.',
        admission='Same native-camera1000-step comparison plus fixed synthetic orbit. Reject colourful view variation without claiming missing geometry was repaired.',
        from_scratch=True,optimizer_state_reused=False,old_gaussian_checkpoint_loaded=False,
        renderer_changed=False,view_only_hiding=False,no_coefficient_sweep=True,
        quality_accepted=False,end_to_end_15min_accepted=False)
    p['source_hashes'][str(Path(__file__).resolve())]=sha(__file__)
    save(OUT/'plan.json',p)
    _,old,new=source();ast.parse(new)
    (OUT/'sh-degree-only-adapter.py').write_text(new)
    (OUT/'sh-degree-only.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))))
    save(OUT/'progress.json',dict(updated=now(),state='prepared_waiting_for_current_GPU_candidate',pid=None,step=0,quality_accepted=False))
    print('DEGREE0_CONTROL_PREPARED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','train']);p.add_argument('--pid',type=int);args=p.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='run':
            control=baseline.eight.direct.control;control.OUT=OUT;control.__file__=__file__;control.run(args.pid)
        else:source()[0]()
    except BaseException as e:
        if OUT.exists():save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
