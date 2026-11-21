"""Single export-only control: preserve the accepted up axis, remove unnecessary yaw permutation."""
import argparse,numpy as np
from types import SimpleNamespace
from capture118_all_subjects import ROOT,REC,ASSETS,rt
from shuolee_portrait.cli import gpu_lock
ap=argparse.ArgumentParser();ap.add_argument('--subject',required=True);a=ap.parse_args();root=ROOT/a.subject
with gpu_lock():
    rt.configure(ASSETS,root)
    from shuolee_portrait.export import rotate_ply
    from shuolee_portrait.training import camera
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    import torch
    failed=rt.read(root/'export/validation.json');before=np.asarray(failed['rotation'])
    # No camera/geometry optimization. Choose a proper diagonal frame only if it
    # gives exactly the same +Y direction as the original rigid export.
    assert np.array_equal(before[1],[0,-1,0])
    R=np.diag([1.,-1.,-1.]);assert np.array_equal(R[1],before[1]) and np.linalg.det(R)==1
    out=root/'export-yaw-control';out.mkdir(exist_ok=False)
    source=root/'gaussian-run/step-018000/world.ply';candidate=out/'candidate.ply'
    report=rotate_ply(source,candidate,R,np.zeros(3))
    report.update(training_changed=False,original_up_axis=before[1].tolist(),same_up_axis=True,
        reason='Unnecessary X/Z swapping changes float32 dot-product summation order; diagonal180degX preserves coordinate ordering and the same +Y convention. This is an export gauge control, not a camera fit.')
    x=GaussianModel(3);x.load_ply(str(source));y=GaussianModel(3);y.load_ply(str(candidate))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    rows=rt.read(root/'data/training-inputs.json')['records'];checks=[]
    with torch.no_grad():
        for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
            row=rows[i];mapped=dict(row);E=np.asarray(row['w2c']).copy();E[:3,:3]=E[:3,:3]@R.T;mapped['w2c']=E.tolist()
            ca,_=camera(row,int(i),768);cb,_=camera(mapped,int(i),768)
            original=render(ca,x,pipe,bg)['render'];again=render(ca,x,pipe,bg)['render'];converted=render(cb,y,pipe,bg)['render'];d=(original-converted).abs()
            checks.append(dict(frame=row['frame'],native_repeat_max=float((original-again).abs().max()),mean_abs=float(d.mean()),max_abs=float(d.max()),components_over_005=int((d>.005).sum())))
    report.update(checks=checks,passed=all(c['mean_abs']<2e-6 and c['max_abs']<.005 for c in checks))
    rt.save(out/'result.json',report);rt.save(REC/f'{a.subject}-export-yaw-control.json',report)
    print(report,flush=True)
