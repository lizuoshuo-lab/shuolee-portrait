"""One export-only numerical control; does not modify cameras or training."""
from pathlib import Path
import sys,json,itertools,numpy as np
from types import SimpleNamespace
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.cli import gpu_lock

with gpu_lock():
    rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',ROOT)
    from shuolee_portrait.export import rotate_ply,rigid_rotation64
    from shuolee_portrait.training import camera
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    import torch
    fit=np.load(ROOT/'fit/fitted.npz');target,_=rigid_rotation64(fit['root_rotation'][0].T)
    rotations=[]
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([-1.,1.],repeat=3):
            R=np.eye(3)[list(permutation)]*np.array(signs)[:,None]
            if np.linalg.det(R)>0:rotations.append(R)
    R=min(rotations,key=lambda m:np.linalg.norm(m-target))
    out=ROOT/'export-numerical-control';out.mkdir()
    source=ROOT/'gaussian-run/step-018000/world.ply';candidate=out/'exact-axis.ply'
    report=rotate_ply(source,candidate,R,np.zeros(3))
    report['basis_reason']='Exact signed axis permutation, no recentering. Preserves original float32 coordinates under a proper rigid frame change.'
    report['up_residual_degrees']=float(np.degrees(np.arccos(np.clip((R@fit['root_rotation'][0][:,1])[1],-1,1))))
    a=GaussianModel(3);a.load_ply(str(source));b=GaussianModel(3);b.load_ply(str(candidate))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');rows=rt.read(ROOT/'data/training-inputs.json')['records'];checks=[]
    with torch.no_grad():
        for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
            row=rows[i];mapped=dict(row);E=np.asarray(row['w2c']);E=E.copy();E[:3,:3]=E[:3,:3]@R.T;mapped['w2c']=E.tolist()
            ca,_=camera(row,int(i),768);cb,_=camera(mapped,int(i),768)
            delta=(render(ca,a,pipe,bg)['render']-render(cb,b,pipe,bg)['render']).abs()
            checks.append(dict(frame=row['frame'],mean_abs=float(delta.mean()),max_abs=float(delta.max()),components_over_005=int((delta>.005).sum())))
    report['checks']=checks;report['passed']=all(x['mean_abs']<2e-6 and x['max_abs']<.005 for x in checks)
    rt.save(out/'result.json',report);print(json.dumps(report,indent=2))
