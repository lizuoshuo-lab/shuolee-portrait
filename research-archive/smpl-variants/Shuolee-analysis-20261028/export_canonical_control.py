"""Validate a conventional +Y Gaussian PLY from the local canonical-motion control."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
from pathlib import Path
import argparse,fcntl,json,sys
from types import SimpleNamespace
import numpy as np
REPO=Path('/mnt/d/Projects/Shuolee');sys.path.insert(0,str(REPO/'src'))
parser=argparse.ArgumentParser();parser.add_argument('--motion',choices=['face','fan'],default='face');args=parser.parse_args()
ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029')/('gray-dynamic-face' if args.motion=='face' else 'gray-dynamic-face-fan')
BASE=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
OUT=ROOT/'export'
lock=Path('/tmp/shuolee-native122-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
try:
    from shuolee_portrait import runtime as rt
    rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',ROOT)
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.export import rotate_ply
    from shuolee_portrait.training import camera,image
    from shuolee_portrait.camera_exchange import write_camera_exchange
    assert rt.read(ROOT/'result.json')['state']=='completed'
    OUT.mkdir(exist_ok=False)
    native=ROOT/'step-018000/canonical.ply';candidate=OUT/'canonical.checking.ply'
    orientation=rt.read(BASE/'export/validation.json');R=np.asarray(orientation['rotation']);origin=np.asarray(orientation['origin_world'])
    report=rotate_ply(native,candidate,R,origin)
    rows=rt.read(BASE/'data/training-inputs.json')['records'];target=[]
    for row in rows:
        E=np.asarray(row['w2c']);mapped=E.copy();mapped[:3,:3]=E[:3,:3]@R.T;mapped[:3,3]=E[:3,:3]@origin+E[:3,3]
        target.append(dict(frame=row['frame'],name=row['frame'],source_frame_index=row['source_frame_index'],time_seconds=row['time_seconds'],wh=row['wh'],K=row['K'],w2c=mapped.tolist()))
    source=GaussianModel(3);source.load_ply(str(native));portable=GaussianModel(3);portable.load_ply(str(candidate))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda');checks=[]
    with torch.no_grad():
        for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
            ca,_=camera(rows[i],int(i),768);cb,_=camera(target[i],int(i),768)
            a=render(ca,source,pipe,bg)['render'];b=render(cb,portable,pipe,bg)['render'];d=abs(a-b)
            checks.append(dict(frame=rows[i]['frame'],mean_abs=float(d.mean()),max_abs=float(d.max())))
    report.update(coordinate_space='Canonical fitted-body world, +Y scene axis permutation; not measured gravity',
        comparison='Canonical render before versus after rigid PLY conversion; no animated-view parity claim',
        source_ply_sha256=rt.sha(native),quality_accepted=False,format_passed=all(x['mean_abs']<2e-6 and x['max_abs']<.005 for x in checks),
        native_render_checks=checks,standard_degree3_gaussian_ply=True,viewer_specific_fields=False)
    rt.save(OUT/'validation.json',report)
    if not report['format_passed']:raise RuntimeError('Canonical +Y export parity failed; candidate preserved')
    candidate.rename(OUT/'point_cloud.ply');write_camera_exchange(OUT,target,world_to_viewer=report)
    review=OUT/'review';review.mkdir()
    fit=dict(np.load(BASE/'fit/fitted.npz'));xyz=portable.get_xyz.detach().cpu().numpy()
    center=np.median(xyz,axis=0);height=float(np.quantile(xyz[:,1],.99)-np.quantile(xyz[:,1],.01));radius=max(height*1.4,.1)
    head=(fit['joints_world'][0,15]-origin)@R.T+(fit['root_rotation'][0][:,1]@R.T)*(.055*float(fit['world_scale']))
    with torch.no_grad():
        for kind,focus,distance,size in [('body',center,radius,[512,768]),('head',head,.40*float(fit['world_scale']),[512,512])]:
            for i,angle in enumerate(np.linspace(0,2*np.pi,12,endpoint=False)):
                eye=focus+distance*np.array([np.sin(angle),.12,np.cos(angle)]);forward=focus-eye;forward/=np.linalg.norm(forward)
                right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
                E=np.eye(4);E[:3,:3]=np.stack([right,down,forward]);E[:3,3]=-E[:3,:3]@eye
                row=dict(frame=f'{kind}_{i:02d}',wh=size,K=[[size[1],0,size[0]/2],[0,size[1],size[1]/2],[0,0,1]],w2c=E.tolist())
                cam,_=camera(row,i);image(render(cam,portable,pipe,bg)['render']).save(review/f'{kind}_{i:02d}.png')
    print(json.dumps(dict(format_passed=True,points=report['points'],sha256=report['sha256'],export=str(OUT/'point_cloud.ply'),quality_accepted=False)),flush=True)
finally:fcntl.flock(lock,fcntl.LOCK_UN)
