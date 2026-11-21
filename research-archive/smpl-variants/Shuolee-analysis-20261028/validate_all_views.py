"""Replay all original training views, with the exact recorded deformation implementation."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
from pathlib import Path
import argparse,ast,fcntl,json,sys,time
from types import SimpleNamespace
import numpy as np
from PIL import Image
parser=argparse.ArgumentParser();parser.add_argument('--motion',choices=['face','fan'],default='fan');args=parser.parse_args()
BASE=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
EVIDENCE=Path('/mnt/d/Projects/Shuolee-analysis-20261028')
suffix='-fan' if args.motion=='fan' else ''
ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029')/f'gray-dynamic-face{suffix}'
OUT=EVIDENCE/f'dynamic-face{suffix}-all-views'
lock=Path('/tmp/shuolee-native122-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
try:
    sys.path.insert(0,'/mnt/d/Projects/Shuolee/src')
    from shuolee_portrait import runtime as rt
    rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',ROOT)
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from scipy.spatial import cKDTree
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.general_utils import build_scaling_rotation,strip_symmetric
    from utils.sh_utils import eval_sh
    from shuolee_portrait.training import Views,image
    assert rt.read(ROOT/'result.json')['state']=='completed'
    OUT.mkdir(exist_ok=False);(OUT/'head-crops').mkdir()
    rows=rt.read(BASE/'data/training-inputs.json')['records'];cfg=rt.read(BASE/'config.json')
    fit=dict(np.load(BASE/'fit/fitted.npz'));motion=dict(np.load(EVIDENCE/f'dynamic-face{suffix}-control/motion.npz'))
    tree=cKDTree(fit['vertices_world'][0]);scale=float(fit['world_scale'])
    g=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    vertex_sequence=g(motion['vertices_world']);affine_sequence=g(motion['vertex_affine']);reference_vertices=vertex_sequence[0]
    code=ROOT/'code/train_head_motion.py'
    classes=[n for n in ast.walk(ast.parse(code.read_text())) if isinstance(n,ast.ClassDef) and n.name=='Deformed']
    assert len(classes)==1
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(code),'exec'),globals())
    original=GaussianModel(3);original.load_ply(str(BASE/'gaussian-run/step-018000/world.ply'))
    model=GaussianModel(3);model.load_ply(str(ROOT/'step-018000/canonical.ply'))
    moving=Deformed(model);moving.update_weights()
    pipe=SimpleNamespace(compute_cov3D_python=True,convert_SHs_python=False,debug=False)
    ordinary=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');views=Views(rows,cfg['training_height'],cfg['image_cache_frames'])
    records=[];parity=[];started=time.monotonic()
    with torch.no_grad():
        for i,row in enumerate(rows):
            cam,gt,alpha=views.get(i);moving.set_frame(i)
            a=image(render(cam,original,ordinary,bg)['render'])
            b=image(render(cam,moving,pipe,bg,override_color=moving.colors(cam))['render'])
            y=np.asarray(image(gt)).astype(float)/255;H,W=y.shape[:2]
            factor=np.array([W,H])/row['wh'];box=np.array(row['face_box_xyxy']).reshape(2,2)*factor
            margin=max(float(np.ptp(box,axis=0).max())*.6,15)
            lo=np.maximum(0,np.floor(box[0]-margin)).astype(int);hi=np.minimum([W,H],np.ceil(box[1]+margin)).astype(int)
            mask=np.asarray(Image.open(row['mask']).convert('L').resize((W,H),Image.Resampling.NEAREST))>127
            crop=np.zeros_like(mask);crop[lo[1]:hi[1],lo[0]:hi[0]]=True
            rec=dict(frame=row['frame'],head_crop_xyxy=[*lo.tolist(),*hi.tolist()])
            for label,im in [('static',a),('head_motion',b)]:
                x=np.asarray(im).astype(float)/255
                rec[label]=dict(head_foreground_rgb_l1=float(np.abs(x-y)[mask&crop].mean()),
                    head_crop_masked_rgb_l1=float(np.abs(x-y*mask[...,None])[crop].mean()),
                    whole_foreground_rgb_l1=float(np.abs(x-y)[mask].mean()))
                path=(BASE/'gaussian-run' if label=='static' else ROOT)/'step-018000'/f'{row["frame"]}_render.png'
                if path.exists():
                    d=abs(np.asarray(im).astype(float)-np.asarray(Image.open(path)).astype(float))/255
                    check=dict(frame=row['frame'],model=label,mean_abs=float(d.mean()),max_abs=float(d.max()))
                    parity.append(check)
                    if check['mean_abs']>1e-5 or check['max_abs']>.01:raise RuntimeError(f'Saved rendering mismatch: {check}')
            records.append(rec)
            triptych=Image.new('RGB',(3*(hi[0]-lo[0]),hi[1]-lo[1]))
            for j,im in enumerate([image(gt),a,b]):triptych.paste(im.crop((*lo,*hi)),(j*(hi[0]-lo[0]),0))
            triptych.save(OUT/'head-crops'/f'{row["frame"]}.png')
            if i%20==0:print(json.dumps(dict(frames_done=i+1,total=len(rows),seconds=time.monotonic()-started)),flush=True)
    average={kind:{key:float(np.mean([r[kind][key] for r in records])) for key in records[0][kind]} for kind in ['static','head_motion']}
    improvements={key:[1-r['head_motion'][key]/r['static'][key] for r in records] for key in average['static']}
    summary={key:dict(improved_frames=int((np.array(v)>0).sum()),regressed_frames=int((np.array(v)<0).sum()),
        median_relative_improvement=float(np.median(v)),minimum=float(min(v)),maximum=float(max(v)),
        relative_improvement_of_mean=1-average['head_motion'][key]/average['static'][key]) for key,v in improvements.items()}
    report=dict(frames=len(rows),step=18000,mean_across_all_views=average,improvement=summary,records=records,
        saved_snapshot_parity=parity,cameras_unchanged=True,original_images_unchanged=True,quality_accepted=False,
        measurement='All original training views; not held-out image quality',deformation_source_sha256=rt.sha(code),
        source_ply_sha256=rt.sha(ROOT/'step-018000/canonical.ply'),baseline_ply_sha256=rt.sha(BASE/'gaussian-run/step-018000/world.ply'))
    rt.save(OUT/'report.json',report)
    print(json.dumps(dict(frames=len(rows),mean=average,improvement=summary,quality_accepted=False)),flush=True)
finally:fcntl.flock(lock,fcntl.LOCK_UN)
