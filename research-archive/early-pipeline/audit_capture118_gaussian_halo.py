"""Read-only frozen-model footprint diagnostic. No training or geometry edits."""
from pathlib import Path
from types import SimpleNamespace
import sys, numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-support')
RECORD=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.cli import gpu_lock
from shuolee_portrait.training import camera,image
rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',ROOT)
out=ROOT/'alignment/gaussian-footprint-audit';out.mkdir()
src=ROOT/'export/point_cloud.ply';digest=rt.sha(src)
rt.save(out/'plan.json',dict(hypothesis='Broad covariance footprints may explain part of outside-mask glow independently of center position.',
    variable='Render the same frozen model once at scale 1.0 and once at 0.5; positions, opacity, SH, cameras and input images stay fixed.',
    source_sha256=digest,budget='Six actual cameras; two render scales; no optimization.',
    scope='Counterfactual image contribution, not a correction and not causal proof of calibration error.',
    stop_conditions='Any nonfinite result or source PLY change.'))
with gpu_lock():
 import torch
 from scene.gaussian_model import GaussianModel
 from gaussian_renderer import render
 model=GaussianModel(3);model.load_ply(str(src))
 pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
 bg=torch.zeros(3,device='cuda');white=torch.ones_like(model.get_xyz)
 rows=rt.read(ROOT/'export/cameras.json')['records'];original=rt.read(ROOT/'data/training-inputs.json')['records'];results=[]
 with torch.no_grad():
  for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
   c,record=camera(rows[i],int(i),768);size=(c.image_width,c.image_height)
   mask=np.array(Image.open(original[i]['mask']).convert('L').resize(size,Image.Resampling.NEAREST))>127
   far=distance_transform_edt(~mask)>4*768/rows[i]['wh'][1]
   rec=dict(frame=rows[i]['frame'],scales={})
   for scale in [1.,.5]:
    pkg=render(c,model,pipe,bg,scaling_modifier=scale)
    alpha=render(c,model,pipe,bg,scaling_modifier=scale,override_color=white)['render'][0].cpu().numpy()
    assert np.isfinite(alpha).all()
    rec['scales'][str(scale)]=dict(total_alpha=float(alpha.sum()),outside_4px_alpha=float(alpha[far].sum()),
        outside_fraction=float(alpha[far].sum()/max(alpha.sum(),1e-8)),
        foreground_alpha_mean=float(alpha[mask].mean()),foreground_fraction_below_half_alpha=float((alpha[mask]<.5).mean()))
    image(pkg['render']).save(out/f'{rows[i]["frame"]}_scale_{scale}.png')
   results.append(rec)
 assert rt.sha(src)==digest
rt.save(RECORD/'gaussian-footprint-audit.json',dict(updated=rt.now(),source_sha256=digest,results=results,
    source_unchanged=True,training_updates=0,quality_accepted=False,
    limitation='Scale reduction changes coverage as well as halo; it is not a trained correction. 1000 steps precedes the configured 10000-step densification start.'))
frames=[Image.open(ROOT/'export/review'/f'novel_{i:03d}.png').convert('RGB') for i in range(36)]
frames[0].save(ROOT/'export/review/orbit.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
print(rt.read(RECORD/'gaussian-footprint-audit.json'))
