"""Fixed-checkpoint background full/native rendering and simple coverage diagnostics."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from scipy.spatial import cKDTree
from PIL import Image
from gsplat import rasterization
from enerf_background_data import load_background,HELDOUT
from audit_enerf_input import digest
from train_enerf_first_frame import tensor,save

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--run',required=True);ap.add_argument('--init',required=True);a=ap.parse_args();run=Path(a.run);out=run/'evaluation';out.mkdir(exist_ok=True)
assert (run/'completion.json').exists() and not (out/'metrics.json').exists()
torch.set_num_threads(8);cp=torch.load(run/'resume.pt',weights_only=False,map_location='cpu');assert cp['step']==6000
params={k:v.cuda() for k,v in cp['params'].items()};assert all(torch.isfinite(v).all() for v in params.values());assert len(params['means'])==532803
init=np.load(a.init);pts=init['points'];N=len(pts);d=cKDTree(pts).query(pts,k=4)[0][:,1:];std=np.clip(np.sqrt((d*d).mean(1)),.001,.1)
initial=dict(means=tensor(pts),scales=tensor(np.log(std)[:,None].repeat(3,axis=1)),quats=tensor(np.tile([1,0,0,0],(N,1))),opacities=tensor(np.full(N,np.log(.1/.9))),sh0=tensor((init['colors']-.5)/.28209479177387814)[:,None,:],shN=torch.zeros(N,8,3,device='cuda'))
rows=[]
for o in load_background(a.root,HELDOUT+['01','08','17']):
    camera=o['camera'];reference=o['rgb'];valid=o['valid'];K=tensor(o['K']);w2c=tensor(o['w2c'])
    Image.fromarray(reference).save(out/f'camera{camera}-reference.png');Image.fromarray(valid.astype('uint8')*255).save(out/f'camera{camera}-valid.png')
    for name,p in [('initial',initial),('trained',params)]:
        with torch.no_grad():
            r,alpha,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),w2c[None],K[None],1920,1080,sh_degree=0 if name=='initial' else 2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
            pred=np.rint((r[0]+.5*(1-alpha[0])).clamp(0,1).cpu().numpy()*255).astype('uint8');al=np.rint(alpha[0,...,0].clamp(0,1).cpu().numpy()*255).astype('uint8')
        Image.fromarray(pred).save(out/f'camera{camera}-{name}.png');Image.fromarray(al).save(out/f'camera{camera}-{name}-alpha.png')
        diff=(pred.astype('float64')-reference.astype('float64'))/255
        masks={'full_valid':valid,'top_rows0_800':valid&(np.indices(valid.shape)[0]<800),'bottom_rows900_1080':valid&(np.indices(valid.shape)[0]>=900)}
        scores={k:dict(pixels=int(m.sum()),psnr_db=float(-10*np.log10(np.mean(diff[m]**2))),mean_absolute_error=float(np.abs(diff[m]).mean()),alpha_below_point95_fraction=float((al[m]<243).mean())) for k,m in masks.items()}
        rows.append(dict(camera=camera,split='heldout_view' if camera in HELDOUT else'train',model=name,reference_sha256=o['sha256'],scores=scores))
        print(camera,name,json.dumps(scores['full_valid']),flush=True)
report=dict(status='12background_renders_complete_independent_and_visual_review_pending',rows=rows,checkpoint_sha256=digest(run/'resume.pt'),points=len(params['means']),steps=6000,
    scope='Empty-background appearance, not moving humans. All3heldout cameras and3fixed training probes, unchanged optimized calibration. Initial and trained have different point counts.',
    rendering='Native1920x1080 with gray background for uncovered pixels; no enhancement. Top/bottom are image bands, not semantic wall/floor segmentations.',
    selection='Final6000 checkpoint fixed before evaluating references; density-capped recovery recorded separately.')
save(out/'metrics.json',report)
