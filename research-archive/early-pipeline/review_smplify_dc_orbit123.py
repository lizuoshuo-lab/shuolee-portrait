"""Read-only, fixed-camera SH colour ablation and matched degree-zero control."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from export_demo_body_snapshot123 import ROOT, R, process
from shuolee_portrait import runtime as rt

BASE = ROOT.parent/'body-gaussian-smplify-eight-laplacian'
DC = ROOT.parent/'body-gaussian-smplify-eight-dc'
OUT = DC/'orbit-comparison'
CAMERAS = ROOT/'delivery/step-030000/novel/cameras.json'
SPECS = [('trained_sh_1000', BASE, 1000, 3, 1),
         ('trained_dc_1000', DC, 1000, 0, 0),
         ('trained_sh_18000', BASE, 18000, 3, 3),
         ('same_18000_dc_ablation', BASE, 18000, 3, 0)]

def prepare():
    assert not OUT.exists()
    files = [Path(__file__).resolve(), CAMERAS]
    for name, run, step, maximum, active in SPECS:
        snap = run/'gaussian-run'/f'step-{step:06d}'
        result = rt.read(snap/'result.json')
        assert rt.sha(snap/'world.ply') == result['world_ply_sha256']
        if not name.endswith('ablation'): assert result['active_sh_degree'] == active
        files += [snap/'result.json', snap/'world.ply', run/'data/training-inputs.json']
    assert rt.sha(BASE/'data/training-inputs.json') == rt.sha(DC/'data/training-inputs.json')
    OUT.mkdir()
    rt.save(OUT/'plan.json', dict(updated=rt.now(), files={str(p):rt.sha(p) for p in files},
        specs=[dict(name=n,run=str(r),step=s,max_sh_degree=m,active_sh_degree=a) for n,r,s,m,a in SPECS],
        views=36, optimization_updates=0, fixed_existing_cameras=True,
        comparisons=['1000-step SH1 versus separately trained DC-only, otherwise identical',
                     '18000-step SH3 versus same model rendered DC-only; geometry and all coefficients unchanged'],
        ground_truth_available=False, quality_accepted=False))

def checked():
    for p,h in rt.read(OUT/'plan.json')['files'].items(): assert rt.sha(p)==h,p

def replay(pid):
    checked(); assert process(pid)[1]=='T' and not (OUT/'result.json').exists()
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, image
    from utils.sh_utils import eval_sh
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    models=[]
    for name,run,step,maximum,active in SPECS:
        model=GaussianModel(maximum); model.load_ply(str(run/'gaussian-run'/f'step-{step:06d}'/'world.ply'))
        model.active_sh_degree=active; models.append(model)
    assert models[1]._features_rest.numel()==0
    for attr in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'):
        assert torch.equal(getattr(models[2],attr),getattr(models[3],attr)),attr
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda'); move=np.eye(4); move[:3,:3]=R
    records=[]; colour_samples=[[] for _ in models]; started=time.monotonic()
    with torch.no_grad():
        for i,saved in enumerate(rt.read(CAMERAS)['records']):
            row=dict(saved,w2c=(np.asarray(saved['w2c'])@move).tolist()); cam,_=camera(row,i)
            rgb=[]; alpha=[]
            for j,model in enumerate(models):
                value=render(cam,model,pipe,bg)['render']
                coverage=render(cam,model,pipe,bg,override_color=torch.ones_like(model._xyz))['render']
                rgb.append(image(value)); alpha.append(coverage)
                rgb[-1].save(OUT/f'{SPECS[j][0]}_{i:03d}.png')
                direction=torch.nn.functional.normalize(model.get_xyz-cam.camera_center.to(model.get_xyz),dim=-1)
                colour_samples[j].append(eval_sh(model.active_sh_degree,model.get_features.transpose(1,2),direction).cpu().numpy())
            assert torch.equal(alpha[2],alpha[3]),'A colour-only ablation must not alter coverage'
            for label,indices in [('matched1000',(0,1)),('ablation18000',(2,3))]:
                pair=Image.new('RGB',(768,512),'black')
                for k,j in enumerate(indices): pair.paste(rgb[j],(384*k,0))
                pair.save(OUT/f'{label}_{i:03d}.png')
            records.append(dict(index=i,camera=row,alpha_ablation_max_error=0))
    pages=[]
    for label in ('matched1000','ablation18000'):
        for start in (0,9,18,27):
            page=Image.new('RGB',(2304,1536),'black')
            for j in range(9):
                with Image.open(OUT/f'{label}_{start+j:03d}.png') as im: page.paste(im,((j%3)*768,(j//3)*512))
            path=OUT/f'{label}_{start:02d}-{start+8:02d}.png';page.save(path)
            pages.append(dict(path=str(path),sha256=rt.sha(path)))
    stats=[]
    for (name,_,_,_,_),model,samples in zip(SPECS,models,colour_samples):
        values=np.stack(samples); span=values.max(0)-values.min(0)
        opacity=model.get_opacity.detach().cpu().numpy()[:,0]
        rgb_span=span.mean(-1)
        stats.append(dict(name=name,points=len(opacity),directional_colour_range_rgb_mean=float(rgb_span.mean()),
            opacity_weighted_range=float((rgb_span*opacity).sum()/opacity.sum()),
            range_p50=float(np.quantile(rgb_span,.5)),range_p90=float(np.quantile(rgb_span,.9)),
            raw_colour_samples_below0=float((values+.5<0).mean()),raw_colour_samples_above1=float((values+.5>1).mean())))
    assert stats[1]['directional_colour_range_rgb_mean']==stats[3]['directional_colour_range_rgb_mean']==0
    checked();assert process(pid)[1]=='T'
    rt.save(OUT/'result.json',dict(updated=rt.now(),views=36,records=records,pages=pages,stats=stats,
        seconds=time.monotonic()-started,alpha_ablation_identical=True,
        per_gaussian_colour_statistics='Same Gaussian,36 fixed view directions, unoccluded coefficient evaluation before clipping; not image quality or physical truth.',
        optimization_updates=0,quality_accepted=False,actual_visual_review=False))
    print('DC_ORBIT_COMPARISON_READY',flush=True)

def run(pid):
    checked();lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt=dict(started=rt.now(),pid=pid,identity=identity,command=cmd,state='suspending');rt.save(OUT/'gpu-handoff.json',receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--pid',str(pid)],check=True,timeout=120)
    finally:
        if process(pid)[2]==identity:os.kill(pid,signal.SIGCONT)
        receipt.update(finished=rt.now(),state='resumed',trainer_state=process(pid)[1]);rt.save(OUT/'gpu-handoff.json',receipt)
        print('BASELINE_RESUMED',pid,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','replay']);p.add_argument('--pid',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:globals()[a.mode](a.pid)
