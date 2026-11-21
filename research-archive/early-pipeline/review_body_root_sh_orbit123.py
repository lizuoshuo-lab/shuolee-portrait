"""Matched immutable 10k PLYs on the same36 synthetic cameras; no optimization."""
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

CONTROL=ROOT.parent/'body-gaussian-root-sh-control'
OUT=CONTROL/'gaussian-run/step-010000/orbit-comparison'
CAMERAS=ROOT/'delivery/step-030000/novel/cameras.json'


def prepare():
    OUT.mkdir(exist_ok=False)
    files=[CAMERAS,Path(__file__).resolve()]
    for run in (ROOT,CONTROL):
        snap=run/'gaussian-run/step-010000'
        files += [snap/'world.ply',snap/'result.json',run/'data/training-inputs.json']
        assert rt.read(snap/'result.json')['world_ply_sha256']==rt.sha(snap/'world.ply')
    assert rt.sha(ROOT/'data/training-inputs.json')==rt.sha(CONTROL/'data/training-inputs.json')
    rt.save(OUT/'plan.json',dict(updated=rt.now(),files={str(f):rt.sha(f) for f in files},
        views=36,step=10000,same_cameras=True,steps_changed=False,ground_truth_available=False,
        purpose='Matched radiance-coordinate comparison; original baseline on left, root-SH control on right.',
        no_old_gaussian_initialization=True,optimization_updates=0,quality_accepted=False))


def checked():
    for f,h in rt.read(OUT/'plan.json')['files'].items(): assert rt.sha(f)==h,f


def replay(pid):
    checked();assert process(pid)[1]=='T' and not (OUT/'result.json').exists()
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera,image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    models=[]
    for run in (ROOT,CONTROL):
        model=GaussianModel(3);model.load_ply(str(run/'gaussian-run/step-010000/world.ply'))
        model.active_sh_degree=3;models.append(model)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');records=[];move=np.eye(4);move[:3,:3]=R
    started=time.monotonic()
    with torch.no_grad():
        for i,saved in enumerate(rt.read(CAMERAS)['records']):
            # Cameras were saved in +Y. Both immutable inputs are source-world.
            row=dict(saved,w2c=(np.asarray(saved['w2c'])@move).tolist())
            cam,_=camera(row,i);rgb=[];alpha=[]
            for j,model in enumerate(models):
                value=render(cam,model,pipe,bg)['render']
                coverage=render(cam,model,pipe,bg,override_color=torch.ones_like(model._xyz))['render']
                rgb.append(image(value));alpha.append(image(coverage))
                rgb[-1].save(OUT/f'{i:03d}_{j}_rgb.png');alpha[-1].save(OUT/f'{i:03d}_{j}_alpha.png')
            pair=Image.new('RGB',(768,512),'black');pair.paste(rgb[0],(0,0));pair.paste(rgb[1],(384,0))
            pair.save(OUT/f'pair_{i:03d}.png')
            apair=Image.new('RGB',(768,512),'black');apair.paste(alpha[0],(0,0));apair.paste(alpha[1],(384,0))
            apair.save(OUT/f'alpha_{i:03d}.png')
            records.append(dict(index=i,camera=row,comparison_sha256=rt.sha(OUT/f'pair_{i:03d}.png'),
                alpha_comparison_sha256=rt.sha(OUT/f'alpha_{i:03d}.png')))
    pages=[]
    for start in (0,9,18,27):
        page=Image.new('RGB',(2304,1536),'black')
        for j in range(9):
            with Image.open(OUT/f'pair_{start+j:03d}.png') as im:page.paste(im,((j%3)*768,(j//3)*512))
        name=f'orbit_{start:02d}-{start+8:02d}.png';page.save(OUT/name);pages.append(dict(path=str(OUT/name),sha256=rt.sha(OUT/name)))
    assert process(pid)[1]=='T';checked()
    rt.save(OUT/'result.json',dict(updated=rt.now(),views=36,records=records,pages=pages,
        seconds=time.monotonic()-started,points=[len(m._xyz) for m in models],same_native_rasterizer=True,
        same_cameras=True,ground_truth_available=False,actual_visual_review=False,quality_accepted=False))
    print('MATCHED_ROOT_SH_ORBIT_READY',flush=True)


def run(pid):
    checked();lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt=dict(started=rt.now(),pid=pid,identity=identity,command=cmd,state='suspending')
    rt.save(OUT/'gpu-handoff.json',receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--pid',str(pid)],check=True,timeout=90)
    finally:
        if process(pid)[2]==identity:os.kill(pid,signal.SIGCONT)
        receipt.update(finished=rt.now(),state='resumed',trainer_state=process(pid)[1]);rt.save(OUT/'gpu-handoff.json',receipt)
        print('BASELINE_RESUMED',pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','replay']);p.add_argument('--pid',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:globals()[a.mode](a.pid)
