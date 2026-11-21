"""Render the audited existing VHAP import and120k meshes; no optimization."""
from pathlib import Path
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,SNAP,OUT as AUDIT,process,verify
from fit_gaussian_teacher_visible123 import active_gpu_pids

OUT=AUDIT/'mesh-initialization-boundary'
FRAMES=['frame_000033','frame_000015','frame_000044','frame_000054']


def checked():
    verify()
    for f,h in io.read(OUT/'plan.json')['hashes'].items():assert io.sha(f)==h,f
    if (OUT/'render-plan.json').exists():
        for f,h in io.read(OUT/'render-plan.json')['hashes'].items():assert io.sha(f)==h,f


def replay(pid):
    checked();assert process(pid)['state']=='T';assert active_gpu_pids()<={pid,os.getpid()}
    started=time.monotonic()
    import torch
    from PIL import Image
    sys.path.insert(0,'/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    torch.set_num_threads(4);renderer=NVDiffRenderer()
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:saved=dict(z)
    fits={str(x):i for i,x in enumerate(fit['frame_ids'])};currents={str(x):i for i,x in enumerate(saved['frame_ids'])}
    rows={r['frame']:r for r in io.read(ROOT/'data/training-inputs.json')['records']}
    faces=torch.tensor(fit['faces'],device='cuda',dtype=torch.long)
    convert=torch.diag(torch.tensor([1.,-1.,-1.,1.],device='cuda'));records=[]
    dest=Path('/mnt/d/00/记录/demo');folder=OUT/'review';folder.mkdir(exist_ok=False)
    with torch.no_grad():
        for frame in FRAMES:
            assert time.monotonic()-started<120;assert active_gpu_pids()<={pid,os.getpid()}
            row=rows[frame];assert io.sha(row['image'])==row['source_sha256'];assert io.sha(row['mask'])==row['mask_sha256']
            raw=np.array(Image.open(row['image']).convert('RGB'));mask=np.array(Image.open(row['mask']).convert('L'))>127
            K=torch.tensor(row['K'],device='cuda',dtype=torch.float32)[None];E=torch.tensor(row['w2c'],device='cuda',dtype=torch.float32)
            RT=(convert@E)[None];w,h=row['wh'];panels=[raw]
            for array in (fit['vertices_world'][fits[frame]],saved['vertices_camera'][currents[frame]]):
                v=torch.tensor(array,device='cuda',dtype=torch.float32)
                rast=renderer.rasterize(v[None],faces,RT,K,(h,w),use_cache=False)['rast_out'][0].flip(0)
                sel=(rast[:,:,-1]>0).cpu().numpy();ids=(rast[:,:,-1].long()-1).clamp_min(0)
                vc=v@E[:3,:3].T+E[:3,3];tri=vc[faces]
                n=torch.nn.functional.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
                shade=.3+.7*(n@torch.tensor([.3,-.2,-.9327379],device='cuda')).abs()
                rgb=(shade[ids,None]*torch.tensor([80.,210.,220.],device='cuda')).cpu().numpy()
                overlay=raw.copy();overlay[sel]=(.6*rgb[sel]+.4*raw[sel]).clip(0,255).astype(np.uint8);panels.append(overlay)
            yy,xx=np.where(mask);box=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21)]
            crops=[Image.fromarray(x).crop(box) for x in panels];sheet=Image.new('RGB',(3*crops[0].width,crops[0].height))
            for i,im in enumerate(crops):sheet.paste(im,(i*im.width,0))
            name=f'分析_网格训练前后_{frame}_原图_VHAP导入_高斯120k网格.png'
            image=folder/name;sheet.save(image);target=dest/name;assert not target.exists();target.write_bytes(image.read_bytes())
            records.append(dict(frame=frame,path=str(image),record_path=str(target),sha256=io.sha(image),crop_integer_xyxy=box,no_text_added=True,no_resampling=True))
    checked();io.save(OUT/'render-result.json',dict(state='four_native_boundary_comparisons_exported',updated=io.now(),
        images=records,renderer='original VHAP NVDiffRenderer',frames=4,all59_numerical_audit=True,
        optimizer_updates=0,mesh_or_Gaussian_modified=False,quality_accepted=False,elapsed_seconds=time.monotonic()-started))
    print('BOUNDARY_REVIEW_RENDERED',len(records),flush=True)


def run(pid):
    checked();assert not (OUT/'render-result.json').exists()
    io.save(OUT/'render-plan.json',dict(created=io.now(),frames=FRAMES,max_seconds=120,optimizer_updates=0,
        hashes={str(Path(__file__)):io.sha(__file__)},
        reason='Show the same59-frame numerical boundary evidence in reference33, side15 and the two failed-face frames44,54. This is diagnostic sampling, not all-frame visual acceptance.'))
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(pid);assert pid==331 and before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert active_gpu_pids()<={pid};child=None
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(50):
            if process(pid)['state']=='T':break
            time.sleep(.05)
        assert process(pid)['state']=='T'
        child=subprocess.Popen([sys.executable,'-B',__file__,'replay','--pid',str(pid)])
        assert child.wait(timeout=120)==0
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
        assert process(pid)['start_identity']==before['start_identity']
        foreign=active_gpu_pids()-{pid}-(set() if child is None else {child.pid})
        if not foreign:os.kill(pid,signal.SIGCONT)
        io.save(OUT/'render-gpu-handoff.json',dict(updated=io.now(),before=before,after=process(pid),foreign_gpu_pids=sorted(foreign)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','replay']);p.add_argument('--pid',type=int);args=p.parse_args();globals()[args.mode](args.pid)
