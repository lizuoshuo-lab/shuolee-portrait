"""Bounded original renderer alpha/gradient check, no geometry optimizer updates."""
from pathlib import Path
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,OUT as AUDIT,process,verify
from fit_gaussian_teacher_import123 import active_gpu_pids

OUT=AUDIT/'native-mask-contract'
CACHE=AUDIT/'original-import-teacher-contract/original-import-lbs-cache.npz'
VHAP=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')
ADAPTER=Path(__file__).with_name('native_mesh_mask123.py')


def prepare():
    verify();OUT.mkdir(exist_ok=False)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];assert len(rows)==59
    files=[Path(__file__),ADAPTER,CACHE,ROOT/'data/training-inputs.json',
           VHAP/'vhap/util/render_nvdiffrast.py',VHAP/'vhap/util/vector_ops.py',VHAP/'vhap/model/tracker.py']
    io.save(OUT/'plan.json',dict(created=io.now(),hashes={str(p):io.sha(p) for p in files},
        frames=59,optimizer_updates=0,max_gpu_seconds=180,
        hypothesis='The original antialiased mesh alpha provides a direct differentiable silhouette constraint to shared canonical clothing offsets, without changing camera, bones, or Gaussian centers.',
        controls=['All59 original native masks/full K; no resize, warp or mask cleaning.',
                  'Same original VHAP import and frozen original LBS; protect head/neck/hands/feet.',
                  'Compare the alpha-only adapter with original render_rgba using constant texture and identical raster inputs.',
                  'One fixed synthetic whole-triangle x-translation derivative vs centered finite difference at1e-4m.',
                  'No parameter fitting, no new Gaussian or geometry export.'],
        thresholds=dict(alpha_max_abs=1e-6,synthetic_derivative_relative=.05,synthetic_derivative_absolute=.001),
        mask_loss='Absolute alpha difference divided by fixed original target foreground pixel count; local explicit normalization, not original enabled default.',
        source_boundary='Original VHAP compute_photometric_energy has silhouette term commented out. Existing RGB+mask files alone do not imply a direct silhouette loss.',
        stop='Source/hash/frame mismatch, nonfinite or zero trainable gradient, broken synthetic derivative or alpha parity. No threshold sweep.',
        quality_accepted=False))
    print('MASK_CONTRACT_REGISTERED',flush=True)


def checked():
    verify();p=io.read(OUT/'plan.json')
    for f,h in p['hashes'].items():assert io.sha(f)==h,f
    return p


def replay(pid):
    p=checked();assert process(pid)['state']=='T';assert active_gpu_pids()<={pid,os.getpid()}
    begin=time.monotonic()
    import torch
    from PIL import Image
    sys.path.insert(0,str(VHAP))
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    from native_mesh_mask123 import native_alpha,foreground_normalized_mask_loss
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda';renderer=NVDiffRenderer(lighting_type='constant')
    conversion=torch.diag(torch.tensor([1.,-1.,-1.,1.],device=device))[None]
    # One predeclared synthetic derivative, independent of learned geometry.
    sv=torch.tensor([[[-.15,-.10,2.],[.16,-.12,2.],[.01,.15,2.]]],device=device)
    sf=torch.tensor([[0,1,2]],device=device,dtype=torch.long)
    sk=torch.tensor([[[300.,0.,47.3],[0.,310.,37.2],[0.,0.,1.]]],device=device)
    dx=torch.tensor([1.,0.,0.],device=device)
    with torch.no_grad():target,_=native_alpha(renderer,sv+.01*dx,sf,conversion,sk,(96,128))
    shift=torch.tensor(0.,device=device,requires_grad=True)
    alpha,_=native_alpha(renderer,sv+shift*dx,sf,conversion,sk,(96,128))
    loss=foreground_normalized_mask_loss(alpha,target);analytic=float(torch.autograd.grad(loss,shift)[0])
    epsilon=1e-4
    with torch.no_grad():
        plus,_=native_alpha(renderer,sv+epsilon*dx,sf,conversion,sk,(96,128))
        minus,_=native_alpha(renderer,sv-epsilon*dx,sf,conversion,sk,(96,128))
        numeric=float((foreground_normalized_mask_loss(plus,target)-foreground_normalized_mask_loss(minus,target))/(2*epsilon))
    difference=abs(analytic-numeric);relative=difference/max(abs(numeric),1e-8)
    synthetic=dict(analytic=analytic,finite_difference=numeric,relative_error=relative,absolute_error=difference,epsilon_m=epsilon)
    io.save(OUT/'synthetic-gradient.json',synthetic)
    assert difference<=p['thresholds']['synthetic_derivative_absolute']+p['thresholds']['synthetic_derivative_relative']*abs(numeric),synthetic
    assert analytic<0 and numeric<0,'Positive x translation must reduce the registered target shift'
    with np.load(CACHE,allow_pickle=False) as z:cache=dict(z)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];assert np.array_equal(cache['frame_ids'],[r['frame'] for r in rows])
    faces=torch.tensor(cache['faces'],device=device,dtype=torch.long)
    base=torch.tensor(cache['base_vertices'],device=device);A=torch.tensor(cache['linear'],device=device)
    learn=torch.tensor(cache['learnable_vertices'],device=device,dtype=torch.bool)
    delta=torch.zeros((10475,3),device=device,requires_grad=True)
    uv=torch.full((10475,2),.5,device=device);texture=torch.ones((1,3,16,16),device=device)
    aggregate=torch.zeros_like(delta);records=[];images=[]
    selected={'frame_000033','frame_000015','frame_000059'}
    for i,row in enumerate(rows):
        assert time.monotonic()-begin<p['max_gpu_seconds']
        assert active_gpu_pids()<={pid,os.getpid()}
        assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
        w,h=row['wh'];mask_array=np.asarray(Image.open(row['mask']).convert('L'))>127
        assert mask_array.shape==(h,w)
        target=torch.tensor(mask_array,device=device,dtype=base.dtype)[None,:,:,None]
        E=torch.tensor(row['w2c'],device=device,dtype=base.dtype)[None];RT=conversion@E
        K=torch.tensor(row['K'],device=device,dtype=base.dtype)[None]
        vertices=(base[i]+(A[i]@(delta*learn[:,None])[:,:,None]).squeeze(-1))[None]
        alpha,rast=native_alpha(renderer,vertices,faces,RT,K,(h,w))
        with torch.no_grad():
            original=renderer.render_rgba(rast,vertices,faces,uv,faces,texture,None,enable_disturbance=False)['rgba'][...,3:]
            parity=float((alpha-original).abs().max());assert parity<=p['thresholds']['alpha_max_abs']
        loss=foreground_normalized_mask_loss(alpha,target)
        grad=torch.autograd.grad(loss,delta)[0]
        assert torch.isfinite(grad).all() and float(grad.norm())>0
        assert not torch.any(grad[~learn]);aggregate+=grad.detach()/59
        record=dict(frame=row['frame'],loss=float(loss),alpha_max_abs_to_original=parity,
                    shared_gradient_norm=float(grad.norm()),nonzero_clothing_gradient_vertices=int((grad.norm(dim=-1)>0).sum()),
                    protected_gradient_max_abs=float(grad[~learn].abs().max()))
        records.append(record)
        if row['frame'] in selected:
            raw=Image.open(row['image']).convert('RGB');pred=(alpha[0,:,:,0].detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            yy,xx=np.where(mask_array);box=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21)]
            panels=[raw,Image.fromarray(mask_array.astype(np.uint8)*255).convert('RGB'),Image.fromarray(pred).convert('RGB')]
            panels=[x.crop(box) for x in panels];sheet=Image.new('RGB',(panels[0].width*3,panels[0].height))
            for k,img in enumerate(panels):sheet.paste(img,(k*img.width,0))
            name=f'分析_原库轮廓接口_{row["frame"]}_原图_原mask_网格alpha.png'
            path=OUT/name;sheet.save(path);destination=Path('/mnt/d/00/记录/demo')/name
            assert not destination.exists();destination.write_bytes(path.read_bytes())
            images.append(dict(frame=row['frame'],path=str(path),record_path=str(destination),sha256=io.sha(path),no_text=True,no_resampling=True))
        io.save(OUT/'progress.json',dict(updated=io.now(),state='native_mask_contract',pid=os.getpid(),frames_completed=i+1))
        print('MASK_FRAME',i+1,row['frame'],record['shared_gradient_norm'],flush=True)
        del alpha,original,rast,vertices,grad,loss
    norm=float(aggregate.norm());assert norm>0 and torch.isfinite(aggregate).all()
    np.savez_compressed(OUT/'mean-mask-gradient-diagnostic-only.npz',gradient=aggregate.cpu().numpy(),learnable_vertices=cache['learnable_vertices'])
    checked()
    value=dict(updated=io.now(),state='passed_native_mask_alpha_gradient_contract',frames=59,frames_result=records,
               synthetic_gradient=synthetic,images=images,mean_clothing_gradient_norm=norm,
               elapsed_seconds=time.monotonic()-begin,max_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
               source='Unchanged original VHAP NVDiffRenderer rasterization and AA; explicit alpha adapter only.',
               optimizer_updates=0,actual_visual_review=False,mesh_or_Gaussian_exported=False,quality_accepted=False,
               limitations=['Mask boundaries do not determine unobserved depth or fix bone/pose errors.',
                            'Handheld objects, hair and mask mistakes remain in original inputs. Anatomical protected groups are fixed.',
                            'Gradient existence and original alpha equality are implementation evidence, not a fitted candidate.'])
    io.save(OUT/'result.json',value);io.save(OUT/'progress.json',dict(updated=io.now(),state=value['state'],pid=None,frames_completed=59))
    print('MASK_CONTRACT_FINISHED',norm,flush=True)


def run(pid):
    checked();assert not (OUT/'result.json').exists()
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(pid);assert pid==331 and before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert 'train_demo_body_upstream_offset123.py train' in before['command'];assert active_gpu_pids()<={pid}
    child=None
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(50):
            if process(pid)['state']=='T':break
            time.sleep(.05)
        assert process(pid)['state']=='T'
        child=subprocess.Popen([sys.executable,'-B',__file__,'replay','--pid',str(pid)])
        code=child.wait(timeout=190);assert code==0,code
    except BaseException as err:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
        io.save(OUT/'failure.json',dict(updated=io.now(),error=repr(err),traceback=traceback.format_exc()));raise
    finally:
        assert process(pid)['start_identity']==before['start_identity']
        foreign=active_gpu_pids()-{pid}-(set() if child is None else {child.pid})
        if not foreign:os.kill(pid,signal.SIGCONT)
        io.save(OUT/'gpu-handoff.json',dict(updated=io.now(),before=before,after=process(pid),foreign_gpu_pids=sorted(foreign)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay']);parser.add_argument('--pid',type=int);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:globals()[args.mode](args.pid)
