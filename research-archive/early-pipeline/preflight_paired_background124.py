"""Verify paired compositing, isolated RNG, unknown-mask gradients and CUDA algebra."""
import fcntl
import importlib.util
from pathlib import Path
import sys,time
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path[:0]=['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src','/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS
from paired_training_background124 import PairedTrainingBackground
from native_target_cache124 import NativeTargetCache
from train_indoor_paired_background124 import OUT,BASE


def main():
    assert not (OUT/'background-preflight.json').exists()
    plan=rt.read(OUT/'plan.json')
    for p,h in plan['source_hashes'].items():assert rt.sha(p)==h,p
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();rt.configure(ASSETS)
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera
    torch.set_num_threads(4)
    test=[]
    def record(name,condition,**details):
        test.append(dict(name=name,passed=bool(condition),**details));assert condition,(name,details)
    frames=['synthetic'];rgb=torch.arange(3*32*32).reshape(3,32,32).remainder(256).to(torch.uint8)
    mask=torch.zeros(1,32,32,dtype=torch.uint8);mask[:,8:24,8:24]=255;mask[:,7,8:24]=128
    cache=NativeTargetCache([rgb],[mask],frames);unchanged=cache.get(0,frames[0]).clone()
    calls=[]
    def fake(cam,model,pipe,bg,**kwargs):calls.append(bg.clone());return {'background':bg}
    a=PairedTrainingBackground(fake,[mask],frames,cache);b=PairedTrainingBackground(fake,[mask],frames,cache)
    cam=SimpleNamespace(supervision_frame='synthetic',original_image=unchanged.clone())
    state=torch.cuda.get_rng_state().clone()
    for i in range(3):
        first=a(cam,None,None,torch.zeros(3,device='cuda'))['background'].clone();target=cam.original_image.clone()
        second=b(cam,None,None,torch.zeros(3,device='cuda'))['background']
        expected=unchanged+(1-mask.to(device='cuda',dtype=torch.float32)/255)*first[:,None,None]
        record('paired_and_reproducible_'+str(i),torch.equal(first,second) and torch.equal(target,expected))
    record('global_cuda_rng_preserved',torch.equal(state,torch.cuda.get_rng_state()))
    record('cached_black_target_unchanged',torch.equal(cache.get(0,'synthetic'),unchanged))
    with torch.no_grad():a(cam,None,None,torch.zeros(3,device='cuda'))
    record('evaluation_no_rng_or_target_mutation',a.calls==3)
    spec=importlib.util.spec_from_file_location('paired_known_loss',OUT/'loss_utils_validity.py');loss=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss)
    valid=torch.ones(1,32,32,dtype=torch.bool,device='cuda');valid[:,12:20,12:20]=False
    pred=torch.rand(3,32,32,device='cuda',requires_grad=True);target=torch.rand_like(pred)
    objective=lambda x,y:.8*loss.l1_loss(x,y,valid)+.2*(1-loss.ssim(x,y,valid_mask=valid))
    value=objective(pred,target);gradient,=torch.autograd.grad(value,pred)
    record('unknown_gradient_zero',float(gradient[:,~valid[0]].abs().max())==0 and float(gradient[:,valid[0]].abs().max())>0)
    modified=target.clone();modified[:,~valid[0]]=float('nan')
    record('unknown_target_cannot_change_loss',torch.equal(value,objective(pred,modified)))
    row=rt.read(BASE/'data/training-inputs.json')['records'][0];cam,_=camera(row,0)
    model=GaussianModel(0);model.load_ply(str(BASE/'gaussian-run/step-018000/world.ply'));model.active_sh_degree=0
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.tensor([.2,.7,.4],device='cuda')
    with torch.no_grad():
        black=render(cam,model,pipe,torch.zeros(3,device='cuda'))['render']
        alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones((len(model._xyz),3),device='cuda'))['render']
        mixed=render(cam,model,pipe,bg)['render']
    err=float((mixed-(black+(1-alpha)*bg[:,None,None])).abs().max())
    record('native_cuda_background_compositing',err<3e-6,max_abs_error=err)
    for p,h in plan['source_hashes'].items():assert rt.sha(p)==h,p
    rt.save(OUT/'background-preflight.json',dict(updated=rt.now(),passed=True,tests=test,optimizer_updates=0,
        native_scene=BASE.name,source_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,quality_accepted=False))
    print('PAIRED_BACKGROUND_PREFLIGHT_PASS',len(test),flush=True)


if __name__=='__main__':main()
