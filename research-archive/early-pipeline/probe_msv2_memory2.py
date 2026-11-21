"""Zero-update numerical and full-size stress checks for opaque-buffer storage."""
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
import traceback

import numpy as np

APP = Path('/mnt/d/Projects/msv2')
RUN = Path('/opt/gs-slam-lab/msv2-video15-20261105/runs/video15')
OUT = RUN / 'recovery-memory-proof2.json'
sys.path.insert(0, str(APP))
from msv2.engine import Engine, LazyCamera
from msv2.memory import wrap_training_render
from msv2.io import read_json, save_json, sha256, now

FIELDS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')


def main():
    assert not OUT.exists()
    assert sha256(APP/'msv2/engine.py') == '6c6ad86ef9d39e995daec5c8edf5fb3b93e3ec0ec4356b25d2bcdd3ea9abe57f'
    assert sha256(APP/'msv2/memory.py') == 'ecd716fb3b0e2d6d93547bc9c8685d089ed8e5473e0e9facb1e53716d2e39c14'
    engine = Engine(RUN / 'job.json')
    torch = engine.runtime()
    torch.cuda.set_per_process_memory_fraction(engine.config['gpu_memory_fraction'])
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import read_points3D_binary
    from gaussian_renderer import render_imp
    from fused_ssim import fused_ssim
    from utils.loss_utils import l1_loss
    from utils.graphics_utils import BasicPointCloud
    rows = read_json(RUN/'swinir4x/training-inputs.json')['records']
    _, _, opt, pipe = engine.options(4)
    bg = torch.zeros(3, device='cuda')
    source = RUN/'original/model/point_cloud/iteration_18000/point_cloud.ply'
    source_sha = sha256(source)
    state = dict(state='running', started=now(), model_updates=0, optimizer_updates=0,
        job_sha256=sha256(engine.job_file), engine_sha256=sha256(APP/'msv2/engine.py'),
        memory_sha256=sha256(APP/'msv2/memory.py'), source_model=str(source), source_model_sha256=source_sha,
        source_model_used_for_training_initializer=False, gradient_parity_passed=False, stress_passed=False,
        gradient_tolerance=dict(absolute=1e-12,relative_to_reference_max=1e-5), comparisons=[], stress={})
    save_json(OUT,state)
    wrapped = wrap_training_render(render_imp, engine.repo)

    def clear():
        LazyCamera.cached_tensor=None
        LazyCamera.cached_id=None
        LazyCamera.last=None
        gc.collect()
        torch.cuda.empty_cache()

    def measure(function, model, camera):
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        result=function(camera,model,pipe,bg)
        image=result['render']
        image.retain_grad()
        target=camera.original_image
        l1=l1_loss(image,target)
        ssim=fused_ssim(image[None],target[None])
        loss=.8*l1+.2*(1-ssim)
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        gradients={}
        for name in FIELDS:
            gradient=getattr(model,name).grad
            if gradient is not None:
                assert torch.isfinite(gradient).all(),name
                gradients[name]=gradient.detach().cpu().clone()
            else:gradients[name]=None
        pixel_gradient=image.grad.detach().cpu().clone()
        return dict(loss=float(loss),l1=float(l1),ssim=float(ssim),seconds=time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3),gradients,image.detach().cpu().clone(),pixel_gradient

    try:
        model=GaussianModel(3)
        model.load_ply(str(source));model.active_sh_degree=3;model.training_setup(opt)
        original={name:getattr(model,name).detach().cpu().clone() for name in FIELDS}
        moments=[torch.zeros_like(p) for group in model.optimizer.param_groups for p in group['params'] for _ in range(2)]
        padding=torch.zeros(512*1024**2,dtype=torch.uint8,device='cuda')
        state['capacity_padding_bytes']=padding.numel()
        state['manual_adam_moment_bytes']=sum(t.numel()*t.element_size() for t in moments)
        for index in (0,len(rows)//2,len(rows)-1):
            camera=LazyCamera(rows[index],index)
            reference,reference_grad,reference_image,reference_pixel_grad=measure(render_imp,model,camera)
            repeat,repeat_grad,repeat_image,repeat_pixel_grad=measure(render_imp,model,camera)
            repeat2,repeat2_grad,repeat2_image,repeat2_pixel_grad=measure(render_imp,model,camera)
            candidate,candidate_grad,candidate_image,candidate_pixel_grad=measure(wrapped,model,camera)
            assert torch.equal(reference_pixel_grad,candidate_pixel_grad), 'SSIM/L1 image gradients changed'
            assert torch.equal(reference_pixel_grad,repeat_pixel_grad) and torch.equal(reference_pixel_grad,repeat2_pixel_grad)
            assert torch.equal(reference_image,repeat_image) and torch.equal(reference_image,repeat2_image)
            image_equal=torch.equal(reference_image,candidate_image)
            delta=(reference_image-candidate_image).abs()
            gradients={}
            for name in FIELDS:
                a,b=reference_grad[name],candidate_grad[name]
                assert (a is None)==(b is None),name
                if a is None:
                    gradients[name]=dict(both_none=True,passed=True);continue
                difference=(a-b).abs()
                maximum=float(a.abs().max()) if a.numel() else 0.
                maximum_error=float(difference.max()) if a.numel() else 0.
                noise=max(float((a-repeat_grad[name]).abs().max()) if a.numel() else 0., float((a-repeat2_grad[name]).abs().max()) if a.numel() else 0.,float((repeat_grad[name]-repeat2_grad[name]).abs().max()) if a.numel() else 0.)
                tolerance=max(1e-12+1e-5*maximum,2*noise)
                gradients[name]=dict(native_repeat_max_abs=noise,within_native_repeat_scale=maximum_error<=2*noise+1e-12,torch_equal=torch.equal(a,b),max_abs=maximum_error,reference_max_abs=maximum,tolerance=tolerance,passed=maximum_error<=tolerance)
                assert gradients[name]['passed'],(name,gradients[name])
            loss_error=abs(reference['loss']-candidate['loss'])
            assert image_equal and loss_error<=1e-7,(image_equal,loss_error)
            for name in FIELDS:assert torch.equal(original[name],getattr(model,name).detach().cpu()),name
            entry=dict(frame=camera.image_name,wh=[camera.width,camera.height],image_bitwise_equal=image_equal,
                image_max_abs=float(delta.max()),image_gradient_bitwise_equal=True,loss_abs_error=loss_error,reference=reference,native_repeat=repeat,native_repeat2=repeat2,candidate=candidate,gradients=gradients)
            state['comparisons'].append(entry)
            state['offload_stats']=dict(wrapped.offload_stats)
            save_json(OUT,state)
            print('MEMORY_PARITY',camera.image_name,loss_error,{k:v.get('torch_equal') for k,v in gradients.items()},flush=True)
            del reference_grad,repeat_grad,repeat2_grad,candidate_grad,reference_image,repeat_image,repeat2_image,candidate_image,reference_pixel_grad,repeat_pixel_grad,repeat2_pixel_grad,candidate_pixel_grad,delta,camera
        assert wrapped.offload_stats['offload_count']==wrapped.offload_stats['restored_count']==3
        assert wrapped.offload_stats['offload_bytes']==wrapped.offload_stats['restored_bytes']
        state['gradient_parity_passed']=True
        state['gradient_parity_scope']='Image/loss/image-gradient bitwise parity; Gaussian parameter gradients within twice observed three-run native atomic accumulation variability or original 1e-5 max-relative threshold, whichever is larger; not bitwise Gaussian gradients'
        del model,moments,padding,original
        clear()

        # Reproduce the same raw-SfM/full4 state that previously exhausted memory.
        xyz,rgb,_=read_points3D_binary(str(RUN/'swinir4x/dataset/sparse/0/points3D.bin'))
        centers=np.asarray([np.linalg.inv(r['w2c'])[:3,3] for r in rows if r['split']=='train'])
        extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
        raw=GaussianModel(0)
        raw.create_from_pcd(BasicPointCloud(xyz,rgb/255.,np.zeros_like(xyz)),extent)
        raw.training_setup(opt)
        camera=LazyCamera(rows[len(rows)//2],len(rows)//2)
        with torch.no_grad():
            ref_image=render_imp(camera,raw,pipe,bg)['render']
            ref_target=camera.original_image
            ref_l1=l1_loss(ref_image,ref_target)
            ref_ssim=fused_ssim(ref_image[None],ref_target[None])
            ref_loss=float(.8*ref_l1+.2*(1-ref_ssim))
            ref_pixels=ref_image.cpu().clone()
        del ref_image,ref_target,ref_l1,ref_ssim
        torch.cuda.empty_cache()
        result,gradients,pixels,pixel_gradient=measure(wrapped,raw,camera)
        assert torch.equal(ref_pixels,pixels)
        assert abs(ref_loss-result['loss'])<=1e-7
        state['stress']=dict(passed=True,points=len(raw.get_xyz),frame=camera.image_name,
            wh=[camera.width,camera.height],reference_no_grad_loss=ref_loss,
            image_bitwise_equal=True,loss_abs_error=abs(ref_loss-result['loss']),**result)
        state['stress_passed']=True
        state['offload_stats']=dict(wrapped.offload_stats)
        assert state['offload_stats']['offload_count']==state['offload_stats']['restored_count']==4
        assert sha256(source)==source_sha
        state.update(state='passed',finished=now(),source_model_file_unchanged=True,
            model_parameters_unchanged=True,native_loss_and_optimizer_unchanged=True,
            scope='Three full4 fitted-baseline image/loss/all-parameter-gradient checks, plus raw58509 full4 stress; zero model/optimizer updates; not a 4x training initializer')
        save_json(OUT,state)
        print('MEMORY_FULL4_STRESS',json.dumps(state['stress']),flush=True)
        print('MEMORY_NUMERICAL_PROOF_PASSED',OUT,flush=True)
    except BaseException as error:
        state.update(state='failed',error=repr(error),traceback=traceback.format_exc(),finished=now())
        save_json(OUT,state)
        raise


if __name__=='__main__':main()
