"""Read-only old-model full-render crop / viewport forward-backward proof."""

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--backend-dir', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--mask-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0,args.package)
    from msv2.io import read_json, save_json, sha256
    from msv2.roi import wrap_renderer, verify_backend, backend_path
    from msv2.memory import wrap_training_render
    from msv2.paired_data import child_roi, child_camera
    from msv2.engine import LazyCamera
    run=Path(args.run); maskrun=Path(args.mask_run); output=Path(args.output)
    job=read_json(run/'job.json'); repo=Path(job['source_dir'])
    sys.path.insert(0,str(run/'source'))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render_imp
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(job['options']['gpu_memory_fraction'])
    torch.manual_seed(516)
    torch.cuda.manual_seed_all(516)
    ply=run/'original/model/point_cloud/iteration_18000/point_cloud.ply'
    inputs=run/'original/training-inputs.json'
    srfile=maskrun/'swinir4x/superresolution.json'
    sourcefiles=[run/'job.json',run/'state.json',run/'plan.json',inputs,srfile,ply,
                 Path(args.package)/'msv2/roi.py',Path(args.package)/'msv2/roi_backend.patch',
                 run/'source/gaussian_renderer/__init__.py',run/'source/scene/gaussian_model.py']
    initial={str(p):sha256(p) for p in sourcefiles}
    rows={r['frame']:r for r in read_json(inputs)['records']}
    sr={r['name']:r for r in read_json(srfile)['records']}
    g=GaussianModel(3)
    g.load_ply(str(ply))
    assert g.get_xyz.shape[0]==539423
    assert g.optimizer is None
    params=[g._xyz,g._features_dc,g._features_rest,g._opacity,g._scaling,g._rotation]
    param_names=['xyz','dc','sh','opacity','scaling','rotation','screen_ndc']
    fullrender=wrap_training_render(render_imp,repo)
    roirender=wrap_renderer(render_imp,repo,args.backend_dir)
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)
    bg=torch.zeros(3,device='cuda')
    proof=dict(schema='msv2-roi-real-model-proof-v1',state='running',model_updates=0,optimizer_updates=0,
        optimizer_constructed=False,old_model_used_as_training_initializer=False,
        scope='Read-only diagnostic of frozen original1 PLY; no model/quality acceptance or training',
        point_count=539423,ply=str(ply),protected_before=initial,
        backend=verify_backend(repo,args.backend_dir),backend_directory=str(backend_path(repo,args.backend_dir)),
        gpu=torch.cuda.get_device_name(),memory_fraction=job['options']['gpu_memory_fraction'],cases=[])
    save_json(output/'proof.json',proof)

    def clear():
        for p in params:p.grad=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()

    try:
        for frame in (1,33,65):
            row=rows[f'frame_{frame:06d}']
            srecord=sr[row['name']]
            face=srecord['faces'][0]
            mask=maskrun/'swinir4x/face-masks'/face['mask_name']
            if not mask.exists():
                mask=maskrun/'swinir4x/face-mask'/face['mask_name']
            assert sha256(mask)==face['mask_sha256']
            initial[str(mask)]=sha256(mask)
            initial[row['native_image']]=sha256(row['native_image'])
            assert sha256(row['native_image'])==row['native_sha256']==srecord['native_sha256']
            with Image.open(mask) as im:
                assert im.size==tuple(row['wh'])
                assert list(im.getbbox())==face['bbox_native']
            roi_native=child_roi(face['bbox_native'],row['wh'])
            for level in (2,1):
                before_case=time.perf_counter()
                mother=LazyCamera(row,uid=0,level=level)
                camera_metadata=child_camera(row['K'],row['w2c'],row['wh'],roi_native,
                    warmup_factor=level)
                fw,fh=camera_metadata['full_wh']
                x,y=camera_metadata['viewport_xy'];w,h=camera_metadata['wh']
                full=SimpleNamespace(**vars(mother));full.image_width,full.image_height=fw,fh
                child=SimpleNamespace(**vars(mother));child.image_width,child.image_height=w,h
                child._msv2_roi=camera_metadata['_msv2_roi']
                assert full.FoVx==child.FoVx==mother.FoVx
                assert full.full_proj_transform is child.full_proj_transform is mother.full_proj_transform
                vw,vh=min(w,fw-x),min(h,fh-y)
                grad=torch.randn(3,h,w,device='cuda') / (3*h*w)
                grad[:,vh:,:]=0;grad[:,:,vw:]=0
                native=[];candidate=[];peaks=[];times=[]
                def execute(render,cam,is_roi):
                    clear();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    package=render(cam,g,pipe,bg,flag_max_count=True)
                    image=package['render']
                    selected=image if is_roi else image[:,y:y+vh,x:x+vw]
                    captured=selected.detach().cpu().clone()
                    selected.backward(grad if is_roi else grad[:,:vh,:vw])
                    torch.cuda.synchronize()
                    gradients=[p.grad.detach().cpu().clone() for p in params]
                    gradients.append(package['viewspace_points'].grad.detach().cpu().clone())
                    peak=dict(allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                              reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                    elapsed=time.perf_counter()-started
                    result=dict(rgb=captured,gradients=gradients)
                    del selected,image,package
                    clear()
                    peaks.append(dict(mode='roi' if is_roi else 'full',**peak));times.append(elapsed)
                    return result
                for _ in range(3):native.append(execute(fullrender,full,False))
                for _ in range(3):candidate.append(execute(roirender,child,True))
                rgb_error=max(float((n['rgb']-r['rgb'][:,:vh,:vw]).abs().max()) for n in native for r in candidate)
                assert rgb_error==0,('RGB',frame,level,rgb_error)
                comparisons=[]
                for index,name in enumerate(param_names):
                    native_spread=max(float((a['gradients'][index]-b['gradients'][index]).abs().max()) for a in native for b in native)
                    delta=max(float((a['gradients'][index]-b['gradients'][index]).abs().max()) for a in native for b in candidate)
                    maximum=max(float(n['gradients'][index].abs().max()) for n in native)
                    tolerance=max(1e-7,1e-5*maximum,2*native_spread)
                    comparisons.append(dict(name=name,max_error=delta,reference_max=maximum,
                        native_three_repeat_max_error=native_spread,tolerance=tolerance,passed=delta<=tolerance))
                case=dict(frame=row['frame'],level=level,full_wh=[fw,fh],roi_native=roi_native,
                    viewport_xy=[x,y],roi_wh=[w,h],source_bbox=face['bbox_native'],
                    mask_path=str(mask),mask_sha256=face['mask_sha256'],source_png_sha256=row['native_sha256'],
                    rgb_bitwise_equal=rgb_error==0,rgb_max_error=rgb_error,gradients=comparisons,
                    peak_memory=peaks,render_backward_seconds=times,seconds=time.perf_counter()-before_case,
                    full_render_memory_offload=fullrender.memory_stats,
                    passed=rgb_error==0 and all(c['passed'] for c in comparisons))
                proof['cases'].append(case)
                save_json(output/'proof.json',proof)
                print(json.dumps(dict(frame=frame,level=level,passed=case['passed'],rgb_max_error=rgb_error,
                    peak_gib=max(p['allocated_gib'] for p in peaks),seconds=case['seconds'])),flush=True)
                assert case['passed'],('Gradient parity',frame,level,comparisons)
                del native,candidate,full,child,mother,grad
                clear()
        proof.update(state='passed',quality_accepted=False)
    except Exception as error:
        proof.update(state='failed',error=str(error),error_type=type(error).__name__)
        raise
    finally:
        final={path:sha256(path) for path in initial}
        proof.update(protected_after=final,protected_unchanged=final==initial)
        save_json(output/'proof.json',proof)
        print('ROI_REAL_PROOF_STATE',proof['state'],'PROTECTED',proof['protected_unchanged'],flush=True)


if __name__=='__main__':main()
