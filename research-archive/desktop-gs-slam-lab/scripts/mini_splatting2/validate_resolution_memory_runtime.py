"""Actual-scene joint/sequential gradient equivalence and cache eviction checks."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001');REPO=RUNTIME/'repo-depth-memory'
sys.path[:0]=[str(RUNTIME/'deps'),str(REPO),'/opt/gs-slam-lab/mini-splatting2-native-001/deps']
os.environ['MINIGS_MOTION_MASKS']=str(ROOT/'motion-masks-001')
import numpy as np
import torch
import diff_gaussian_rasterization_ms as native_ms
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from native_depth_guidance import DepthGuidance
from gaussian_renderer import render_imp
from fused_ssim import fused_ssim
spec=importlib.util.spec_from_file_location('bounded_trainer',REPO/'msv2/train_depth_guided.py');trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
spec=importlib.util.spec_from_file_location('unbounded_depth',RUNTIME/'repo-depthguide/native_depth_guidance.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def main():
    assert str(RUNTIME/'deps') in native_ms._C.__file__
    torch.set_num_threads(4);out=ROOT/'native-resolution-memory-runtime';target=out/'validation.json';assert not target.exists()
    work=out/'fixture-attempt002';work.mkdir()
    p=argparse.ArgumentParser();mp=ModelParams(p);pp=PipelineParams(p)
    args=p.parse_args(['-s',str(ROOT/'source-resolution-rectified-002/dataset'),'-m',str(work),'-r','1','--eval','--data_device','cpu'])
    pc=GaussianModel(3);scene=Scene(mp.extract(args),pc,shuffle=False,resolution_scales=[1,2])
    pc.load_ply(str(ROOT/'native-resolution-low-18000-s0/model/point_cloud/iteration_18000/point_cloud.ply'))
    pc._mip_cameras=scene.getTrainCameras();pc.init_culling(51);pc._mip_iteration=6000
    bounded=DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),work,.003)
    unbounded=old.DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),work,.003)
    cache_checks=[]
    # Identical prior values before AND after eviction, both input resolutions.
    sequence=[scene.getTrainCameras(s)[i] for s in [1,2] for i in [0,1,2,3,4,0]]
    for c in sequence:
        a=bounded.prior(c);b=unbounded.prior(c)
        errors=[float((x-y).abs().max()) for x,y in zip(a,b)]
        valid=trainer.motion_valid(c,c.original_image.cuda())
        assert max(errors)==0 and len(bounded.cache)<=4 and len(trainer.MOTION_CACHE)<=4
        cache_checks.append(dict(frame=c.image_name,size=[c.image_width,c.image_height],max_errors=errors,cache_entries=len(bounded.cache)))
    del unbounded,sequence,a,b,valid;bounded.cache.clear();trainer.MOTION_CACHE.clear();gc.collect();torch.cuda.empty_cache()
    params=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
    background=torch.zeros(3,device='cuda');rows=[]
    for scale,index in [(2,0),(1,0),(1,21),(1,49)]:
        c=scene.getTrainCameras(scale)[index];gt=c.original_image.cuda();valid=trainer.motion_valid(c,gt);bounded.prior(c)
        def calculate(sequential):
            for parameter in params:parameter.grad=None
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.time()
            pkg=render_imp(c,pc,pp.extract(args),background,culling=pc._culling[:,c.uid]);im=pkg['render']
            l1=((im-gt).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
            rgb=.8*l1+.2*(1-fused_ssim((im*valid+gt*(1-valid))[None],gt[None]))
            if sequential:rgb.backward()
            depth=bounded(c,pc,6001,valid,pc._culling[:,c.uid])
            if sequential:depth.backward()
            else:(rgb+depth).backward()
            gradients=[parameter.grad.detach().cpu().clone() for parameter in params]
            screen=pkg['viewspace_points'].grad.detach().cpu().clone()
            torch.cuda.synchronize()
            return dict(loss=float(rgb.detach()+depth.detach()),peak_mib=torch.cuda.max_memory_allocated()/2**20,
                seconds=time.time()-start),gradients,screen
        a,ga,sa=calculate(False);b,gb,sb=calculate(True)
        errors=[dict(max_absolute=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-10))) for x,y in zip(ga,gb)]
        passed=abs(a['loss']-b['loss'])<1e-7 and all(e['max_absolute']<1e-5 and e['relative_l2']<2e-4 for e in errors) and float((sa-sb).abs().max())<1e-5
        rows.append(dict(frame=c.image_name,size=[c.image_width,c.image_height],joint=a,sequential=b,gradient_errors=errors,
            screen_gradient_max_error=float((sa-sb).abs().max()),passed=passed))
        (out/'validation-attempt002.json').write_text(json.dumps(dict(status='running' if passed else 'failed',rows=rows),indent=2))
        print(json.dumps(rows[-1]),flush=True)
        assert passed,'Inspect saved per-parameter discrepancies before further changes'
        del ga,gb,sa,sb
    result=dict(status='passed',created=time.time(),cache_checks=cache_checks,rows=rows,
        loss_and_schedule_unchanged=True,gradients_equivalent_within_float_tolerance=True,
        rgb_screen_gradient_preserved=True,cuda_library=native_ms._C.__file__,
        scope='Actual202kGaussian fixture,51training cameras,4RGB/depth checks including3fullresolution frames. Not a quality claim.')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],memory=[dict(size=r['size'],joint=r['joint']['peak_mib'],sequential=r['sequential']['peak_mib']) for r in rows])),flush=True)


if __name__=='__main__':main()
