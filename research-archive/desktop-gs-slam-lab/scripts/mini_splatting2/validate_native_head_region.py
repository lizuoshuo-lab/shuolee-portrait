"""Validate the prepared head loss with real native RGB+depth parameter gradients.

Read-only model fixture. No optimizer steps, training schedule changes or quality
claims. Run only after the active training/evaluation GPU queue has completed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')
REPO=RUNTIME/'repo-depth-cache'
sys.path[:0]=[str(RUNTIME/'deps'),str(REPO),'/opt/gs-slam-lab/mini-splatting2-native-001/deps']
os.environ['MINIGS_MOTION_MASKS']=str(ROOT/'motion-masks-001')
import torch
import diff_gaussian_rasterization_ms as native
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render_imp
from fused_ssim import fused_ssim
from native_depth_guidance import DepthGuidance
from head_region_l1 import TrainingHeadL1

spec=importlib.util.spec_from_file_location('head_fixture_trainer',REPO/'msv2/train_depth_guided.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def main():
    assert str(RUNTIME/'deps') in native._C.__file__
    for name in ['native-late-queue.json','native-late-publication-queue.json','native-training-fit-queue.json']:
        state=json.loads((ROOT/name).read_text())['status']
        assert state.startswith('complete'),f'GPU work not finished: {name}: {state}'
    torch.set_num_threads(4)
    out=ROOT/'native-head-gradient-validation-001';out.mkdir(exist_ok=False)
    model=ROOT/'native-resolution-cache-full-18000-s0/model'
    model_file=model/'point_cloud/iteration_18000/point_cloud.ply'
    before=hashlib.sha256(model_file.read_bytes()).hexdigest()
    protocol=dict(created=time.time(),fixture=str(model_file),fixture_sha256=before,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        head_loss_sha256=hashlib.sha256(Path(__file__).with_name('head_region_l1.py').read_bytes()).hexdigest(),
        frames_by_training_index=[0,21,49],scales=[1,2],coefficient=2.,
        backward='original joint RGB L1 + SSIM + depth',training_updated=False,
        culling='all false for every branch of this read-only fixture',
        gate='neutral loss/gradients within float tolerance, repeatability reported, candidate nonzero finite gradients')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    p=argparse.ArgumentParser();mp=ModelParams(p);pp=PipelineParams(p)
    args=p.parse_args(['-s',str(ROOT/'source-resolution-rectified-002/dataset'),'-m',str(model),'-r','1','--eval','--data_device','cpu'])
    pc=GaussianModel(3);scene=Scene(mp.extract(args),pc,load_iteration=18000,shuffle=False,resolution_scales=[1,2])
    pc._mip_cameras=scene.getTrainCameras();pc._mip_iteration=18000
    prior=DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),out,.003)
    neutral=TrainingHeadL1(ROOT/'training-head-regions-003',1.)
    weighted=TrainingHeadL1(ROOT/'training-head-regions-003',2.)
    parameters=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
    names=['xyz','features_dc','features_rest','opacity','scaling','rotation']
    background=torch.zeros(3,device='cuda');culled=torch.zeros(len(pc.get_xyz),device='cuda',dtype=torch.bool)
    rows=[]
    for scale,index in [(2,0),(1,0),(1,21),(1,49)]:
        camera=scene.getTrainCameras(scale)[index]
        gt=camera.original_image.cuda();valid=trainer.motion_valid(camera,gt)
        def calculate(mode):
            for parameter in parameters:parameter.grad=None
            torch.cuda.reset_peak_memory_stats()
            package=render_imp(camera,pc,pp.extract(args),background,culling=culled)
            image=package['render']
            if mode=='original':l1=((image-gt).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
            else:l1=(neutral if mode=='neutral' else weighted)(camera.image_name,image,gt,valid)
            rgb=.8*l1+.2*(1-fused_ssim((image*valid+gt*(1-valid))[None],gt[None]))
            depth=prior(camera,pc,18001,valid,culled)
            loss=rgb+depth;loss.backward();torch.cuda.synchronize()
            gradients=[parameter.grad.detach().cpu().clone() for parameter in parameters]
            gradients.append(package['viewspace_points'].grad.detach().cpu().clone())
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
            return dict(loss=float(loss),l1=float(l1),depth=float(depth),peak_mib=torch.cuda.max_memory_allocated()/2**20),gradients
        original,ga=calculate('original');repeat,gr=calculate('original');control,gc=calculate('neutral');candidate,gw=calculate('weighted')
        def errors(a,b):
            return {name:dict(max_absolute=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-10)))
                for name,x,y in zip(names+['viewspace'],a,b)}
        e=errors(ga,gc);repeat_error=errors(ga,gr);change=errors(ga,gw)
        passed=abs(original['loss']-control['loss'])<1e-7 and all(x['max_absolute']<1e-5 and x['relative_l2']<2e-4 for x in e.values())
        passed=passed and abs(candidate['l1']-original['l1'])>1e-8 and any(x['relative_l2']>1e-4 for x in change.values())
        row=dict(frame=camera.image_name,size=[camera.image_width,camera.image_height],original=original,neutral=control,weighted=candidate,
            neutral_gradient_error=e,original_repeat_error=repeat_error,weighted_gradient_change=change,passed=passed)
        rows.append(row);(out/'partial.json').write_text(json.dumps(dict(status='running' if passed else 'failed',rows=rows),indent=2))
        print(json.dumps(dict(frame=camera.image_name,size=row['size'],passed=passed)),flush=True)
        assert passed,'Inspect the frozen per-parameter diagnostic before integration'
        del ga,gr,gc,gw,gt,valid
    after=hashlib.sha256(model_file.read_bytes()).hexdigest();assert before==after
    result=dict(status='passed',created=time.time(),protocol=protocol,rows=rows,model_unchanged=True,
        libraries=dict(ms_path=native._C.__file__,ms_sha256=hashlib.sha256(Path(native._C.__file__).read_bytes()).hexdigest()),
        training_updated=False,integrated_into_trainer=False,quality_improvement_proven=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='passed',checks=len(rows))),flush=True)


if __name__=='__main__':main()
