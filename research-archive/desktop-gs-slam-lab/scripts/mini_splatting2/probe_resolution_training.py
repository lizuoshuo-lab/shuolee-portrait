"""Validate calibrated paired inputs and full native backward memory before training.

Uses an existing model only as a realistic memory stress fixture. No model is
saved or evaluated for quality here. CPU image residency applies to both arms.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')
REPO=RUNTIME/'repo-depthguide';BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001')
sys.path[:0]=[str(RUNTIME/'deps'),str(REPO),str(BASE/'deps')]
os.environ['MINIGS_MOTION_MASKS']=str(ROOT/'motion-masks-001')

import torch
from arguments import ModelParams,PipelineParams,OptimizationParams
from scene import Scene,GaussianModel
from gaussian_renderer import render_imp
from native_depth_guidance import DepthGuidance
from fused_ssim import fused_ssim

spec=importlib.util.spec_from_file_location('resolution_probe_trainer',REPO/'msv2/train_depth_guided.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def main():
    torch.set_num_threads(4)
    out=ROOT/'native-resolution-probe-002';out.mkdir(exist_ok=False)
    import diff_gaussian_rasterization as nr
    import diff_gaussian_rasterization_ms as nm
    libraries={k:v._C.__file__ for k,v in [('rgb',nr),('ms',nm)]}
    assert all(str(RUNTIME/'deps') in v for v in libraries.values())
    target=out/'summary.json'
    state=dict(status='running',started=time.time(),rows=[],arms={},accepted=False,libraries=libraries,
        purpose='Input, mask, prior and memory check only; not training or quality acceptance',
        image_residency='cpu in BOTH arms; loss inputs transferred to cuda per frame')
    def save():target.write_text(json.dumps(state,indent=2))
    save()
    try:
        pair=ROOT/'source-resolution-rectified-002'
        inputs=json.loads((pair/'resolution-ablation-inputs.json').read_text())
        assert len(inputs['rows'])==59
        camera_geometry={}
        for arm,folder in [('low','matched-low-resolution-control'),('full','dataset')]:
            armout=out/arm;armout.mkdir()
            parser=argparse.ArgumentParser();mp=ModelParams(parser);pp=PipelineParams(parser);op=OptimizationParams(parser)
            args=parser.parse_args(['-s',str(pair/folder),'-m',str(armout),'-r','1','--eval','--data_device','cpu'])
            pc=GaussianModel(3);scene=Scene(mp.extract(args),pc,shuffle=False,resolution_scales=[1,2])
            train=scene.getTrainCameras();test=scene.getTestCameras()
            assert len(train)==51 and len(test)==8
            assert all(int(c.image_name)%8!=0 for c in train)
            assert all(int(c.image_name)%8==0 for c in test)
            camera_geometry[arm]={c.image_name:dict(fov=[c.FoVx,c.FoVy],
                view=c.world_view_transform.cpu().tolist(),size=[c.image_width,c.image_height]) for c in train+test}
            pc.load_ply(str(ROOT/'native-depthspread000-18000-s0/model/point_cloud/iteration_18000/point_cloud.ply'))
            pc._mip_cameras=train;pc.init_culling(len(train));pc.training_setup(op.extract(args))
            guidance=DepthGuidance(ROOT/'depth-priors',train,armout,.003)
            trainer.MOTION_CACHE.clear();trainer.MOTION_USAGE.clear()
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            cached=[]
            for scale in [1,2]:
                for c in scene.getTrainCameras(scale):
                    gt=c.original_image.cuda();valid=trainer.motion_valid(c,gt)
                    inv,confidence,normalizer=guidance.prior(c)
                    assert inv.shape==confidence.shape==valid.shape[-2:]
                    assert torch.isfinite(inv).all() and torch.isfinite(confidence).all()
                    assert set(valid.unique().tolist()) <= {0.,1.}
                    cached.append(dict(frame=c.image_name,scale=scale,size=[c.image_width,c.image_height],
                        excluded_fraction=float((1-valid).mean()),confidence_fraction=float((confidence>0).float().mean())))
                    del gt
            background=torch.zeros(3,device='cuda')
            for c in train:
                if int(c.image_name) not in [1,9,17,25,33,41,49,57]:continue
                pc._mip_iteration=6000
                start=time.time();gt=c.original_image.cuda();valid=trainer.motion_valid(c,gt)
                pkg=render_imp(c,pc,pp.extract(args),background,culling=pc._culling[:,c.uid])
                pred=pkg['render']
                l1=((pred-gt).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
                structural=1-fused_ssim((pred*valid+gt*(1-valid))[None],gt[None])
                dl=guidance(c,pc,6000,valid,pc._culling[:,c.uid])
                loss=.8*l1+.2*structural+dl;loss.backward()
                params=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
                assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                pc.optimizer.step(pkg['radii']>0,pkg['radii'].shape[0]);pc.optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
                state['rows'].append(dict(arm=arm,frame=c.image_name,loss=float(loss),depth_loss=float(dl),
                    gaussians=len(pc.get_xyz),seconds=time.time()-start,free_mib=free/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                save();del pkg,pred,gt,valid,l1,structural,dl,loss,params
            state['arms'][arm]=dict(cameras_cached=len(cached),cached=cached,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            del pc,scene,train,test,guidance,background,inv,confidence,normalizer
            trainer.MOTION_CACHE.clear();trainer.MOTION_USAGE.clear();gc.collect();torch.cuda.empty_cache();save()
        for name,a in camera_geometry['low'].items():
            b=camera_geometry['full'][name]
            assert a['view']==b['view'] and a['fov']==b['fov']
            assert [2*x for x in a['size']]==b['size']
        low=state['arms']['low']['cached'];full=state['arms']['full']['cached']
        paired={r['frame']:r for r in low if r['scale']==1}
        same=[r for r in full if r['scale']==2]
        assert all(abs(r['excluded_fraction']-paired[r['frame']]['excluded_fraction'])<1e-7 for r in same)
        minimum_free=min(r['free_mib'] for r in state['rows'])
        assert minimum_free>2000,'Insufficient GPU headroom; do not start full training'
        state.update(status='passed',finished=time.time(),geometry_equivalence=True,
            masks_at_same_resolution_identical=True,minimum_free_mib=minimum_free,
            training_ready=True,limitation='192k Gaussian backward fixture; full densification peak not guaranteed; smoke and monitor next')
        save();print(json.dumps({k:v for k,v in state.items() if k not in ['arms','rows']}),flush=True)
    except Exception as e:
        state.update(status='failed',error=str(e),finished=time.time());save();raise


if __name__=='__main__':main()
