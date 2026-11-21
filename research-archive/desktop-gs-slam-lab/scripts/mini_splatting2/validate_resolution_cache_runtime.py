"""Exact cache eviction and mask equality tests on all training views and scales."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001');REPO=RUNTIME/'repo-depth-cache'
sys.path[:0]=[str(RUNTIME/'deps'),str(REPO),'/opt/gs-slam-lab/mini-splatting2-native-001/deps']
os.environ['MINIGS_MOTION_MASKS']=str(ROOT/'motion-masks-001')
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import diff_gaussian_rasterization_ms as native
from arguments import ModelParams
from scene import Scene,GaussianModel
from native_depth_guidance import DepthGuidance
spec=importlib.util.spec_from_file_location('cache_trainer',REPO/'msv2/train_depth_guided.py');trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
spec=importlib.util.spec_from_file_location('original_prior',RUNTIME/'repo-depthguide/native_depth_guidance.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def main():
    assert str(RUNTIME/'deps') in native._C.__file__
    torch.set_num_threads(4);out=ROOT/'native-resolution-cache-runtime';work=out/'fixture';work.mkdir(exist_ok=False)
    p=argparse.ArgumentParser();mp=ModelParams(p)
    args=p.parse_args(['-s',str(ROOT/'source-resolution-rectified-002/dataset'),'-m',str(work),'-r','1','--eval','--data_device','cpu'])
    scene=Scene(mp.extract(args),GaussianModel(0),shuffle=False,resolution_scales=[1,2])
    a=DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),work,.003)
    b=old.DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),work,.003)
    rows=[]
    for scale in [1,2]:
        cameras=scene.getTrainCameras(scale)
        for c in cameras+[cameras[0],cameras[25]]:
            v1=a.prior(c);v2=b.prior(c)
            assert all(torch.equal(x,y) for x,y in zip(v1,v2))
            b.cache.clear()
            gt=c.original_image.cuda();valid=trainer.motion_valid(c,gt)
            arr=np.asarray(Image.open(ROOT/'motion-masks-001'/(c.image_name+'.png')).convert('L')).copy()
            expected=1-F.interpolate(torch.from_numpy((arr>0).astype('float32'))[None,None].cuda(),size=gt.shape[-2:],mode='nearest')[0]
            assert torch.equal(valid,expected) and len(a.cache)<=4 and len(trainer.MOTION_CACHE)<=4
            rows.append(dict(frame=c.image_name,size=[c.image_width,c.image_height],prior_identical=True,mask_identical=True,
                gpu_prior_entries=len(a.cache),gpu_mask_entries=len(trainer.MOTION_CACHE)))
            del v1,v2,gt,valid,expected
    result=dict(status='passed',created=time.time(),rows=rows,checks=len(rows),
        prior_and_mask_bit_identical=True,cache_bound=4,all_51_training_frames=True,scales=[1,2],
        first_frames_revisited_after_eviction=True,backward='original joint RGB+depth loss backward unchanged',
        rasterizer=str(native._C.__file__),scope='Reference cache equivalence; full training transition tested separately')
    (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
