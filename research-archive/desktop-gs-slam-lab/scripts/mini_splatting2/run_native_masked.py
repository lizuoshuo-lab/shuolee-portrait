"""Controlled native Mini-Splatting2 RGB-mask intervention.

Only RGB loss excludes the reviewed moving background person. Native depth
reinitialization remains unchanged in this first ablation and is disclosed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-001')

def main():
    masks=ROOT/'motion-masks-001'
    review=json.loads((masks/'visual-review.json').read_text());assert review['accepted_for_rgb_loss']
    baseline=json.loads((ROOT/'native-baseline-18000-s0/manifest.json').read_text());assert baseline['status']=='complete'
    out=ROOT/'native-motion-rgb-18000-s0';out.mkdir(exist_ok=False)
    repo=RUNTIME/'repo';source=(repo/'msv2/train.py').read_text()
    helper='''
import json as motion_json
from pathlib import Path as MotionPath
from PIL import Image as MotionImage
import numpy as motion_np
MOTION_ROOT=MotionPath(os.environ['MINIGS_MOTION_MASKS'])
MOTION_CACHE={}
MOTION_USAGE={}
def motion_valid(camera,gt):
    key=(camera.image_name,gt.shape[-2],gt.shape[-1])
    MOTION_USAGE[camera.image_name]=MOTION_USAGE.get(camera.image_name,0)+1
    if key not in MOTION_CACHE:
        x=motion_np.asarray(MotionImage.open(MOTION_ROOT/(camera.image_name+'.png')).convert('L')).copy()
        mask=torch.from_numpy((x>0).astype('float32'))[None,None].to(gt.device)
        mask=torch.nn.functional.interpolate(mask,size=gt.shape[-2:],mode='nearest')[0]
        MOTION_CACHE[key]=1-mask
    return MOTION_CACHE[key]
'''
    assert source.count('import torch\n')==1
    source=source.replace('import torch\n','import torch\n'+helper)
    needle='        gt_image = viewpoint_cam.original_image.cuda()'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'\n        valid = motion_valid(viewpoint_cam, gt_image)')
    source=source.replace('Ll1 = l1_loss(image, gt_image)','Ll1 = ((image-gt_image).abs()*valid).sum()/(3*valid.sum().clamp_min(1))')
    source=source.replace('ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))',
        'ssim_value = fused_ssim((image*valid+gt_image*(1-valid)).unsqueeze(0), gt_image.unsqueeze(0))')
    source+='\nif __name__ == "__main__":\n    (MotionPath(args.model_path)/"mask-usage.json").write_text(motion_json.dumps(MOTION_USAGE,indent=2))\n'
    script=repo/'msv2/train_motion_rgb.py';script.write_text(source)
    (out/'train_motion_rgb.py').write_text(source)
    config=out/'config';config.write_text((repo/'config/fast').read_text())
    target=out/'model';env=os.environ.copy()
    env.update(PYTHONPATH=str(RUNTIME/'deps')+os.pathsep+str(repo),OMP_NUM_THREADS='4',MINIGS_MOTION_MASKS=str(masks))
    command=[sys.executable,str(script),'-s',str(RUNTIME/'dataset'),'-m',str(target),'-r','1','--eval',
        '--imp_metric','indoor','--config_path',str(config),'--test_iterations','-1','--port','6119']
    manifest=dict(status='running',command=command,steps=18000,seed=0,started=time.time(),
        intervention='reviewed training-only SAM2 mask on moving background person; RGB L1 and SSIM only',
        mask_review_sha256=hashlib.sha256((masks/'visual-review.json').read_bytes()).hexdigest(),
        training_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        compared_to='native-baseline-18000-s0',depth_reinitialization_masked=False,
        equivalence='same initialization, split, resolution, RNG seed, native topology operations and iteration budget; topology trajectory may diverge',
        limitation='static scene experiment excluding one moving background person, not dynamic person reconstruction')
    mp=out/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    with (out/'train.log').open('w') as log:result=subprocess.run(command,cwd=repo/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(returncode=result.returncode,elapsed_seconds=time.time()-manifest['started'],status='complete' if result.returncode==0 else 'failed')
    ply=target/'point_cloud/iteration_18000/point_cloud.ply'
    if result.returncode==0:
        assert ply.exists();manifest['ply']=str(ply);manifest['ply_sha256']=hashlib.sha256(ply.read_bytes()).hexdigest()
    mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
