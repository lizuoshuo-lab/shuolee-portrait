"""Fixed out-of-capture camera paths, real training reference, image diffusion pilot.

Never substitutes generated images for held-out photographs. Outputs are clearly
separated from 3D model updates and retain depth/camera records for consistency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch

S=Path(__file__).parent; LAB=S.parents[1]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001'
OUT=ROOT/'difix-novel-paths-001'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


@torch.no_grad()
def prepare():
    from local_refinement import read_model, render
    OUT.mkdir(exist_ok=False); torch.set_num_threads(4)
    for folder in ['input','depth','reference','output']: (OUT/folder).mkdir()
    manifest=json.loads((ROOT/'native-resolution-cache-full-18000-s0/manifest.json').read_text())
    model_path=Path(manifest['baked_ply']); assert digest(model_path)==manifest['baked_ply_sha256']
    cameras=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda x:x['img_name'])
    anchor=next(x for x in cameras if x['img_name']=='000024')
    training=[x for i,x in enumerate(cameras) if i%8!=0]
    rotation=np.array(anchor['rotation']);position=np.array(anchor['position'])
    def distance(c): return np.arccos(np.clip((np.trace(rotation.T@np.array(c['rotation']))-1)/2,-1,1))+np.linalg.norm(position-np.array(c['position']))/5
    reference=min(training,key=distance); assert int(reference['img_name'])%8!=0
    reference_path=ROOT/'source-resolution-rectified-002/dataset/images'/(reference['img_name']+'.png')
    Image.open(reference_path).convert('RGB').resize((576,1024),Image.Resampling.LANCZOS).save(OUT/'reference/training.png')
    paths_path=ROOT/'geometry-evaluation/paths.json'
    paths=json.loads(paths_path.read_text())['frames']
    assert len(paths)==120 and {x['kind'] for x in paths}=={'distance','elevation'}
    protocol=dict(status='frozen_before_generation',created=time.time(),model='native-resolution-cache-full-18000-s0',
        model_sha256=digest(model_path),paths_sha256=digest(paths_path),reference=reference['img_name'],
        reference_original_sha256=digest(reference_path),reference_policy='One real training photo fixed across both paths; no reference switching',
        resolution=[576,1024],frames_per_path=60,paths=['distance','elevation'],fps=10,
        diffusion=dict(name='nvidia/difix_ref',revision='d4830559772a5795c9d136302c5b197d6418d3fb',
                       seed=0,prompt='remove degradation',steps=1,timestep=199,dtype='float16'),
        rendering='gsplat with baked3Dfilter once and view-dependent2DMip compensation',
        requested_evaluation=['subjective detail clarity','appearance/identity preservation','anatomy and occlusion','motion continuity','overall preference'],
        no_gt_for_extrapolated_cameras=True,psnr_not_used_for_path_ranking=True,
        geometry_updated=False,distilled_to_3d=False,accepted=False,
        caveat='Image diffusion pilot on actual3D projections; neither new geometry nor temporally joint video diffusion',
        sources=['https://github.com/nv-tlabs/Difix3D','https://research.nvidia.com/labs/toronto-ai/difix3d/'])
    dump(OUT/'protocol.json',protocol)
    model=read_model(model_path);rows=[]
    for path in paths:
        c=dict(anchor);c.update(width=576,height=1024,fx=anchor['fx']*576/anchor['width'],fy=anchor['fy']*1024/anchor['height'])
        R=np.array(path['rotation']);P=np.array(path['position']);V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@P
        c['viewmat']=torch.tensor(V,device='cuda',dtype=torch.float32)[None]
        rgb,alpha,_=render(model,c,mode='mip',depth=True)
        name=f'{path["kind"]}-{path["index"]:03d}'
        Image.fromarray((rgb[...,:3].clamp(0,1)*255).round().byte().cpu().numpy()).save(OUT/'input'/(name+'.png'))
        np.savez_compressed(OUT/'depth'/(name+'.npz'),depth=rgb[...,3].cpu().numpy(),alpha=alpha[...,0].cpu().numpy())
        rows.append(dict(name=name,**path,width=576,height=1024,fx=c['fx'],fy=c['fy'],viewmat=V.tolist()))
    assert digest(model_path)==protocol['model_sha256']
    dump(OUT/'cameras.json',rows);dump(OUT/'status.json',dict(status='inputs_ready',frames=len(rows),created=time.time()))
    print(json.dumps(dict(status='inputs_ready',frames=len(rows),reference=reference['img_name'])),flush=True)


def infer():
    # Reuse the pinned isolated dependency folder; never replace training packages.
    sys.path.insert(0,'/opt/gs-slam-lab/mini-splatting2-refine/diffusion-deps')
    from evaluate_difix import load_class, DifixPipeline
    torch.set_num_threads(4)
    root=ROOT/'sources/difix_ref'
    unet=load_class('unet_2d_condition','unet','UNet2DConditionModel').from_pretrained(root/'unet',torch_dtype=torch.float16,local_files_only=True)
    vae=load_class('autoencoder_kl','vae','AutoencoderKL').from_pretrained(root/'vae',torch_dtype=torch.float16,local_files_only=True)
    pipe=DifixPipeline.from_pretrained(root,unet=unet,vae=vae,torch_dtype=torch.float16,local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    protocol=json.loads((OUT/'protocol.json').read_text());rows=json.loads((OUT/'cameras.json').read_text())
    reference=Image.open(OUT/'reference/training.png').convert('RGB');results=[];start=time.time()
    assert not (OUT/'summary.json').exists()
    # Keep source order, weights, seed and reference fixed, without selecting successful frames.
    for row in rows:
        path=OUT/'output'/(row['name']+'.png');assert not path.exists()
        image=Image.open(OUT/'input'/(row['name']+'.png')).convert('RGB');began=time.time()
        with torch.inference_mode():
            result=pipe('remove degradation',image=image,ref_image=reference,height=1024,width=576,
                num_inference_steps=1,timesteps=[199],guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(0)).images[0]
        result.save(path);torch.cuda.synchronize()
        record=dict(name=row['name'],seconds=time.time()-began,input_sha256=digest(OUT/'input'/(row['name']+'.png')),
            output_sha256=digest(path),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        results.append(record)
        dump(OUT/'status.json',dict(status='generating',completed=len(results),total=len(rows),last=record,elapsed_seconds=time.time()-start))
        if len(results)%10==0: print(json.dumps(dict(completed=len(results),total=len(rows),elapsed_seconds=time.time()-start)),flush=True)
    dump(OUT/'summary.json',dict(status='complete_pending_visual_and_temporal_review',protocol=protocol,rows=results,
        elapsed_seconds=time.time()-start,geometry_updated=False,distilled_to_3d=False,accepted=False))
    dump(OUT/'status.json',dict(status='generation_complete',completed=len(results),total=len(rows),elapsed_seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','infer']);args=p.parse_args()
    if args.action=='prepare': prepare()
    else: infer()
