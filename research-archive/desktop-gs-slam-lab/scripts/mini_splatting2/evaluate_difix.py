"""Bounded diffusion appearance test; no generated view enters 3D training.

Only real TRAINING images may be used as references. Evaluation photos never
condition the model. Improvements here do not demonstrate improved geometry.
"""
import importlib.util
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001'
sys.path.insert(0,str(LAB/'repos/Difix3D/src'))
from pipeline_difix import DifixPipeline


def load_class(name,folder,classname):
    path=ROOT/'sources/difix_ref'/folder/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return getattr(module,classname)


def psnr(a,b):
    return float(-10*np.log10(max(float(np.mean((a-b)**2)),1e-12)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='difix-appearance-test-r1');args=parser.parse_args()
    out=ROOT/args.output_name;out.mkdir(exist_ok=False)
    root=ROOT/'sources/difix_ref'
    unet_class=load_class('unet_2d_condition','unet','UNet2DConditionModel')
    vae_class=load_class('autoencoder_kl','vae','AutoencoderKL')
    unet=unet_class.from_pretrained(root/'unet',torch_dtype=torch.float16,local_files_only=True)
    vae=vae_class.from_pretrained(root/'vae',torch_dtype=torch.float16,local_files_only=True)
    pipe=DifixPipeline.from_pretrained(root,unet=unet,vae=vae,torch_dtype=torch.float16,local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    cams=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda c:c['img_name'])
    train=[c for i,c in enumerate(cams) if i%8];tests=[c for i,c in enumerate(cams) if not i%8]
    rows=[];start=time.time()
    for c in tests:
        R=np.array(c['rotation']);C=np.array(c['position'])
        def proximity(o):
            a=np.arccos(np.clip((np.trace(R.T@np.array(o['rotation']))-1)/2,-1,1))
            return a+np.linalg.norm(C-np.array(o['position']))/5
        reference=min(train,key=proximity)
        image=Image.open(ROOT/'mip-depth-refine-s0/evaluation'/f'{c["img_name"]}-s1.png').convert('RGB').resize((576,1024),Image.Resampling.LANCZOS)
        ref=Image.open(BASE/'hover-sfm-001/dataset/images'/f'{reference["img_name"]}.jpg').convert('RGB').resize((576,1024),Image.Resampling.LANCZOS)
        gt=Image.open(BASE/'hover-sfm-001/dataset/images'/f'{c["img_name"]}.jpg').convert('RGB').resize((576,1024),Image.Resampling.LANCZOS)
        t=time.time()
        with torch.inference_mode():
            result=pipe('remove degradation',image=image,ref_image=ref,height=1024,width=576,
                        num_inference_steps=1,timesteps=[199],guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(0)).images[0]
        torch.cuda.synchronize()
        result.save(out/f'{c["img_name"]}-difix.png');image.save(out/f'{c["img_name"]}-input.png');gt.save(out/f'{c["img_name"]}-gt.png');ref.save(out/f'{c["img_name"]}-reference.png')
        target=np.asarray(gt).astype('float32')/255
        row=dict(frame=c['img_name'],reference=reference['img_name'],seconds=time.time()-t,
                 input_psnr=psnr(np.asarray(image).astype('float32')/255,target),
                 output_psnr=psnr(np.asarray(result).astype('float32')/255,target),
                 peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30)
        rows.append(row);print(row,flush=True)
        (out/'progress.json').write_text(json.dumps(rows,indent=2))
    report=dict(status='complete',method='Difix reference-conditioned single-step image repair',
                weights_revision='d4830559772a5795c9d136302c5b197d6418d3fb',
                code_commit='c76edc595586e16732c91ddee82f3a6d83a8a9cc',dtype='float16',resolution=[576,1024],
                reference_policy='closest pose among 51 training frames only; never a held-out image',
                input_model='mip-depth-refine-s0',steps=1,timestep=199,seed=0,rows=rows,
                mean_input_psnr=float(np.mean([r['input_psnr'] for r in rows])),
                mean_output_psnr=float(np.mean([r['output_psnr'] for r in rows])),
                elapsed_seconds=time.time()-start,
                geometry_changed=False,pseudo_training_views_added=False,
                interpretation='2D appearance test only; need temporal consistency and independent geometry evidence before distillation')
    (out/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
