"""Compare matched training steps, retaining the exact original camera and crop."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=6000);parser.add_argument('--motion',choices=['head','face','fan'],default='head');args=parser.parse_args()
BASE=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
TRIAL=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029')/dict(head='gray-head-motion',face='gray-dynamic-face',fan='gray-dynamic-face-fan')[args.motion]
OUT=Path('/mnt/d/Projects/Shuolee-analysis-20261028')/dict(head='head-motion-render-comparison',face='dynamic-face-render-comparison',fan='dynamic-face-fan-render-comparison')[args.motion]/f'step-{args.step:06d}'
OUT.mkdir(parents=True,exist_ok=False)
rows=json.loads((BASE/'data/training-inputs.json').read_text())['records']
selected=np.unique(np.linspace(0,len(rows)-1,6).astype(int));records=[]
fig,axes=plt.subplots(6,3,figsize=(10,12.6),squeeze=False)
full,fullaxes=plt.subplots(6,3,figsize=(10,23),squeeze=False)
fit=dict(np.load(BASE/'fit/fitted.npz'))
for line,i in enumerate(selected):
    row=rows[i];name=row['frame'];old=BASE/'gaussian-run'/f'step-{args.step:06d}';new=TRIAL/f'step-{args.step:06d}'
    original=np.asarray(Image.open(old/f'{name}_input.png').convert('RGB'))
    static=np.asarray(Image.open(old/f'{name}_render.png').convert('RGB'))
    motion=np.asarray(Image.open(new/f'{name}_render.png').convert('RGB'))
    assert np.array_equal(original,np.asarray(Image.open(new/f'{name}_input.png').convert('RGB')))
    H,W=original.shape[:2];factor=np.array([W,H])/row['wh'];box=np.array(row['face_box_xyxy']).reshape(2,2)*factor
    margin=max(float(np.ptp(box,axis=0).max())*.6,15)
    lo=np.maximum(0,np.floor(box[0]-margin)).astype(int);hi=np.minimum([W,H],np.ceil(box[1]+margin)).astype(int)
    mask=np.asarray(Image.open(row['mask']).convert('L').resize((W,H),Image.Resampling.NEAREST))>127
    y=original.astype(float)/255;crop=np.zeros_like(mask);crop[lo[1]:hi[1],lo[0]:hi[0]]=True
    rec=dict(frame=name,head_crop_xyxy=[*lo.tolist(),*hi.tolist()],quality_measure='Same training views; no independent unseen-view claim')
    for label,im in [('static',static),('head_motion',motion)]:
        x=im.astype(float)/255
        rec[label]=dict(head_foreground_rgb_l1=float(np.abs(x-y)[mask&crop].mean()),
                        head_crop_masked_rgb_l1=float(np.abs(x-y*mask[...,None])[crop].mean()),
                        whole_foreground_rgb_l1=float(np.abs(x-y)[mask].mean()))
    for j,(im,title) in enumerate(zip([original,static,motion],['Original',f'Static, {args.step} steps',f'{args.motion.title()} motion, {args.step} steps'])):
        ax=axes[line,j];ax.imshow(im);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal')
        fa=fullaxes[line,j];fa.imshow(im);fa.set_xticks([]);fa.set_yticks([])
        if line==0:ax.set_title(title,fontsize=10);fa.set_title(title,fontsize=10)
        if j==0:ax.set_ylabel(name,fontsize=9);fa.set_ylabel(name,fontsize=9)
    records.append(rec)
fig.suptitle('Fixed cameras and original pixels | matched training schedule | head detail',fontsize=12)
fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(OUT/'head-comparison.png',dpi=160);plt.close(fig)
full.suptitle('Fixed cameras and original pixels | matched training schedule | full frame',fontsize=12)
full.tight_layout(rect=[0,0,1,.98]);full.savefig(OUT/'full-comparison.png',dpi=140);plt.close(full)
average={kind:{key:float(np.mean([r[kind][key] for r in records])) for key in records[0][kind]} for kind in ['static','head_motion']}
report=dict(step=args.step,records=records,mean_across_six_views=average,original_inputs_identical=True,
            cameras_unchanged=True,quality_accepted=False)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(step=args.step,mean=average)),flush=True)
