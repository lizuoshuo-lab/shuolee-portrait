"""Compare existing 18k renders with their original training images, read-only."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--label',required=True)
parser.add_argument('--step',type=int,default=18000)
args=parser.parse_args()
root=args.run; out=args.output;out.mkdir(parents=True,exist_ok=True)
rows=json.loads((root/'data/training-inputs.json').read_text())['records']
result=json.loads((root/'gaussian-run/result.json').read_text()) if args.step==18000 else dict(steps=args.step)
assert result['steps']==args.step
snap=root/'gaussian-run'/f'step-{args.step:06d}'
assert json.loads((snap/'native-ply-parity.json').read_text())['passed']
selected=np.unique(np.linspace(0,len(rows)-1,6).astype(int))
fig,axs=plt.subplots(len(selected),3,figsize=(10,2.1*len(selected)),squeeze=False)
metrics=[]
for line,i in enumerate(selected):
    row=rows[i];frame=row['frame']
    images=[np.asarray(Image.open(snap/f'{frame}_{suffix}.png').convert('RGB')) for suffix in ['input','masked_input','render']]
    assert images[0].shape==images[1].shape==images[2].shape
    H,W=images[0].shape[:2]
    factor=np.array([W/row['wh'][0],H/row['wh'][1]])
    box=np.array(row['face_box_xyxy'],float).reshape(2,2)*factor
    margin=max(float(np.ptp(box,axis=0).max())*.6,15)
    low=np.maximum(0,np.floor(box[0]-margin)).astype(int)
    high=np.minimum([W,H],np.ceil(box[1]+margin)).astype(int)
    mask=np.asarray(Image.open(row['mask']).convert('L').resize((W,H),Image.Resampling.NEAREST))>127
    original=images[0].astype(float)/255;render=images[2].astype(float)/255
    cropmask=np.zeros_like(mask);cropmask[low[1]:high[1],low[0]:high[0]]=True;valid=mask&cropmask
    error=np.abs(original-render)
    metrics.append(dict(frame=frame,head_crop_xyxy=[*low.tolist(),*high.tolist()],training_image_wh=[W,H],
                        head_foreground_pixels=int(valid.sum()),head_foreground_rgb_l1=float(error[valid].mean()),
                        whole_foreground_rgb_l1=float(error[mask].mean()),training_view_not_independent_holdout=True))
    for col,(image,title) in enumerate(zip(images,['Original training image','Masked supervision',f'{args.step}-step render'])):
        ax=axs[line,col];ax.imshow(image);ax.set_xlim(low[0],high[0]);ax.set_ylim(high[1],low[1]);ax.set_aspect('equal')
        ax.set_xticks([]);ax.set_yticks([])
        if line==0:ax.set_title(title,fontsize=10)
    axs[line,0].set_ylabel(frame,fontsize=9)
fig.suptitle(f'{args.label} | same training camera, no image alignment correction',fontsize=11)
fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(out/'training-face-comparison.png',dpi=160);plt.close(fig)
(out/'training-comparison.json').write_text(json.dumps(dict(run=str(root),steps=result['steps'],
                records=metrics,quality_accepted=False),indent=2)+'\n')
print(json.dumps(dict(label=args.label,steps=result['steps'],head_rgb_l1=[x['head_foreground_rgb_l1'] for x in metrics])))
