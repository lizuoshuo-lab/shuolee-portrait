"""Bounded native boundary evidence for the completed input-only preflight."""
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from probe_hand_identity_guard124 import ROOT,OUT,SUBJECTS
from run_official_gaussians123 import read,save,sha,now

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert read(OUT/'result.json')['passed']
assert not (OUT/'boundary-review-plan.json').exists()
records_dir={'grass':'005','winterhat':'001','sofa':'006','indoor':'008'}
selection={};sources={str(OUT/'result.json'):sha(OUT/'result.json')}
for subject in SUBJECTS:
    path=OUT/(subject+'-input-preview.npz');sources[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:flags=z['ambiguous']
    choices=set();intervals=[]
    for side in range(2):
        change=np.diff(np.r_[False,flags[:,side],False].astype(int))
        for start,end in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)-1):
            indexes=[max(0,int(start)-1),int(start),int((start+end)//2),int(end)]
            if end+1<len(flags):indexes.append(int(end+1))
            choices.update(indexes)
            intervals.append(dict(side=('left','right')[side],first=int(start)+1,last=int(end)+1))
    selection[subject]=dict(indexes=sorted(choices),intervals=intervals)
assert sum(len(v['indexes']) for v in selection.values())<=20
save(OUT/'boundary-review-plan.json',dict(updated=now(),state='registered_before_images',
    selection_rule='Every contiguous quarantine interval, both sides and all subjects: predecessor, first, integer midpoint, last, immediate recovery if present. Deduplicate. No fit-error selection.',
    selection=selection,budget=dict(images_max=20,gpu_calls=0,fitter_calls=0),sources=sources,
    limits='Boundary and midpoint review does not certify visibility of all suppressed observations. No physical3Dtruth. Preserve the failed old fits.'))
images=[]
for subject in SUBJECTS:
    rows=read(ROOT/(subject+'-input')/'data/inputs.json')['records']
    with np.load(OUT/(subject+'-input-preview.npz'),allow_pickle=False) as z:
        original=z['original_raster_checked'];guarded=z['guarded_preview']
    for i in selection[subject]['indexes']:
        row=rows[i];assert sha(row['image'])==row['source_sha256']
        raw=Image.open(row['image']).convert('RGB');w,h=raw.size
        points=original[i,np.r_[[2,3,4,5,6,7],np.arange(25,67)],:2]
        valid=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
        points=points[valid];assert len(points)>0
        margin=35
        box=[max(0,int(np.floor(points[:,0].min()))-margin),max(0,int(np.floor(points[:,1].min()))-margin),
             min(w,int(np.ceil(points[:,0].max()))+margin),min(h,int(np.ceil(points[:,1].max()))+margin)]
        crop=raw.crop(box);panels=[crop]
        for observations in [original[i],guarded[i]]:
            im=crop.copy();draw=ImageDraw.Draw(im)
            for indexes,color,radius in [(range(25,46),(255,150,0),2),(range(46,67),(0,220,255),2),([7],(255,0,0),6),([4],(0,70,255),6)]:
                for k in indexes:
                    if observations[k,2]<=0:continue
                    x,y=observations[k,:2]-np.array(box[:2])
                    draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
            panels.append(im)
        combined=Image.new('RGB',(crop.width*3,crop.height))
        for j,im in enumerate(panels):combined.paste(im,(j*crop.width,0))
        path=Path('/mnt/d/00/记录')/records_dir[subject]/f'分析_手部身份隔离124_{subject}_{row["frame"]}_原图_原观测_隔离预览.png'
        assert not path.exists();combined.save(path)
        images.append(dict(subject=subject,frame=row['frame'],path=str(path),sha256=sha(path),native_crop=box,
                           resized=False,text_on_image=False))
for path,h in sources.items():assert sha(path)==h,path
save(OUT/'boundary-review-images.json',dict(updated=now(),images=images,actual_visual_review=False))
print('HAND_IDENTITY_BOUNDARY_IMAGES',len(images),flush=True)
