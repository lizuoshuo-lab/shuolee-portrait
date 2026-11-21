import os
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,now
from review_full_sequence123 import mesh_renderer,joined,crop_box
root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/winterhat-input')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sel=read(root/'selection.json'); row=next(r for r in read(root/'data/inputs.json')['records'] if r['frame']==sel['reference_frame'])
with np.load(root/'reference-fit/winner-camera-space.npz',allow_pickle=False) as z: mesh=dict(z)
raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L');box=crop_box(mask.getbbox(),raw.size,30)
render,_=mesh_renderer();mesh['K_integer_centers']=np.asarray(row['K_integer_centers'])
original=raw.crop(box); overlay=render(original,np.asarray(box),mesh)
dest=root/'reference-fit/reference-review.png';joined([original,overlay]).save(dest)
save(root/'reference-fit/review-source.json',dict(updated=now(),frame=row['frame'],image=row['image'],crop=list(box),actual_visual_review=False,path=str(dest)))
print(dest)
