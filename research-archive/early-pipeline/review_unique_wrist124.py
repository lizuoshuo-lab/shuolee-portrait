"""Render unchanged raw pixels and both fitted meshes; no labels or resizing."""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import control_unique_wrist124 as c
from review_full_sequence123 import mesh_renderer,crop_box

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (c.OUT/'visual-review.json').exists()
r=c.read(c.OUT/'result.json');assert r['state']=='complete_pending_visual_review'
row,targets=c.base.target()
raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
old=c.a.load_npz(c.a.frame(c.a.RUN,16));new=c.a.load_npz(c.OUT/'winner-camera-space.npz')
for field in ('K_integer_centers','camera_translation','betas','faces'):
    assert np.array_equal(old[field],new[field]),field
assert np.array_equal(targets,new['original_native_observations'])
boxes={'body':crop_box(mask.getbbox(),raw.size,30)}
for label,lo,hi in [('face',67,118),('hands',25,67),('feet',19,25)]:
    points=targets[0,lo:hi];points=points[points[:,2]>0,:2]
    boxes[label]=crop_box(np.r_[points.min(0),points.max(0)],raw.size,55)
render,_=mesh_renderer();images={}
for label,coords in boxes.items():
    box=np.asarray(coords,dtype=int);panels=[raw.crop(box)]
    for arrays in (old,new):
        panels.append(render(raw.crop(box),box,dict(vertices_camera=arrays['vertices_camera'],
            faces=arrays['faces'],K_integer_centers=arrays['K_integer_centers'])))
    w,h=panels[0].size;panel=Image.new('RGB',(w*3,h))
    for j,im in enumerate(panels):assert im.size==(w,h);panel.paste(im,(j*w,0))
    path=Path('/mnt/d/00/记录/006')/f'smpl-x_唯一手腕124_frame000016_{label}_原图_原输入_唯一手腕目标_原生像素.png'
    assert not path.exists();panel.save(path)
    images[label]=dict(path=str(path),sha256=c.sha(path),native_crop=box.tolist())
c.save(c.OUT/'visual-review.json',dict(updated=c.now(),state='images_generated_pending_actual_review',
    images=images,columns=['raw','original all-parameter fit','unique wrist all-parameter fit'],
    rescaled=False,geometry_changed=False,all_original_observations_retained=True,
    actual_visual_review=False,quality_accepted=False,gaussian_admitted=False,full_sequence_admitted=False))
print({k:v['path'] for k,v in images.items()},flush=True)
