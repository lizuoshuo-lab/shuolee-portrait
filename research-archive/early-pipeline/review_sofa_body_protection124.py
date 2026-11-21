"""Native unscaled review of the bounded parameter-partition control."""
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import control_sofa_body_protection124 as c
from review_full_sequence123 import mesh_renderer,crop_box

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
parser=argparse.ArgumentParser();parser.add_argument('--local-joints',action='store_true')
args=parser.parse_args()
out=c.a.RUN/'local-head-wrist-final-stage-control' if args.local_joints else c.OUT
assert not (out/'visual-review.json').exists()
result=c.read(out/'result.json')
assert result['state']=='complete_pending_visual_review'
rawrow,targets=c.base.target()
raw=Image.open(rawrow['image']).convert('RGB');mask=Image.open(rawrow['mask']).convert('L')
old=c.a.load_npz(c.a.frame(c.a.RUN,16));new=c.a.load_npz(out/'winner-camera-space.npz')
for field in ('K_integer_centers','camera_translation','betas','faces'):
    assert np.array_equal(old[field],new[field]),field
boxes={'body':crop_box(mask.getbbox(),raw.size,30)}
for label,lo,hi in [('face',67,118),('hands',25,67),('feet',19,25)]:
    q=targets[0,lo:hi];q=q[q[:,2]>0,:2]
    assert len(q)>0
    bounds=np.r_[q.min(0),q.max(0)]
    boxes[label]=crop_box(bounds,raw.size,55)
render,_=mesh_renderer();images={}
for label,coords in boxes.items():
    box=np.asarray(coords,dtype=int)
    panels=[raw.crop(box)]
    for arrays in (old,new):
        panels.append(render(raw.crop(box),box,dict(vertices_camera=arrays['vertices_camera'],
                      faces=arrays['faces'],K_integer_centers=arrays['K_integer_centers'])))
    w,h=panels[0].size
    panel=Image.new('RGB',(3*w,h))
    for j,im in enumerate(panels):
        assert im.size==(w,h)
        panel.paste(im,(j*w,0))
    suffix='头颈手腕局部精修' if args.local_joints else '身体冻结脸手精修'
    path=Path('/mnt/d/00/记录/006')/f'smpl-x_身体保护124_frame000016_{label}_原图_原联合阶段_{suffix}_原生像素.png'
    assert not path.exists();panel.save(path)
    images[label]=dict(path=str(path),sha256=c.sha(path),native_crop=box.tolist())
c.save(out/'visual-review.json',dict(updated=c.now(),state='images_generated_pending_actual_review',
    images=images,columns=['raw','original all-parameter final stage','local neck/head/wrist final stage' if args.local_joints else 'body frozen final face-hand stage'],
    rescaled=False,geometry_changed=False,actual_visual_review=False,quality_accepted=False,
    gaussian_admitted=False,full_sequence_admitted=False))
print({k:v['path'] for k,v in images.items()},flush=True)
