"""Native-pixel pictures of the saved stage transition; no fitting."""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import audit_sofa_knee_objective124 as a
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, crop_box

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
out = a.RUN / 'knee-stopping-replay'
assert not (out/'visual-review.json').exists()
result = read(out/'result.json')
assert result['endpoint_parity_passed']
stages = [s for s in result['stages'] if s['kind']=='body']
assert len(stages)==2
before, after = stages[0]['after_state'], stages[1]['after_state']
with np.load(out/'trajectory.npz',allow_pickle=False) as z:
    data = dict(z)
row = read(a.SRC/'data/inputs.json')['records'][15]
raw = Image.open(row['image']).convert('RGB')
mask = Image.open(row['mask']).convert('L')
box = np.array(crop_box(mask.getbbox(), raw.size, 30))
K = np.array(row['K_integer_centers'])
render,_ = mesh_renderer()
panels = [raw.crop(box)]
for index in (before,after):
    panels.append(render(raw.crop(box),box,dict(vertices_camera=data[f'state_{index:03d}_vertices_camera'],
                    faces=data['faces'],K_integer_centers=K)))
width,height=panels[0].size
sheet=Image.new('RGB',(3*width,height))
for j,panel in enumerate(panels):
    assert panel.size==(width,height)
    sheet.paste(panel,(j*width,0))
path=Path('/mnt/d/00/记录/006/smpl-x_连续帧124_frame000016_原图_身体阶段后_脸手联合阶段后_原生像素.png')
assert not path.exists()
sheet.save(path)
save(out/'visual-review.json',dict(updated=now(),state='images_generated_pending_actual_review',
    image=str(path),image_sha256=sha(path),source_image_sha256=sha(row['image']),
    exact_native_pixel_crop=box.tolist(),columns=['raw','after original stage3','after original stage4'],
    stage_states=[before,after],rescaled=False,geometry_changed=False,
    actual_visual_review=False,quality_accepted=False,gaussian_admitted=False))
print(str(path),sheet.size,flush=True)
