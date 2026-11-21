"""Reuse native evidence renderer; evaluate ALL original targets, including withheld."""
import argparse
import inspect
from pathlib import Path
import numpy as np
from PIL import Image
import run_grass_temporal124 as c
import review_grass_temporal124 as review
from run_grass_identity124 import OUT,PREVIOUS

parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=int,choices=[13,27,94],required=True)
checkpoint=parser.parse_args().checkpoint
if checkpoint==94:
    p=c.read(OUT/'temporal-progress.json');assert p['state']=='completed' and p['frames']==94
    request=dict(frames=c.read(OUT/'plan.json')['ordered_fit_frames'],last_frame=None)
else:
    request=c.read(OUT/('review-request.json' if checkpoint==13 else 'review-request-027.json'))
original=inspect.getsource(review.main)
hook="    request = c.read(c.OUT/('review-request'+suffix+'.json'))"
assert original.count(hook)==1
source=original.replace(hook,'    request = provided_request')
source=source.replace("        assert np.array_equal(old['camera_translation'],new['camera_translation'])", "        assert np.isfinite(new['camera_translation']).all()")
source=source.replace('草地连续帧124','草地手部身份124').replace('独立拟合_连续约束','连续精修_身份隔离')
source=source.replace("'independent'","'previous_refinement'").replace("'temporal'","'hand_identity'")
source=source.replace('same_K_shape_camera_translation=True','same_K_shape=True,both_optimize_original_body_translation=True')
c.OUT=OUT;c.BASE=PREVIOUS/'official-fit';review.provided_request=request
exec(compile(source,'<grass-identity-native-review>','exec'),review.__dict__)
review.main(checkpoint)
path=OUT/f'review-{checkpoint:03d}/result.json';result=c.read(path)
parity=[]
for row in result['records']:
    name=row['frame'];p=OUT/'official-fit/frames'/name/'winner-camera-space.npz'
    if not p.exists():p=OUT/('review-pause-pose.npz' if checkpoint==13 else 'review-pause-pose-027.npz')
    with np.load(p,allow_pickle=False) as z:new=dict(z)
    with np.load(PREVIOUS/'official-fit/frames'/name/'winner-camera-space.npz',allow_pickle=False) as z:old=dict(z)
    delta={k:float(np.max(np.abs(old[k]-new[k]))) for k in ['vertices_camera','joints_projected_native','pose_embedding','global_orient','camera_translation','betas']}
    parity.append(dict(frame=name,raw_input_unchanged_before74=int(name[-6:])<74,max_abs=delta))
result.update(adapter_source_sha256=c.sha(__file__),baseline=str(PREVIOUS),
    comparison='Same-camera previous94refinement vs new identity-input control; no independent-pose mixture.',
    all_original_raw_targets_evaluated=True,quarantined_observations_kept_in_evaluation=True,baseline_array_differences=parity)
c.save(path,result)
tiles=[];pages=[]
for row in result['records']:
    ims=[Image.open(row['images'][r]['path']).convert('RGB') for r in ['face','feet']]
    tile=Image.new('RGB',(max(im.width for im in ims),sum(im.height for im in ims)))
    y=0
    for im in ims:tile.paste(im,(0,y));y+=im.height
    tiles.append((row['frame'],tile))
for start in range(0,len(tiles),2):
    batch=tiles[start:start+2];canvas=Image.new('RGB',(sum(im.width for _,im in batch),max(im.height for _,im in batch)))
    x=0
    for _,im in batch:canvas.paste(im,(x,0));x+=im.width
    p=path.parent/f'face-feet-{start//2+1:02d}.png';canvas.save(p)
    pages.append(dict(path=str(p),frames=[f for f,_ in batch],resized=False,text_on_image=False))
c.save(path.parent/'face-feet-pages.json',pages)
print('GRASS_IDENTITY_REVIEW_READY',checkpoint,flush=True)
