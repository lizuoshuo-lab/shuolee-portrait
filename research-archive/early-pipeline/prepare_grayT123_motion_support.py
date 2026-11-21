"""One fixed-frame material-support inventory; no detector or fitting calls."""
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native

out=REC/'grayT-relative-motion-support-review'
if (out/'preparation-result.json').exists():raise RuntimeError('Inventory already prepared; no repeat.')
out.mkdir(exist_ok=True)
plan=read(REC/'grayT-relative-motion-support-plan.json');assert plan['frames']==[40,54,66,80]
sources=[REC/'grayT-relative-motion-support-plan.json',ROOT/'data/training-inputs.json',REC/'grayT-skin-localization-review/result.json']
hashes=[sha(p) for p in sources]
rows=read(sources[1])['records'];skin={o['frame_number']:o for o in read(sources[2])['observations']}
images=[];inputs=[]
for frame in plan['frames']:
    row=rows[frame-1];p=native(row['image']);h=sha(p);assert h==row['source_sha256']
    inputs.append(dict(frame=frame,path=str(p),sha256=h))
    x,y=skin[frame]['xy'];x0,y0,x1,y1=row['initial_box_xyxy']
    areas=[('head',int(round(x)),int(round(y+40)),384),
           ('torso',int(round((x0+x1)/2)),int(round(y0+.35*(y1-y0))),512)]
    with Image.open(p) as im:
        im=im.convert('RGB')
        for region,cx,cy,size in areas:
            crop=[cx-size//2,cy-size//2,cx+size//2,cy+size//2]
            assert crop[0]>=0 and crop[1]>=0 and crop[2]<=im.width and crop[3]<=im.height
            name=f'frame_{frame:03d}_{region}_native.png';im.crop(crop).save(out/name)
            images.append(dict(frame=row['frame'],region=region,file=name,columns=['native pixels, no resize or warp'],crop=crop))
assert all(sha(p)==h for p,h in zip(sources,hashes))
save(out/'index.json',dict(images=images))
save(out/'preparation-result.json',dict(created=datetime.now(timezone.utc).isoformat(),images=8,frames=plan['frames'],
    sources=[dict(path=str(p),sha256=h) for p,h in zip(sources,hashes)],inputs=inputs,
    no_new_detection_tracking_fitting=True,quality_accepted=False))
print('Prepared8native crops for a single material-support inventory.')
