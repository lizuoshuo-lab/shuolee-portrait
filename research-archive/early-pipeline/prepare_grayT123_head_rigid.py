"""One prefit identity/rank check and all41 native proxy crops; no fitting."""
from datetime import datetime,timezone
import time,json
import numpy as np
from PIL import Image,ImageDraw
from grayT123_head_rigid_common import *

def main():
    started=time.monotonic()
    if (OUT/'preparation-result.json').exists():raise RuntimeError('Preparation already completed; do not repeat.')
    OUT.mkdir(exist_ok=True);d=load();images=[];checks=[];tiles=[];source_images=[]
    for q,frame in enumerate(d['frames']):
        E=d['E'][q:q+1];weight=d['weight'][q];uv=d['uv'][q]
        fun=lambda p:((project(d['X'],E,d['K'],transform(p,d['center'],d['radius']))[0][0]-uv)*np.sqrt(weight[:,None])).ravel()
        J=jacobian(fun,np.zeros(6));sv=np.linalg.svd(J,compute_uv=False);rank=int(np.linalg.matrix_rank(J))
        checks.append(dict(frame_number=int(frame),positive_weight_count=int((weight>0).sum()),weights=weight.tolist(),
            jacobian_rank=rank,jacobian_singular_values=sv.tolist(),condition_number=float(sv[0]/sv[-1])))
        row=d['rows'][frame-1];impath=native(row['image']);h=sha(impath);assert h==row['source_sha256']
        source_images.append(dict(frame=int(frame),path=str(impath),sha256=h))
        cx,cy=map(int,np.rint(uv.mean(0)));crop=[cx-96,cy-96,cx+96,cy+96]
        with Image.open(impath) as im:raw=im.convert('RGB').crop(crop)
        overlay=raw.copy();draw=ImageDraw.Draw(overlay)
        for xy,ww in zip(uv,weight):
            x,y=xy-np.array(crop[:2]);color=(0,240,255) if ww>0 else (160,160,160)
            draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
        tile=Image.new('RGB',(384,192));tile.paste(raw,(0,0));tile.paste(overlay,(192,0));tiles.append(tile)
        name=f'frame_{frame:03d}_nose_raw_proxy.png'
        if (OUT/name).exists():
            with Image.open(OUT/name) as old:assert np.array_equal(np.asarray(old),np.asarray(tile))
        else:tile.save(OUT/name)
        images.append(dict(frame=row['frame'],region='鼻部语义FIT代理：原生图／原检测点；不是物理真值',file=name,
            columns=['原生192px','原检测9点'],crop=crop,weights=weight.tolist()))
    sheets=[]
    for i in range(0,len(tiles),6):
        canvas=Image.new('RGB',(768,576),(245,245,245))
        for j,t in enumerate(tiles[i:i+6]):canvas.paste(t,((j%2)*384,(j//2)*192))
        name=f'prefit_sheet_{i//6+1:02d}.png'
        if (OUT/name).exists():
            with Image.open(OUT/name) as old:assert np.array_equal(np.asarray(old),np.asarray(canvas))
        else:canvas.save(OUT/name)
        sheets.append(dict(file=name,frames=d['frames'][i:i+6].tolist()))
    passed=all(c['positive_weight_count']>=6 and c['jacobian_rank']==6 for c in checks)
    assert all(sha(f)==h for f,h in zip(d['files'],d['hashes']))
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,frames=d['frames'].tolist(),
        independent_barycentric_projection_max_abs_px=d['projection_delta'],nose_3D_singular_values=np.linalg.svd(d['X']-d['center'],compute_uv=False).tolist(),
        nose_rms_radius=d['radius'],checks=checks,numerical_preparation_passed=passed,sheets=sheets,
        sources=[dict(path=str(f),sha256=h) for f,h in zip(d['files'],d['hashes'])],source_images=source_images,
        no_fit_or_training=True,physical_truth_established=False,quality_accepted=False,
        preparation_repairs=['Initial module syntax2e corrected before import.',
            'Native crop bounds used NumPy integers; JSON serialization TypeError reproduced. Bounds changed to native Python integers. Existing41crops and7sheets were pixel-verified and retained, no refit or new selection.'])
    save(OUT/'index.json',dict(images=images,image_labels_burned_in=False,actual_review_pending=True));save(OUT/'preparation-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','sources','source_images']},ensure_ascii=False))

if __name__=='__main__':main()
