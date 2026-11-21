"""Project the dominant side-view rows onto all original training photos."""

from collections import defaultdict
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

from trace_round22_face_pixels import checked, code, need, pin, pixel, preprocess, read, sha
from trace_round25_garment_pixels import OUT, PROJECT, R24_REVIEW, loaded_drive, np_tensor
from trace_round25_native_garment_material import native_scene


SIDE_SHA = 'bdcc7e62dfe3a34ef746ebf22409da052bbae7b392e69df19e492621aa2b3bcd'
DISPLAY_IDS = {'016': (43202, 69017, 8202, 21056, 4447), '006': (166263, 165373, 168357, 174991)}


def select(details, subject):
    candidates = {}
    for view in details['views'].values():
        for ray in view['pairs']:
            members=ray['contributors']
            selected=[members[0],max(members,key=lambda item:item['alpha_weight'])]
            selected.extend(item for item in members if item['stable_id'] in DISPLAY_IDS[subject])
            for item in selected:
                candidates.setdefault(item['row'],dict(stable_id=item['stable_id'],part=item['part'],face=item['face'],observations=[]))
    return candidates


def source_scan(subject,details,review):
    plan=read(checked(review['plan']))
    arm='r24'
    manifest=checked(plan['graphs'][arm]['manifest'])
    need(sha(manifest)==details['input_proof']['drive_sha256'],'The attribution and native drive differ.')
    scene,info=loaded_drive(manifest.parent)
    candidates=select(details,subject)
    rows=np.asarray(list(candidates),dtype=np.int64)
    config=read(checked(plan['training_config']))
    drives={r['frame']:r for r in plan['drives'][arm]['records']}
    original={r['frame']:r for r in plan['frozen']['rows']}
    renders=review['native'][arm]
    opacity=np_tensor(scene.get_opacity)[:,0]
    rgb=np.maximum(np.float32(.28209479177387814)*np_tensor(scene.get_features)[:,0,:]+np.float32(.5),0.)
    media,provenance={},{}
    for slot,(frame,source_bin) in enumerate(zip(info['frame_ids'][:24],config['training_source_bins'],strict=True)):
        r,truth=original[frame],renders[frame]
        camera=truth['camera']
        need(all(camera[k]==info['cameras'][slot][k] for k in ('frame','w2c','K','wh','pose_index')) and
             truth['source_sha256']==r['source_sha256'] and truth['mask_sha256']==r['mask_sha256'],
             'A signed native frame, camera or mask changed.')
        source=pin(r['image'],r['source_sha256'])
        matte=pin(r['mask'],r['mask_sha256'])
        with Image.open(matte) as image:
            mask=np.asarray(image.convert('L'))
        x0,y0,x1,y1=drives[frame]['box_xyxy']
        pieces=[]
        for role in ('black','white','alpha'):
            with Image.open(checked(truth['delivered_media'][role])) as image:
                pieces.append(np.asarray(image.convert('RGB'),dtype=np.int16))
        pane=np.stack(pieces)
        need(pane.shape==(3,y1-y0,x1-x0,3),'Saved native pane dimensions differ.')
        pre=preprocess(native_scene(scene,slot,opacity,rgb),camera)
        xy=np.rint(pre['xy'][rows]).astype(np.int64)
        for row,(x,y) in zip(rows,xy,strict=True):
            if not(x0<=x<x1 and y0<=y<y1):
                continue
            recovered=pixel(pre,(int(x),int(y)))
            codes=np.stack((code(recovered['rgb']),code(recovered['rgb']+1-recovered['opacity']),
                           np.full(3,code(recovered['opacity']),dtype=np.int16)))
            parity=int(abs(codes-pane[:,y-y0,x-x0]).max())
            indices=np.where(recovered['rows']==row)[0]
            weight=float(recovered['weights'][indices[0]]) if len(indices) else 0.
            candidates[int(row)]['observations'].append(dict(frame=frame,source_bin=source_bin,xy=[int(x),int(y)],
                native_center_alpha=weight,source_matte=int(mask[y,x]),max_code_error=parity,
                visible=parity<=1 and weight>=.01))
        media[frame]=source
        provenance[frame]=dict(source_sha256=r['source_sha256'],mask_sha256=r['mask_sha256'],rendered_media=truth['delivered_media'])
    return candidates,media,provenance


def board(directory,subject,rows,media):
    rows={row['stable_id']:row for row in rows.values()}
    named=[rows[k] for k in DISPLAY_IDS[subject] if k in rows]
    cell,header=106,30
    size=(cell*8, len(named)*(header+cell))
    canvas=Image.new('RGB',size,'#181818')
    draw=ImageDraw.Draw(canvas)
    for j,row in enumerate(named):
        y0=j*(header+cell)
        draw.text((3,y0+3),f'{subject} ID {row["stable_id"]} {row["part"]} | highest visible native center weights per source bin',fill='white')
        strongest={}
        for item in sorted(row['observations'],key=lambda v:v['native_center_alpha'],reverse=True):
            if item['visible']:
                strongest.setdefault(item['source_bin'],item)
        for i,item in enumerate(list(strongest.values())[:8]):
            x,y=item['xy']
            with Image.open(media[item['frame']]) as im:
                crop=im.crop((x-22,y-22,x+22,y+22)).resize((88,88),Image.Resampling.NEAREST)
            xx,yy=i*cell+3,y0+header-7
            canvas.paste(crop,(xx,yy))
            draw.ellipse((xx+42,yy+42,xx+46,yy+46),outline='#24eaf6')
            draw.text((xx,yy+89),f'{item["frame"][-3:]} w={item["native_center_alpha"]:.2f}',fill='white')
    path=directory/f'{subject}-dominant-native-source-pixels.png'
    with path.open('xb') as f:
        canvas.save(f,format='PNG')
    return dict(file=str(path),sha256=sha(path))


def main():
    sys.path.insert(0,PROJECT+'24/src')
    torch.set_num_threads(2)
    base=OUT/'smpl-delivery-round24'
    index=read(pin(base/'review-receipt.json',R24_REVIEW))
    directory=base/'input-audit'
    old=pin(directory/'side-waist-hem-attribution.json',SIDE_SHA)
    subjects={}
    for subject,details in read(old)['subjects'].items():
        rows,media,provenance=source_scan(subject,details,read(checked(index['subjects'][subject]['review'])))
        subjects[subject]=dict(candidates=list(rows.values()),sources=provenance,board=board(directory,subject,rows,media))
        print(subject,len(rows),sum(item['visible'] for row in rows.values() for item in row['observations']),flush=True)
    path=directory/'dominant-garment-native-visibility.json'
    with path.open('x',encoding='utf-8') as f:
        json.dump(dict(schema='shuolee-r24-dominant-side-row-native-visibility/1',side_attribution_sha256=SIDE_SHA,
            source_sha256=sha(__file__),subjects=subjects,
            scope='Top absolute direct contrast and top alpha rows in each fixed sample, plus named mixed lower-hem rows. Native visibility requires center weight >=.01 and PNG parity within 1 code.'),f,indent=2,allow_nan=False)
        f.write('\n')
    print(dict(file=str(path),sha256=sha(path)),flush=True)


if __name__=='__main__':
    main()
