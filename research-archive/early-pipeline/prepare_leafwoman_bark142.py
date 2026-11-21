"""Read and bind the 022 above-hat native source strip on all front frames."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from probe_video22_background135 import BASE, CONTACT, VIDEO, decodes
from review_full_sequence123 import relocate
from update_leafwoman129_state import atomic, read


OUT = BASE/'near-bark142'
BOX = (930,0,1832,256)  # old background135 corridor, exposed above the hat


def prepare():
    assert not (OUT/'source.json').exists()
    OUT.mkdir(exist_ok=True)
    fit=read(BASE/'fit-plan.json'); bg=read(BASE/'background-motion135/plan.json')
    assert list(BOX[::2])==bg['excluded_x_full_height']
    contacts=read(CONTACT); by={r['frame']:r for r in contacts['records']}
    assert sha(VIDEO)==fit['source_video_sha256'] and sha(CONTACT)==fit['source_contacts_sha256']
    roster=fit['front_arc']; ids=[r['frame'] for r in roster]
    indices=[r['source_frame_index'] for r in roster]
    assert ids==bg['frames'] and indices==bg['indices'] and len(indices)==24 and not fit['rear_arc']
    x0,y0,x1,y1=BOX
    strips=[];bound=[]
    for row,im in zip(roster,decodes(indices),strict=True):
        entry=by[row['frame']]; assert row['source_frame_index']==entry['source_frame_index']
        body=entry['paths']['body']; box=body['crop_original_xyxy']
        path=relocate(body['path']); assert sha(path)==body['sha256'] and row['source_sha256']==entry['source_sha256']
        with Image.open(path) as f:
            pixels=np.asarray(f.convert('RGB'))
        assert np.array_equal(im[box[1]:box[3],box[0]:box[2]],pixels),row['frame']
        strips.append(im[y0:y1,x0:x1].copy())
        bound.append(dict(frame=row['frame'],native_index=row['source_frame_index'],body_crop_sha256=body['sha256']))
    np.savez_compressed(OUT/'native-strips.npz',rgb=np.stack(strips),frame_ids=np.asarray(ids))
    pages=[]
    for start in range(0,len(ids),8):
        page=Image.new('RGB',(994,624),'#21252e'); d=ImageDraw.Draw(page)
        for j,i in enumerate(range(start,start+8)):
            u=20+(j%2)*497; v=18+(j//2)*156
            page.paste(Image.fromarray(strips[i]).resize((451,128),Image.Resampling.NEAREST),(u,v))
            d.text((u,v+130),ids[i]+'    top strip y=0..255',fill='white')
            for x in range(1000,1801,100):
                xp=u+(x-x0)//2; d.text((xp-10,v-15),str(x),fill='white')
                d.line((xp,v-4,xp,v-1),fill='white')
            for y in (0,100,200):
                yp=v+y//2; d.text((u-19,yp-4),str(y),fill='white')
        p=OUT/f'source-bark-{start//8+1}.png';page.save(p);pages.append(dict(path=str(p),sha256=sha(p)))
    atomic(OUT/'source.json',dict(source_video_sha256=sha(VIDEO),fit_plan_sha256=sha(BASE/'fit-plan.json'),
        old_background_plan_sha256=sha(BASE/'background-motion135/plan.json'),strip_box=list(BOX),
        raw_body_crop_matches=24,frames=bound,strip_sha256=sha(OUT/'native-strips.npz'),pages=pages,
        source_strip_native=True,display_half_size=True,fit_or_training_started=False))
    print('BARK142_NATIVE_STRIPS',len(strips),flush=True)


if __name__=='__main__':
    prepare()
