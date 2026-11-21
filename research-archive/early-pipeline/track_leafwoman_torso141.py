"""Follow visible 022 cardigan material in native pixels; no geometry fit."""
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from probe_video22_background135 import BASE, CONTACT, VIDEO
from review_full_sequence123 import joined, relocate
from track_native_continuity122 import FB_LIMIT, PARAMS, controls, step
from update_leafwoman129_state import atomic, read


OUT = BASE/'torso-material141'
SEEDS = [
    ('top_button_stitch', (307,692), 'fit'),
    ('middle_button_stitch', (312,875), 'fit'),
    ('bottom_button_stitch', (310,963), 'held'),
    ('top_sewn_slit_tip', (291,771), 'held'),
    ('bottom_sewn_slit_tip', (276,865), 'fit'),
    ('placket_white_stripe_join', (266,795), 'fit'),
    ('left_knit_color_transition', (228,881), 'held'),
    ('flexible_scarf_weave', (393,837), 'control'),
    ('moving_sleeve_boundary', (204,665), 'control'),
]


def decode(indices, box, reverse):
    x0,y0,x1,y1 = box
    lo,hi=min(indices),max(indices)
    exclusions=sorted(set(range(lo,hi+1))-set(indices))
    select=f'between(n\\,{lo}\\,{hi})'+''.join(f'*(1-eq(n\\,{i}))' for i in exclusions)
    filt=f'select={select},format=rgb24,crop={x1-x0}:{y1-y0}:{x0}:{y0}'
    if reverse: filt+=',reverse'
    args=[shutil.which('ffmpeg'),'-nostdin','-v','error','-hwaccel','none','-threads','2',
        '-noautorotate','-i',str(VIDEO),'-filter_threads','1','-vf',filt,'-fps_mode','passthrough',
        '-frames:v',str(len(indices)),'-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    with subprocess.Popen(args,stdout=subprocess.PIPE) as p:
        try:
            for i in indices:
                data=p.stdout.read((x1-x0)*(y1-y0)*3)
                assert len(data)==(x1-x0)*(y1-y0)*3,(i,'incomplete RGB')
                yield i,np.frombuffer(data,np.uint8).reshape(y1-y0,x1-x0,3)
            assert p.wait(timeout=15)==0
        finally:
            if p.poll() is None: p.kill();p.wait()


def verify_crop(rgb, row, box):
    x0,y0,x1,y1=box
    path=relocate(row['path'])
    assert sha(path)==row['sha256'] and row['native_integer_crop'] and not row['resized']
    u,v,U,V=row['crop_original_xyxy']
    with Image.open(path) as im:
        stored=np.asarray(im.convert('RGB'))
    a,b,c,d=max(x0,u),max(y0,v),min(x1,U),min(y1,V)
    assert c>a and d>b
    assert np.array_equal(rgb[b-y0:d-y0,a-x0:c-x0],stored[b-v:d-v,a-u:c-u]), (path,'native overlap mismatch')


def draw_roster(upper, source_idx, tracks, valid, xy0):
    crops=[]
    for point,seen in zip(tracks,valid,strict=True):
        if not seen:
            crops.append(Image.new('RGB',(66,66),'#32353e'))
            continue
        x,y=point-xy0
        px,py=map(round,(x,y))
        raw=upper.crop((px-32,py-32,px+34,py+34))
        d=ImageDraw.Draw(raw)
        d.ellipse((30,30,34,34),outline='#16edf5',width=1)
        crops.append(raw)
    overview=joined(crops)
    pen=ImageDraw.Draw(overview)
    pen.text((3,3),str(source_idx),fill='white',stroke_width=1,stroke_fill='#191919')
    return overview


def run():
    cv2.setNumThreads(2)
    t0=time.monotonic()
    state=OUT/'result.json'; plan=OUT/'plan.json'
    assert not state.exists() and not plan.exists()
    fit=read(BASE/'fit-plan.json');contacts=read(CONTACT); rows={r['frame']:r for r in contacts['records']}
    assert sha(CONTACT)==fit['source_contacts_sha256'] and sha(VIDEO)==fit['source_video_sha256']
    frames=fit['front_arc']; assert len(frames)==24 and not fit['rear_arc']
    roster={r['source_frame_index']:r['frame'] for r in frames}
    ix=[r['source_frame_index'] for r in frames]
    assert ix==sorted(set(ix))
    reserve=set(fit['retained_holdouts'])|{f'frame_{n:06d}' for n in (5,15,26,36)}
    skipped=sorted(rows[f]['source_frame_index'] for f in reserve)
    assert not set(skipped)&set(ix)
    inputs=[rows[f]['paths']['upper'] for f in roster.values()]
    x0=min(r['crop_original_xyxy'][0] for r in inputs);x1=max(r['crop_original_xyxy'][2] for r in inputs)
    y0=min((r['crop_original_xyxy'][1]+r['crop_original_xyxy'][3])//2 for r in inputs)
    y1=max(r['crop_original_xyxy'][3] for r in inputs)
    box=[x0,y0,x1,y1]
    origin=np.array([x0,y0])
    seed='frame_000020'; idx=rows[seed]['source_frame_index']
    source=rows[seed]['paths']['upper']; u0,v0,*_=source['crop_original_xyxy']
    points=np.array([r[1] for r in SEEDS])+[u0,v0]
    assert np.isin(idx,ix) and np.all(points>=origin+16) and np.all(points<np.array([x1,y1])-16)
    scope=dict(source_video_sha256=sha(VIDEO),contacts_sha256=sha(CONTACT),fit_plan_sha256=sha(BASE/'fit-plan.json'),
        flow_source_sha256=sha(Path(__file__).with_name('track_native_continuity122.py')),runner_sha256=sha(__file__),
        source_roster=roster,skipped_source_indices=skipped,seed_frame=seed,seed_index=idx,native_crop=box,
        seeds=[dict(id=name,source_native_xy=xy.tolist(),role=role) for (name,_,role),xy in zip(SEEDS,points)],
        lk=PARAMS,forward_backward_limit_px=FB_LIMIT,budget_seconds=120,
        protocol='Bidirectional adjacent-frame LK from the fixed source-only seed. Decode fixed integer crop with no rescale. Skip the exact eight protected sources; intermediate frames only carry trajectories. '
                 'Stop at first numerical loss, no reacquisition. Label original physical region in source previews before considering geometry. Whole source seed families stay in their declared roles.')
    atomic(plan,scope)
    ctrl=controls()
    total,n=ix[-1]+1,len(points)
    uv=np.full((total,n,2),np.nan,np.float32)
    backward=np.full_like(uv,np.nan)
    fb=np.full((total,n),np.nan,np.float32)
    status=np.full((total,n),-1,np.int16)
    uv[idx]=points;status[idx]=0
    checked={}
    for reverse,src in ((True,range(ix[0],idx+1)),(False,range(idx,ix[-1]+1))):
        seq=[i for i in src if i not in skipped]
        if reverse: seq.reverse()
        current=None
        active=np.arange(n)
        for i,rgb in decode(seq,box,reverse):
            assert time.monotonic()-t0<scope['budget_seconds']
            if i in roster:
                verify_crop(rgb,rows[roster[i]]['paths']['upper'],box)
                checked[i]=True
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            if current is not None and len(active):
                q,z,err,_,reason=step(current,gray,uv[prev,active]-origin)
                uv[i,active]=q+origin
                backward[i,active]=z+origin
                fb[i,active]=err;status[i,active]=reason
                active=active[reason==0]
            current,prev=gray,i
        print('TORSO141_DIRECTION',reverse,len(seq),len(active),flush=True)
    assert set(checked)==set(roster) and (status[skipped]==-1).all()
    np.savez_compressed(OUT/'tracked-pixels.npz',uv=uv,backward=backward,forward_backward=fb,status=status)
    pages=[]
    for first in range(0,len(ix),8):
        page=Image.new('RGB',(66*n,66*min(8,len(ix)-first)),'#23242a')
        for j,i in enumerate(ix[first:first+8]):
            row=rows[roster[i]]['paths']['upper'];xy0=np.array(row['crop_original_xyxy'][:2])
            with Image.open(relocate(row['path'])) as im:
                line=draw_roster(im.convert('RGB'),i,uv[i],status[i]==0,xy0)
            page.paste(line,(0,j*66))
        path=OUT/f'roster-atlas-{first//8+1}.png';page.save(path)
        pages.append(dict(path=str(path),sha256=sha(path),rows=[roster[i] for i in ix[first:first+8]]))
    details=[dict(frame=roster[i],native_index=i,valid_seed_ids=[SEEDS[j][0] for j in np.flatnonzero(status[i]==0)]) for i in ix]
    atomic(state,dict(plan_sha256=sha(plan),seconds=time.monotonic()-t0,flow_controls=ctrl,
        roster_exact_native_crop_matches=len(checked),tracks_sha256=sha(OUT/'tracked-pixels.npz'),
        rows=details,pages=pages,visual_identity_review_pending=True,new_camera_written=False,fit_started=False,quality_accepted=False))
    print('TORSO141',len(checked),[len(r['valid_seed_ids']) for r in details],flush=True)


if __name__=='__main__':
    run()
