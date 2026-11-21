"""Complete the tiny seven-track face-ROI census and show held background checks.
Read-only, error-independent visual inspection supplement; no new solver run.
"""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,OUT,REVIEW,read,save,sha,native

def main():
    pp=REVIEW/'complete-review-plan.json'
    if pp.exists():raise RuntimeError('Read the existing visual supplement; do not repeat.')
    d=dict(np.load(OUT/'diagnostic-observations.npz'));idx=read(REVIEW/'index.json')
    old={x['frame'] for x in idx['images'] if x['region']=='FIT_face_ROI_candidates'}
    allids=sorted(set(d['fit_ids'][d['fit_region_flags'][:,4]]));remaining=sorted(set(allids)-old)
    assert len(allids)==7 and len(remaining)==4
    selected=read(OUT/'selected-samples.json')['samples']
    bg=[s['track_id'] for s in selected if s['split']=='HELD' and s['category']=='all_nodes_outside_person_boxes']
    save(pp,dict(created=datetime.now(timezone.utc).isoformat(),reason='Point-location review showed that every initial FIT face-ROI sample was actually background; inspect all remaining IDs, no error selection.',
        remaining_FIT_face_ROI_track_ids=remaining,original_seven_track_census=allids,
        held_background_ids=bg,held_background_views='First chronological check plus nearest anchor on each side; no residual-based selection.',
        no_feature_extraction=True,no_optimization=True,no_split_or_observation_changes=True,images_have_no_text=True))
    rows=read(ROOT/'data/training-inputs.json')['records']
    tracks={t['id']:t for t in read(ROOT/'camera/appearance-tracks.json')['records']}
    held={t['id']:t for t in read(ROOT/'camera/held-evaluation.json')['records']}
    tasks=[]
    for ident in remaining:
        nodes=tracks[ident]['nodes']; chosen=[nodes[i] for i in np.unique(np.linspace(0,len(nodes)-1,3).astype(int))]
        tasks.append(('FIT_face_ROI_census',ident,chosen))
    for ident in bg:
        nodes=held[ident]['nodes'];tasks.append(('HELD_background_anchor_check_anchor',ident,nodes[:3]))
    results=[]
    for category,ident,nodes in tasks:
        parts=[];sources=[]
        for i,j in nodes:
            xy=np.load(ROOT/f'camera/features-{i:03d}.npz')['xy'][j];cx,cy=map(int,np.rint(xy));crop=(cx-128,cy-128,cx+128,cy+128)
            with Image.open(native(rows[i]['image'])) as im:parts.append(im.convert('RGB').crop(crop))
            src=dict(frame=rows[i]['frame'],source_frame_index=rows[i]['source_frame_index'],xy=xy.tolist(),crop=crop)
            if ident in held:
                tr=held[ident];src['role']='CHECK' if i in tr['test_frames'] else 'ANCHOR'
                src['error_px']=tr['errors'][tr['test_frames'].index(i)] if i in tr['test_frames'] else None
            sources.append(src)
        im=Image.new('RGB',(256*len(parts),256))
        for j,p in enumerate(parts):im.paste(p,(256*j,0))
        name=f'{category}_{ident[:12]}.png';im.save(REVIEW/name);overlay=im.copy();draw=ImageDraw.Draw(overlay)
        for j,s in enumerate(sources):
            x=s['xy'][0]-s['crop'][0]+j*256;y=s['xy'][1]-s['crop'][1]
            draw.ellipse((x-10,y-10,x+10,y+10),outline=(0,255,255),width=1)
            for a,b,c,e in [(x-20,y,x-12,y),(x+12,y,x+20,y),(x,y-20,x,y-12),(x,y+12,x,y+20)]:draw.line((a,b,c,e),fill=(0,255,255),width=1)
        overlay_name=name.replace('.png','_locator.png');overlay.save(REVIEW/overlay_name)
        entry=dict(frame=ident,region=category,file=name,locator=overlay_name,columns=[s['frame'] for s in sources],sources=sources,
            sha256=sha(REVIEW/name),native_pixels_no_warp=True,labels_in_pixels=False,selected_without_error=True)
        results.append(entry)
    save(REVIEW/'complete-review-index.json',dict(subject='grayT',images=results,actual_review_pending=True))
    # Extend the same existing gallery, preserving the original raw strips.
    idx['images'].extend(results);save(REVIEW/'index.json',idx)
    print(json.dumps(results,ensure_ascii=False))

if __name__=='__main__':main()
