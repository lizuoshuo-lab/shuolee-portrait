"""Bounded raw-observation audit for half-circle8; no fit or target changes."""
import inspect,os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import run_sofa_selected_views124 as prior
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import crop_box,joined

OUT=prior.original.ROOT/'sofa-half8-input-audit124'

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    OUT.mkdir();source=read(prior.SRC/'selection.json')
    rows=read(prior.SRC/'data/inputs.json')['records'];old=read(prior.OUT/'plan.json')['frames']
    rejected=read(prior.OUT/'source-review.json')['rejected_input_frames']
    held=set(source['review_only_frames']);times={r['frame']:r['time_seconds'] for r in rows}
    scores={r['frame']:r['score'] for r in source['candidates']}
    guards={r['frame']:r['screen_flags'] for r in read(prior.original.ROOT/'selected-observation-audit124/wrist-preselection/sofa.json')['records']}
    eligible=[r['frame'] for r in rows if r['frame'] not in set(old)|held|set(rejected) and not guards[r['frame']]]
    ranked=sorted(eligible,key=lambda f:(-min(abs(times[f]-times[g]) for g in old),scores[f],f))
    names=ranked[:6]
    save(OUT/'plan.json',dict(updated=now(),scope='Single CPU input/coverage audit; no fitting, no edits to observation coordinates or confidence.',
        capture_arc='front_half_or_less_from_original_images; no observed rear',target_count=8,retained_seven=old,
        rejected_preserved=rejected,heldout_preserved=sorted(held),
        candidate_rank_rule='Maximum nearest chronological distance to retained7, tie by frozen prefit score then frame ID. Chronology is only a coverage proxy, not equal-angle certification.',
        candidate_limit=6,candidate_priority=names,
        choice_rule='First raw-source ownership-consistent candidate in the frozen list. If none, keep7with explicit gap; never use heldout or restore the rejected53.',
        optimizer_updates=0,quality_accepted=False))
    namespace=dict(prior.scope)
    code=inspect.getsource(prior.original.data).replace("read(SRC/'selection.json')['training_frames']","[r['frame'] for r in rows]")
    exec(compile(code,'<same-all56-observation-mapping>','exec'),namespace)
    allrows,targets=namespace['data']();lookup={r['frame']:i for i,r in enumerate(allrows)}
    records=[];tiles=[]
    for f in names:
        i=lookup[f];row=allrows[i];target=targets[i]
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        box=crop_box(mask.getbbox(),raw.size,35);crop=raw.crop(box);points=np.asarray(crop).copy()
        for ids,color in [(prior.original.entry.COCO,(0,255,0)),(range(25,46),(255,60,60)),(range(46,67),(60,100,255))]:
            for j in ids:
                p=target[j]
                if p[2]>0:cv2.circle(points,tuple(np.round(p[:2]-box[:2]).astype(int)),3,color,1,cv2.LINE_AA)
        panel=joined([crop,Image.fromarray(points)])
        path=Path('/mnt/d/00/记录/006')/f'分析_沙发半圈补帧124_{f}_原图与2D观测.png'
        assert not path.exists();panel.save(path)
        thumb=panel.copy();thumb.thumbnail((700,700));tiles.append(thumb)
        hp=target[25:67,:2];hb=crop_box([hp[:,0].min(),hp[:,1].min(),hp[:,0].max()+1,hp[:,1].max()+1],raw.size,30)
        hand=raw.crop(hb);hand_points=np.asarray(hand).copy()
        for ids,color in [(range(25,46),(255,60,60)),(range(46,67),(60,100,255))]:
            for j in ids:
                p=target[j]
                if p[2]>0:cv2.circle(hand_points,tuple(np.round(p[:2]-hb[:2]).astype(int)),3,color,1,cv2.LINE_AA)
        hpath=OUT/(f+'-hands.png');joined([hand,Image.fromarray(hand_points)]).save(hpath)
        records.append(dict(frame=f,body=str(path),hands=str(hpath),body_sha256=sha(path),hands_sha256=sha(hpath),
            score=scores[f],nearest_time_distance=min(abs(times[f]-times[g]) for g in old),actual_visual_review=False))
    pages=[]
    for first in range(0,6,3):
        sheet=Image.new('RGB',(2100,700))
        for i,im in enumerate(tiles[first:first+3]):sheet.paste(im,(700*i,0))
        path=OUT/f'candidates-{first//3+1}.png';sheet.save(path);pages.append(str(path))
    save(OUT/'result.json',dict(updated=now(),records=records,pages=pages,actual_visual_review=False,
        selected_additional_frame=None,training_started=False,source_observations_unchanged=True))
    print('SOFA_HALF8_CANDIDATES',names,flush=True)

if __name__=='__main__':main()
