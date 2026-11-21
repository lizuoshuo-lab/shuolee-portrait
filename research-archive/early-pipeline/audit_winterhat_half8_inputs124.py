"""One bounded native-source ownership audit, before any new winterhat fit."""
import inspect, os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import run_grass_selected_views124 as original
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import crop_box, joined

SRC = original.ROOT/'winterhat-input'
OUT = original.ROOT/'winterhat-half8-input-audit124'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUT.exists()
    selection = read(SRC/'selection.json')
    rows = read(SRC/'data/inputs.json')['records']
    guards = {r['frame']:r['screen_flags'] for r in read(original.ROOT/'selected-observation-audit124/wrist-preselection/winterhat.json')['records']}
    retained = [f for f in selection['training_frames'] if not guards[f]]
    rejected = [f for f in selection['training_frames'] if guards[f]]
    assert rejected == ['frame_000013'] and len(retained) == 7
    held = set(selection['review_only_frames'])
    times = {r['frame']:r['time_seconds'] for r in rows}
    scores = {r['frame']:r['score'] for r in selection['candidates']}
    eligible = [r['frame'] for r in rows if r['frame'] not in set(retained)|held|set(rejected) and not guards[r['frame']]]
    ranked = sorted(eligible,key=lambda f:(-min(abs(times[f]-times[g]) for g in retained),scores[f],f))
    candidates = ranked[:6]
    assert candidates == ['frame_000001','frame_000002','frame_000003','frame_000028','frame_000029','frame_000083']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),retained_seven=retained,rejected_preserved=rejected,
        heldout_preserved=sorted(held),candidate_priority=candidates,candidate_limit=6,
        rule='Maximum nearest chronological distance to retained7, tie frozen prefit score then frame ID; same as sofa. Time is a coverage proxy only.',
        choice='First source-ownership-consistent candidate in this frozen list; if none, do not fill with heldout or known conflict.',
        targets_and_confidence_unchanged=True,optimizer_updates=0,quality_accepted=False))
    source = inspect.getsource(original.data).replace("selected=read(SRC/'selection.json')['training_frames']", "selected=[r['frame'] for r in rows]")
    namespace = dict(original.__dict__,SRC=SRC)
    exec(compile(source,'<same-all112-winterhat-observation-mapping>','exec'),namespace)
    allrows, targets = namespace['data']()
    lookup = {r['frame']:i for i,r in enumerate(allrows)}
    records,tiles = [],[]
    for f in rejected+candidates:
        i=lookup[f];row=allrows[i];target=targets[i]
        assert sha(row['image']) == row['source_sha256'] and sha(row['mask']) == row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        box=crop_box(mask.getbbox(),raw.size,35);crop=raw.crop(box);points=np.asarray(crop).copy()
        for ids,color in [(original.entry.COCO,(0,255,0)),(range(25,46),(255,60,60)),(range(46,67),(60,100,255))]:
            for j in ids:
                p=target[j]
                if p[2]>0:cv2.circle(points,tuple(np.round(p[:2]-box[:2]).astype(int)),3,color,1,cv2.LINE_AA)
        panel=joined([crop,Image.fromarray(points)])
        path=Path('/mnt/d/00/记录/001')/f'分析_冬帽半圈补帧124_{f}_原图与2D观测.png'
        assert not path.exists();panel.save(path)
        thumb=panel.copy();thumb.thumbnail((700,700));tiles.append(thumb)
        hands=target[25:67,:2];hb=crop_box(np.r_[hands.min(0),hands.max(0)],raw.size,30)
        hand=raw.crop(hb);hp=np.asarray(hand).copy()
        for ids,color in [(range(25,46),(255,60,60)),(range(46,67),(60,100,255))]:
            for j in ids:
                p=target[j]
                if p[2]>0:cv2.circle(hp,tuple(np.round(p[:2]-hb[:2]).astype(int)),3,color,1,cv2.LINE_AA)
        hpath=OUT/(f+'-hands.png');joined([hand,Image.fromarray(hp)]).save(hpath)
        records.append(dict(frame=f,body=str(path),hands=str(hpath),body_sha256=sha(path),hands_sha256=sha(hpath),
            existing_screen_flags=guards[f],prefit_score=scores[f],actual_visual_review=False))
    pages=[]
    for start in range(0,len(tiles),3):
        sheet=Image.new('RGB',(2100,700))
        for i,im in enumerate(tiles[start:start+3]):sheet.paste(im,(700*i,0))
        path=OUT/f'candidates-{start//3+1}.png';sheet.save(path);pages.append(str(path))
    save(OUT/'result.json',dict(updated=now(),records=records,pages=pages,actual_visual_review=False,
        selected_additional_frame=None,training_started=False,source_observations_unchanged=True))
    print('WINTERHAT_HALF8_SOURCE_AUDIT_READY',candidates,flush=True)


if __name__ == '__main__':main()
