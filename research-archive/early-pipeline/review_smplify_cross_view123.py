"""Pair-matched existing-scene negative control and unwarped evidence images."""
from itertools import combinations
from pathlib import Path
import json, hashlib, time
import numpy as np
from PIL import Image, ImageDraw
import cv2
import audit_smplify_observation_contract123 as a

O=a.OUT; read=a.read; save=a.save; sha=a.sha
assert (O/'result.json').exists() and not (O/'pair-negative-control.json').exists()
tracks_path=a.ROOT/'camera/appearance-tracks.json'
rows=read(a.ROOT/'data/inputs.json')['records']
save(O/'pair-negative-control-plan.json',dict(registered=a.now(),
    purpose='Check the same three view-pair epipolar convention against all existing shared SIFT tracks; split by original FIT/HELD and mask support without error selection.',
    pairs=[[0,31],[31,62],[0,62]],
    source_sha256=sha(tracks_path),mask_classification='At both rounded native pixels: both outside person, both inside, or mixed. These labels do not prove static physical identity.',
    budget=dict(cpu_seconds=60,new_features=0,new_fits=0),
    no_new_or_reassigned_tracks=True,no_residual_filter=True))
started=time.monotonic(); tracks=read(tracks_path)['records']
xy={s:np.load(a.ROOT/f'camera/features-{s:03d}.npz')['xy'] for s in [0,31,62]}
masks={s:np.array(Image.open(rows[s]['mask']))>0 for s in [0,31,62]}
out=[]
for i,j in [(0,31),(31,62),(0,62)]:
    group={};F=a.fundamental(rows[i],rows[j]);shared=[]
    for tr in tracks:
        nodes=dict(tr['nodes'])
        if i not in nodes or j not in nodes:continue
        p,q=xy[i][nodes[i]],xy[j][nodes[j]]
        flags=[]
        for s,v in [(i,p),(j,q)]:
            x,y=np.rint(v).astype(int);assert 0<=x<masks[s].shape[1] and 0<=y<masks[s].shape[0]
            flags.append(bool(masks[s][y,x]))
        region='both_person_mask' if all(flags) else 'both_outside_person_mask' if not any(flags) else 'mixed_mask'
        split='HELD' if tr['held'] else 'FIT';key=split+'_'+region
        err=float(a.epipolar(F,p[None],q[None])[0]); group.setdefault(key,[]).append(err)
        shared.append(dict(id=tr['id'],split=split,region=region,uv=[p.tolist(),q.tolist()],error_px=err))
    out.append(dict(slots=[i,j],shared_tracks=len(shared),groups={k:a.stats(v) for k,v in group.items()},records=shared))
assert time.monotonic()-started<60
save(O/'pair-negative-control.json',dict(updated=a.now(),pairs=out,tracks_sha256=sha(tracks_path),
    original_splits_preserved=True,no_error_selection=True,no_camera_changes=True,
    physical_identity_and_staticity_proven=False))

crops={r['slot']:r for r in read(a.REF/'input-crops.json')}
fits={s:dict(np.load(a.FIT/f'slot{s:03d}-fit.npz')) for s in [0,31,62]}
record=Path('/mnt/d/00/记录/005/分析/跨视角与重复目标')
record.mkdir(parents=True,exist_ok=True)
images=[]

def publish(im,name,kind,extra):
    path=record/name;assert not path.exists();im.save(path)
    # One readback validates the file without moving the underlying source data.
    with Image.open(path) as check:check.load();assert check.size==im.size
    images.append(dict(path=str(path),filename=name,kind=kind,wh=list(im.size),sha256=sha(path),**extra))

def combine(parts):
    sheet=Image.new('RGB',(sum(p.width for p in parts),parts[0].height));x=0
    for p in parts:sheet.paste(p,(x,0));x+=p.width
    return sheet

for s in [0,31,62]:
    raw=Image.open(rows[s]['image']).convert('RGB');fit=fits[s]
    for side,slots,label in [('left_hand',[7,25],'左腕'),('right_hand',[4,46],'右腕')]:
        b=np.array(crops[s]['regions'][side]['crop_integer']);base=raw.crop(tuple(b));end=base.copy();d=ImageDraw.Draw(end)
        points=fit['targets'][0,slots]-b[:2]
        for p,color in zip(points,[(0,255,80),(255,70,30)]):d.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),outline=color,width=2)
        d.line([tuple(points[0]),tuple(points[1])],fill=(255,210,0),width=1)
        sheet=combine([base,end]);sheet=sheet.resize((sheet.width*4,sheet.height*4),Image.Resampling.NEAREST)
        publish(sheet,f'20261029_视频帧{rows[s]["source_frame_index"]:06d}_{label}_原图_身体腕绿与手根红.png','duplicate_wrist_targets',dict(slot=s,target_slots=slots,crop=b.tolist()))

# Nose 30, outer eye corners 36/45 and mouth corners 48/54 in IBUG68.
face_targets=np.array([30,36,45,48,54])-17+67
for i,j in [(0,31),(31,62),(0,62)]:
    F=a.fundamental(rows[i],rows[j]);b=np.array(crops[j]['regions']['face']['crop_integer'])
    raw=Image.open(rows[j]['image']).convert('RGB').crop(tuple(b));own=raw.copy();epi=raw.copy()
    for im in [own,epi]:
        d=ImageDraw.Draw(im)
        for p in fits[j]['projected'][0,face_targets]-b[:2]:d.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=(255,0,220))
    d=ImageDraw.Draw(epi)
    for p in fits[i]['projected'][0,face_targets]:
        l=F@np.r_[p,1.];line=l.copy();line[2]+=line[0]*b[0]+line[1]*b[1]
        # Full mathematical line clipped only to the displayed native crop.
        if abs(line[1])>abs(line[0]):
            ends=[(0,-line[2]/line[1]),(raw.width-1,-(line[0]*(raw.width-1)+line[2])/line[1])]
        else:
            ends=[(-line[2]/line[0],0),(-(line[1]*(raw.height-1)+line[2])/line[0],raw.height-1)]
        d.line(ends,fill=(0,235,255),width=1)
    sheet=combine([raw,own,epi]);sheet=sheet.resize((sheet.width*4,sheet.height*4),Image.Resampling.NEAREST)
    publish(sheet,f'20261029_视频帧{rows[i]["source_frame_index"]:06d}到{rows[j]["source_frame_index"]:06d}_面部_原图_本视角模型紫_另一视角极线青.png',
            'same_camera_cross_view_epipolar',dict(slots=[i,j],crop=b.tolist(),target_slots=face_targets.tolist()))
save(O/'rendered-evidence.json',dict(updated=a.now(),images=images,no_text_on_images=True,no_image_warp=True,
    diagram_semantics='Diagnostic targets and epipolar lines, not verified physical correspondences. Mathematical lines may leave the crop.',
    source_view_projection_fixed=True,camera_unchanged=True,new_work_log_images=False))
print(json.dumps({'negative_control':[{k:v for k,v in r.items() if k!='records'} for r in out],'images':len(images)},ensure_ascii=False))
