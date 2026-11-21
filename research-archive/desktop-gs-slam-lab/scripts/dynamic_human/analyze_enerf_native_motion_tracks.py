"""Independent endpoint arithmetic and bounded native-photo correspondence panels."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np, cv2
from PIL import Image, ImageDraw

L=Path(__file__).resolve().parents[2]
R=L/'outputs/dynamic-human-enerf-native-motion-tracks-075'
I=L/'outputs/dynamic-human-enerf-articulated-control-064'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (R/'model-comparison-completion.json').is_file()
(R/'review').mkdir(exist_ok=True)

def stats(a):
    a=np.asarray(a)
    return dict(n=int(a.size),median=float(np.median(a)),mean=float(np.mean(a)),p90=float(np.percentile(a,90)),max=float(np.max(a))) if a.size else dict(n=0)

rows=[];detail=[];checks=[];arrays={};worst=None
for record in read(R/'preparation.json')['records']:
    c=record['camera'];f=R/'tracks'/f'camera{c}.npz';assert sha(f)==record['tracks_sha256']
    tr=dict(np.load(f));a=dict(np.load(R/'baseline067'/f'camera{c}.npz'));b=dict(np.load(R/'correctRGB074'/f'camera{c}.npz'))
    for pred in [a,b]:
        assert np.array_equal(pred['source_xy'],tr['source_xy']) and np.array_equal(pred['target_frame'],tr['target_frame'])
        assert np.isfinite(pred['flow']).all()
        assert np.allclose(pred['flow'],pred['flow_numerator']/np.maximum(pred['alpha'][None,:,None],1e-8),atol=1e-6)
        assert np.allclose(pred['actor_mass'].sum(-1),pred['alpha'],atol=2e-6)
        assert np.allclose(pred['predicted_target_xy'],tr['source_xy'][None]+pred['flow'],atol=2e-5)
    accepted=tr['accepted']
    assert np.all(tr['fb'][accepted]<.5) and np.all(tr['direct_fb'][accepted]<.5)
    assert np.all(tr['direct_disagreement'][accepted]<1) and np.all(tr['patch_ncc'][accepted]>=.97) and np.all(tr['patch_mae'][accepted]<=12)
    # Both models must explain the same source-person pixel for the paired subset.
    common=accepted.copy();gates=[]
    for pred in [a,b]:
        fraction=pred['actor_mass'][np.arange(len(tr['actor'])),tr['actor']]/np.maximum(pred['alpha'],1e-8)
        gate=(pred['alpha'][None]>=.95)&(fraction[None]>=.9)&(pred['positive_depth_mass']/np.maximum(pred['alpha'][None],1e-8)>=.99)
        gates.append(gate);common&=gate
    e0=np.linalg.norm(a['predicted_target_xy']-tr['target_xy'],axis=-1)
    e1=np.linalg.norm(b['predicted_target_xy']-tr['target_xy'],axis=-1)
    travel=np.linalg.norm(tr['target_xy']-tr['source_xy'][None],axis=-1)
    arrays[c]=(tr,a,b,common,e0,e1)
    for j,t in enumerate(tr['target_frame']):
        valid=common[j]
        for actor in [0,1]:
            for region in ['all','right_hand_box','other_actor_interior']:
                q=valid&(tr['actor']==actor)
                if region!='all':q&=tr['region']==region
                detail.append(dict(camera=c,frame=int(t),actor=actor,region=region,
                    before=stats(e0[j,q]),after=stats(e1[j,q]),observed_displacement=stats(travel[j,q]),
                    model_flow_dispersion_after=stats(np.sqrt(b['variance'][j,q].sum(-1)))))
        for k in np.flatnonzero(valid):
            row=dict(camera=c,frame=int(t),feature=int(k),actor=int(tr['actor'][k]),region=str(tr['region'][k]),
                before_endpoint_px=float(e0[j,k]),after_endpoint_px=float(e1[j,k]),observed_displacement_px=float(travel[j,k]),
                source_xy=tr['source_xy'][k].tolist(),tracked_xy=tr['target_xy'][j,k].tolist(),model_xy=b['predicted_target_xy'][j,k].tolist())
            rows.append(row)
            if worst is None or row['after_endpoint_px']>worst['after_endpoint_px']:worst=row
        # Recompute native patch similarity for a bounded deterministic sample.
        ids=np.flatnonzero(accepted[j])[::max(1,int(accepted[j].sum())//12)]
        original=np.load(I/'native-input'/f'camera{c}-frame000750.npz')['rgb']
        target=np.load(I/'native-input'/f'camera{c}-frame{t:06d}.npz')['rgb']
        ga=cv2.cvtColor(original,cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
        for k in ids:
            p=cv2.getRectSubPix(ga,(11,11),tuple(map(float,tr['source_xy'][k]))).astype(float).ravel()
            q=cv2.getRectSubPix(gb,(11,11),tuple(map(float,tr['target_xy'][j,k]))).astype(float).ravel()
            ncc=float(np.dot(p-p.mean(),q-q.mean())/(np.linalg.norm(p-p.mean())*np.linalg.norm(q-q.mean())))
            mae=float(np.mean(abs(p-q)))
            assert abs(ncc-float(tr['patch_ncc'][j,k]))<1e-6 and abs(mae-float(tr['patch_mae'][j,k]))<1e-5
            checks.append(dict(camera=c,frame=int(t),feature=int(k),ncc=ncc,mae=mae))
    print('TRACK_ANALYSIS',c,int(accepted.sum()),int(common.sum()),flush=True)

summary=dict(status='native_motion_diagnostic_numeric_complete_visual_review_pending',
    feature_seeds=read(R/'preparation.json')['reference_seeds'],
    accepted_image_track_observations=read(R/'preparation.json')['accepted_observations'],
    paired_model_visible_observations=len(rows),
    endpoint_error_to_image_track_px=dict(baseline067=stats([r['before_endpoint_px'] for r in rows]),correctRGB074=stats([r['after_endpoint_px'] for r in rows])),
    right_hand_box=dict(observations=sum(r['region']=='right_hand_box' for r in rows),
        before=stats([r['before_endpoint_px'] for r in rows if r['region']=='right_hand_box']),after=stats([r['after_endpoint_px'] for r in rows if r['region']=='right_hand_box'])),
    worst_model_disagreement=worst,detail=detail,
    qualification='Source-visibility-filtered projected Gaussian-center motion compared with sparse LK hypotheses, not true surface motion or anatomical finger labels. No endpoint-error selection. Repeated frame observations are not independent subjects.',
    model_updates=0,goal_complete=False)
save(R/'motion-summary.json',summary)
save(R/'endpoint-records.json',dict(records=rows,goal_complete=False))
save(R/'independent-arithmetic-audit.json',dict(status='saved_moments_actor_mass_endpoint_arithmetic_and_bounded_native_patch_scores_recomputed',
    model_camera_arrays=26,native_patch_checks=len(checks),checks=checks,goal_complete=False))

def cross(im,xy,color,size=3):
    d=ImageDraw.Draw(im);x,y=map(float,xy);d.line((x-size,y,x+size,y),fill=color,width=1);d.line((x,y-size,x,y+size),fill=color,width=1)

def panel(c,t,box,ids,path):
    tr,a,b,common,_,_=arrays[c];j=list(tr['target_frame']).index(t)
    src=np.load(I/'native-input'/f'camera{c}-frame000750.npz')['rgb'];dst=np.load(I/'native-input'/f'camera{c}-frame{t:06d}.npz')['rgb']
    x0,y0,x1,y1=map(int,box);w=x1-x0;h=y1-y0
    originals=[Image.fromarray(src[y0:y1,x0:x1]),Image.fromarray(dst[y0:y1,x0:x1]),Image.fromarray(dst[y0:y1,x0:x1])]
    images=[x.copy() for x in originals]
    for k in ids:
        for column,xy,color in [(0,tr['source_xy'][k],(255,190,0)),(1,tr['target_xy'][j,k],(30,220,70)),(2,b['predicted_target_xy'][j,k],(255,80,30))]:
            cross(images[column],xy-np.array([x0,y0]),color)
    out=Image.new('RGB',(3*max(w,200),h+54),'white');draw=ImageDraw.Draw(out)
    for n,(im,title) in enumerate(zip(images,['Original750 + feature seeds',f'Original{t} + tracked patches',f'Original{t} + model motion'])):
        out.paste(im,(n*max(w,200),54));draw.text((n*max(w,200)+4,8),title,fill='black')
    draw.text((4,30),f'Camera{c} | {len(ids)} accepted corresponding features | overlays on photographs, NOT reconstruction renders',fill='black')
    out.save(path)
    return dict(file=str(path.relative_to(R)),camera=c,frame=t,bbox=box,features=list(map(int,ids)),source='Original photo pixels plus explicit feature crosses; no enhancement, no generated scene content.')

panels=[]
for c in ['01','04','07','08']:
    tr,a,b,common,e0,e1=arrays[c]
    for t in [744,756]:
        j=list(tr['target_frame']).index(t);ids=np.flatnonzero(common[j])
        # Deterministic spatial spacing, not largest or smallest endpoint error.
        selected=[]
        for actor in [0,1]:
            picked=[]
            for k in ids[tr['actor'][ids]==actor]:
                if all(np.linalg.norm(tr['source_xy'][k]-tr['source_xy'][v])>=30 for v in picked):picked.append(k)
                if len(picked)>=12:break
            selected.extend(picked)
        bb=np.load(I/'native-input'/f'camera{c}-frame000750.npz')['bbox'].tolist()
        bt=np.load(I/'native-input'/f'camera{c}-frame{t:06d}.npz')['bbox'].tolist()
        box=[min(bb[0],bt[0]),min(bb[1],bt[1]),max(bb[2],bt[2]),max(bb[3],bt[3])]
        panels.append(panel(c,t,box,selected,R/'review'/f'camera{c}-frame{t:06d}-coverage.png'))
        for actor in [0,1]:
            hb=tr['hand_boxes'][actor];center=(hb[:2]+hb[2:])/2
            # Follow accepted hand-box correspondences only; if absent, display
            # the source neighborhood without pretending it is a target-hand crop.
            hand=ids[(tr['actor'][ids]==actor)&(tr['region'][ids]=='right_hand_box')]
            points=[center]
            if len(hand):points.extend(tr['target_xy'][j,hand].tolist())
            points=np.array(points);lo=np.floor(points.min(0)-60).astype(int);hi=np.ceil(points.max(0)+60).astype(int)
            box=[int(max(0,lo[0])),int(max(0,lo[1])),int(min(1920,hi[0])),int(min(1080,hi[1]))]
            panels.append(panel(c,t,box,hand,R/'review'/f'camera{c}-frame{t:06d}-actor{actor}-handbox.png'))
w=worst;c=w['camera'];t=w['frame'];pts=np.array([w['source_xy'],w['tracked_xy'],w['model_xy']]);lo=np.floor(pts.min(0)-35).astype(int);hi=np.ceil(pts.max(0)+35).astype(int)
box=[int(max(0,lo[0])),int(max(0,lo[1])),int(min(1920,hi[0])),int(min(1080,hi[1]))]
panels.append(panel(c,t,box,[w['feature']],R/'review'/'largest-model-disagreement.png'))
save(R/'visual-review-plan.json',dict(status='fixed_camera_time_panels_plus_maximum_endpoint_disagreement',records=panels,actual_review_complete=False,goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps({k:v for k,v in summary.items() if k not in ['detail']},ensure_ascii=False))
