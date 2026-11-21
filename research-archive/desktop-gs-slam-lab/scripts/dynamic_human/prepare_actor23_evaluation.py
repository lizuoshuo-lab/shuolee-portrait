"""Freeze native evaluation sources and detail windows before candidate rendering."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';D=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor2_3');E=O/'evaluation-input';E.mkdir(exist_ok=True);V=O/'review-plan';V.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'evaluation-protocol.json').exists();assert not any((O/b/'native-evaluation/metrics.json').exists() for b in ['static','dynamic_nominal','dynamic_clock'])
plan=read(O/'validation-reservation.json');cal=read(O/'cameras.json');train_clock=read(O/'geometry-time/candidate-clock-offsets.json')['offsets'];dc=read(O/'evaluation-time/candidate-clock-offsets.json')['offsets'];check=read(O/'evaluation-time/moving-track-clock-audit.json');assert check['jacobian_rank']==17 and check['validation']['all_track_fit']['median']<check['validation']['zero']['median']
changes={c:dc[c]-train_clock[c] for c in plan['train_cameras']};eval_clock={**train_clock,**{c:dc[c] for c in plan['heldout_cameras']}}
if not (O/'evaluation-clock-decision.json').exists():save(O/'evaluation-clock-decision.json',dict(status='frozen_before_reserved_appearance_access_and_candidate_inspection',recorded_utc=datetime.now(timezone.utc).isoformat(),evaluation_offsets=eval_clock,train_vs_disjoint_offset_difference_frames=changes,max_absolute_difference=float(max(abs(x) for x in changes.values())),median_absolute_difference=float(np.median([abs(x) for x in changes.values()])),disjoint_moving_track_validation=check['validation'],selection='Use original training-only offsets for15knowncameras and the predeclared disjoint-time all-track estimator for00/06/12. Same camera04reference. Neither estimator sees reserved appearance images.',limitations='Empirical offsets, not hardware timestamps. Other cameras were seen only for calibration at582..594, so600..624is appearance/geometry-heldout, not unseen sensors. No candidate image used for selection.',goal_complete=False))
audit={(r['camera'],r['frame']):r for r in read(O/'training-input-audit.json')['records']};rows=[];full=[];details=[];fallbacks=[];cv2.setNumThreads(3)
initial=np.load(O/'model/initial.npz');major_components=np.argsort(-np.bincount(initial['component']))[:2]
review_cams=['00','04','06','07','12','13'];review_frames=[601,613,623];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for c in sorted(cal):
    K=np.array(cal[c]['K'],float);W=np.array(cal[c]['w2c'],float);dist=np.array(cal[c]['D'],float);mx,my=cv2.initUndistortRectifyMap(K,dist,None,K,(1920,1080),cv2.CV_32FC1);valid=cv2.remap(np.ones((1080,1920),np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>.999
    for f in plan['frame_window']:
        split='validation_camera_time' if c in plan['heldout_cameras'] and f%2 else 'validation_camera' if c in plan['heldout_cameras'] else 'validation_time' if f%2 else 'train_check' if c in ['01','04','13'] and f in [600,612,624] else None
        if split is None:continue
        if split=='train_check':
            path=O/audit[c,f]['input'];z=np.load(path);rgb,matte,v,bbox=z['rgb'],z['matte'],z['valid'],z['bbox'];input_sha=sha(path);assert input_sha==audit[c,f]['input_sha256'];source_hash=audit[c,f]['source_rgb_sha256'];source_matte_hash=audit[c,f]['source_matte_sha256']
        else:
            source=D/'images'/c/f'{f:06d}.jpg';mask=D/'bgmtv2'/c/f'{f:06d}.jpg';rgb=cv2.cvtColor(cv2.imread(str(source)),cv2.COLOR_BGR2RGB);matte=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255;assert rgb.shape==(1080,1920,3) and matte.shape==(1080,1920);rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);matte=cv2.remap(matte,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);v=valid;yy,xx=np.where((matte>.05)&v);bbox=np.array([max(0,xx.min()-24),max(0,yy.min()-24),min(1920,xx.max()+25),min(1080,yy.max()+25)],np.int64);path=E/f'camera{c}-frame{f:06d}.npz';np.savez_compressed(path,rgb=rgb,matte=matte,valid=v,K=K,w2c=W,bbox=bbox) if not path.exists() else None;stored=np.load(path);assert all(np.array_equal(stored[k],value) for k,value in dict(rgb=rgb,matte=matte,valid=v,K=K,w2c=W,bbox=bbox).items());input_sha=sha(path);source_hash=sha(source);source_matte_hash=sha(mask)
        row=dict(camera=c,frame=f,split=split,input=path.relative_to(O).as_posix(),input_sha256=input_sha,source_rgb_sha256=source_hash,source_matte_sha256=source_matte_hash,bbox=bbox.tolist());rows.append(row)
        if c in review_cams and f in review_frames:
            full.append(dict(camera=c,frame=f,bbox=bbox.tolist(),input_sha256=input_sha));x0,y0,x1,y1=map(int,bbox);height=y1-y0
            n,labels,st,_=cv2.connectedComponentsWithStats(((matte>.5)&v).astype(np.uint8),connectivity=8);ii=np.argsort(-st[1:,cv2.CC_STAT_AREA])[:2]+1
            window_basis='two_largest_source_matte_components'
            if len(ii)<2 or min(st[ii,cv2.CC_STAT_AREA])<=5000:
                # Touching silhouettes cannot identify two people. Keep this hard
                # case and use train-only reference geometry to bound contexts.
                boxes=[]
                for comp in major_components:
                    xyz=initial['points'][initial['component']==comp];q=(xyz@W[:3,:3].T+W[:3,3])@K.T;uv=q[q[:,2]>.1,:2]/q[q[:,2]>.1,2:];lo=np.maximum(np.floor(uv.min(0))-24,[0,0]);hi=np.minimum(np.ceil(uv.max(0))+24,[1920,1080]);xx,yy=lo.astype(int);ww,hh=(hi-lo).astype(int);boxes.append([xx,yy,ww,hh,ww*hh])
                st=np.array(boxes);ii=np.array([0,1]);window_basis='projected_two_major_training_initialization_components_touching_matte_fallback';fallbacks.append(dict(camera=c,frame=f,reason='source silhouettes not separable into two sufficiently large components',boxes=st[:,:4].tolist()))
            tall=ii[np.argmax(st[ii,cv2.CC_STAT_HEIGHT])];short=ii[np.argmin(st[ii,cv2.CC_STAT_HEIGHT])]
            def box(i,lo,hi,pad=64):
                x,y,w,h,_=st[i];return [max(0,int(x-pad)),max(0,int(y+lo*h)),min(1920,int(x+w+pad)),min(1080,int(y+hi*h))]
            local=[('standing_hands_object_context',box(tall,.20,.55)),('crouching_hands_object_context',box(short,.35,1.20)),('clothing_context',box(tall,.18,.72,24)),('lower_limbs_context',[x0,max(0,int(y0+.68*height)),x1,min(1080,y1+24)])]
            for name,bb in local:
                xx0,yy0,xx1,yy1=bb;assert xx1>xx0 and yy1>yy0;details.append(dict(camera=c,frame=f,name=name,bbox=bb,input_sha256=input_sha,window_basis=window_basis))
            if c in ['04','13'] and f==613:
                source_panel=Image.fromarray(rgb).crop(tuple(bbox));draw=ImageDraw.Draw(source_panel)
                for j,(name,bb) in enumerate(local):
                    xx0,yy0,xx1,yy1=bb;draw.rectangle([xx0-x0,yy0-y0,xx1-x0,yy1-y0],outline=['orange','cyan','magenta','lime'][j],width=2);draw.text((max(0,xx0-x0+3),max(0,yy0-y0+3)),name,font=font,fill=['orange','cyan','magenta','lime'][j],stroke_width=1,stroke_fill='black')
                source_panel.save(V/f'camera{c}-frame000613-source-windows.png')
        save(O/'evaluation-preparation-progress.json',dict(inputs=len(rows)))
    print('ACTOR23_EVAL_INPUTS',c,len(rows),flush=True)
assert len(rows)==264 and len(full)==18 and len(details)==72
save(O/'evaluation-protocol.json',dict(status='264native_sources_and_fixed_review_windows_before_candidate_rendering',records=rows,split_counts={s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},evaluation_clock_sha256=sha(O/'evaluation-clock-decision.json'),reserved_appearance_images_used_for_training_or_clocks=False,scope='Same site, independent actors.00/06/12appearance and geometry heldout at600..624; calibration-only outside thiswindow. Oddtimes heldout from all Gaussian loss. Source matting is uncertain and includes objects; original RGB retained.',goal_complete=False))
save(V/'protocol.json',dict(status='source_defined_windows_frozen_before_candidate_rendering',full_person=full,details=details,touching_matte_fallbacks=fallbacks,selection='Sixcameras00/04/06/07/12/13at601/613/623forfullpersons and72detail contexts. Also inspect worst aggregate regressions after full evaluation. Connected-component box fractions are contexts, not exact semantic bodypart annotations. Touching silhouettes use projected training-only reference component boxes; all such hard cases remain in evaluation.',scope='Selected independently of candidate model outputs. The images used to define detail windows never enter training, and the model protocol was already frozen.',goal_complete=False));print('ACTOR23_EVAL_PREPARATION_COMPLETE',flush=True)
