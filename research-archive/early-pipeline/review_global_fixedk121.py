"""Unwarped source patches for the 121 fixed-K diagnostic, before/after the solve."""
from pathlib import Path
import json,sys,cv2,numpy as np
from global_fixedk121 import R, ROOT, SUBJECTS, input_data, read, save, now, apply_material_gate

def crop(im,uv,size=160):
    x,y=np.rint(uv).astype(int);r=size//2;out=np.full((size,size,3),240,np.uint8)
    h,w=im.shape[:2];x0,x1=max(0,x-r),min(w,x+r);y0,y1=max(0,y-r),min(h,y+r)
    out[y0-y+r:y1-y+r,x0-x+r:x1-x+r]=im[y0:y1,x0:x1];return out

out=R/'visual-review/global-fixedk';out.mkdir(exist_ok=True)
for s in SUBJECTS:
    obs,tracks,hashes=input_data(s);rows=obs['records'];images={i:cv2.imread(r['image']) for i,r in enumerate(rows)}
    if '--static' in sys.argv:
        plan=read(R/'global-fixedk-plan.json')
        if plan['material_preflight'][s]['decision']=='stop_before_pose': continue
        apply_material_gate(s,obs,tracks,plan)
    if '--after' not in sys.argv:
        region_name='background_static_support_candidate' if '--static' in sys.argv else 'background_candidate'
        suffix='_static_support' if '--static' in sys.argv else ''
        candidates=[t for t in tracks if all(v==region_name for v in t['regions'].values())]
        chosen=[]
        # Two candidates per temporal quartile, one fit and one heldout. No geometric residual selection.
        for lo in [0,6,12,18]:
            for split in ['fit','heldout']:
                ss=[t for t in candidates if t['split']==split and any(lo<=k<lo+6 for k in t['by_frame']) and t['track_id'] not in {x['track_id'] for x in chosen}]
                if ss: chosen.append(sorted(ss,key=lambda t:(-len(t['observations']),t['track_id']))[0])
        montage=[];receipt=[]
        for t in chosen:
            keys=sorted(t['by_frame']);picks=[keys[0],keys[len(keys)//2],keys[-1]]
            montage.append(np.concatenate([crop(images[k],t['by_frame'][k]['uv']) for k in picks],axis=1))
            receipt.append(dict(track_id=t['track_id'],split=t['split'],columns=[dict(frame=rows[k]['frame'],uv=t['by_frame'][k]['uv']) for k in picks]))
        if montage: cv2.imwrite(str(out/f'{s}{suffix}_背景候选_原生三视角.png'),np.concatenate(montage,axis=0))
        save(out/f'{s}{suffix}_background_samples.json',dict(updated=now(),tracks=receipt,selection='longest track per temporal quartile per frozen split; no geometry selection',pixels='160x160 native unscaled patches, center is observation, no labels or point overlay'))
    else:
        p=ROOT/s/'global-fixedk/heldout-predictions.json'
        if not p.exists(): continue
        predictions=read(p);receipt={}
        for reg in ['background_static_support_candidate','background_excluded_material','face_roi','body_roi']:
            valid=[x for x in predictions if x['region']==reg and x['state']=='predicted']; valid.sort(key=lambda x:-x['error_px'])
            picked=[];seen=set()
            for x in valid:
                if x['track_id'] in seen: continue
                picked.append(x);seen.add(x['track_id'])
                if len(picked)==8: break
            montage=[]
            for x in picked:
                i=x['frame_index'];im=images[i]
                # Two separate native source crops around observed vs predicted pixels; no warped composite.
                montage.append(np.concatenate([crop(im,x['observed_uv']),crop(im,x['predicted_uv'])],axis=1))
            if montage: cv2.imwrite(str(out/f'{s}_{reg}_留出误差原图裁片.png'),np.concatenate(montage,axis=0))
            receipt[reg]=picked
        save(out/f'{s}_heldout_error_samples.json',dict(updated=now(),samples=receipt,columns=['original crop centered on observed feature','same original image crop centered on LOO predicted location'],selection='top8 errors with unique track IDs, for failure diagnosis only, no fitting or filtering'))
print('NATIVE_PATCHES_SAVED',str(out))
