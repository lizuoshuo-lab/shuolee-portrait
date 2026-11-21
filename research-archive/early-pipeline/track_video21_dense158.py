"""Observe real intermediate 021 cameras with screened, short raw tracks."""
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import types

import cv2
import numpy as np
from scipy.spatial import cKDTree

from bridge_video21_background157 import OUT as BRIDGE
from fit_video21_native148 import NATIVE
from review_full_sequence123 import relocate
from screen_video21_background156 import patch_pair
from track_leafwoman_torso141 import decode
from track_native_continuity122 import step
from track_video21_background154 import OUT as ORIGINAL, family_split, load_gray, mask_for, seed_points
from update_leafwoman129_state import atomic, read, sha


OUT = ORIGINAL.with_name('dense-camera-tracks158')


def protect_old_split(uv, family, split, ids, old_uv, old_split):
    touched = {0:set(),1:set()}
    for current_row, old_row in ids:
        known = np.flatnonzero(np.isfinite(uv[current_row,:,0]))
        for role in (0,1):
            other = old_uv[old_row, (old_split==role)&np.isfinite(old_uv[old_row,:,0])]
            if not len(other) or not len(known):
                continue
            distances, _ = cKDTree(other).query(uv[current_row,known],distance_upper_bound=12.)
            touched[role].update(family[known[np.isfinite(distances)]].tolist())
    output = split.copy()
    mixed = touched[0]&touched[1]
    for i in np.flatnonzero(split>=0):
        group = family[i]
        if group in mixed:
            output[i] = -1
        elif group in touched[1]:
            output[i] = 1
        elif group in touched[0]:
            output[i] = 0
    assert not (set(family[output==0])&touched[1])
    assert not (set(family[output==0])&set(family[output==1]))
    return output, dict(old_fit_contact_families=len(touched[0]),old_heldout_contact_families=len(touched[1]),
                        mixed_quarantined_families=len(mixed))


def main():
    assert not OUT.exists()
    original_plan, original_result = read(ORIGINAL/'plan.json'),read(ORIGINAL/'result.json')
    bridge_plan, bridge = read(BRIDGE/'plan.json'),read(BRIDGE/'result.json')
    assert bridge['plan_sha256']==sha(BRIDGE/'plan.json')
    assert bridge['tracks_sha256']==sha(BRIDGE/'tracks.npz')
    assert bridge['candidate_frame_grayscale_bitexact']
    with np.load(ORIGINAL/'tracks.npz',allow_pickle=False) as raw:
        old_uv,old_split=raw['uv'],raw['split']
    assert original_result['tracks_sha256']==sha(ORIGINAL/'tracks.npz')==bridge_plan['parent_tracks_sha256']
    native_plan=read(NATIVE/'plan.json')
    video=relocate(native_plan['source_video'])
    intake=relocate(read(NATIVE/'result.json')['data_root'])/'data/inputs.json'
    assert sha(video)==original_plan['source_video_sha256'] and sha(intake)==original_plan['input_sha256']
    rows={r['frame']:r for r in read(intake)['records']}
    names=original_plan['frames']
    source={rows[f]['source_frame_index']:f for f in names}
    original_indices=list(source)
    blocked=set(bridge_plan['excluded_original_source_indices'])
    dense=bridge_plan['calibration_only_source_indices']
    assert not set(dense)&blocked
    camera_indices=sorted(set(original_indices)|{i for i in dense if i%4==0})
    assert not set(camera_indices)&blocked
    frames=[source.get(i,f'native_{i:06d}') for i in camera_indices]
    slots={i:j for j,i in enumerate(camera_indices)}
    same_frames=[(slots[i],j) for j,i in enumerate(original_indices)]
    wh=original_plan['wh']
    decoder=types.FunctionType(decode.__code__,dict(decode.__globals__,VIDEO=video))
    plan_path=OUT/'plan.json'
    atomic(plan_path,dict(original_plan,updated=datetime.now(timezone.utc).isoformat(),
        worker_sha256=sha(__file__), frames=frames,native_indices=camera_indices,
        source_parent_sha256=sha(ORIGINAL/'tracks.npz'), bridge_result_sha256=sha(BRIDGE/'result.json'),
        hypothesis='Actual auxiliary cameras at uniform four-frame source steps add rigid overlap near sparse-pair cuts.',
        tracking_indices=dense,additional_camera_frames=len(camera_indices)-len(source),
        heldout_native_indices=sorted(blocked),tracker_appearance='Same 154 FB and 156 template checks at each actual intermediate; independent two-hop agreement within 2 px.',
        split_policy='Co-visible 12px family separation. Prior heldout overlap stays heldout; mixed prior overlap is quarantined. Fit and heldout are separate throughout.',
        input_images_are_source_only=True,no_pose_interpolation=True,gaussian_started=False))
    cv2.setNumThreads(4)
    started=time.monotonic()
    active=np.empty(0,np.int64)
    xy=np.empty((0,2),np.float32)
    older_ids=active.copy()
    older_xy=xy.copy()
    older=previous=None
    histories, camera_receipts = [], []
    masks={}
    def get_mask(index):
        if index not in masks:
            masks[index]=mask_for(rows[source[index]],original_plan['excluded_subject_chair_xyxy'],wh)
        return masks[index]

    for native_index,rgb in decoder(dense,(0,0,*wh),False):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        bracket=bisect_left(original_indices,native_index)
        lower=original_indices[max(0,bracket-1)]
        upper=original_indices[min(bracket,len(original_indices)-1)]
        allowed=cv2.bitwise_and(get_mask(lower),get_mask(upper))
        masks={i:masks[i] for i in (lower,upper)}
        previous_ids,previous_xy=active,xy
        if len(active):
            proposed,_,_,_,rejected=step(previous,gray,xy)
            keep=[]
            for k in np.flatnonzero(rejected==0):
                x,y=np.rint(proposed[k]).astype(int)
                if allowed[y,x] and patch_pair(previous,gray,xy[k],proposed[k])[3]:
                    keep.append(k)
            active,xy=active[keep],proposed[keep]
            common,old_i,new_i=np.intersect1d(older_ids,active,return_indices=True)
            if len(common):
                direct,_,_,_,rejected=step(older,gray,older_xy[old_i])
                stable=np.ones(len(active),bool)
                stable[new_i]=(rejected==0)&(np.linalg.norm(direct-xy[new_i],axis=1)<=2.)
                active,xy=active[stable],xy[stable]
        if native_index in slots:
            index=slots[native_index]
            if native_index in source:
                assert np.array_equal(gray,load_gray(rows[source[native_index]],wh))
            seed=seed_points(gray,allowed,xy)
            new_ids=np.arange(len(histories),len(histories)+len(seed))
            histories.extend([] for _ in seed)
            active=np.r_[active,new_ids]
            xy=np.concatenate([xy,seed],axis=0)
            for k,p in zip(active,xy,strict=True):
                histories[k].append((index,p.copy()))
            camera_receipts.append(dict(frame=frames[index],source_index=native_index,
                grayscale_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),tracked=len(active)))
            if index%16==0:
                print('DENSE_CAMERA_TRACKS',index+1,len(frames),len(active),flush=True)
        older,previous=previous,gray
        older_ids,older_xy=previous_ids,previous_xy
    assert [r['frame'] for r in camera_receipts]==frames
    uv=np.full((len(frames),len(histories),2),np.nan,np.float32)
    for j,track in enumerate(histories):
        for i,p in track:
            uv[i,j]=p
    family,split=family_split(uv)
    split,protection=protect_old_split(uv,family,split,same_frames,old_uv,old_split)
    path=OUT/'tracks.npz'
    np.savez_compressed(path,frames=np.asarray(frames),uv=uv,family=family,split=split)
    seen=np.isfinite(uv[...,0])
    pairs=[]
    for i in range(len(frames)-1):
        for j in range(i+1,min(i+3,len(frames))):
            common=seen[i]&seen[j]
            pairs.append(dict(indices=[i,j],frames=[frames[i],frames[j]],fit=int((common&(split==0)).sum()),
                              heldout=int((common&(split==1)).sum())))
    atomic(OUT/'result.json',dict(seconds=time.monotonic()-started,plan_sha256=sha(plan_path),
        tracks_sha256=sha(path),frames=len(frames),fit_tracks=int((split==0).sum()),heldout_tracks=int((split==1).sum()),
        split_protection=protection,camera_frame_receipts=camera_receipts,pair_support=pairs,
        source_frames_bitexact=len(same_frames),calibrated_camera=False,gaussian_started=False))
    print('DENSE_CAMERA_TRACKS_COMPLETE',len(frames),int((split==0).sum()),int((split==1).sum()),flush=True)


if __name__=='__main__':
    main()
