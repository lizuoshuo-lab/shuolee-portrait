"""Verify the same 021 correspondences via real intermediate source images."""
from datetime import datetime, timezone
from pathlib import Path
import time
import types

import cv2
import numpy as np

from fit_video21_native148 import NATIVE
from review_full_sequence123 import relocate
from screen_video21_background156 import OUT as SPARSE, longest_segment, patch_pair
from track_leafwoman_torso141 import decode
from track_native_continuity122 import step
from track_video21_background154 import OUT as TRACKS, load_gray, mask_for
from update_leafwoman129_state import atomic, read, sha


OUT = TRACKS.with_name('dense-bridge-background157')


def main():
    assert not OUT.exists()
    track_plan = read(TRACKS/'plan.json')
    track_result = read(TRACKS/'result.json')
    sparse = read(SPARSE/'result.json')
    assert track_result['plan_sha256'] == sha(TRACKS/'plan.json')
    assert sparse['plan_sha256'] == sha(SPARSE/'plan.json')
    archive = TRACKS/'tracks.npz'
    assert track_result['tracks_sha256'] == sha(archive)
    with np.load(archive, allow_pickle=False) as z:
        frames, uv, family, split = (z[k] for k in ('frames','uv','family','split'))
    native = read(NATIVE/'result.json')
    native_plan = read(NATIVE/'plan.json')
    video = relocate(native_plan['source_video'])
    intake = relocate(native['data_root'])/'data/inputs.json'
    assert native_plan['video_sha256'] == track_plan['source_video_sha256'] == sha(video)
    assert sha(intake) == track_plan['input_sha256']
    rows = {r['frame']:r for r in read(intake)['records']}
    original_indices = [rows[f]['source_frame_index'] for f in frames]
    holdouts = {rows[f]['source_frame_index'] for f in track_plan['retained_portrait_holdouts']}
    assert original_indices == track_plan['native_indices']
    assert not set(original_indices) & holdouts
    dense_indices = sorted((set(range(original_indices[0],original_indices[-1]+1,2))
                            | set(original_indices)) - holdouts)
    assert max(np.diff(dense_indices)) <= 4
    wh = track_plan['wh']
    box = (0,0,*wh)
    decoder = types.FunctionType(decode.__code__, dict(decode.__globals__, VIDEO=video))
    plan_path = OUT/'plan.json'
    atomic(plan_path,dict(track_plan,updated=datetime.now(timezone.utc).isoformat(),
        worker_sha256=sha(__file__), parent_tracks_sha256=sha(archive),
        sparse_screen_result_sha256=sha(SPARSE/'result.json'),
        appearance_worker_sha256=sha(Path(__file__).with_name('screen_video21_background156.py')),
        source_decode_sha256=sha(Path(__file__).with_name('track_leafwoman_torso141.py')),
        hypothesis='The 3 fps appearance gate breaks where viewpoint change is large; validating the same endpoints through real intermediate images can preserve identities.',
        calibration_only_source_indices=dense_indices,
        excluded_original_source_indices=sorted(holdouts),
        max_native_index_step=int(max(np.diff(dense_indices))),
        screen='Original .75px forward-backward check and 156 native patch acceptance at every dense step. Confirm the existing endpoint within 2px without supplying it to flow.',
        segment_policy='One longest contiguous run with at least 2 validated 3fps links; preserve original track families and both split roles.',
        optical_flow_coords_used_for_camera_fit=False,geometry_residual_used=False))

    cv2.setNumThreads(4)
    started = time.monotonic()
    stream = decoder(dense_indices,box,False)
    first_i, rgb = next(stream)
    assert first_i == original_indices[0]
    first = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    assert np.array_equal(first,load_gray(rows[frames[0]],wh))
    appearance = np.zeros((len(frames)-1,len(split)),bool)
    supported = np.isfinite(uv[...,0]) & (split>=0)[None]
    receipts = []
    for i,(left,right) in enumerate(zip(original_indices[:-1],original_indices[1:],strict=True)):
        ids = np.flatnonzero(supported[i]&supported[i+1])
        previous = first
        points = uv[i,ids].copy()
        safe_mask = cv2.bitwise_and(mask_for(rows[frames[i]],track_plan['excluded_subject_chair_xyxy'],wh),
                                  mask_for(rows[frames[i+1]],track_plan['excluded_subject_chair_xyxy'],wh))
        tested = len(ids)
        for j, rgb in stream:
            current = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            if len(ids):
                q,_,_,_,rejected = step(previous,current,points)
                accepted=[]
                for k in np.flatnonzero(rejected==0):
                    x,y = np.rint(q[k]).astype(int)
                    if safe_mask[y,x] and patch_pair(previous,current,points[k],q[k])[3]:
                        accepted.append(k)
                ids,points = ids[accepted],q[accepted]
            previous = current
            if j == right:
                break
        assert j == right and np.array_equal(current,load_gray(rows[frames[i+1]],wh))
        error = np.linalg.norm(points-uv[i+1,ids],axis=1)
        stable = ids[error<=2.]
        appearance[i,stable] = True
        receipts.append(dict(from_frame=frames[i],to_frame=frames[i+1],native_interval=[left,right],
            original_correspondences=tested,reached_endpoint=len(ids),verified_endpoint_count=len(stable),
            verified_fit=int((split[stable]==0).sum()),verified_heldout=int((split[stable]==1).sum())))
        first = current
        if i%8==0 or i==len(frames)-2:
            print('DENSE_BACKGROUND_LINKS',i+1,len(frames)-1,len(stable),flush=True)
    assert next(stream,None) is None

    points_out = np.full_like(uv,np.nan)
    split_out = np.full_like(split,-1)
    for j in np.flatnonzero(split>=0):
        region = longest_segment(appearance[:-1,j]&appearance[1:,j])
        if region.start!=region.stop:
            points_out[region,j] = uv[region,j]
            split_out[j] = split[j]
    present = np.isfinite(points_out[...,0])
    assert np.array_equal(points_out[present],uv[present])
    assert np.array_equal(split_out[split_out>=0],split[split_out>=0])
    tracks_path = OUT/'tracks.npz'
    np.savez_compressed(tracks_path,frames=frames,uv=points_out,family=family,split=split_out)
    pairs=[]
    for i in range(len(frames)-1):
        for j in range(i+1,min(len(frames),i+3)):
            shared=present[i]&present[j]
            pairs.append(dict(indices=[i,j],frames=[frames[i],frames[j]],
                fit=int((shared&(split_out==0)).sum()),heldout=int((shared&(split_out==1)).sum())))
    atomic(OUT/'result.json',dict(seconds=time.monotonic()-started,plan_sha256=sha(plan_path),
        tracks_sha256=sha(tracks_path),frames=len(frames),dense_frames=len(dense_indices),
        candidate_frame_grayscale_bitexact=bool(len(receipts)==len(frames)-1),
        fit_tracks=int((split_out==0).sum()),heldout_tracks=int((split_out==1).sum()),
        edge_verification=receipts,pair_support=pairs,
        no_person_pose_or_camera_solve=True,gaussian_started=False))
    print('DENSE_BACKGROUND_BRIDGE_COMPLETE',len(frames),len(dense_indices),int((split_out==0).sum()),int((split_out==1).sum()),flush=True)


if __name__ == '__main__':
    main()
