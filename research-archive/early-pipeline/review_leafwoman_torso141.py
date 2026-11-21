"""Freeze the visual identity result of the native 022 material tracks."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from track_leafwoman_torso141 import CONTACT, OUT, atomic, read
from review_full_sequence123 import joined, relocate


def review():
    result=read(OUT/'result.json'); plan=read(OUT/'plan.json')
    path=OUT/'visual-review.json'
    assert not path.exists() and result['plan_sha256']==sha(OUT/'plan.json')
    pixels=OUT/'tracked-pixels.npz'
    assert result['tracks_sha256']==sha(pixels) and sha(CONTACT)==plan['contacts_sha256']
    for page in result['pages']: assert sha(page['path'])==page['sha256']
    source={r['frame']:r for r in read(CONTACT)['records']}
    numbers=[r['native_index'] for r in result['rows']]
    # The actual crop sheets were inspected. Deliberately only count these
    # clearly visible button regions, not correspondence of an exact thread.
    visible=[(64,304),(48,232),(24,264)]
    definitely_wrong={0:[0,1,2],288:[1],296:[1],304:[1]}
    seed_ids=[r['id'] for r in plan['seeds']]
    records=[]
    with np.load(pixels,allow_pickle=False) as f:
        xy=f['uv']; fb=f['forward_backward']; status=f['status']
        assert (status[numbers]==0).all() and (status[plan['skipped_source_indices']]==-1).all()
        common=[i for i in numbers if all(lo<=i<=hi for lo,hi in visible)]
        b=xy[common,:3].astype(float)
        major=b[:,2]-b[:,0]; middle=b[:,1]-b[:,0]
        line=np.linalg.norm(major,axis=1)
        orth=np.abs(major[:,0]*middle[:,1]-major[:,1]*middle[:,0])/line
        sig=np.linalg.svd(b-b.mean(1,keepdims=True),compute_uv=False)
        for i,row in zip(numbers,result['rows'],strict=True):
            frame=row['frame']
            regions=[seed_ids[j] for j,(lo,hi) in enumerate(visible) if lo<=i<=hi]
            records.append(dict(frame=frame,native_index=i,visible_button_region_candidates=regions,
                numeric_fb_passes=seed_ids,physical_point_established_ids=[],
                visually_wrong_or_occluded_button_ids=[seed_ids[j] for j in definitely_wrong.get(i,[])]))
        images=[]
        for frame in ('frame_000001','frame_000020','frame_000039'):
            row=source[frame]; i=row['source_frame_index']; crop=row['paths']['upper']
            original=relocate(crop['path']);assert sha(original)==crop['sha256']
            ox,oy,*_=crop['crop_original_xyxy']
            with Image.open(original) as im: raw=im.convert('RGB')
            q=xy[i]-[ox,oy]
            x0,y0=np.maximum(np.floor(q.min(0)-38).astype(int),[0,0])
            x1,y1=np.minimum(np.ceil(q.max(0)+38).astype(int),raw.size)
            assert x1>x0 and y1>y0
            raw=raw.crop((x0,y0,x1,y1)); overlay=raw.copy(); draw=ImageDraw.Draw(overlay)
            for j,(x,y) in enumerate(q-[x0,y0]):
                if j in definitely_wrong.get(i,[]): color='#ff5065'
                elif j<3 and visible[j][0]<=i<=visible[j][1]: color='#15ecf6'
                else: color='#eac668'
                draw.ellipse((x-5,y-5,x+5,y+5),outline='#141419',width=3)
                draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
            dest=OUT/(frame+'-cardigan-native.png');joined([raw,overlay]).save(dest)
            images.append(dict(frame=frame,path=str(dest),sha256=sha(dest),
                crop_native_xyxy=[int(ox+x0),int(oy+y0),int(ox+x1),int(oy+y1)]))
        metrics=dict(roster_views=len(numbers),co_visible_button_region_views=len(common),
            candidate_button_endpoint_span_median_px=float(np.median(line)),
            candidate_middle_button_distance_to_endpoint_line_median_px=float(np.median(orth)),
            candidate_middle_button_distance_to_endpoint_line_max_px=float(np.max(orth)),
            candidate_centered_position_singular_ratio_median=float(np.median(sig[:,1]/sig[:,0])),
            forward_backward_p50_px=float(np.nanmedian(fb)),forward_backward_max_px=float(np.nanmax(fb)))
    atomic(path,dict(track_result_sha256=sha(OUT/'result.json'),roster_visual_review_all24=True,
        review_scope='Three atlas pages inspected. Tracks remaining on a visible button region are candidate 2D trajectories. '
                     'A stitch-level correspondence, visibility across all intermediate frames, and static rigidity were not established. '
                     'Slit tips, stripe/knit and flexible material controls do not furnish spatially distinct verified point families.',
        obvious_false_survivals='First 3 roster top-button estimates and last 3 roster middle-button estimates. Optical flow succeeds numerically even on the occluder.',
        metrics=metrics,rows=records,images=images,physical_independent_3D_support_admitted=False,
        relative_pose_solved=False,new_fitting_started=False,gaussian_started=False,quality_accepted=False))
    print('TORSO141_REVIEW',metrics,flush=True)


if __name__=='__main__':
    review()
