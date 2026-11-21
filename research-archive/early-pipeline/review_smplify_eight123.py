"""Eight optimized images and51 excluded images reported separately."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import train_smplify_eight123 as e
from run_official_gaussians123 import read,save,sha,now


def main():
    selected=set(read(e.OUT/'selection.json')['selected_frames'])
    replay=e.OUT.parent/'body-gaussian-smplify-eight-failure-capture'
    snap=replay/'gaussian-run/step-000900';base=e.BASE/'gaussian-run/step-001000'
    rows=read(e.OUT/'data/training-inputs.json')['records']
    boxes={r['frame']:r['images'] for r in read(base/'review/result.json')['records']}
    records=[]
    for row in rows:
        raw=np.asarray(Image.open(row['image']).convert('RGB')).astype(np.float32)
        mask=np.asarray(Image.open(row['mask']).convert('L'))>127
        a=np.asarray(Image.open(snap/(row['frame']+'_gaussian.png')).convert('RGB')).astype(np.float32)
        b=np.asarray(Image.open(base/(row['frame']+'_gaussian.png')).convert('RGB')).astype(np.float32)
        assert raw.shape==a.shape==b.shape==(1520,2704,3)
        err_a=np.abs(a-raw).mean(-1)/255;err_b=np.abs(b-raw).mean(-1)/255
        regions={}
        for item in boxes[row['frame']]:
            x0,y0,x1,y1=item['crop_integer_xyxy'];m=mask[y0:y1,x0:x1]
            if m.any():
                va=float(err_a[y0:y1,x0:x1][m].mean());vb=float(err_b[y0:y1,x0:x1][m].mean())
                regions[item['region']]=dict(eight=va,all59=vb,relative_change=va/vb-1,pixels=int(m.sum()))
        records.append(dict(frame=row['frame'],split='train8' if row['frame'] in selected else 'excluded51',regions=regions))
    groups={}
    for split in ('train8','excluded51','all59'):
        members=[r for r in records if split=='all59' or r['split']==split]
        groups[split]={}
        for key in ('body','face','left_hand','right_hand','feet'):
            values=[r['regions'][key] for r in members if key in r['regions']]
            change=np.asarray([v['relative_change'] for v in values])
            groups[split][key]=dict(frames=len(values),relative_change_median=float(np.median(change)),
                relative_change_mean=float(change.mean()),improved_frames=int((change<0).sum()),
                eight_mae_median=float(np.median([v['eight'] for v in values])),all59_mae_median=float(np.median([v['all59'] for v in values])))
    with np.load(e.OUT/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(snap/'current-meshes.npz',allow_pickle=False) as z:mesh=dict(z)
    with np.load(e.OUT/'fit/surface-cache.npz',allow_pickle=False) as z:cache=dict(z)
    assert np.array_equal(mesh['frame_ids'],fit['frame_ids'])
    face_error=float(np.abs(mesh['vertices_camera'][:,cache['protected_face_vertices']]-cache['base_vertices'][:,cache['protected_face_vertices']]).max())
    assert face_error==0
    sampling=read(snap/'sampling.json')['counts'];assert sum(sampling)==900
    assert all(n in (112,113) if r['frame'] in selected else n==0 for r,n in zip(rows,sampling))
    run=read(replay/'gaussian-run/result.json')
    result=dict(updated=now(),state='eight_frame_failure_replay_numeric_review_complete',training_count=8,excluded_count=51,groups=groups,records=records,
        snapshot_step=900,baseline_step=1000,step_matched_comparison=False,snapshot_source='One diagnostic replay to preserve the same900-step failure; original failure untouched.',
        protected_face_motion_m=face_error,training_observations_per_selected_frame=[112,113],excluded_RGB_updates=0,
        timing_seconds=run['seconds'],baseline59_timing_seconds=read(e.BASE/'gaussian-run/result.json')['seconds'],
        measurement='Original masks and fixed native ROI RGBMAE. The51 excluded images have existing SMPLify poses; not independent calibration or new-video generalization.',
        causal_change='Gaussian training image roster only; fixed8 selection was frozen before training and uses no Gaussian loss. RGB comparisons have unequal900/1000 budgets, so they cannot establish an8-versus59 causal quality advantage.',
        quality_accepted=False,end_to_end_15min_accepted=False,model_promoted=False,
        limitation='This reuses the old59-frame38.3-minute SMPLify preprocessing. It does not demonstrate an8-frame end-to-end15-minute pipeline.',
        source_hashes={str(p):sha(p) for p in [e.OUT/'selection.json',snap/'result.json',base/'result.json',snap/'current-meshes.npz']})
    save(e.OUT/'quality-review.json',result)
    print(json.dumps(dict(groups=groups,seconds=run['seconds'],face_motion=face_error)))


if __name__=='__main__':main()
