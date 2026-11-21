"""Same-step frozen-mask RGB comparison; no fitting or frame filtering."""
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from train_smplify_eight_offsetlr123 import OUT

BASE = OUT.parent/'body-gaussian-smplify-eight-dc-reviewedfit'


def main():
    oldsnap = BASE/'gaussian-run/step-001000'
    newsnap = OUT/'gaussian-run/step-001000'
    target = newsnap/'same-step-metrics.json'
    assert not target.exists()
    for name in ('data/training-inputs.json','selection.json','fit/fitted.npz','fit/surface-cache.npz'):
        assert sha(BASE/name) == sha(OUT/name), name
    assert read(oldsnap/'result.json')['active_sh_degree'] == read(newsnap/'result.json')['active_sh_degree'] == 0
    rows = read(OUT/'data/training-inputs.json')['records']
    selected = set(read(OUT/'selection.json')['selected_frames'])
    boxes = {r['frame']:r['images'] for r in read(oldsnap/'review/result.json')['records']}
    records = []
    for row in rows:
        frame = row['frame']
        raw = np.asarray(Image.open(row['image']).convert('RGB'), dtype=np.float32)
        mask = np.asarray(Image.open(row['mask']).convert('L')) > 127
        old = np.asarray(Image.open(oldsnap/(frame+'_gaussian.png')).convert('RGB'), dtype=np.float32)
        new = np.asarray(Image.open(newsnap/(frame+'_gaussian.png')).convert('RGB'), dtype=np.float32)
        assert old.shape == new.shape == raw.shape == (1520,2704,3)
        errors = {k:np.abs(v-raw).mean(-1)/255 for k,v in [('lr1e6',old),('lr1e5',new)]}
        regions = {}
        for item in boxes[frame]:
            x0,y0,x1,y1 = item['crop_integer_xyxy']; m = mask[y0:y1,x0:x1]
            if m.any():
                regions[item['region']] = {k:float(e[y0:y1,x0:x1][m].mean()) for k,e in errors.items()}
        records.append(dict(frame=frame,train8=frame in selected,regions=regions))
    groups = {}
    for group,rs in [('train8',[r for r in records if r['train8']]),('excluded51',[r for r in records if not r['train8']]),('all59',records)]:
        groups[group] = {}
        for region in ('body','face','left_hand','right_hand','feet'):
            values = [r['regions'][region] for r in rs if region in r['regions']]
            old = np.array([v['lr1e6'] for v in values]); new = np.array([v['lr1e5'] for v in values])
            groups[group][region] = dict(frames=len(values),improved=int((new<old).sum()),
                baseline_mean=float(old.mean()),candidate_mean=float(new.mean()),
                relative_median=float(np.median(new/old-1)))
    save(target,dict(updated=now(),step=1000,records=records,groups=groups,
        baseline=str(oldsnap),candidate=str(newsnap),both_SH0=True,frame_filtering=False,
        baseline_geometry=read(BASE/'gaussian-run/geometry-step-001000.json'),
        candidate_geometry=read(OUT/'gaussian-run/geometry-step-001000.json'),
        baseline_offset=read(oldsnap/'offset-update.json'),candidate_offset=read(newsnap/'offset-update.json'),
        quality_accepted=False,scope='Only offset LR changed. Same native masks and fixed ROIs. 51 excluded frames were previously used by SMPLify-X; not independent full-pipeline truth.'))
    print('OFFSET_LR_SAME_STEP_METRICS',groups,flush=True)


if __name__ == '__main__':
    main()
