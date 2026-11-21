"""Matched-step all59 native fixed-mask comparison for the fitted-input change."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now
from train_smplify_eight_reviewedfit123 import OUT,BASE,REPLAY


def main(step):
    oldroot=BASE.parent/'body-gaussian-smplify-eight-dc' if step==1000 else BASE
    oldsnap=oldroot/'gaussian-run'/f'step-{step:06d}';newsnap=OUT/'gaussian-run'/f'step-{step:06d}'
    target=newsnap/'matched-input-metrics.json';assert not target.exists()
    assert sha(oldroot/'data/training-inputs.json')==sha(OUT/'data/training-inputs.json')
    assert sha(oldroot/'selection.json')==sha(OUT/'selection.json')
    assert read(oldsnap/'result.json')['active_sh_degree']==read(newsnap/'result.json')['active_sh_degree']==0
    rows=read(OUT/'data/training-inputs.json')['records']
    chosen=set(read(OUT/'selection.json')['selected_frames'])
    changed=set(read(REPLAY/'input-audit.json')['changed_frames'])
    boxes={r['frame']:r['images'] for r in read(BASE/'gaussian-run/step-018000/review/result.json')['records']}
    records=[]
    for row in rows:
        f=row['frame'];raw=np.asarray(Image.open(row['image']).convert('RGB'),dtype=np.float32)
        mask=np.asarray(Image.open(row['mask']).convert('L'))>127
        old=np.asarray(Image.open(oldsnap/(f+'_gaussian.png')).convert('RGB'),dtype=np.float32)
        new=np.asarray(Image.open(newsnap/(f+'_gaussian.png')).convert('RGB'),dtype=np.float32)
        assert old.shape==new.shape==raw.shape==(1520,2704,3)
        errors={k:np.abs(v-raw).mean(-1)/255 for k,v in [('early_fit',old),('reviewed_fit',new)]}
        regions={}
        for item in boxes[f]:
            x0,y0,x1,y1=item['crop_integer_xyxy'];m=mask[y0:y1,x0:x1]
            if m.any():regions[item['region']]={k:float(e[y0:y1,x0:x1][m].mean()) for k,e in errors.items()}
        records.append(dict(frame=f,train8=f in chosen,pose_changed=f in changed,regions=regions))
    groups={}
    for group,rs in [('train8',[r for r in records if r['train8']]),('excluded51',[r for r in records if not r['train8']]),
                     ('pose_changed12',[r for r in records if r['pose_changed']]),('pose_unchanged47',[r for r in records if not r['pose_changed']]),('all59',records)]:
        groups[group]={}
        for region in ('body','face','left_hand','right_hand','feet'):
            values=[r['regions'][region] for r in rs if region in r['regions']]
            old=np.array([v['early_fit'] for v in values]);new=np.array([v['reviewed_fit'] for v in values]);relative=new/old-1
            groups[group][region]=dict(frames=len(values),improved_frames=int((relative<0).sum()),
                early_fit_mae_mean=float(old.mean()),reviewed_fit_mae_mean=float(new.mean()),
                early_fit_mae_median=float(np.median(old)),reviewed_fit_mae_median=float(np.median(new)),
                relative_change_mean=float(relative.mean()),relative_change_median=float(np.median(relative)))
    save(target,dict(updated=now(),step=step,records=records,groups=groups,
        baseline=str(oldsnap),candidate=str(newsnap),early_fit_points=read(oldsnap/'result.json')['points'],
        reviewed_fit_points=read(newsnap/'result.json')['points'],both_SH0=True,
        old_geometry=read(oldroot/'gaussian-run'/f'geometry-step-{step:06d}.json'),
        new_geometry=read(OUT/'gaussian-run'/f'geometry-step-{step:06d}.json'),
        same_fixed_ROIs=True,same_masks=True,frame_filtering=False,quality_accepted=False,
        scope='Same-step independently trained SH0 candidates; only fitted-input source differs. Native RGB scores are not independent physical geometry truth; excluded51 were previously used by SMPLify-X.'))
    print('MATCHED_FIT_SOURCE_METRICS',groups['all59'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);main(p.parse_args().step)
