"""Compare the shared 500-update checkpoint window after allocator-only restart."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]/'outputs/dynamic-human-street-joint-009'


def main():
    original=ROOT/'baseline'
    bounded=ROOT/'baseline-bounded'
    for filename in ('config.yaml','person-rois.json'):
        assert (original/filename).read_bytes()==(bounded/filename).read_bytes(),filename
    old=json.loads((original/'initialization.json').read_text())
    new=json.loads((bounded/'initialization.json').read_text())
    for key in ('gaussians','instances','lidar_points','roi_count','main_person_roi_count'):
        assert old[key]==new[key],key
    def rows(directory):
        lines=(directory/'training.jsonl').read_text().splitlines()
        return {r['step']:r for line in lines if line.strip()
                for r in [json.loads(line)] if r['step']<=450}
    a,b=rows(original),rows(bounded)
    assert set(a)==set(b)==set(range(0,500,50))
    assert all(a[k]['image_id']==b[k]['image_id'] for k in a)
    comparisons=[]
    old_dir=original/'evaluation/updates-00500'
    new_dir=bounded/'evaluation/updates-00500'
    assert (old_dir/'metrics.json').exists() and (new_dir/'metrics.json').exists()
    for old_path in sorted(old_dir.glob('*-prediction.png')):
        x=np.asarray(Image.open(old_path),dtype=np.int16)
        y=np.asarray(Image.open(new_dir/old_path.name),dtype=np.int16)
        assert x.shape==y.shape
        difference=np.abs(x-y)
        comparisons.append(dict(image=old_path.name,maximum_channel_difference_8bit=int(difference.max()),
            mean_channel_difference_8bit=float(difference.mean()),
            changed_pixel_fraction=float(np.mean(np.any(difference!=0,axis=-1)))))
    assert len(comparisons)==3
    metrics=[]
    before=json.loads((old_dir/'metrics.json').read_text())['views']
    after=json.loads((new_dir/'metrics.json').read_text())['views']
    assert len(before)==len(after)==3
    for x,y in zip(before,after):
        assert (x['frame'],x['camera'])==(y['frame'],y['camera'])
        metrics.append(dict(frame=x['frame'],camera=x['camera'],psnr_change_db=y['psnr_db']-x['psnr_db'],
            ssim_change=y['ssim']-x['ssim']))
    report=dict(status='same_configuration_but_500_update_renders_not_pixelwise_equivalent',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        same_configuration=True,same_initialization_counts_and_instance_maps=True,
        same_evaluation_regions=True,same_logged_training_views=True,
        pixelwise_equivalent=all(x['maximum_channel_difference_8bit']==0 for x in comparisons),
        image_comparisons=comparisons,metric_comparisons=metrics,
        interpretation='Same configuration and logged sampled views, but rendered pixels differ. Treat restart as an independent run, not exact continuation; no quality improvement attributed to allocator changes. Cause of numerical differences not isolated.')
    (ROOT/'allocator-restart-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
