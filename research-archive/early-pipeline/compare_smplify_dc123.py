"""Frozen native ROIs and all59 frames, matched training-step SH3/DC comparison."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData
from run_official_gaussians123 import read,save,sha,now
from train_smplify_eight_dc18000_123 import OUT
BASE=OUT.parent/'body-gaussian-smplify-eight-laplacian'

def main(step):
    snap=OUT/'gaussian-run'/f'step-{step:06d}';base=BASE/'gaussian-run'/f'step-{step:06d}'
    assert not (snap/'matched-metrics.json').exists()
    assert sha(OUT/'data/training-inputs.json')==sha(BASE/'data/training-inputs.json')
    assert sha(OUT/'selection.json')==sha(BASE/'selection.json')
    inputs=read(OUT/'data/training-inputs.json')['records'];selected=set(read(OUT/'selection.json')['selected_frames'])
    boxes={r['frame']:r['images'] for r in read(base/'review/result.json')['records']}
    records=[]
    for row in inputs:
        f=row['frame'];raw=np.asarray(Image.open(row['image']).convert('RGB'),dtype=np.float32)
        mask=np.asarray(Image.open(row['mask']).convert('L'))>127
        ref=np.asarray(Image.open(base/(f+'_gaussian.png')).convert('RGB'),dtype=np.float32)
        new=np.asarray(Image.open(snap/(f+'_gaussian.png')).convert('RGB'),dtype=np.float32)
        assert raw.shape==ref.shape==new.shape==(1520,2704,3)
        err={'SH3':np.abs(raw-ref).mean(-1)/255,'DC':np.abs(raw-new).mean(-1)/255}
        regions={}
        for box in boxes[f]:
            x0,y0,x1,y1=box['crop_integer_xyxy'];m=mask[y0:y1,x0:x1]
            if m.any():regions[box['region']]={k:float(v[y0:y1,x0:x1][m].mean()) for k,v in err.items()}
        records.append(dict(frame=f,split='train8' if f in selected else 'excluded51',regions=regions))
    groups={}
    for split in ('train8','excluded51','all59'):
        rows=[r for r in records if split=='all59' or r['split']==split];groups[split]={}
        for region in ('body','face','left_hand','right_hand','feet'):
            vals=[r['regions'][region] for r in rows if region in r['regions']]
            changes=np.array([x['DC']/x['SH3']-1 for x in vals])
            groups[split][region]=dict(frames=len(vals),relative_mae_change_median=float(np.median(changes)),
                relative_mae_change_mean=float(changes.mean()),improved_frames=int((changes<0).sum()),
                sh3_mae_median=float(np.median([x['SH3'] for x in vals])),dc_mae_median=float(np.median([x['DC'] for x in vals])))
    offsets={};gaussians={}
    with np.load(OUT/'fit/surface-cache.npz',allow_pickle=False) as z:clothing=z['clothing_vertices']
    for label,p in [('SH3',base),('DC',snap)]:
        with np.load(p/'binding.npz',allow_pickle=False) as z:delta=z['static_offset']
        length=np.linalg.norm(delta[clothing],axis=-1)*1000
        offsets[label]=dict(median_mm=float(np.median(length)),p90_mm=float(np.quantile(length,.9)),max_mm=float(length.max()))
        vertex=PlyData.read(str(p/'world.ply'))['vertex'].data
        scales=np.exp(np.column_stack([vertex[f'scale_{i}'] for i in range(3)]))
        ratio=scales.max(1)/scales.min(1)
        higher=[n for n in vertex.dtype.names if n.startswith('f_rest_')]
        gaussians[label]=dict(points=len(vertex),high_order_fields=len(higher),
            axis_ratio_p50=float(np.quantile(ratio,.5)),axis_ratio_p90=float(np.quantile(ratio,.9)),
            axis_ratio_p99=float(np.quantile(ratio,.99)),axis_ratio_max=float(ratio.max()),
            max_axis_median_mm=float(np.median(scales.max(1)))*1000,
            max_axis_p99_mm=float(np.quantile(scales.max(1),.99))*1000,
            all_geometry_fields_finite=bool(np.isfinite(scales).all()))
    assert gaussians['DC']['high_order_fields']==0
    save(snap/'matched-metrics.json',dict(updated=now(),step=step,records=records,groups=groups,offsets=offsets,gaussians=gaussians,
        native_pixel_shape=[2704,1520],same_fixed_masks=True,same_fixed_ROIs=True,filtered_by_error=False,
        scope='All59 existing poses; masked native RGB MAE, not independent geometry truth. Lower error on8 training views does not prove novel-view quality.',
        baseline_geometry=read(BASE/'gaussian-run'/f'geometry-step-{step:06d}.json'),
        candidate_geometry=read(OUT/'gaussian-run'/f'geometry-step-{step:06d}.json'),quality_accepted=False))
    print({k:{r:round(v['relative_mae_change_median']*100,3) for r,v in x.items()} for k,x in groups.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,default=18000);main(p.parse_args().step)
