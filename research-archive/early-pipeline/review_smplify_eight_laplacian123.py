"""Fixed native-mask ROI comparison; no pose selection or error filtering."""
import argparse, json
import numpy as np
from PIL import Image
from train_smplify_eight_laplacian123 import OUT, eight
from run_official_gaussians123 import read,save,sha,now

def main(step):
    snap=OUT/'gaussian-run'/f'step-{step:06d}'
    refs={'same8_step900':eight.OUT.parent/'body-gaussian-smplify-eight-failure-capture/gaussian-run/step-000900',
          'all59_step1000':eight.BASE/'gaussian-run/step-001000'}
    selected=set(read(OUT/'selection.json')['selected_frames'])
    rows=read(OUT/'data/training-inputs.json')['records']
    boxes={r['frame']:r['images'] for r in read(refs['all59_step1000']/'review/result.json')['records']}
    rec=[]
    for row in rows:
        frame=row['frame'];raw=np.asarray(Image.open(row['image']).convert('RGB'),dtype=np.float32)
        mask=np.asarray(Image.open(row['mask']).convert('L'))>127
        predictions={'candidate':snap/(frame+'_gaussian.png'),**{k:p/(frame+'_gaussian.png') for k,p in refs.items()}}
        errors={}
        for k,p in predictions.items():
            im=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32);assert im.shape==raw.shape==(1520,2704,3)
            errors[k]=np.abs(im-raw).mean(-1)/255
        regions={}
        for item in boxes[frame]:
            x0,y0,x1,y1=item['crop_integer_xyxy'];m=mask[y0:y1,x0:x1]
            if m.any(): regions[item['region']]={k:float(v[y0:y1,x0:x1][m].mean()) for k,v in errors.items()}
        rec.append(dict(frame=frame,split='train8' if frame in selected else 'excluded51',regions=regions))
    groups={}
    for split in ('train8','excluded51','all59'):
        members=[r for r in rec if split=='all59' or r['split']==split];groups[split]={}
        for region in ('body','face','left_hand','right_hand','feet'):
            vals=[r['regions'][region] for r in members if region in r['regions']]
            groups[split][region]={}
            for key in refs:
                change=np.asarray([v['candidate']/v[key]-1 for v in vals])
                groups[split][region][key]=dict(frames=len(vals),median_change=float(np.median(change)),
                    mean_change=float(change.mean()),improved_frames=int((change<0).sum()),
                    candidate_mae_median=float(np.median([v['candidate'] for v in vals])))
    with np.load(OUT/'fit/surface-cache.npz',allow_pickle=False) as z:cache=dict(z)
    with np.load(snap/'current-meshes.npz',allow_pickle=False) as z:mesh=dict(z)
    with np.load(snap/'binding.npz',allow_pickle=False) as z:delta=z['static_offset']
    face=float(np.abs(mesh['vertices_camera'][:,cache['protected_face_vertices']]-cache['base_vertices'][:,cache['protected_face_vertices']]).max())
    assert face==0
    lengths=np.linalg.norm(delta[cache['clothing_vertices']],axis=-1)
    sample=read(snap/'sampling.json');assert sum(sample['counts'])==step
    assert all(n==0 for r,n in zip(rows,sample['counts']) if r['frame'] not in selected)
    result=dict(updated=now(),state='fixed_ROI_review_complete',step=step,groups=groups,records=rec,
        protected_face_motion_m=face,clothing_offset_mm=dict(median=float(np.median(lengths))*1000,
            p90=float(np.quantile(lengths,.9))*1000,max=float(lengths.max())*1000),
        comparators={k:dict(path=str(p),step=900 if '900' in k else 1000,step_matched=(step==(900 if '900' in k else 1000))) for k,p in refs.items()},
        fixed_training_frames=read(OUT/'selection.json')['selected_frames'],training_count=8,excluded_count=51,
        scope='Masked native-pixel RGBMAE on fixed ROIs.59 existing fitted poses are not independent physical3D truth. Different-step comparisons are descriptive only.',
        geometry=read(OUT/'gaussian-run'/f'geometry-step-{step:06d}.json'),quality_accepted=False,
        model_promoted=False,end_to_end_15min_accepted=False)
    save(snap/'numeric-review.json',result)
    print(json.dumps(dict(step=step,groups=groups,clothing_offset_mm=result['clothing_offset_mm'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',required=True,type=int);main(p.parse_args().step)
