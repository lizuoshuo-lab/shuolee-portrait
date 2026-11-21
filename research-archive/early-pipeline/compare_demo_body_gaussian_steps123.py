"""Read-only matched-frame photometric comparison, including the observed arm."""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from run_official_gaussians123 import read,save,sha,now

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-upstream-offset')
OBS=Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')


def main(before,after,new_root=None):
    new_root=Path(new_root) if new_root else ROOT
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and (before<after or new_root!=ROOT)
    assert sha(ROOT/'data/training-inputs.json')==sha(new_root/'data/training-inputs.json')
    old=ROOT/'gaussian-run'/f'step-{before:06d}';new=new_root/'gaussian-run'/f'step-{after:06d}'
    for folder in (old,new):
        assert read(folder/'result.json')['world_ply_sha256']==sha(folder/'world.ply')
    rows=read(ROOT/'data/training-inputs.json')['records']
    crops={r['frame']:r['images'] for r in read(old/'review/result.json')['records']}
    with np.load(OBS,allow_pickle=False) as z:
        observation={str(f):p for f,p in zip(z['frame_ids'],z['keypoints'])}
    results=[]
    for row in rows:
        f=row['frame'];assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        target=np.array(Image.open(row['image']).convert('RGB'),dtype=np.float32)/255
        mask=np.array(Image.open(row['mask']).convert('L'))>127
        images=[np.array(Image.open(folder/(f+'_gaussian.png')).convert('RGB'),dtype=np.float32)/255 for folder in (old,new)]
        assert all(im.shape==target.shape for im in images)
        error=[np.abs(im-target).mean(axis=2) for im in images]
        regions={}
        for item in crops[f]:
            x0,y0,x1,y1=item['crop_integer_xyxy']; valid=mask[y0:y1,x0:x1]
            regions[item['region']]=dict(mask_pixels=int(valid.sum()),before=float(error[0][y0:y1,x0:x1][valid].mean()) if valid.any() else None,after=float(error[1][y0:y1,x0:x1][valid].mean()) if valid.any() else None)
        # COCO133 body order: left shoulder5, elbow7, wrist9. Geometry is measured
        # on native observed pixels, not on the fitted (possibly wrong) skeleton.
        points=observation[f][[5,7,9]];assert np.isfinite(points).all()
        arm=Image.new('L',(target.shape[1],target.shape[0]),0)
        width=max(6,round(.03*(row['initial_box_xyxy'][3]-row['initial_box_xyxy'][1])))
        ImageDraw.Draw(arm).line([tuple(p) for p in points],fill=255,width=width,joint='curve')
        valid=(np.array(arm)>0)&mask
        regions['observed_left_arm']=dict(mask_pixels=int(valid.sum()),before=float(error[0][valid].mean()) if valid.any() else None,after=float(error[1][valid].mean()) if valid.any() else None)
        results.append(dict(frame=f,regions=regions))
    summary={}
    for region in results[0]['regions']:
        values=np.array([[r['regions'][region]['before'],r['regions'][region]['after']] for r in results],dtype=float)
        valid=np.isfinite(values).all(axis=1);v=values[valid]
        summary[region]=dict(valid_frames=int(valid.sum()),before_median=float(np.median(v[:,0])),after_median=float(np.median(v[:,1])),improved_frames=int((v[:,1]<v[:,0]).sum()),worsened_frames=int((v[:,1]>v[:,0]).sum()))
    out=dict(updated=now(),before_step=before,after_step=after,before_run=str(ROOT),after_run=str(new_root),frames=len(rows),summary=summary,records=sorted(results,key=lambda r:r['frame']),observations_sha256=sha(OBS),inputs_sha256=sha(ROOT/'data/training-inputs.json'),script_sha256=sha(__file__),metrics='Native RGB MAE in target-mask interior, normalized0..1; arm uses native COCO shoulder-elbow-wrist polyline at3%person-height width.',not_silhouette_or_independent_geometric_truth=True,quality_accepted=False)
    save(new/'matched-step-comparison.json',out)
    print({k:out[k] for k in ('before_step','after_step','frames','summary')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=int,required=True);p.add_argument('--after',type=int,required=True)
    p.add_argument('--new-root')
    args=p.parse_args();main(args.before,args.after,args.new_root)
