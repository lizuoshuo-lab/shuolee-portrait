"""CPU-only same-frame native comparisons for the bound whole-body run."""
import argparse
import math
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, joined
from train_demo_body_upstream123 import DEMO


def main(step, run_name='body-gaussian-upstream-offset', label='全身原版共享偏移'):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import cv2
    cv2.setNumThreads(1)
    assert Path(run_name).name == run_name and run_name.startswith('body-gaussian-')
    root=DEMO/run_name; snap=root/'gaussian-run'/f'step-{step:06d}'
    receipt=read(snap/'result.json'); assert receipt['step']==step
    assert receipt['world_ply_sha256']==sha(snap/'world.ply')
    out=snap/'review'; out.mkdir(exist_ok=True)
    record=Path('/mnt/d/00/记录/demo'); record.mkdir(exist_ok=True)
    rows=read(root/'data/training-inputs.json')['records']
    boxes=read(DEMO/'smplify-vhap-pose-prior-long/comparison-epoch120/result.json')['per_frame']
    boxes={r['frame_id']:r['images'] for r in boxes}
    with np.load(snap/'current-meshes.npz',allow_pickle=False) as z: meshes=dict(z)
    assert list(meshes['frame_ids'])==[r['frame'] for r in rows]
    render,renderer_source=mesh_renderer(); records=[]; started=time.monotonic()
    for i,row in enumerate(rows):
        frame=row['frame']; raw=Image.open(row['image']).convert('RGB')
        assert raw.size==(2704,1520) and sha(row['image'])==row['source_sha256']
        gs=Image.open(snap/(frame+'_gaussian.png')).convert('RGB'); assert gs.size==raw.size
        mesh=render(raw,np.array([0,0,*raw.size]),dict(vertices_camera=meshes['vertices_camera'][i:i+1],
            faces=meshes['faces'],K_integer_centers=np.asarray(row['K_integer_centers'])))
        images=[]
        for region in ('body','face','left_hand','right_hand','feet'):
            item=next(x for x in boxes[frame] if x['region']==region);box=item['crop_integer_xyxy']
            image=joined([p.crop(box) for p in (raw,mesh,gs)])
            filename=f'高斯_{label}_{step:06d}_{frame}_{region}_原图_当前mesh_高斯.png'
            path=record/filename;image.save(path)
            images.append(dict(region=region,path=str(path),sha256=sha(path),crop_integer_xyxy=box,
                panels=['source','current_trained_mesh','Gaussian'],resampling=False,text_added=False))
        records.append(dict(frame=frame,images=images))
    records.sort(key=lambda r:r['frame'])
    pages=[]
    for region in ('body','face','left_hand','right_hand','feet'):
        count=6 if region=='body' else 12
        for start in range(0,59,count):
            group=records[start:start+count];pictures=[]
            for row in group:
                item=next(x for x in row['images'] if x['region']==region)
                image=Image.open(item['path']).convert('RGB')
                if region=='body':
                    scale=min(1.,360/image.height);image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
                pictures.append(image)
            w=max(p.width for p in pictures);h=max(p.height for p in pictures)
            page=Image.new('RGB',(2*w,math.ceil(len(pictures)/2)*h),'white')
            for j,picture in enumerate(pictures):page.paste(picture,((j%2)*w,(j//2)*h))
            name=f'{region}-{start//count+1:02d}.png';page.save(out/name)
            pages.append(dict(region=region,path=str(out/name),sha256=sha(out/name),frames=[r['frame'] for r in group]))
    save(out/'result.json',dict(updated=now(),frames=59,step=step,records=records,pages=pages,
        renderer_source=str(renderer_source),renderer_sha256=sha(renderer_source),seconds=time.monotonic()-started,
        world_ply_sha256=receipt['world_ply_sha256'],actual_visual_review=False,quality_accepted=False))
    print('BODY_GAUSSIAN_REVIEW_IMAGES_READY',step,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--run-name',default='body-gaussian-upstream-offset')
    parser.add_argument('--label',default='全身原版共享偏移')
    args=parser.parse_args()
    main(args.step,args.run_name,args.label)
