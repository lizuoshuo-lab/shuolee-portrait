"""CPU evidence from saved native images and named poses, no fitting or GPU."""
import argparse, math, os, time
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, joined, crop_box

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')

def main(subject,step):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    cv2.setNumThreads(1)
    out=ROOT/subject;p=read(out/'plan.json');rows=read(out/'data/training-inputs.json')['records']
    selected=read(out/'selection.json')['selected_frames']
    snap=out/'gaussian-run'/f'step-{step:06d}' if step else out
    dest=snap/'review';dest.mkdir(exist_ok=True)
    record=Path('/mnt/d/00/记录')/p['record_id'];record.mkdir(exist_ok=True)
    with np.load(out/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    if step:
        with np.load(snap/'current-meshes.npz',allow_pickle=False) as z:mesh=dict(z)
        assert list(mesh['frame_ids'])==[r['frame'] for r in rows]
    else:
        vc=np.einsum('nij,nvj->nvi',fit['w2c'][:,:3,:3],fit['vertices_world'])+fit['w2c'][:,None,:3,3]
        assert float(np.max(abs(vc-fit['vertices_camera'])))<2e-6
        mesh=dict(vertices_camera=vc,faces=fit['faces'])
    render,source=mesh_renderer();results=[];start=time.monotonic()
    for i,row in enumerate(rows):
        if not step and row['frame'] not in selected:continue
        raw=Image.open(row['image']).convert('RGB');assert sha(row['image'])==row['source_sha256']
        mask=Image.open(row['mask']).convert('L');assert sha(row['mask'])==row['mask_sha256']
        # Fixed source-derived crop, reused exactly for mesh/Gaussian panels.
        bbox=mask.getbbox();assert bbox is not None
        body=crop_box(bbox,raw.size,30)
        face=crop_box(row['face_box_xyxy'],raw.size,35)
        bx0,by0,bx1,by1=body
        feet=crop_box([bx0,by0+.72*(by1-by0),bx1,by1],raw.size)
        obs=fit['observed_native_keypoints'][i]
        crops=dict(body=body,face=face,feet=feet)
        for name,ids in [('left_hand',range(25,46)),('right_hand',range(46,67))]:
            xy=obs[list(ids),:2];crops[name]=crop_box(np.r_[xy.min(0)-18,xy.max(0)+18],raw.size)
        gs=Image.open(snap/(row['frame']+'_gaussian.png')).convert('RGB') if step else None
        if gs:assert gs.size==raw.size
        images=[]
        for region,box in crops.items():
            original=raw.crop(box)
            overlay=render(original,np.asarray(box),dict(vertices_camera=mesh['vertices_camera'][i:i+1],faces=mesh['faces'],K_integer_centers=np.asarray(row['K_integer_centers'])))
            panels=[original,overlay]+([gs.crop(box)] if step else [])
            pic=joined(panels)
            filename=f'{"高斯" if step else "smpl-x"}_跨人物SH0_{step:06d}_{row["frame"]}_{region}_原图_当前mesh'+('_高斯' if step else '')+'.png'
            path=record/filename;pic.save(path)
            images.append(dict(region=region,path=str(path),sha256=sha(path),crop_integer_xyxy=box,resampling=False,text_added=False))
        results.append(dict(frame=row['frame'],training=row['frame'] in selected,images=images))
    results.sort(key=lambda x:x['frame']);pages=[]
    for region in ('body','face','feet'):
        count=12
        for begin in range(0,len(results),count):
            group=results[begin:begin+count];pictures=[]
            for r in group:
                item=next(j for j in r['images'] if j['region']==region);pic=Image.open(item['path']).convert('RGB')
                maxh=280 if region=='body' else 190
                scale=min(1,maxh/pic.height);pic=pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.LANCZOS)
                pictures.append(pic)
            w=max(x.width for x in pictures);h=max(x.height for x in pictures);page=Image.new('RGB',(3*w,math.ceil(len(group)/3)*h),'white')
            for j,pic in enumerate(pictures):page.paste(pic,((j%3)*w,(j//3)*h))
            path=dest/f'{region}-{begin//count+1:02d}.png';page.save(path)
            pages.append(dict(region=region,path=str(path),frames=[r['frame'] for r in group],sha256=sha(path)))
    save(dest/'result.json',dict(updated=now(),subject=subject,step=step,records=results,pages=pages,
        renderer='Existing CPU depth-sorted mesh overlay, not native raster parity',renderer_sha256=sha(source),
        full_resolution_unwarped_crops=True,contact_pages_uniformly_reduced=True,seconds=time.monotonic()-start,
        actual_visual_review=False,quality_accepted=False))
    print('CROSS_SUBJECT_REVIEW_READY',subject,step,len(results),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('subject');ap.add_argument('--step',type=int,default=0);args=ap.parse_args();main(args.subject,args.step)
