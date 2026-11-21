"""All six frozen indoor views, compare ROI2x to baseline at original pixels."""
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, joined, crop_box

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
Z=ROOT/'indoor-person-zoom2x124'
B=ROOT/'indoor-selected-sh0-pipeline124'

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    cv2.setNumThreads(1)
    result=read(Z/'gaussian-run/result.json')
    assert result['state']=='bounded_control_completed' and result['snapshot_step']==1000
    snap=Z/'gaussian-run/step-001000';dest=snap/'review';dest.mkdir(exist_ok=False)
    record=Path('/mnt/d/00/记录/008');record.mkdir(exist_ok=True)
    rows=read(Z/'data/training-inputs.json')['records']
    sources=read(Z/'source-native-inputs.json')['records']
    maps=read(Z/'pixel-mapping.json')['records']
    with np.load(snap/'current-meshes.npz',allow_pickle=False) as z:mesh=dict(z)
    with np.load(Z/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    assert list(mesh['frame_ids'])==[r['frame'] for r in rows]
    assert len(rows)==len(sources)==len(maps)==6
    render,renderer_source=mesh_renderer();records=[];start=time.monotonic()
    for i,(row,source,mapping) in enumerate(zip(rows,sources,maps)):
        frame=row['frame'];assert frame==source['frame']==mapping['frame']
        assert sha(source['image'])==source['source_sha256'] and sha(source['mask'])==source['mask_sha256']
        assert sha(source['validity'])==source['validity_sha256']
        raw=Image.open(source['image']).convert('RGB')
        mask=Image.open(source['mask']).convert('L');valid=Image.open(source['validity']).convert('L')
        baseline=Image.open(B/'gaussian-run/step-001000'/(frame+'_gaussian.png')).convert('RGB')
        enlarged=Image.open(snap/(frame+'_gaussian.png')).convert('RGB')
        assert enlarged.size==tuple(row['wh']) and baseline.size==raw.size
        roi=tuple(mapping['crop_xyxy']);x0,y0,x1,y1=roi
        # Box averaging explicitly reconstructs the original-pixel sampling scale.
        candidate=enlarged.resize((x1-x0,y1-y0),Image.Resampling.BOX)
        candidate.save(dest/(frame+'-candidate-original-pixel-roi.png'))
        bb=mask.getbbox();assert bb is not None
        body=crop_box(bb,raw.size,30)
        face=crop_box(source['face_box_xyxy'],raw.size,35)
        crops={'body':body,'face':face}
        observed=fit['observed_native_keypoints'][i]
        for name,ids in [('left_hand',range(25,46)),('right_hand',range(46,67))]:
            xy=observed[list(ids),:2]
            crops[name]=crop_box(np.r_[xy.min(0)-18,xy.max(0)+18],raw.size)
        images=[];metrics=[]
        full_mask=np.asarray(mask,dtype=np.float64)/255
        known=np.asarray(valid)>0
        for region,box in crops.items():
            bx0,by0,bx1,by1=box
            box=(max(x0,bx0),max(y0,by0),min(x1,bx1),min(y1,by1))
            bx0,by0,bx1,by1=map(int,box)
            if bx1<=bx0 or by1<=by0:continue
            box=(bx0,by0,bx1,by1)
            original=raw.crop(box)
            overlay=render(original,np.asarray(box),dict(vertices_camera=mesh['vertices_camera'][i:i+1],
                faces=mesh['faces'],K_integer_centers=np.asarray(source['K_integer_centers'])))
            old=baseline.crop(box);new=candidate.crop((bx0-x0,by0-y0,bx1-x0,by1-y0))
            picture=joined([original,overlay,old,new])
            name=f'高斯_室内人物放大2倍124_001000_{frame}_{region}_原图_当前mesh_原图训练_放大训练.png'
            path=record/name;assert not path.exists();picture.save(path)
            images.append(dict(region=region,path=str(path),sha256=sha(path),crop_original_xyxy=list(box),
                panel_order=['source','current_mesh','baseline1000','crop2x1000_downsampled'],text_added=False,
                native_coordinate_comparison=True,candidate_resampling='2x render BOX averaged to original pixel size'))
            ma=full_mask[by0:by1,bx0:bx1];va=known[by0:by1,bx0:bx1]
            gt=np.asarray(original,dtype=np.float64)/255
            target=gt*ma[...,None]
            fg=(ma>0)&va
            entry={'region':region,'known_foreground_pixels':int(fg.sum()),'known_pixels':int(va.sum())}
            for key,img in [('baseline',old),('zoom2x',new)]:
                error=np.abs(np.asarray(img,dtype=np.float64)/255-target).mean(axis=2)
                entry[key+'_foreground_mae']=float(error[fg].mean()) if fg.any() else None
                entry[key+'_known_composite_mae']=float(error[va].mean()) if va.any() else None
            metrics.append(entry)
        records.append(dict(frame=frame,images=images,metrics=metrics,source_sha256=source['source_sha256'],
            candidate_png_sha256=sha(snap/(frame+'_gaussian.png')),baseline_png_sha256=sha(B/'gaussian-run/step-001000'/(frame+'_gaussian.png'))))
    records.sort(key=lambda r:r['frame']);pages=[]
    for region,maxh in [('body',410),('face',260),('left_hand',220),('right_hand',220)]:
        pictures=[];frames=[]
        for r in records:
            entries=[x for x in r['images'] if x['region']==region]
            if not entries:continue
            pic=Image.open(entries[0]['path']).convert('RGB')
            scale=min(1,maxh/pic.height,1400/pic.width)
            pic=pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.LANCZOS)
            pictures.append(pic);frames.append(r['frame'])
        width=max(x.width for x in pictures);height=sum(x.height for x in pictures)
        page=Image.new('RGB',(width,height),(30,30,30));y=0
        for pic in pictures:page.paste(pic,(0,y));y+=pic.height
        path=dest/(region+'-all6.png');page.save(path)
        pages.append(dict(region=region,path=str(path),sha256=sha(path),frames=frames))
    save(dest/'result.json',dict(updated=now(),subject=Z.name,step=1000,records=records,pages=pages,
        renderer='Existing CPU depth-sorted mesh overlay; no fitting or mesh editing',renderer_sha256=sha(renderer_source),
        evaluation_coordinates='Original video integer pixel coordinates; same original source crop for four panels',
        no_warp=True,source_and_baseline_resampled=False,zoom2x_area_downsampled_for_fair_comparison=True,
        occluded_and_out_of_frame_quality_excluded=True,feet_acceptance_omitted=True,
        metrics_scope='Visible known pixels; same mask and original RGB. MAE is not geometry or novel-view quality proof.',
        seconds=time.monotonic()-start,actual_visual_review=False,quality_accepted=False))
    print('PERSON_ZOOM_REVIEW_READY',len(records),flush=True)

if __name__=='__main__':main()
