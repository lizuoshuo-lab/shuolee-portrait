"""Read-only source-pixel raw/mask/current-mesh/GS/alpha review for a new person.

No old-subject baseline, GPU renderer, pose solver or model is loaded here.
Source-pixel reconstruction error is not independent geometric ground truth.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, joined, crop_box
from train_football_half12_sh0126 import OUT, N


def source_pixel_prediction(snapshot, row, mapping):
    image = Image.open(snapshot/(row['frame']+'_gaussian.png')).convert('RGB')
    assert image.size == tuple(row['wh'])
    x0,y0,x1,y1 = mapping['crop_xyxy']
    reduced = image.resize((x1-x0,y1-y0),Image.Resampling.BOX)
    canvas = Image.new('RGB',tuple(mapping['source']['wh']),'black')
    canvas.paste(reduced,(x0,y0))
    return canvas


def source_pixel_alpha(snapshot, row, mapping):
    path = snapshot/(row['frame']+'_cumulative-alpha.npy')
    alpha = np.load(path,allow_pickle=False)
    assert alpha.dtype == np.float32 and alpha.shape == (row['wh'][1],row['wh'][0])
    assert np.isfinite(alpha).all()
    receipt = read(snapshot/(row['frame']+'_cumulative-alpha.json'))
    assert sha(path) == receipt['alpha_npy_sha256']
    x0,y0,x1,y1 = mapping['crop_xyxy']
    reduced = Image.fromarray(alpha,mode='F').resize((x1-x0,y1-y0),Image.Resampling.BOX)
    canvas = np.zeros((mapping['source']['wh'][1],mapping['source']['wh'][0]),np.float32)
    canvas[y0:y1,x0:x1] = np.asarray(reduced)
    return canvas, receipt


def main(step):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert step in (1000,10000,18000)
    import cv2
    cv2.setNumThreads(1)
    snap = OUT/'gaussian-run'/f'step-{step:06d}'
    assert read(snap/'result.json')['step'] == step
    dest = snap/'review'
    dest.mkdir(exist_ok=False)
    archive = Path('/mnt/d/00/记录/011')
    archive.mkdir(exist_ok=True)
    assert '只要图片不要文本' in (archive.parent/'规则.md').read_text(encoding='utf-8-sig')
    rows = read(OUT/'data/training-inputs.json')['records']
    sources = read(OUT/'source-native-inputs.json')['records']
    mappings = read(OUT/'pixel-mapping.json')['records']
    with np.load(snap/'current-meshes.npz',allow_pickle=False) as z:
        mesh = dict(z)
    with np.load(OUT/'fit/fitted.npz',allow_pickle=False) as z:
        fit = dict(z)
    assert len(rows) == len(sources) == len(mappings) == N
    assert mesh['frame_ids'].astype(str).tolist() == fit['frame_ids'].astype(str).tolist() == [r['frame'] for r in rows]
    render, renderer_source = mesh_renderer()
    records = []
    started = time.monotonic()
    for i,(row,source,mapping) in enumerate(zip(rows,sources,mappings)):
        frame = row['frame']
        assert frame == source['frame'] == mapping['frame'] == mapping['source']['frame']
        assert mapping['source'] == source
        for key,digest in [('image','source_sha256'),('mask','mask_sha256'),('validity','validity_sha256')]:
            assert sha(source[key]) == source[digest]
        assert np.array_equal(source['w2c'],np.eye(4))
        raw = Image.open(source['image']).convert('RGB')
        mask = Image.open(source['mask']).convert('L')
        assert raw.size == mask.size == (2704,1520)
        valid = np.asarray(Image.open(source['validity'])) > 0
        candidate = source_pixel_prediction(snap,row,mapping)
        alpha, alpha_receipt = source_pixel_alpha(snap,row,mapping)
        alpha_display = Image.fromarray(np.rint(np.clip(alpha,0,1)*255).astype(np.uint8)).convert('RGB')
        body = crop_box(mask.getbbox(),raw.size,30)
        crops = {'body':body,'feet':crop_box([body[0],body[1]+.72*(body[3]-body[1]),body[2],body[3]],raw.size)}
        if source.get('face_box_xyxy') is not None:
            crops['face'] = crop_box(source['face_box_xyxy'],raw.size,35)
        for name,ids in [('left_hand',range(25,46)),('right_hand',range(46,67))]:
            xy = fit['observed_native_keypoints'][i,list(ids),:2]
            crops[name] = crop_box(np.r_[xy.min(0)-18,xy.max(0)+18],raw.size)
        images,metrics = [],[]
        for region,box in crops.items():
            x0,y0,x1,y1 = map(int,box)
            original = raw.crop(box)
            overlay = render(original,np.asarray(box),dict(vertices_camera=mesh['vertices_camera'][i:i+1],
                faces=mesh['faces'],K_integer_centers=np.asarray(source['K_integer_centers'])))
            prediction = candidate.crop(box)
            picture = joined([original,mask.crop(box).convert('RGB'),overlay,prediction,alpha_display.crop(box)])
            path = archive/f'高斯_足球场半圈12帧126_{step:06d}_{frame}_{region}_原图_mask_当前mesh_高斯_累计alpha.png'
            assert not path.exists()
            picture.save(path)
            images.append(dict(region=region,path=str(path),sha256=sha(path),crop_original_xyxy=list(map(int,box)),
                panel_order=['source','source_mask','current_mesh','current_gaussian','cumulative_alpha_display']))
            ma = np.asarray(mask.crop(box),dtype=np.float64)/255
            target = np.asarray(original,dtype=np.float64)/255*ma[...,None]
            known = valid[y0:y1,x0:x1]
            foreground = (ma>0)&known
            err = np.abs(np.asarray(prediction,dtype=np.float64)/255-target).mean(2)
            av = alpha[y0:y1,x0:x1][foreground]
            metrics.append(dict(region=region,known_pixels=int(known.sum()),foreground_pixels=int(foreground.sum()),
                foreground_mae=float(err[foreground].mean()) if foreground.any() else None,
                known_composite_mae=float(err[known].mean()) if known.any() else None,
                alpha_quantile_probabilities=[0.,.05,.5,.95,1.],
                alpha_foreground_quantiles=np.quantile(av,[0.,.05,.5,.95,1.]).tolist() if av.size else None))
        records.append(dict(frame=frame,images=images,metrics=metrics,
            source_image_sha256=source['source_sha256'],source_mask_sha256=source['mask_sha256'],
            candidate_png_sha256=sha(snap/(frame+'_gaussian.png')),
            alpha_measurement=alpha_receipt,baseline=None,source_pixel_metrics_are_not_geometric_truth=True))
    records.sort(key=lambda r:r['frame'])
    pages = []
    for region,max_height in [('body',430),('face',240),('left_hand',200),('right_hand',200),('feet',220)]:
        entries = [(r['frame'],x) for r in records for x in r['images'] if x['region']==region]
        for start in range(0,len(entries),4):
            batch = entries[start:start+4]
            panels = []
            for frame,entry in batch:
                pic = Image.open(entry['path']).convert('RGB')
                scale = min(1,max_height/pic.height,1800/pic.width)
                panels.append(pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.LANCZOS))
            sheet = Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),(25,25,25))
            y = 0
            for pic in panels:
                sheet.paste(pic,(0,y));y += pic.height
            path = dest/f'{region}-{start//4+1:02d}.png'
            sheet.save(path)
            pages.append(dict(region=region,path=str(path),frames=[x[0] for x in batch],sha256=sha(path)))
    save(dest/'result.json',dict(updated=now(),step=step,records=records,pages=pages,
        baseline=None,comparison_scope='New-person source reconstruction only; no old-subject baseline or independent physical accuracy claim.',
        renderer_sha256=sha(renderer_source),seconds=time.monotonic()-started,actual_visual_review=False,
        coordinates='Original source integer pixels;ROI2x RGB/alpha box-downsampled without geometric warp.',
        hands_near_pockets_unknown=True,unseen_rear_unknown=True,quality_accepted=False))
    print('FOOTBALL_TWELVE_GAUSSIAN_REVIEW_READY',step,len(records),flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
