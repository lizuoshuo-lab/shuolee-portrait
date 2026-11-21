"""All16 source-pixel panels; compare common six frames without mixing poses."""
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import mesh_renderer, joined, crop_box
from train_indoor_circle16_sh0124 import OUT, BASELINE


def original_pixel_render(root, step, row, mapping):
    image = Image.open(root/'gaussian-run'/f'step-{step:06d}'/(row['frame']+'_gaussian.png')).convert('RGB')
    assert image.size == tuple(row['wh'])
    x0,y0,x1,y1 = mapping['crop_xyxy']
    reduced = image.resize((x1-x0,y1-y0), Image.Resampling.BOX)
    canvas = Image.new('RGB', tuple(mapping['source']['wh']), 'black')
    canvas.paste(reduced, (x0,y0))
    return canvas


def main(step):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert step in (1000,10000,18000)
    import cv2
    cv2.setNumThreads(1)
    snap = OUT/'gaussian-run'/f'step-{step:06d}'
    assert read(snap/'result.json')['step'] == step
    dest = snap/'review'
    dest.mkdir(exist_ok=False)
    archive = Path('/mnt/d/00/记录/008')
    rows = read(OUT/'data/training-inputs.json')['records']
    sources = read(OUT/'source-native-inputs.json')['records']
    mappings = read(OUT/'pixel-mapping.json')['records']
    oldrows = {r['frame']:r for r in read(BASELINE/'data/training-inputs.json')['records']}
    oldmaps = {r['frame']:r for r in read(BASELINE/'pixel-mapping.json')['records']}
    with np.load(snap/'current-meshes.npz') as z:
        mesh = dict(z)
    with np.load(OUT/'fit/fitted.npz') as z:
        fit = dict(z)
    assert mesh['frame_ids'].astype(str).tolist() == [r['frame'] for r in rows]
    assert len(rows) == len(sources) == len(mappings) == 16
    render, renderer_source = mesh_renderer()
    records = []
    started = time.monotonic()
    for i,(row,source,mapping) in enumerate(zip(rows,sources,mappings)):
        frame = row['frame']
        assert frame == source['frame'] == mapping['frame']
        for key,digest in [('image','source_sha256'),('mask','mask_sha256'),('validity','validity_sha256')]:
            assert sha(source[key]) == source[digest]
        raw = Image.open(source['image']).convert('RGB')
        mask = Image.open(source['mask']).convert('L')
        valid = np.asarray(Image.open(source['validity'])) > 0
        candidate = original_pixel_render(OUT,step,row,mapping)
        old = original_pixel_render(BASELINE,step,oldrows[frame],oldmaps[frame]) if frame in oldrows else None
        crops = {'body':crop_box(mask.getbbox(),raw.size,30)}
        if source.get('face_box_xyxy') is not None:
            crops['face'] = crop_box(source['face_box_xyxy'],raw.size,35)
        for name,ids in [('left_hand',range(25,46)),('right_hand',range(46,67))]:
            xy = fit['observed_native_keypoints'][i,list(ids),:2]
            crops[name] = crop_box(np.r_[xy.min(0)-18,xy.max(0)+18],raw.size)
        images, metrics = [], []
        for region,box in crops.items():
            x0,y0,x1,y1 = map(int,box)
            if x1<=x0 or y1<=y0:
                continue
            original = raw.crop(box)
            overlay = render(original,np.asarray(box),dict(vertices_camera=mesh['vertices_camera'][i:i+1],
                faces=mesh['faces'],K_integer_centers=np.asarray(source['K_integer_centers'])))
            prediction = candidate.crop(box)
            picture = joined([original,overlay,prediction])
            path = archive/f'高斯_室内一圈16帧124_{step:06d}_{frame}_{region}_原图_当前mesh_高斯.png'
            assert not path.exists()
            picture.save(path)
            images.append(dict(region=region,path=str(path),sha256=sha(path),
                crop_original_xyxy=list(map(int,box)),panel_order=['source','current_mesh','sixteen_view_gaussian']))
            ma = np.asarray(mask.crop(box),dtype=np.float64)/255
            target = np.asarray(original,dtype=np.float64)/255*ma[...,None]
            known = valid[y0:y1,x0:x1]
            foreground = (ma>0)&known
            item = dict(region=region,known_pixels=int(known.sum()),foreground_pixels=int(foreground.sum()))
            for key,pic in [('sixteen',prediction),('old_six',old.crop(box) if old is not None else None)]:
                if pic is None:
                    continue
                err = np.abs(np.asarray(pic,dtype=np.float64)/255-target).mean(2)
                item[key+'_foreground_mae'] = float(err[foreground].mean()) if foreground.any() else None
                item[key+'_known_composite_mae'] = float(err[known].mean()) if known.any() else None
            metrics.append(item)
        records.append(dict(frame=frame,images=images,metrics=metrics,
            candidate_png_sha256=sha(snap/(frame+'_gaussian.png')),same_original_source_comparison=old is not None))
    records.sort(key=lambda r:r['frame'])
    pages = []
    for region,max_height in [('body',430),('face',240),('left_hand',200),('right_hand',200)]:
        entries = [(r['frame'],x) for r in records for x in r['images'] if x['region']==region]
        for start in range(0,len(entries),4):
            batch = entries[start:start+4]
            panels = []
            for frame,entry in batch:
                pic = Image.open(entry['path']).convert('RGB')
                scale = min(1,max_height/pic.height,1400/pic.width)
                pic = pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.LANCZOS)
                panels.append(pic)
            sheet = Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),(25,25,25))
            y = 0
            for pic in panels:
                sheet.paste(pic,(0,y));y += pic.height
            path = dest/f'{region}-{start//4+1:02d}.png'
            sheet.save(path)
            pages.append(dict(region=region,path=str(path),frames=[x[0] for x in batch],sha256=sha(path)))
    common_comparison = {}
    for region in ('body','face','left_hand','right_hand'):
        common = [m for r in records for m in r['metrics'] if m['region']==region
                  and m.get('old_six_foreground_mae') is not None]
        count = sum(m['foreground_pixels'] for m in common)
        common_comparison[region] = dict(frames=len(common),foreground_pixels=count,
            **{key:sum(m[key+'_foreground_mae']*m['foreground_pixels'] for m in common)/count
               if count else None for key in ('old_six','sixteen')})
    save(dest/'result.json',dict(updated=now(),step=step,records=records,pages=pages,
        common_six_foreground_mae=common_comparison,
        renderer_sha256=sha(renderer_source),seconds=time.monotonic()-started,actual_visual_review=False,
        common_six_causal_scope='Same source pixels; new roster and new joint fit change together, not view-count-only ablation.',
        coordinates='Original source integer pixel coordinates;2x renders box-downsampled; no geometric warp.',
        feet_quality_excluded=True,quality_accepted=False))
    print('SIXTEEN_GAUSSIAN_REVIEW_READY',step,len(records),flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
