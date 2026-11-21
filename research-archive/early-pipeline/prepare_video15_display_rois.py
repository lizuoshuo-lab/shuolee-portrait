"""Verify immutable source RGB and reuse only historical 2D display rectangles."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


NEW = Path('D:/GS-SLAM-Data/msv2-video15-20261105')
OLD = Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-cross-subject-128/graywoman')
OUT = Path('D:/00/记录/2026-11-05_MSV2_video15')
VIDEO_SHA = '965befa346fdb0b5e5140745ab5651ef6ed7f8a87ae32777e1d88017bc45e547'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def padded(raw, wh, fraction):
    a = np.asarray(raw[:2], dtype=float)
    b = np.asarray(raw[2:], dtype=float)
    assert np.isfinite(a).all() and np.isfinite(b).all() and (b > a).all()
    pad = np.maximum((b - a) * fraction, 8)
    lo = np.maximum(np.floor(a - pad), 0).astype(int)
    hi = np.minimum(np.ceil(b + pad), wh).astype(int)
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def main():
    lineage = json.loads((NEW / 'pixel-lineage.json').read_text(encoding='utf-8'))
    assert lineage['passed'] and lineage['no_spatial_resizing']
    assert lineage['video_sha256'] == VIDEO_SHA
    assert sha(Path(lineage['source'])) == VIDEO_SHA
    # Retain only frame identities and 2D rectangles; never consume old geometry,
    # K, w2c, masks, priors, optimizer parameters or fitted state.
    old_rows = [{key: row[key] for key in (
        'frame', 'source_frame_index', 'source_sha256', 'wh',
        'initial_box_xyxy', 'face_box_xyxy', 'face_region_source')}
        for row in json.loads((OLD / 'data/training-inputs.json').read_text())['records']]
    by_index = {row['source_frame_index']: row for row in old_rows}
    current = lineage['records']
    assert len(old_rows) == 64 and len(current) == 65
    records, checks = [], []
    for row in current:
        index, wh = row['source_index'], row['wh']
        assert wh == [1520, 2704]
        new_path = NEW / 'dataset/images' / row['image']
        assert sha(new_path) == row['sha256']
        with Image.open(new_path) as im:
            assert im.mode == 'RGB' and list(im.size) == wh
            new_rgb = np.asarray(im)
        assert hashlib.sha256(new_rgb.tobytes()).hexdigest() == row['rgb_sha256']
        if index in by_index:
            prior = by_index[index]
            old_path = OLD / 'raw' / (prior['frame'] + '.png')
            assert sha(old_path) == prior['source_sha256']
            with Image.open(old_path) as im:
                assert im.mode == 'RGB' and list(im.size) == wh == prior['wh']
                old_rgb = np.asarray(im)
            assert np.array_equal(new_rgb, old_rgb), f'RGB mismatch at source {index}'
            rgb_sha = hashlib.sha256(old_rgb.tobytes()).hexdigest()
            assert rgb_sha == row['rgb_sha256']
            checks.append(dict(frame=row['frame'], source_index=index,
                               new_rgb_sha256=rgb_sha, old_rgb_sha256=rgb_sha,
                               old_png_sha256=prior['source_sha256'],
                               new_png_sha256=row['sha256'], RGB_bitwise_equal=True))
            method = 'exact_source_index_RGB_verified_2D_detection'
        else:
            assert index == 512
            nearest = min(by_index, key=lambda x: (abs(x-index), x))
            assert nearest == 505
            prior = by_index[nearest]
            method = 'nearest_display_only'
        body = padded(prior['initial_box_xyxy'], wh, .08)
        face = padded(prior['face_box_xyxy'], wh, .45)
        records.append(dict(frame=row['frame'], image=row['image'], source_index=index,
            time_seconds=row['time_seconds'], wh=wh,
            image_sha256=row['sha256'], image_rgb_sha256=row['rgb_sha256'],
            body=body, face=face, provenance=dict(method=method,
                detector_source_index=prior['source_frame_index'],
                detector_frame=prior['frame'],
                body_detector='YOLOv8n historical 2D image box, 4% detector margin already included',
                face_detector='RTMW133 approximate face ROI only',
                raw_body_box_xyxy=prior['initial_box_xyxy'],
                raw_face_box_xyxy=prior['face_box_xyxy'],
                face_region_source=prior['face_region_source'],
                display_only=True, used_for_training=False, used_for_camera_recovery=False,
                current_frame_is_direct_detection=(method != 'nearest_display_only'))))
        print('DISPLAY_ROI', index, method, flush=True)
    report = dict(schema='msv2-display-regions-v1', created=datetime.now(timezone.utc).isoformat(),
        video_sha256=VIDEO_SHA, source_video_sha256=VIDEO_SHA, wh=[1520,2704],
        pixel_lineage_path=str(NEW/'pixel-lineage.json'),
        pixel_lineage_sha256=sha(NEW/'pixel-lineage.json'),
        old_2D_manifest_path=str(OLD/'data/training-inputs.json'),
        old_2D_manifest_sha256=sha(OLD/'data/training-inputs.json'),
        source_domain='Original full RGB1520x2704, no crop/resize/rotation in either path',
        exact_source_index_matches=64, RGB_bitwise_verified_count=len(checks),
        pixel_checks=checks, nearest_display_only_count=1,
        padding={'body_fraction':.08, 'face_fraction':.45, 'minimum_each_axis_px':8},
        no_old_geometry_used=True, no_masks_read=True,
        used_for_training=False, used_for_camera_recovery=False, training_updates=0,
        records=records, last_face_crop_visual_reviewed=False)
    OUT.mkdir(parents=True, exist_ok=True)
    destination=OUT/'展示区域.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    panel=Image.new('RGB',(960,570*3),'#ededed')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    draw=ImageDraw.Draw(panel)
    for slot,i in enumerate((0,32,64)):
        r=records[i]
        y=slot*570
        with Image.open(NEW/'dataset/images'/r['image']) as im:
            display=im.copy();display.thumbnail((300,530),Image.Resampling.LANCZOS)
            ratio=display.width/im.width
            dd=ImageDraw.Draw(display)
            for key,color in [('body','#009dd4'),('face','#f05020')]:
                dd.rectangle(tuple(round(v*ratio) for v in r[key]),outline=color,width=3)
            panel.paste(display,(0,y+32))
            crop=im.crop(r['face']);crop=crop.resize((crop.width*3,crop.height*3),Image.Resampling.LANCZOS)
            panel.paste(crop,(330,y+70))
            crop.close()
        draw.text((12,y+4),f'{r["frame"]} · source{r["source_index"]}',font=font,fill='black')
        draw.text((330,y+4),'面部框抽查 · 3倍插值仅便于查看',font=font,fill='black')
        draw.text((330,y+33),r['provenance']['method'],font=small,fill='black')
    panel.save(OUT/'展示区域抽查.jpg',quality=94)
    print('DISPLAY_ROIS_READY',destination,flush=True)


if __name__ == '__main__':
    main()
