"""Saved-observation visibility and sampling audit; never fits or trains."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image
from review_full_sequence123 import relocate, read, sha
from smplify_score_contract123 import relative_confidence
from official_smplifyx_entry123 import COCO


def stats(a):
    a = np.asarray(a)
    return dict(minimum=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), maximum=float(a.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = relocate(args.root)
    out = root/'reviews/observation-visibility-audit'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    provenance = read(root/'input-provenance.json')
    for filename,digest in provenance['input_files'].items():
        assert sha(relocate(filename)) == digest, filename
    obs_path = next(relocate(s) for s in provenance['input_files'] if s.endswith('/observations.npz'))
    with np.load(obs_path,allow_pickle=False) as f:
        obs = dict(f)
    rows = read(root/'data/training-inputs.json')['records']
    assert obs['frame_ids'].astype(str).tolist() == [r['frame'] for r in rows]
    conf,q90 = relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    assert np.array_equal(q90,provenance['per_joint_score_q90'])
    dest = np.r_[COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source = np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    target = np.zeros((len(rows),118,3),np.float32)
    target[:,dest,:2] = obs['keypoints'][:,source]
    target[:,dest,2] = conf[:,source]
    selected = {0,31,62,78,93}  # existing reviewed samples, selected before this audit
    records = relocate('D:/00/记录/005')
    results, images = [], []
    for i,row in enumerate(rows):
        if time.monotonic()-start > 120:
            raise RuntimeError('Read-only CPU audit budget exceeded.')
        directory = root/'frames'/row['frame']
        with np.load(directory/'winner-camera-space.npz',allow_pickle=False) as f:
            projected = f['joints_projected_native'][0]
            assert str(f['frame_id'].item()) == row['frame']
            assert np.array_equal(f['K_integer_centers'],row['K_integer_centers'])
        mask_path = relocate(row['mask'])
        assert sha(mask_path) == row['mask_sha256']
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert('L'))
        w,h = row['wh']; resized = (round(w*899/h),899)
        scale = np.array(resized)/[w,h]
        errors = np.linalg.norm(projected-target[i,:,:2],axis=1)
        face_extent = np.ptp(target[i,67:118,:2],axis=0)
        item = dict(frame_id=row['frame'],source_frame=row['source_frame_index'],
            face51_confidence=stats(target[i,67:118,2]),
            face51_squared_weight_sum=float((target[i,67:118,2]**2).sum()),
            face51_error_px=stats(errors[67:118]),feet6_error_px=stats(errors[19:25]),
            feet6_confidence=stats(target[i,19:25,2]),
            face_landmark_extent_native=face_extent.tolist(),
            face_landmark_extent_training=(face_extent*scale).tolist(),
            mask_fraction=float((mask>0).mean()),full_K_or_pose_changed=False)
        results.append(item)
        if i not in selected:
            continue
        receipt = read(directory/'saved-mesh-review.json')
        feature = next(v for v in receipt['files'] if '二维训练观测' in v['panels'])
        feature_path = relocate(feature['path']); assert sha(feature_path)==feature['sha256']
        bx0,by0,bx1,by1 = feature['crop_integer_xyxy']; bw=bx1-bx0
        with Image.open(feature_path) as im: triptych=im.convert('RGB')
        assert triptych.size == (3*bw,by1-by0)
        for region,label in [('面部','face'),('双脚','feet')]:
            matching=[v for v in receipt['files'] if region in Path(v['path']).name and '原图_官方SMPLX透明覆盖'==v['panels']]
            assert len(matching)==1
            x0,y0,x1,y1=matching[0]['crop_integer_xyxy']
            assert bx0<=x0<x1<=bx1 and by0<=y0<y1<=by1
            panels=[triptych.crop((j*bw+x0-bx0,y0-by0,j*bw+x1-bx0,y1-by0)) for j in range(3)]
            sheet=Image.new('RGB',((x1-x0)*3,y1-y0))
            for j,panel in enumerate(panels):sheet.paste(panel,(j*panel.width,0))
            name=f'20261030_官方首轮观测核查_{row["frame"]}_{region}_原图_二维检测绿_三维投影紫.png'
            path=records/'特征点'/name;assert not path.exists();sheet.save(path)
            copy=out/f'{row["frame"]}_{label}.png';sheet.save(copy)
            images.append(dict(frame_id=row['frame'],region=label,path=str(path),review_copy=str(copy),
                sha256=sha(path),crop_integer_xyxy=[x0,y0,x1,y1],source_feature_sha256=feature['sha256'],
                no_text=True,no_new_resize=True,actual_viewed=False))
        if i in (31,93):
            image_path=relocate(row['image']);assert sha(image_path)==row['source_sha256']
            with Image.open(image_path) as im: raw=im.convert('RGB')
            crop=next(v for v in receipt['files'] if '面部' in Path(v['path']).name)['crop_integer_xyxy']
            native=raw.crop(crop)
            box=[int(np.floor(crop[0]*scale[0])),int(np.floor(crop[1]*scale[1])),
                 int(np.ceil(crop[2]*scale[0])),int(np.ceil(crop[3]*scale[1]))]
            train=raw.resize(resized,Image.Resampling.LANCZOS).crop(box)
            sheet=Image.new('RGB',(native.width+train.width,max(native.height,train.height)),(35,35,35))
            sheet.paste(native,(0,0));sheet.paste(train,(native.width,0))
            name=f'20261030_官方首轮输入采样_{row["frame"]}_左原生面部_右实际899高训练面部_均不再次缩放.png'
            path=records/'分析'/name;assert not path.exists();sheet.save(path)
            copy=out/f'{row["frame"]}_native_vs_training.png';sheet.save(copy)
            images.append(dict(frame_id=row['frame'],region='sampling',path=str(path),review_copy=str(copy),
                sha256=sha(path),native_crop_wh=list(native.size),training_crop_wh=list(train.size),
                no_text=True,no_additional_resize=True,actual_viewed=False))
    result=dict(created=datetime.now(timezone.utc).isoformat(),state='completed_visual_review_pending',
        source_observation_sha256=sha(obs_path),source_provenance_sha256=sha(root/'input-provenance.json'),
        training_config_sha256=sha(root/'gaussian-run/config.json'),frames=len(rows),rows=results,images=images,
        all_face_observations_positive=bool((target[:,67:118,2]>0).all()),
        face_weight_not_physical_visibility=True,source_images_unchanged=True,
        parameters_updated=False,new_training_started=False,GPU_calls=0,physical_truth_verified=False,
        timing_seconds=time.monotonic()-start,
        limitation='Fitted detector errors/relative confidence are not independent physical truth. Sampling density alone does not identify the cause of ghosting.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'result':str(out/'result.json'),'frames':len(rows),'images':len(images),'seconds':result['timing_seconds']}),flush=True)


if __name__=='__main__':
    main()
