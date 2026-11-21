"""CPU evidence for an actual completed24-view fit and8unfitted heldout sources.

Reuses the existing same-pixel mesh reviewer. A source-unobserved face has no
face crop or facial RMS here. Heldouts show source/mask/2D only, never a mesh
or inferred camera. No result becomes quality-accepted by generating images.
"""
import ast
import difflib
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import review_football_half12_fit126 as previous
import run_officewoman_circle24_fit127 as fit
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import joined,crop_box
from officewoman_fit_recovery_contract127 import review_completion

N=24
ARCHIVE=Path('/mnt/d/00/记录/019')
JOINT_GROUP_SOURCE=dict(left_knee=[13],right_knee=[14],left_foot=[15,17,18,19],right_foot=[16,20,21,22])
JOINT_GROUP_TARGET=dict(left_knee=[13],right_knee=[10],left_foot=[14,19,20,21],right_foot=[11,22,23,24])


def joint_group_support(row):
    flags=row.get('source_joint_group_supported')
    if type(flags) is not dict or set(flags)!=set(JOINT_GROUP_TARGET) or any(type(v) is not bool for v in flags.values()):
        raise ValueError('Unknown source knee/foot support: four reviewed booleans are required; no implicit true.')
    return flags


def foot_review_crop(arrays,flags,wh):
    """Crop only actually supported, finite, in-raster foot observations.

    No lower-body fallback is allowed: trousers at the image boundary are not
    a foot observation. This read-only crop never changes fitting data.
    """
    points=np.asarray(arrays['observed_native_keypoints'])[0]
    if points.shape!=(118,3):
        raise ValueError('Expected one mapped118 observation row.')
    supported=[]
    for group in ('left_foot','right_foot'):
        if flags[group]:
            values=points[JOINT_GROUP_TARGET[group]]
            good=(values[:,2]>0)&np.isfinite(values).all(1)&(values[:,0]>=0)&(values[:,0]<wh[0])&(values[:,1]>=0)&(values[:,1]<wh[1])
            if good.any():
                supported.append(values[good,:2])
    if not supported:
        return None
    xy=np.concatenate(supported)
    return crop_box(np.r_[xy.min(0),xy.max(0)],wh,55)


def arm_review_crop(arrays,body_box,wh):
    points=np.asarray(arrays['observed_native_keypoints'])[0,1:8]
    good=(points[:,2]>0)&np.isfinite(points).all(1)
    xy=points[good,:2]
    return crop_box(np.r_[xy.min(0),xy.max(0)],wh,55) if len(xy) else list(body_box)


def review_regions(row,arrays,body_box,wh,hand_box):
    observed=row['face_region_is_observed']
    if type(observed) is not bool:
        raise ValueError('Unknown source face visibility: review is required.')
    support=joint_group_support(row)
    boxes=dict(body=body_box,arms=arm_review_crop(arrays,body_box,wh),hands=hand_box)
    feet=foot_review_crop(arrays,support,wh)
    if feet is not None:
        boxes['feet']=feet
    if observed and row.get('face_box_xyxy') is not None:
        boxes['face']=crop_box(row['face_box_xyxy'],wh,35)
    return boxes


def compiled_reviewer():
    original=previous.source
    code=original
    edits=[
        ("    out=fit.OUT;assert read(out/'process-exit.json')['exit_code']==0",
         '    out=fit.OUT;completion=review_completion(out,fit.__file__)',1),
        ("Path('/mnt/d/00/记录/011')","Path('/mnt/d/00/记录/019')",2),
        ('足球人物半圈12帧126','办公室环绕24帧127',2),
        ('FOOTBALL_HALF12_NATIVE_REVIEW_READY','OFFICEWOMAN24_NATIVE_REVIEW_READY',1),
        ("    p=read(out/'plan.json');r=read(out/'result.json')",
         "    p=read(out/'plan.json');r=read(out/'result.json')\n    assert len(p['rows'])==N\n    source_visibility=read(out/'source-observation-visibility.json')\n    visibility=source_visibility['face_visibility']\n    limb_visibility=source_visibility['joint_group_visibility']\n    assert all(type(visibility[x['frame']]) is bool and visibility[x['frame']] is x['face_region_is_observed'] for x in p['rows'])\n    assert all(joint_group_support(x)==limb_visibility[x['frame']] for x in p['rows'])",1),
        ("        boxes=dict(body=bb,face=crop_box(row['face_box_xyxy'],raw.size,35),\n            hands=hand_box,\n            feet=crop_box([bb[0],bb[1]+.72*(bb[3]-bb[1]),bb[2],bb[3]],raw.size))",
         '        assert raw.size==m.size==(1520,2704)\n        boxes=review_regions(row,a,bb,raw.size,hand_box)',1),
        ('records.append(dict(frame=name,images=images,fit_sha256=sha(path),actual_visual_review=False))',
         "records.append(dict(frame=name,images=images,fit_sha256=sha(path),actual_visual_review=False,face_region_is_observed=visibility[name],face_rms_px=None,face_metric_status='Not computed by this visual reviewer; invisible face has no crop or detector-error truth.',source_joint_group_supported=joint_group_support(row),feet_region_has_supported_observations=('feet' in images),feet_rms_px=None,feet_metric_status='Not computed by this visual reviewer; unsupported or empty foot observations have no crop or metric.'))",1),
        ("renderer_sha256=sha(source),reviewer_sha256=sha(__file__)))",
         "renderer_sha256=sha(source),reviewer_sha256=sha(__file__),original_process_exit_code=completion['original_process_exit_code'],metadata_postprocess_completion=completion))",1),
    ]
    for old,new,count in edits:
        assert code.count(old)==count,(old,code.count(old),count)
        code=code.replace(old,new)
    ast.parse(code)
    namespace=dict(previous.namespace,fit=fit,N=N,review_regions=review_regions,joint_group_support=joint_group_support,review_completion=review_completion,__file__=__file__)
    exec(compile(code,'<same-native24-fit-review-source-visibility-aware>','exec'),namespace)
    return namespace['main'],original,code


def heldout_sources():
    import cv2
    cv2.setNumThreads(1)
    p=fit.source_plan()
    assert len(p['training_frames'])==24 and len(p['heldout_frames'])==8
    doc=read(p['source_inputs']);rows={r['frame']:r for r in doc['records']}
    with np.load(p['source_observations'],allow_pickle=False) as z:
        obs=dict(z)
    slots={f:i for i,f in enumerate(obs['frame_ids'].astype(str))}
    dest=fit.OUT/'review/heldout-source-only';dest.mkdir(exist_ok=False)
    records=[];tiles=[]
    for frame in p['heldout_frames']:
        row=rows[frame];i=slots[frame];observed=p['face_visibility'][frame]
        support=joint_group_support(dict(source_joint_group_supported=p['joint_group_visibility'][frame]))
        assert type(observed) is bool
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        assert raw.size==mask.size==(1520,2704)
        box=crop_box(mask.getbbox(),raw.size,30);crop=raw.crop(box);overlay=np.asarray(crop).copy()
        hidden_head=set(range(5))|set(range(23,91))
        hidden_limb={j for group,ids in JOINT_GROUP_SOURCE.items() if not support[group] for j in ids}
        for j,xy in enumerate(obs['keypoints'][i]):
            if (not observed and j in hidden_head) or j in hidden_limb:
                continue
            if not np.isfinite(xy).all() or not np.isfinite(obs['scores'][i,j]) or obs['scores'][i,j]<=0:
                continue
            point=tuple(np.rint(xy-np.asarray(box[:2])).astype(int))
            cv2.circle(overlay,point,3,(0,255,80) if j<23 else (255,180,0) if j<91 else (70,120,255),1,cv2.LINE_AA)
        ma=np.asarray(mask.crop(box),dtype=np.float32)/255
        person=Image.fromarray((np.asarray(crop,dtype=np.float32)*ma[...,None]).astype(np.uint8))
        picture=joined([crop,Image.fromarray(overlay),mask.crop(box).convert('RGB'),person])
        path=ARCHIVE/f'分析_办公室环绕24帧127_独立留出_{frame}_原图_2D_mask_人像.png'
        assert not path.exists();picture.save(path)
        records.append(dict(frame=frame,path=str(path),sha256=sha(path),source_image_sha256=row['source_sha256'],
            source_mask_sha256=row['mask_sha256'],crop_original_xyxy=box,face_region_is_observed=observed,
            source_joint_group_supported=support,
            panel_order=['original','raw_detector2D_with_unsupported_head_knee_foot_hidden','original_mask','masked_person'],
            raw_detector_is_not_physical_truth=True,mesh_or_camera_fitted=False,face_rms_px=None,
            fitting_or_train_normalizer_used=False,actual_visual_review=False))
        thumb=picture.copy();thumb.thumbnail((1440,650));tiles.append(thumb)
    pages=[]
    for first in range(0,len(tiles),4):
        group=tiles[first:first+4]
        image=Image.new('RGB',(max(x.width for x in group),sum(x.height for x in group)),(25,25,25))
        y=0
        for tile in group:
            image.paste(tile,(0,y));y+=tile.height
        path=dest/f'page-{first//4+1:02d}.png';image.save(path);pages.append(str(path))
    result_path=fit.OUT/'review/result.json';result=read(result_path)
    result.update(heldout_source_only=dict(records=records,pages=pages,heldout_fit_or_camera_available=False,
        role='Independent of this fit; source-only review is not unseen-video generalization or physical correspondence ground truth.'),
        face_visibility_aware=True,source_joint_group_visibility_aware=True,face_RMS_generated_by_reviewer=False,feet_RMS_generated_by_reviewer=False,quality_accepted=False)
    save(result_path,result)


def render():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ARCHIVE.mkdir(exist_ok=True)
    assert '只要图片不要文本' in (ARCHIVE.parent/'规则.md').read_text(encoding='utf-8-sig')
    compiled_reviewer()[0]()
    heldout_sources()


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert JOINT_GROUP_SOURCE==fit.GROUP_SOURCE and JOINT_GROUP_TARGET==fit.GROUP_TARGET
    _,before,code=compiled_reviewer()
    arrays=dict(observed_native_keypoints=np.zeros((1,118,3),np.float32))
    arrays['observed_native_keypoints'][0,2]=[400,1000,.8]
    arrays['observed_native_keypoints'][0,4]=[500,1300,.8]
    bbox=[100,200,1200,2600]
    support={key:False for key in JOINT_GROUP_TARGET}
    input_before=arrays['observed_native_keypoints'].copy()
    for observed in (False,True):
        row=dict(face_region_is_observed=observed,face_box_xyxy=[500,300,700,600],source_joint_group_supported=support)
        boxes=review_regions(row,arrays,bbox,[1520,2704],[350,900,800,1500])
        assert ('face' in boxes) is observed and all(k in boxes for k in ('body','arms','hands')) and 'feet' not in boxes
    cases=[]
    for case in ('unsupported_guessed_feet','supported_but_empty','left_supported_only','supported_outside_raster'):
        current=arrays['observed_native_keypoints'].copy()
        flags=dict(support)
        if case=='unsupported_guessed_feet':
            current[0,JOINT_GROUP_TARGET['left_foot']]=[700,2600,.8]
            expected=None
        elif case=='supported_but_empty':
            flags['left_foot']=True
            expected=None
        elif case=='left_supported_only':
            flags['left_foot']=True
            current[0,14]=[700,2450,.8]
            current[0,19]=[720,2500,.6]
            current[0,11]=[1100,1300,.9]
            expected=crop_box([700,2450,720,2500],[1520,2704],55)
        else:
            flags['right_foot']=True
            current[0,11]=[800,2800,.9]
            expected=None
        copy_before=current.copy()
        result=foot_review_crop(dict(observed_native_keypoints=current),flags,[1520,2704])
        assert result==expected and np.array_equal(current,copy_before)
        cases.append(dict(case=case,feet_crop=result,observations_modified=False))
    assert np.array_equal(arrays['observed_native_keypoints'],input_before)
    rejected=False
    try:
        review_regions(dict(face_region_is_observed=None),arrays,bbox,[1520,2704],bbox)
    except ValueError:
        rejected=True
    assert rejected
    for flags in (None,{},dict(support,left_foot=None)):
        try:
            joint_group_support(dict(source_joint_group_supported=flags))
        except ValueError:
            continue
        raise AssertionError('Unknown joint-group support was accepted.')
    ast.parse(Path(__file__).read_text())
    receipt=dict(updated=now(),state='CPU_static_fit_review_prepared',helper_sha256=sha(__file__),
        fit_helper_sha256=sha(fit.__file__),source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        count=24,native_wh=[1520,2704],record='019',all_training_rows_rendered_when_executed=True,
        invisible_face_crop_and_RMS_omitted=True,unknown_visibility_rejected=True,
        joint_group_schema={key:dict(source133=JOINT_GROUP_SOURCE[key],mapped118=JOINT_GROUP_TARGET[key]) for key in JOINT_GROUP_TARGET},
        no_unsupported_or_empty_feet_crop_or_RMS=True,feet_crop_cases=cases,
        unknown_source_foot_or_knee_support_rejected=True,source_observations_modified=False,
        heldout_source_only_count=8,heldout_mesh_or_camera_fabricated=False,
        actual_fit_exit_zero_or_verified_specific_CPU_recovery_required=True,original_exit1_not_rewritten=True,
        recovery_contract_sha256=sha(Path(__file__).with_name('officewoman_fit_recovery_contract127.py')),images_text_free=True,
        interface_diff=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),
            fromfile='existing-football-review',tofile='new-officewoman24-visibility-aware-review')),
        actual_fit_read=False,renders_executed=False,GPU_started=False,actual_visual_review=False,quality_accepted=False)
    save(Path(__file__).with_suffix('.audit.json'),receipt)
    print('OFFICEWOMAN24_FIT_REVIEW_CPU_STATIC_READY',flush=True)


if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'render'
    assert mode in ('audit','render')
    audit() if mode=='audit' else render()
