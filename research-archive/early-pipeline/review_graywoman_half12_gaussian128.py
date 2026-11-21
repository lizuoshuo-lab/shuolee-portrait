"""Read-only CPU128 source review; novel native review stays a separate task.

The1000 snapshot is a known-pose-failure diagnostic. This module never starts a
GPU renderer, changes training, repairs pose, admits18k or tests new opacity.
Unknown hand observations do not justify hiding any saved mesh/Gaussian limb.
"""
import argparse
import ast
import difflib
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
import review_officewoman_circle24_gaussian127 as base
from review_graywoman_half12_fit128 import joint_group_support
from review_officewoman_circle24_fit127 import foot_review_crop
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import crop_box

N,NATIVE_WH=12,(1520,2704)
OUT=base.OUT.parent/'graywoman-half12-paired18k128'
FIT=OUT.parent/'graywoman-half12-pose128'
ARCHIVE=Path('/mnt/d/00/记录/015')
REFERENCE='frame_000023'
BASE_SHA='2d2e4e5121266fd684355dc2dfa7073f114ac678da35e08b9fcf36afe0dff9ab'
EXPECTED_SOURCE_PLAN_SHA='67a862ec25118e2d938819d8d1c353a64d0f1d796fbd4b463517775aafc97ee8'


def rendered_roi(mapping,wh):
    x0,y0,x1,y1=map(int,mapping['crop_xyxy'])
    assert 0<=x0<x1<=wh[0] and 0<=y0<y1<=wh[1]
    region=np.zeros((wh[1],wh[0]),bool);region[y0:y1,x0:x1]=True
    return region


def show_unrendered_as_gray(prediction,alpha_display,mapping):
    """Display only: gray outside derivedROI means unknown, never alpha0."""
    known=rendered_roi(mapping,prediction.size);pictures=[]
    for image in (prediction,alpha_display):
        array=np.asarray(image).copy();array[~known]=48;pictures.append(Image.fromarray(array))
    return tuple(pictures)


def crops_for_source(source,body,observations,wh):
    flags=joint_group_support(source)
    if type(source.get('face_region_is_observed')) is not bool:
        raise ValueError('Unknown128source face visibility; no implicit true.')
    x0,y0,x1,y1=body;h=y1-y0
    crops=dict(full_frame=[0,0,wh[0],wh[1]],body=body,arms=crop_box([x0,y0+.18*h,x1,y0+.72*h],wh))
    feet=foot_review_crop(dict(observed_native_keypoints=observations),flags,wh)
    if feet is not None:crops['feet']=feet
    if source['face_region_is_observed'] and source.get('face_box_xyxy') is not None:
        crops['face']=crop_box(source['face_box_xyxy'],wh,35)
    # No source-supported independent hands in this128 roster. Actual limbs
    # remain in complete raw/body/arms/Gaussian/mesh images, without deletion.
    assert flags['left_wrist'] is False and flags['right_wrist'] is False
    return crops


def compiled_review():
    assert sha(base.__file__)==BASE_SHA
    import inspect
    original=inspect.getsource(base.previous.main)
    code=original
    # Reapply the frozen127 literal connector list to the real source function;
    # inspect.getsource cannot read an exec-compiled generated filename.
    tree=ast.parse(inspect.getsource(base.compiled_review))
    edits_node=next(n for n in tree.body[0].body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='edits' for t in n.targets))
    for old,new in ast.literal_eval(edits_node.value):
        assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
    original=code
    old_crops="        crops = {'body':body}\n        feet = foot_review_crop(dict(observed_native_keypoints=fit['observed_native_keypoints'][i:i+1]),support,raw.size)\n        if feet is not None:\n            crops['feet'] = feet\n        if source['face_region_is_observed'] is True and source.get('face_box_xyxy') is not None:\n            crops['face'] = crop_box(source['face_box_xyxy'],raw.size,35)\n        for name,ids in [('left_hand',range(25,46)),('right_hand',range(46,67))]:\n            xy = fit['observed_native_keypoints'][i,list(ids),:2]\n            crops[name] = crop_box(np.r_[xy.min(0)-18,xy.max(0)+18],raw.size)"
    edits=[
        ('assert step in (1000,10000,18000)','assert step == 1000',1),
        ("Path('/mnt/d/00/记录/019')","Path('/mnt/d/00/记录/015')",1),
        ('Unknown127source face visibility','Unknown128source face visibility',1),
        ('高斯_办公室环绕24帧127_','高斯_灰衣半圈12帧128_',1),
        ('OFFICEWOMAN24_GAUSSIAN_REVIEW_READY','GRAYWOMAN12_GAUSSIAN_REVIEW_READY',1),
        ("        valid = np.asarray(Image.open(source['validity'])) > 0",
         "        valid = np.asarray(Image.open(source['validity'])) > 0\n        valid = valid & rendered_roi(mapping,raw.size)",1),
        ("        alpha_display = Image.fromarray(np.rint(np.clip(alpha,0,1)*255).astype(np.uint8)).convert('RGB')",
         "        alpha_display = Image.fromarray(np.rint(np.clip(alpha,0,1)*255).astype(np.uint8)).convert('RGB')\n        candidate,alpha_display = show_unrendered_as_gray(candidate,alpha_display,mapping)",1),
        ('records.append(dict(frame=frame,images=images,metrics=metrics,',
         "records.append(dict(frame=frame,images=images,metrics=metrics,rendered_source_roi_xyxy=mapping['crop_xyxy'],outside_ROI='not_rendered_unknown_gray48_display_only_excluded_from_metrics',",1),
        (old_crops,"        crops = crops_for_source(source,body,fit['observed_native_keypoints'][i:i+1],raw.size)",1),
        ("[('body',430),('face',240),('left_hand',200),('right_hand',200),('feet',220)]","[('full_frame',430),('body',430),('face',240),('arms',350),('feet',220)]",1),
    ]
    for old,new,count in edits:
        assert code.count(old)==count,(old,code.count(old),count)
        code=code.replace(old,new)
    ast.parse(code)
    namespace=dict(base.previous.main.__globals__,OUT=OUT,N=N,NATIVE_WH=NATIVE_WH,
                   joint_group_support=joint_group_support,foot_review_crop=foot_review_crop,crops_for_source=crops_for_source,
                   rendered_roi=rendered_roi,show_unrendered_as_gray=show_unrendered_as_gray)
    exec(compile(code,str(__file__)+'::existing_CPU_native_gaussian_review','exec'),namespace)
    return namespace['main'],original,code


def review(step):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and step==1000
    source_plan=FIT.parent/'graywoman-half12-input128/plan.json'
    assert sha(source_plan)==EXPECTED_SOURCE_PLAN_SHA
    snapshot=OUT/'gaussian-run'/f'step-{step:06d}';result=read(snapshot/'result.json')
    assert result['step']==step and result['reference_frame']==REFERENCE and result['active_sh_degree']==0
    assert sha(snapshot/'world.ply')==result['world_ply_sha256']
    source_hashes={str(p):sha(p) for p in [source_plan,FIT/'selected-fit.npz',FIT/'result.json',
        OUT/'source-native-inputs.json',OUT/'pixel-mapping.json',OUT/'fit/fitted.npz',snapshot/'world.ply',snapshot/'result.json',snapshot/'current-meshes.npz']}
    compiled_review()[0](step)
    path=snapshot/'review/result.json';data=read(path)
    data.update(subject='graywoman',reference_frame=REFERENCE,training_views=N,diagnostic_only=True,
        known_source_mesh_arm_gesture_failure=True,quality_accepted=False,
        direct18000_admitted=False,opacity_candidate_implemented=False,source_flags_are_not_physical_GT=True,
        unknown_fingers_and_wrists_not_used_for_hand_GT_crops_or_MAE=True,
        actual_mesh_and_gaussian_limbs_not_hidden=True,
        derived_ROI_scope='Saved Gaussian/alpha renders cover only each pixel-mapping crop_xyxy. Gray48 outsideROI is not rendered/unknown and excluded from metrics; no absence-of-floaters or alpha0 claim outsideROI. Full native mesh remains shown independently.',
        alpha_scope='Native rasterizer1-T accumulated over all sorted retained layers with native truncation/early-stop; not front clothing coverage or independent per-point opacity. Float32 input unchanged; display clip only.',
        source_coordinates='Same existing ROI2x BOX downsample and paste into original integer source raster; not native input rescaling, geometric warp or new camera.',
        continuous_novel_views=dict(state='not_rendered_not_reviewed',GPU_started_by_this_helper=False,
            existing_reusable_entries=['export_cross_subject_sh0124.py','export_demo_body_snapshot123.py::replay'],
            entry_scope='Root may separately adapt existing native replay/export to this actual1000/reference23. No orbit, camera list, parity or quality acceptance is claimed here.',
            reference_frame=REFERENCE,ground_truth_available=False,old_subject_orbit_reused=False),
        source_hashes=source_hashes,review_helper_sha256=sha(__file__))
    save(path,data)
    assert all(sha(p)==h for p,h in source_hashes.items())
    print('GRAYWOMAN128_SOURCE12_GAUSSIAN_ALPHA_REVIEW_READY',step,len(data['records']),flush=True)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    _,before,code=compiled_review()
    display=Image.new('RGB',(8,6),(10,20,30));mapping=dict(crop_xyxy=[2,1,6,5])
    known=rendered_roi(mapping,display.size);pictures=show_unrendered_as_gray(display,display,mapping)
    assert all(np.all(np.asarray(x)[~known]==48) and np.array_equal(np.asarray(x)[known],np.asarray(display)[known]) for x in pictures)
    flags={k:True for k in ('left_knee','right_knee','left_foot','right_foot','left_elbow','right_elbow')}
    flags.update(left_wrist=False,right_wrist=False)
    row=dict(face_region_is_observed=True,face_box_xyxy=[500,300,700,600],source_joint_group_supported=flags)
    observed=np.zeros((1,118,3),np.float32);observed[0,14]=[700,2450,.8];untouched=observed.copy()
    crops=crops_for_source(row,[100,200,1200,2600],observed,NATIVE_WH)
    assert set(crops)=={'full_frame','body','arms','face','feet'} and np.array_equal(observed,untouched)
    row['face_region_is_observed']=False;flags['left_foot']=flags['right_foot']=False
    crops=crops_for_source(row,[100,200,1200,2600],observed,NATIVE_WH)
    assert set(crops)=={'full_frame','body','arms'} and np.array_equal(observed,untouched)
    row['face_region_is_observed']=None
    try:crops_for_source(row,[100,200,1200,2600],observed,NATIVE_WH)
    except ValueError:pass
    else:raise AssertionError('Unknown face support accepted.')
    assert "for i,(row,source,mapping) in enumerate(zip(rows,sources,mappings))" in code
    assert "mesh['vertices_camera'][i:i+1]" in code and "faces=mesh['faces']" in code
    save(Path(__file__).with_suffix('.audit.json'),dict(updated=now(),state='CPU_static_gaussian128_review_ready',
        helper_sha256=sha(__file__),base_helper_sha256=BASE_SHA,compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        N=N,native_wh=list(NATIVE_WH),reference_frame=REFERENCE,record_id='015',only_step=1000,
        snapshot_or_unfinished_model_read=False,review_render_executed=False,GPU_started=False,
        native_same_camera_raw_mask_current_mesh_gaussian_alpha=True,arms_face_supported_feet_crops=True,
        independent_hand_GT_or_crop_fabricated=False,all_actual_mesh_Gaussian_limbs_retained=True,
        saved_gaussian_domain='derivedROI only; gray48 outside means not rendered/unknown, metric-excluded, never proof of alpha0/no external Gaussians',
        continuous_novel_views='Pending, not rendered or verified. Root may separately reuse the existing native export/replay entry; this helper neither creates cameras nor starts GPU.',
        source_observations_masks_raw_pixels_or_training_modified=False,quality_accepted=False,direct18000_admitted=False,opacity_candidate_implemented=False,
        interface_diff=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='frozen127_CPU_GSreview',tofile='new128_CPU_GSreview'))))
    print('GRAYWOMAN128_GAUSSIAN_REVIEW_CPU_STATIC_READY',sha(__file__),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','review']);p.add_argument('--step',type=int,default=1000);args=p.parse_args()
    if args.mode=='audit':audit()
    else:review(args.step)
