"""CPU-only128 review adapter: actual12 fitted meshes,4 unfitted heldouts.

Reuses the existing CPU native mesh/synthetic-side renderer. Region support
only gates point overlays; every mesh vertex/triangle remains visible to the
renderer, including implausible hidden limbs. No fitting or source changes.
"""
import ast
import difflib
import hashlib
import inspect
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw
import review_officewoman_circle24_fit127 as base
import run_graywoman_half12_fit128 as fit
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import crop_box,joined

N,HOLD_COUNT=12,4
ARCHIVE=Path('/mnt/d/00/记录/015')
JOINT_GROUP_SOURCE=dict(fit.GROUP_SOURCE)
JOINT_GROUP_TARGET=dict(fit.GROUP_TARGET)
SOURCE_PLAN_SHA='67a862ec25118e2d938819d8d1c353a64d0f1d796fbd4b463517775aafc97ee8'
BASE_SHA='0b3c1b7cff2cb5126d5b29a6bedae644c4377262c9bcf3e983959f386a74bd97'


def joint_group_support(row):
    flags=row.get('source_joint_group_supported')
    if type(flags) is not dict or set(flags)!=set(JOINT_GROUP_TARGET) or any(type(v) is not bool for v in flags.values()):
        raise ValueError('Eight reviewed knee/foot/wrist/elbow booleans required; unknown is never true.')
    return flags


def review_completion(out,helper):
    """128 has no exceptional recovery path: original process must exit0."""
    assert sha(fit.AUDIT/'plan.json')==SOURCE_PLAN_SHA
    exit_record=read(out/'process-exit.json');assert exit_record['exit_code']==0,'Actual128 fit must finish successfully before CPU render.'
    plan=read(out/'plan.json');result=read(out/'result.json')
    assert plan['reference_frame']==plan['reference_for_root_seed']=='frame_000023'
    assert len(plan['rows'])==N and len(plan['review_only_excluded'])==HOLD_COUNT
    assert result['frames_count']==N and result['reference_frame']=='frame_000023'
    assert sha(out/'selected-fit.npz')==result['fit_sha256']
    return dict(original_process_exit_code=0,CPU_metadata_recovery_used=False,
                fit_helper_sha256=sha(helper),source_plan_sha256=SOURCE_PLAN_SHA,
                result_sha256=sha(out/'result.json'),selected_fit_sha256=result['fit_sha256'],quality_accepted=False)


def supported_overlay_indices(row):
    flags=joint_group_support(row)
    observed=row.get('face_region_is_observed')
    if type(observed) is not bool:raise ValueError('Source face support must be explicit.')
    hidden=set(range(25,67)) # both finger sets are source-occluded; no new hand truth
    if not observed:hidden.update([0,*range(15,19),*range(67,118)])
    for group,ids in JOINT_GROUP_TARGET.items():
        if not flags[group]:hidden.update(ids)
    return [i for i in range(118) if i not in hidden]


def validate_saved_frame(row,arrays):
    """Identity/fullK/arrays check, not a new physical quality threshold."""
    assert np.asarray(arrays['vertices_camera']).shape==(1,10475,3)
    assert np.asarray(arrays['joints_camera']).shape==(1,118,3)
    assert np.asarray(arrays['observed_native_keypoints']).shape==(1,118,3)
    assert np.isfinite(arrays['vertices_camera']).all() and np.isfinite(arrays['joints_camera']).all()
    assert np.asarray(arrays['K_integer_centers']).shape==(3,3)
    assert np.array_equal(arrays['K_integer_centers'],np.asarray(row['K_integer_centers']))
    groups=joint_group_support(row)
    for group,ids in JOINT_GROUP_TARGET.items():
        if not groups[group]:assert np.all(arrays['observed_native_keypoints'][0,ids,2]==0)
    with np.load(fit.OUT/'selected-fit.npz',allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();assert len(ids)==N and len(set(ids))==N
        i=ids.index(row['frame'])
        for key in ('vertices_camera','joints_camera','observed_native_keypoints'):
            assert np.array_equal(arrays[key],z[key][i:i+1]),(row['frame'],key)
        assert np.array_equal(arrays['faces'],z['faces'])


def review_regions(row,arrays,body_box,wh,unused_hand_box):
    flags=joint_group_support(row);supported_overlay_indices(row)
    x0,y0,x1,y1=body_box;h=y1-y0
    boxes=dict(body=body_box,arms=crop_box([x0,y0+.18*h,x1,y0+.72*h],wh))
    foot=base.foot_review_crop(arrays,flags,wh)
    if foot is not None:boxes['feet']=foot
    if row['face_region_is_observed'] and row.get('face_box_xyxy') is not None:
        boxes['face']=crop_box(row['face_box_xyxy'],wh,35)
    # No hidden-hand crop is fabricated. Complete mesh hands are still rendered
    # in body/arms and all synthetic directions without deleting geometry.
    return boxes


def make_panel(raw,box,arrays,row,renderer):
    indices=supported_overlay_indices(row)
    observations=np.asarray(arrays['observed_native_keypoints'])[0]
    joints=np.asarray(arrays['joints_camera'])[0]
    K=np.asarray(arrays['K_integer_centers']);homogeneous=joints@K.T
    uv=homogeneous[:,:2]/homogeneous[:,2:]
    assert np.isfinite(uv).all()
    source2d=raw.copy();current3d=raw.copy()
    d2,d3=ImageDraw.Draw(source2d),ImageDraw.Draw(current3d)
    for i in indices:
        point=observations[i]
        if not np.isfinite(point).all() or point[2]<=0:continue
        for draw,xy,color in ((d2,point[:2],'#00ff50'),(d3,uv[i],'#ff00dc')):
            x,y=(xy-np.asarray(box[:2])).tolist()
            if 0<=x<raw.width and 0<=y<raw.height:
                draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
    return joined([raw,source2d,current3d,renderer(raw,np.asarray(box),arrays)])


def compiled_reviewer():
    assert sha(base.__file__)==BASE_SHA
    _,_,original=base.compiled_reviewer();code=original
    edits=[
        ("Path('/mnt/d/00/记录/019')","Path('/mnt/d/00/记录/015')",2),
        ('办公室环绕24帧127','灰衣半圈12帧128',2),
        ('OFFICEWOMAN24_NATIVE_REVIEW_READY','GRAYWOMAN12_NATIVE_REVIEW_READY',1),
        ("        with np.load(path,allow_pickle=False) as z:a=dict(z)","        with np.load(path,allow_pickle=False) as z:a=dict(z)\n        validate_saved_frame(row,a)",1),
        ('        hand_box,hand_crop_basis=hand_review_crop(a,bb,raw.size)',"        hand_box,hand_crop_basis=bb,'hidden_source_hands_no_crop_generated'",1),
        ('im=raw.crop(b);panel=joined([im,renderer(im,np.array(b),a)])','im=raw.crop(b);panel=make_panel(im,np.array(b),a,row,renderer)',1),
        ('feet_rms_px=None,feet_metric_status=',"point_overlay_does_not_modify_source=True,hidden_hand_crops_generated=False,all_mesh_geometry_rendered=True,feet_rms_px=None,feet_metric_status=",1),
    ]
    for old,new,count in edits:
        assert code.count(old)==count,(old,code.count(old),count)
        code=code.replace(old,new)
    ast.parse(code)
    namespace=dict(base.previous.namespace,fit=fit,N=N,review_regions=review_regions,joint_group_support=joint_group_support,
                   review_completion=review_completion,validate_saved_frame=validate_saved_frame,make_panel=make_panel,__file__=__file__)
    exec(compile(code,str(__file__)+'::same_CPU_native12_mesh_review','exec'),namespace)
    return namespace['main'],original,code


def heldout_sources():
    source=inspect.getsource(base.heldout_sources)
    edits=[
        ("assert len(p['training_frames'])==24 and len(p['heldout_frames'])==8","assert len(p['training_frames'])==N and len(p['heldout_frames'])==HOLD_COUNT",1),
        ('办公室环绕24帧127','灰衣半圈12帧128',1),
        ('            if (not observed and j in hidden_head) or j in hidden_limb:',"            if (not observed and j in hidden_head) or j in hidden_limb or j>=91:",1),
        ('raw_detector2D_with_unsupported_head_knee_foot_hidden','raw_detector2D_with_unsupported_head_knee_foot_wrist_elbow_and_hidden_fingers_hidden',1),
    ]
    for old,new,count in edits:
        assert source.count(old)==count,(old,source.count(old),count);source=source.replace(old,new)
    namespace=dict(base.__dict__,fit=fit,N=N,HOLD_COUNT=HOLD_COUNT,ARCHIVE=ARCHIVE,
                   joint_group_support=joint_group_support,JOINT_GROUP_SOURCE=JOINT_GROUP_SOURCE)
    exec(compile(source,str(__file__)+'::four_unfitted_heldout_sources','exec'),namespace)
    namespace['heldout_sources']()
    path=fit.OUT/'review/result.json';result=read(path)
    result.update(reference_frame='frame_000023',training_frame_count=N,heldout_source_only_count=HOLD_COUNT,
        point_panel_order=['original','supported_source2D_predictions','current_saved3D_joints_projected_with_completeK','current_saved_mesh_overlay'],
        point_colors=dict(source2D='green',current3D_projection='magenta'),point_overlays_modify_raw_observations=False,
        unsupported_wrist_elbow_and_hidden_fingers_not_displayed_as_source_truth=True,
        all_mesh_vertices_and_triangles_retained_for_native_and_sides=True,
        mesh_renderer_scope='Existing CPU depth-sorted triangle overlay; visual diagnostic, not exact rasterizer acceptance or physical correspondence truth.',
        synthetic_sides_scope='12 actual fitted meshes x4 synthetic directions, not photographed unseen-view truth. Held4 have no fitted mesh or camera.',
        physical_joint_or_camera_calibration_accepted=False,actual_visual_review=False,quality_accepted=False)
    save(path,result)


def render():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ARCHIVE.mkdir(exist_ok=True)
    assert '只要图片不要文本' in (ARCHIVE.parent/'规则.md').read_text(encoding='utf-8-sig')
    assert sha(fit.AUDIT/'plan.json')==SOURCE_PLAN_SHA
    compiled_reviewer()[0]()
    heldout_sources()


def audit():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    _,before,code=compiled_reviewer()
    flags=dict.fromkeys(JOINT_GROUP_SOURCE,False)
    row=dict(face_region_is_observed=True,face_box_xyxy=[500,300,700,600],source_joint_group_supported=flags)
    target=np.zeros((1,118,3),np.float32);target[0,2]=[800,900,.8];target[0,4]=[900,1300,.9];target[0,25]=[900,1400,.9]
    joints=np.zeros((1,118,3),np.float32);joints[:,:,:2]=target[:,:,:2];joints[:,:,2]=2
    arrays=dict(observed_native_keypoints=target,joints_camera=joints,K_integer_centers=np.eye(3))
    original={k:v.copy() for k,v in arrays.items()}
    regions=review_regions(row,arrays,[100,100,1300,2600],[1520,2704],None)
    assert 'hands' not in regions and 'feet' not in regions and set(regions)=={'body','arms','face'}
    indices=supported_overlay_indices(row)
    assert 2 in indices and not set([3,4,6,7])&set(indices) and not set(range(25,67))&set(indices)
    im=Image.new('RGB',(128,256),(10,20,30));panel=make_panel(im,[0,0,128,256],arrays,row,lambda raw,box,data:raw.copy())
    assert panel.size==(512,256) and np.array_equal(np.asarray(panel)[:,:128],np.asarray(im))
    assert all(np.array_equal(v,original[k]) for k,v in arrays.items())
    for bad in (None,{},dict(flags,left_wrist=None)):
        try:joint_group_support(dict(source_joint_group_supported=bad))
        except ValueError:continue
        raise AssertionError('Unknown source support accepted.')
    assert not torch.cuda.is_initialized()
    save(Path(__file__).with_suffix('.audit.json'),dict(updated=now(),state='CPU_static_review128_ready',
        helper_sha256=sha(__file__),base_helper_sha256=BASE_SHA,fit_helper_sha256=sha(fit.__file__),
        compiled_CPU_reviewer_sha256=hashlib.sha256(code.encode()).hexdigest(),N=N,heldout_source_only=HOLD_COUNT,
        native_wh=[1520,2704],reference_frame='frame_000023',record_id='015',source_plan_sha256=SOURCE_PLAN_SHA,
        expected_group_source=JOINT_GROUP_SOURCE,expected_group_target=JOINT_GROUP_TARGET,
        actual_fit_completion_read=False,actual_meshes_rendered=False,actual_visual_review=False,quality_accepted=False,
        same_native_crop_panels_no_warp=True,source_observations_modified=False,unknown_visibility_halts=True,
        hidden_hand_crop_not_fabricated=True,all_actual_mesh_geometry_retained=True,heldout_mesh_or_camera_fabricated=False,
        original_process_exit0_required=True,no_127_recovery_override=True,CUDA_initialized=False,GPU_started=False,
        interface_diff=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='frozen127_CPU_review',tofile='new128_CPU_review'))))
    print('GRAYWOMAN128_FIT_REVIEW_CPU_STATIC_READY',sha(__file__),flush=True)


if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'audit'
    assert mode in ('audit','render')
    audit() if mode=='audit' else render()
