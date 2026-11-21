"""CPU129 review adapter:24 actual fit candidates and4 unfitted heldouts.

The filename retains the eventual GS~12 route name; this reviewer never picks
that subset. It reuses the existing raw/2D/3D/mesh and four-direction renderer.
Cropped knee/feet stay unknown: no fabricated foot crop, metric or negative.
"""
import argparse
import ast
import difflib
import hashlib
import inspect
import os
from pathlib import Path
import re

import numpy as np
import review_graywoman_half12_fit128 as previous
import run_leafwoman_half12_fit129 as fit
from run_official_gaussians123 import read,save,sha,now

N,HOLD_COUNT=24,4
NATIVE_WH=(2704,1520)
ARCHIVE=Path('/mnt/d/00/记录/022')
SOURCE_PLAN_SHA='0115b08fed63b9c93042ee00f985dedf81f2634ead26fac0c79ea78044ff2373'
BASE_SHA='019190cfc348837b6e9e9c07e091aef670ff976bf3a4091bb48e6fabad163f9b'
JOINT_GROUP_SOURCE=dict(fit.GROUP_SOURCE)
JOINT_GROUP_TARGET=dict(fit.GROUP_TARGET)
joint_group_support=previous.joint_group_support
supported_overlay_indices=previous.supported_overlay_indices
review_regions=previous.review_regions
make_panel=previous.make_panel


def review_completion(out,helper):
    assert SOURCE_PLAN_SHA is not None and re.fullmatch('[0-9a-f]{64}',SOURCE_PLAN_SHA)
    assert sha(fit.AUDIT/'plan.json')==SOURCE_PLAN_SHA
    source=read(fit.AUDIT/'plan.json');plan=read(out/'plan.json');result=read(out/'result.json')
    assert read(out/'process-exit.json')['exit_code']==0,'Actual129 fit must finish exit0 before CPU render.'
    assert len(plan['rows'])==N and len(plan['review_only_excluded'])==HOLD_COUNT
    assert plan['reference_frame']==plan['reference_for_root_seed']==source['reference_frame']==result['reference_frame']
    assert result['frames_count']==N and [r['frame'] for r in plan['rows']]==source['training_frames']
    assert sha(out/'selected-fit.npz')==result['fit_sha256']
    assert all(sha(path)==digest for path,digest in plan['source_hashes'].items())
    return dict(original_process_exit_code=0,CPU_metadata_recovery_used=False,
                fit_helper_sha256=sha(helper),source_plan_sha256=SOURCE_PLAN_SHA,
                result_sha256=sha(out/'result.json'),selected_fit_sha256=result['fit_sha256'],
                reference_frame=source['reference_frame'],quality_accepted=False)


# Rebind only the existing validator's globals to this24-view source. Its
# completeK, frame identity and saved-array equality checks stay unchanged.
_scope=dict(previous.__dict__,fit=fit,N=N)
exec(compile(inspect.getsource(previous.validate_saved_frame),str(__file__)+'::existing_saved_array_validator','exec'),_scope)
validate_saved_frame=_scope['validate_saved_frame']


def compiled_reviewer():
    assert sha(previous.__file__)==BASE_SHA
    main,_,original=previous.compiled_reviewer();code=original
    edits=[
        ("Path('/mnt/d/00/记录/015')","Path('/mnt/d/00/记录/022')",2),
        ('灰衣半圈12帧128','持叶候选24帧129',2),
        ('GRAYWOMAN12_NATIVE_REVIEW_READY','LEAFWOMAN24_NATIVE_REVIEW_READY',1),
        ('assert raw.size==m.size==(1520,2704)','assert raw.size==m.size==NATIVE_WH',1),
    ]
    for old,new,count in edits:
        assert code.count(old)==count,(old,code.count(old),count);code=code.replace(old,new)
    ast.parse(code)
    namespace=dict(main.__globals__,fit=fit,N=N,NATIVE_WH=NATIVE_WH,
                   review_completion=review_completion,validate_saved_frame=validate_saved_frame,__file__=__file__)
    exec(compile(code,str(__file__)+'::same_CPU_actual24_mesh_review','exec'),namespace)
    return namespace['main'],original,code


def heldout_sources():
    base=previous.base
    source=inspect.getsource(base.heldout_sources)
    edits=[
        ("assert len(p['training_frames'])==24 and len(p['heldout_frames'])==8","assert len(p['training_frames'])==N and len(p['heldout_frames'])==HOLD_COUNT",1),
        ('办公室环绕24帧127','持叶候选24帧129',1),
        ('assert raw.size==mask.size==(1520,2704)','assert raw.size==mask.size==NATIVE_WH',1),
        ('            if (not observed and j in hidden_head) or j in hidden_limb:',
         '            if (not observed and j in hidden_head) or j in hidden_limb or j>=91:',1),
        ('raw_detector2D_with_unsupported_head_knee_foot_hidden','raw_detector2D_with_unsupported_head_knee_foot_wrist_elbow_and_unaccepted_finger_data_hidden',1),
    ]
    for old,new,count in edits:
        assert source.count(old)==count,(old,source.count(old),count);source=source.replace(old,new)
    namespace=dict(base.__dict__,fit=fit,N=N,HOLD_COUNT=HOLD_COUNT,ARCHIVE=ARCHIVE,NATIVE_WH=NATIVE_WH,
                   joint_group_support=joint_group_support,JOINT_GROUP_SOURCE=JOINT_GROUP_SOURCE)
    exec(compile(source,str(__file__)+'::four_unfitted_heldout_sources','exec'),namespace)
    namespace['heldout_sources']()
    path=fit.OUT/'review/result.json';result=read(path);source_plan=read(fit.AUDIT/'plan.json')
    result.update(subject='leafwoman',reference_frame=source_plan['reference_frame'],fit_candidate_count=N,
        prospective_Gaussian_view_target=12,Gaussian_subset_selected_by_reviewer=False,
        heldout_source_only_count=HOLD_COUNT,source_plan_sha256=SOURCE_PLAN_SHA,
        point_panel_order=['original','supported_source2D_predictions','current_saved3D_joints_projected_with_completeK','current_saved_mesh_overlay'],
        point_colors=dict(source2D='green',current3D_projection='magenta'),point_overlays_modify_raw_observations=False,
        source_visibility_scope='Clothed sleeve/cuff/face region support only; unsupported/unknown wrist/elbow/knee/foot dots are not fabricated. Finger data is not accepted as independent physical truth.',
        lower_body_scope='Cropped/occluded knees and feet remain unknown; no foot crop or foot RMS when no supported observations. Complete actual mesh still appears in body and all synthetic directions.',
        all_mesh_vertices_and_triangles_retained_for_native_and_sides=True,
        mesh_renderer_scope='Existing CPU depth-sorted triangle overlay; visual diagnostic, not exact rasterizer acceptance or physical correspondence truth.',
        synthetic_sides_scope='24 actual saved fitted meshes x4 synthetic directions, not photographed ground truth. Held4 have no fitted mesh/camera.',
        physical_joint_or_camera_calibration_accepted=False,actual_visual_review=False,quality_accepted=False)
    save(path,result)


def render():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert SOURCE_PLAN_SHA is not None and re.fullmatch('[0-9a-f]{64}',SOURCE_PLAN_SHA)
    assert sha(fit.AUDIT/'plan.json')==SOURCE_PLAN_SHA
    assert '只要图片不要文本' in (ARCHIVE.parent/'规则.md').read_text(encoding='utf-8-sig')
    ARCHIVE.mkdir(exist_ok=True)
    compiled_reviewer()[0]()
    heldout_sources()


def audit():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert JOINT_GROUP_SOURCE==previous.JOINT_GROUP_SOURCE and JOINT_GROUP_TARGET==previous.JOINT_GROUP_TARGET
    _,before,code=compiled_reviewer()
    flags=dict.fromkeys(JOINT_GROUP_TARGET,False);row=dict(face_region_is_observed=True,face_box_xyxy=[1200,300,1400,500],source_joint_group_supported=flags)
    target=np.zeros((1,118,3),np.float32);target[0,2]=[1300,700,.8];target[0,4]=[1400,900,.8];target[0,14]=[1200,1500,.8]
    unchanged=target.copy();regions=review_regions(row,dict(observed_native_keypoints=target),[900,100,1700,1520],NATIVE_WH,None)
    assert set(regions)=={'body','arms','face'} and np.array_equal(target,unchanged)
    assert not set([3,4,6,7])&set(supported_overlay_indices(row))
    flags.update(left_wrist=True,right_wrist=True,left_elbow=True,right_elbow=True)
    assert set([3,4,6,7])<=set(supported_overlay_indices(row))
    assert 'feet' not in review_regions(row,dict(observed_native_keypoints=target),[900,100,1700,1520],NATIVE_WH,None)
    assert np.array_equal(target,unchanged)
    assert 'for row in p[\'rows\']:' in code and 'NATIVE_WH' in code
    assert not torch.cuda.is_initialized()
    save(Path(__file__).with_suffix('.audit.json'),dict(updated=now(),state='CPU_static_fit129_review_ready',
        helper_sha256=sha(__file__),base_helper_sha256=BASE_SHA,fit_helper_sha256=sha(fit.__file__),
        compiled_CPU_reviewer_sha256=hashlib.sha256(code.encode()).hexdigest(),fit_candidate_count=N,heldout_source_only_count=HOLD_COUNT,
        native_wh=list(NATIVE_WH),record_id='022',reference='Read frozen sourceplan reference dynamically; never first-frame default.',
        source_plan_sha256=SOURCE_PLAN_SHA,render_requires_matching_source_plan_sha_parameter=True,
        expected_reference_frame_bound_by_source_hash='frame_000014',
        actual_fit_or_unfrozen_source_read=False,actual_meshes_rendered=False,OUT_review_created=False,D022_images_created=False,
        same_native_4panels_and_original_renderer=True,all24_saved_meshes_and4synthetic_directions=True,
        no_unsupported_or_empty_foot_crop_or_metric=True,source_wrists_and_elbows_follow_explicit_flags=True,
        point_overlays_do_not_modify_observations=True,all_actual_mesh_geometry_retained=True,
        heldout_mesh_or_camera_fabricated=False,Gaussian_subset_selected=False,
        original_process_exit0_required=True,CUDA_initialized=False,GPU_started=False,quality_accepted=False,
        interface_diff=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='executed128_CPU_fitreview',tofile='new129_CPU24_fitreview'))))
    print('LEAFWOMAN129_FIT24_REVIEW_CPU_STATIC_READY',sha(__file__),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','render']);p.add_argument('--source-plan-sha');args=p.parse_args()
    if args.mode=='audit':audit()
    else:
        assert args.source_plan_sha and re.fullmatch('[0-9a-f]{64}',args.source_plan_sha)
        assert args.source_plan_sha==SOURCE_PLAN_SHA
        render()
