"""Read-only CPU source review of the frozen fit24 -> GS12 leafwoman view.

Reuse the existing five-panel source/mask/current-mesh/GS/alpha renderer.
Only identities, native raster, record and explicit support connectors change.
No GPU rendering, source changes, pose repair, subset selection or admission.
"""
import argparse
import ast
import difflib
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
import review_graywoman_half12_gaussian128 as previous
from review_leafwoman_half12_fit129 import joint_group_support
from review_officewoman_circle24_fit127 import foot_review_crop
from review_full_sequence123 import crop_box
from run_official_gaussians123 import read,save,sha,now

N,FIT_N,HOLD_N=12,24,4
NATIVE_WH=(2704,1520)
OUT=previous.OUT.parent/'leafwoman-half12-paired18k129'
FIT=OUT.parent/'leafwoman-half12-pose129'
SOURCE_PLAN=OUT.parent/'leafwoman-half12-input129/plan.json'
SELECTION=FIT/'gs-selection/selection.json'
ARCHIVE=Path('/mnt/d/00/记录/022')
BASE_SHA='04e2afa7c0998c0108ab293ff4a51e013b13b1bd9e2d20581c9e8f464faa590f'
SOURCE_PLAN_SHA='0115b08fed63b9c93042ee00f985dedf81f2634ead26fac0c79ea78044ff2373'
SELECTION_SHA='954ce63fa33678a20933e0bfd28b622f4af80fc2e7fd623dccf9bdc3cea0ec21'
rendered_roi=previous.rendered_roi
show_unrendered_as_gray=previous.show_unrendered_as_gray


def crops_for_source(source,body,observations,wh):
    flags=joint_group_support(source)
    if type(source.get('face_region_is_observed')) is not bool:
        raise ValueError('Unknown129source face visibility; no implicit true.')
    x0,y0,x1,y1=body;h=y1-y0
    crops=dict(full_frame=[0,0,wh[0],wh[1]],body=body,
               arms=crop_box([x0,y0+.18*h,x1,y0+.72*h],wh))
    feet=foot_review_crop(dict(observed_native_keypoints=observations),flags,wh)
    if feet is not None:crops['feet']=feet
    if source['face_region_is_observed'] and source.get('face_box_xyxy') is not None:
        crops['face']=crop_box(source['face_box_xyxy'],wh,35)
    # The whole actual mesh/Gaussian stays in the rendered images. Source
    # region support does not certify fingers or justify a hand GT crop/MAE.
    return crops


def compiled_review():
    assert sha(previous.__file__)==BASE_SHA
    main,_,before=previous.compiled_review();code=before
    edits=[
        ("Path('/mnt/d/00/记录/015')","Path('/mnt/d/00/记录/022')",1),
        ('Unknown128source face visibility','Unknown129source face visibility',1),
        ('高斯_灰衣半圈12帧128_','高斯_持叶半圈12帧129_',1),
        ('GRAYWOMAN12_GAUSSIAN_REVIEW_READY','LEAFWOMAN12_GAUSSIAN_REVIEW_READY',1),
    ]
    for old,new,count in edits:
        assert code.count(old)==count,(old,code.count(old),count)
        code=code.replace(old,new)
    ast.parse(code)
    namespace=dict(main.__globals__,OUT=OUT,N=N,NATIVE_WH=NATIVE_WH,
        joint_group_support=joint_group_support,crops_for_source=crops_for_source,
        rendered_roi=rendered_roi,show_unrendered_as_gray=show_unrendered_as_gray)
    exec(compile(code,str(__file__)+'::existing_CPU_source12_Gaussian_review','exec'),namespace)
    return namespace['main'],before,code


def input_context():
    """Read only canonical fit24/source/selection, never an unfinished snapshot."""
    assert sha(SOURCE_PLAN)==SOURCE_PLAN_SHA and sha(SELECTION)==SELECTION_SHA
    source,selection,plan,result=(read(p) for p in (SOURCE_PLAN,SELECTION,FIT/'plan.json',FIT/'result.json'))
    assert read(FIT/'process-exit.json')['exit_code']==0
    assert selection['source_plan_sha256']==sha(FIT/'plan.json')
    assert selection['input_source_plan_sha256']==SOURCE_PLAN_SHA
    assert selection['fit_sha256']==result['fit_sha256']==sha(FIT/'selected-fit.npz')
    assert len(plan['rows'])==FIT_N and len(plan['review_only_excluded'])==HOLD_N
    assert [row['frame'] for row in plan['rows']]==source['training_frames']
    assert selection['heldout_frames']==plan['review_only_excluded']==source['heldout_frames']
    frames=selection['training_frames'];reference=selection['reference_frame']
    assert len(frames)==len(set(frames))==N and reference in frames
    assert set(frames)<=set(source['training_frames'])
    assert not set(frames)&set(selection['excluded_holdout_reserves'])
    assert selection['original_fit_candidate_count']==FIT_N and selection['selected_Gaussian_count']==N
    assert selection['shared_betas_or_pose_refit'] is False
    assert selection['heldout_used_for_selection'] is False
    with np.load(FIT/'selected-fit.npz',allow_pickle=False) as z:original=dict(z)
    ids=original['frame_ids'].astype(str).tolist()
    assert ids==source['training_frames'] and len(ids)==len(set(ids))==FIT_N
    assert original['K_integer_centers'].shape==(3,3)
    assert original['betas'].shape==(1,10) and original['body_pose_axis_angle'].shape==(1,63)
    rows={row['frame']:row for row in source['rows']}
    for frame in frames:
        row=rows[frame];joint_group_support(row)
        assert type(row['face_region_is_observed']) is bool
        assert tuple(row['wh'])==NATIVE_WH
        assert np.array_equal(row['K_integer_centers'],original['K_integer_centers'])
        assert not any(row['source_joint_group_supported'][g] for g in ('left_knee','right_knee','left_foot','right_foot'))
    return dict(source=source,selection=selection,plan=plan,original=original,
        frames=frames,reference=reference,rows=rows,fit24_indices={f:ids.index(f) for f in frames})


def validate_prepared_subset(context,snapshot):
    """Initial handoff by frame ID; current vertices may contain learned offset."""
    rows=read(OUT/'data/training-inputs.json')['records']
    sources=read(OUT/'source-native-inputs.json')['records']
    mappings=read(OUT/'pixel-mapping.json')['records']
    expected=[context['reference']]+[f for f in context['frames'] if f!=context['reference']]
    with np.load(OUT/'fit/fitted.npz',allow_pickle=False) as z:handoff=dict(z)
    with np.load(snapshot/'current-meshes.npz',allow_pickle=False) as z:current=dict(z)
    assert [r['frame'] for r in rows]==[r['frame'] for r in sources]==[r['frame'] for r in mappings]==expected
    assert handoff['frame_ids'].astype(str).tolist()==current['frame_ids'].astype(str).tolist()==expected
    assert current['vertices_camera'].shape==handoff['vertices_camera'].shape==(N,10475,3)
    assert np.isfinite(current['vertices_camera']).all()
    assert np.array_equal(current['faces'],handoff['faces']) and np.array_equal(current['faces'],context['original']['faces'])
    assert handoff['K_integer_centers'].shape==(N,3,3)
    for i,source in enumerate(sources):
        frame=source['frame'];j=context['fit24_indices'][frame];row=context['rows'][frame]
        assert source['source_joint_group_supported']==row['source_joint_group_supported']
        assert source['face_region_is_observed']==row['face_region_is_observed']
        assert source['source_sha256']==row['source_sha256'] and source['mask_sha256']==row['mask_sha256']
        assert tuple(source['wh'])==NATIVE_WH
        assert np.array_equal(source['K_integer_centers'],context['original']['K_integer_centers'])
        assert np.array_equal(handoff['K_integer_centers'][i],context['original']['K_integer_centers'])
        for key in ('vertices_camera','joints_camera','observed_native_keypoints'):
            assert np.array_equal(handoff[key][i],context['original'][key][j]),(frame,key)
    # Do not overwrite or equate the current learned mesh to its initializer.
    return expected


def review(step):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and step==1000
    context=input_context();snapshot=OUT/'gaussian-run'/f'step-{step:06d}'
    result=read(snapshot/'result.json')
    assert result['step']==step and result['reference_frame']==context['reference'] and result['active_sh_degree']==0
    assert result['optimizer_updates']==1000 and result['checkpoint_phase']=='after_density_adam_zero_grad'
    assert sha(snapshot/'world.ply')==result['world_ply_sha256']
    checkpoint=OUT/'gaussian-run/chkpnt1000.pth'
    assert sha(checkpoint)==result['native_checkpoint_sha256']
    expected=validate_prepared_subset(context,snapshot)
    paths=[SOURCE_PLAN,SELECTION,FIT/'selected-fit.npz',FIT/'plan.json',FIT/'result.json',
        OUT/'source-native-inputs.json',OUT/'pixel-mapping.json',OUT/'fit/fitted.npz',
        snapshot/'world.ply',snapshot/'result.json',snapshot/'current-meshes.npz',checkpoint]
    frozen={str(p):sha(p) for p in paths}
    compiled_review()[0](step)
    path=snapshot/'review/result.json';data=read(path)
    assert {r['frame'] for r in data['records']}==set(expected)
    data.update(subject='leafwoman',reference_frame=context['reference'],training_views=N,
        original_fit_candidate_count=FIT_N,fit24_frame_indices=context['fit24_indices'],
        fit_reference_frame=context['selection']['fit_reference_frame'],
        same_frozen_fit24_subset_no_refit=True,source_plan_sha256=SOURCE_PLAN_SHA,selection_sha256=SELECTION_SHA,
        optimizer_updates_in_reviewed_checkpoint=1000,native_checkpoint_sha256=result['native_checkpoint_sha256'],
        diagnostic_only=True,quality_accepted=False,direct18000_admitted=False,opacity_candidate_implemented=False,
        source_flags_are_not_physical_GT=True,finger_GT_or_MAE_fabricated=False,
        unknown_knee_feet_not_rendered_as_negative_GT=True,actual_mesh_and_gaussian_limbs_not_hidden=True,
        derived_ROI_scope='Saved GS/alpha covers only pixel-mapping crop_xyxy; gray48 outside is not rendered/unknown and metric-excluded. No alpha0 or absence-of-floaters claim outsideROI. Actual full mesh remains shown.',
        alpha_scope='Native rasterizer1-T over all retained sorted layers with truncation/early-stop; not front clothing coverage. Float32 unchanged; display clip only.',
        source_coordinates='Original integer source raster; existing ROI2x RGB/alpha BOX-downsample and paste, no geometric warp.',
        reference_warning=context['selection']['reference_supported_observation_warning'],
        continuous_novel_views=dict(state='not_rendered_not_reviewed',GPU_started_by_this_helper=False,
            existing_reusable_entries=['export_cross_subject_sh0124.py','export_demo_body_snapshot123.py::replay'],
            reference_frame=context['reference'],ground_truth_available=False,old_subject_orbit_reused=False),
        source_hashes=frozen,review_helper_sha256=sha(__file__))
    save(path,data)
    assert all(sha(p)==h for p,h in frozen.items())
    print('LEAFWOMAN129_SOURCE12_GAUSSIAN_ALPHA_REVIEW_READY',step,len(data['records']),flush=True)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    context=input_context();_,before,code=compiled_review()
    observed=np.zeros((1,118,3),np.float32);untouched=observed.copy()
    for frame in context['frames']:
        source=context['rows'][frame]
        crops=crops_for_source(source,[500,200,1900,1520],observed,NATIVE_WH)
        assert set(crops)=={'full_frame','body','arms','face'}
    assert np.array_equal(observed,untouched)
    bad=dict(source,face_region_is_observed=None)
    try:crops_for_source(bad,[500,200,1900,1520],observed,NATIVE_WH)
    except ValueError:pass
    else:raise AssertionError('Unknown source face support accepted.')
    test=Image.new('RGB',(8,6),(10,20,30));mapping=dict(crop_xyxy=[2,1,6,5])
    known=rendered_roi(mapping,test.size)
    pictures=show_unrendered_as_gray(test,test,mapping)
    assert all(np.all(np.asarray(x)[~known]==48) and np.array_equal(np.asarray(x)[known],np.asarray(test)[known]) for x in pictures)
    assert "mesh['vertices_camera'][i:i+1]" in code and "faces=mesh['faces']" in code
    assert "for i,(row,source,mapping) in enumerate(zip(rows,sources,mappings))" in code
    save(Path(__file__).with_suffix('.audit.json'),dict(updated=now(),state='CPU_static_gaussian129_review_ready',
        helper_sha256=sha(__file__),base_helper_sha256=BASE_SHA,compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        source_plan_sha256=SOURCE_PLAN_SHA,selection_sha256=SELECTION_SHA,
        source_fit24_sha256=sha(FIT/'selected-fit.npz'),fit24_frame_indices=context['fit24_indices'],
        selected_frames=context['frames'],N=N,original_fit_count=FIT_N,reference_frame=context['reference'],
        native_wh=list(NATIVE_WH),record_id='022',only_step=1000,
        snapshot_read=False,review_render_executed=False,GPU_started=False,source_or_training_modified=False,
        actual_saved_current_mesh_uses_GS12_frame_IDs_not_first12_fit_slots=True,
        source_pixel_ROI_unknown_preserved=True,no_fabricated_feet_or_hand_GT_crop=True,
        all_actual_mesh_and_Gaussian_geometry_retained=True,novel_views='Pending separate actual review.',
        quality_accepted=False,direct18000_admitted=False,
        interface_diff=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='frozen128_CPU_GSreview',tofile='new129_CPU_GSreview'))))
    print('LEAFWOMAN129_GAUSSIAN_REVIEW_CPU_STATIC_READY',sha(__file__),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','review']);p.add_argument('--step',type=int,default=1000);args=p.parse_args()
    audit() if args.mode=='audit' else review(args.step)
