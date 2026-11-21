"""Verify the original 021 detector-to-fit-to-mesh coordinate path on saved results."""
import argparse
import ast
from pathlib import Path
import sys

import numpy as np

import fit_video21_native148 as base
from fit_video21_seated149 import OUT as SEATED
import official_smplifyx_entry123 as entry
from review_full_sequence123 import relocate
from update_leafwoman129_state import atomic, read, sha


def main(root):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(base.original.SMPLX_ROOT))
    import smplx
    from smplx.joint_names import JOINT_NAMES
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    from facial_geometry_boundary123 import protected_face_vertices

    assert Path(smplx.__file__).is_relative_to(base.original.SMPLX_ROOT)
    plan = read(base.OUT/'plan.json')
    result_path = root/'result.json'
    result = read(result_path)
    assert result['fitting_complete'] and result['plan_sha256']==sha(root/'plan.json')
    assert [r['frame'] for r in result['frames']]==plan['source_frames']
    assert not set(plan['source_frames'])&set(plan['retained_holdouts'])
    native = read(base.NATIVE/'result.json')
    data = relocate(native['data_root'])/'data/observations.npz'
    assert sha(data)==native['observations_sha256']==plan['source_obs_sha256']
    assert sha(base.CONTACTS)==plan['contacts_sha256'] and sha(base.ARCHIVE)==plan['baseline_observation_sha256']
    with np.load(data,allow_pickle=False) as z:
        source = dict(z)
    utils, _, _, _ = entry.load_official(base.original.OFFICIAL)
    mapping = utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False)
    assert len(mapping)==118
    for source_id,target in zip(base.SOURCE_IDX[:23],base.TARGET_IDX[:23],strict=True):
        assert source['joint_names'][source_id]==JOINT_NAMES[mapping[target]], (source_id,target)
    contacts = read(base.CONTACTS)['records']
    names = source['frame_ids'].tolist()
    K = np.asarray(plan['K_integer_centers'])
    K_corner=K.copy();K_corner[:2,2]+=.5
    arrays = []
    with np.load(base.ARCHIVE,allow_pickle=False) as z:
        targets = z['observed_native_keypoints']
    for i,(row,record) in enumerate(zip(contacts,result['frames'],strict=True)):
        assert row['frame']==record['frame'] and sha(record['output'])==record['sha256']
        assert np.array_equal(row['K_integer_centers'],K) and np.array_equal(row['K'],K_corner)
        assert np.array_equal(row['w2c'],np.eye(4))
        with np.load(record['output'],allow_pickle=False) as z:
            current = dict(z)
        assert np.array_equal(current['K_integer_centers'],K)
        assert np.array_equal(current['observed_native_keypoints'][0],targets[i])
        pixels = source['keypoints'][names.index(row['frame']),base.SOURCE_IDX].astype(np.float32)
        assert np.array_equal(targets[i,base.TARGET_IDX,:2],pixels)
        arrays.append(current)
    params = ('global_orient','expression','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','body_pose_axis_angle','camera_translation')
    fit = {k:np.concatenate([a[k] for a in arrays],axis=0) for k in params}
    shape = arrays[0]['betas']
    assert all(np.array_equal(a['betas'],shape) and np.array_equal(a['faces'],arrays[0]['faces']) for a in arrays)
    fit.update(faces=arrays[0]['faces'],shared_betas=shape,frame_ids=np.asarray(plan['source_frames']),
               w2c=np.broadcast_to(np.eye(4,dtype=np.float32),(len(arrays),4,4)).copy())
    bridge = FrozenSkeletonSurface(fit,np.zeros((10475,3),np.float32),base.original.ASSET)
    replay = bridge.full_model_joints[:,mapping.tolist()].numpy()
    jerr = float(np.max(abs(replay-np.concatenate([a['joints_camera'] for a in arrays],axis=0))))
    verr = float(np.max(abs(bridge.base_vertices.numpy()-np.concatenate([a['vertices_camera'] for a in arrays],axis=0))))
    q = replay@K.T
    uv = q[...,:2]/q[...,2:]
    uerr = float(np.max(abs(uv-np.concatenate([a['joints_projected_native'] for a in arrays],axis=0))))
    assert max(jerr,verr)<2e-5 and uerr<.002
    with np.load(base.original.ASSET,allow_pickle=False) as model:
        face = protected_face_vertices(bridge.model.lbs_weights,bridge.faces,torch.as_tensor(model['lmk_faces_idx']),
                                       torch.as_tensor(model['dynamic_lmk_faces_idx']))
    delta = torch.zeros_like(bridge.initial_offset)
    cloth = torch.isin(bridge.model.lbs_weights.argmax(-1),torch.tensor([3,6,9])) & ~face
    delta[cloth,2]=.001
    assert torch.count_nonzero(delta)>0 and torch.count_nonzero((bridge.vertices(delta)-bridge.base_vertices)[:,face])==0
    off = base.original.OFFICIAL
    init = ast.parse((off/'smplifyx/fit_single_frame.py').read_text(encoding='utf-8'))
    body = next(node for node in init.body if isinstance(node,ast.FunctionDef) and node.name=='fit_single_frame')
    defaults = {arg.arg:value for arg,value in zip(body.args.args[-len(body.args.defaults):],body.args.defaults,strict=True)}
    ids = ast.literal_eval(defaults['init_joints_idxs'])
    assert tuple(ids)==(9,12,2,5)
    upstream,_ = entry.adapted_entry_source(off)
    assert "body_tri_idxs=[[5,12],[2,9]]" in Path(entry.__file__).read_text(encoding='utf-8')
    assert 'pose_embedding = torch.zeros' in upstream
    assert 'joint_weights[:,[1,9,12]]' not in upstream
    atomic(root/'pose-step-audit149.json',dict(frames=len(arrays),fit_sha256=sha(result_path),
        verified_named_body_and_foot_correspondences=23,raw_to_fit_xy_exact_float32=True,complete_K_and_corner_convention_exact=True,
        original_smplx_vertex_max_difference_model_m=verr,original_smplx_joint_max_difference_model_m=jerr,
        original_projection_max_difference_px=uerr,protected_face_zero_displacement=True,
        original_camera_init_body25_ids=list(ids),hip_initialization_scope='Raw hip projections are used in original camera initialization even when their later body loss is disabled. The seated pelvis is partially occluded and hip depth is unverified.',
        original_fit_pose_starts_zero_latent=True,background_camera_or_world_depth_verified=False,
        no_optimizer_updates=True,gaussian_started=False,quality_accepted=False))
    print('VIDEO21_POSE_PATH_OK',root.name,len(arrays),verr,jerr,uerr,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seated',action='store_true')
    main(SEATED if parser.parse_args().seated else base.OUT)
