"""Exact pinned 022 CPU SMPL-X replay using saved decoded body angles."""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np

import official_smplifyx_entry123 as official
from update_leafwoman129_state import FIT, ROOT, T, atomic, read

WSL = os.name == 'posix' and 'microsoft' in os.uname().release.lower()


def mounted(path):
    drive, rest = str(path).replace('\\', '/').split(':/', 1)
    return Path('/mnt') / drive.lower() / rest


if WSL:
    FIT, ROOT, T = (mounted(p) for p in (FIT, ROOT, T))
OUT=ROOT/'leafwoman-fit40-132/arm-refine138'
ARCHIVE=FIT/'selected-fit.npz'
OFFICIAL=ROOT.parent/'smplifyx-reference-control/official'
SMPLX_ROOT=T/'shuolee/current/novel-view-face/head-structure/private-runtime/python-deps/smplx-official'
ASSET=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
if WSL:
    ASSET = mounted(ASSET)
    SMPLX_ROOT = Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/lib/python3.10/site-packages')
PARAMS=('betas','global_orient','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression')


def register():
    p=read(FIT/'plan.json')['source_hashes'];bound={}
    for short,local in [('smplx/body_models.py',SMPLX_ROOT/'smplx/body_models.py'),
                        ('smplx/lbs.py',SMPLX_ROOT/'smplx/lbs.py'),('SMPLX_NEUTRAL.npz',ASSET)]:
        digest,={v for k,v in p.items() if k.endswith(short)}
        assert official.sha(local)==digest;bound[str(local)]=digest
    bound[str(ARCHIVE)]=official.sha(ARCHIVE);assert bound[str(ARCHIVE)]==read(FIT/'result.json')['fit_sha256']
    with np.load(ARCHIVE,allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();valid=np.flatnonzero((z['observed_native_keypoints'][:,2:8,2]>0).all(axis=1))
    chosen=[int(valid[x]) for x in (0,len(valid)//2,-1)]
    assert [ids[i] for i in chosen]==['frame_000007','frame_000020','frame_000033']
    official.adapted_entry_source(OFFICIAL)
    plan=dict(updated=datetime.now(timezone.utc).isoformat(),source_hashes=bound,full_arm_supported_frames=[ids[i] for i in valid],
        slots=chosen,frames=[ids[i] for i in chosen],frame_selection='First/middle/last chronologically among all six positive saved arm observations; no residual ranking.',
        zero_replay_limits=dict(max_mesh_component_model_units=2e-5,max_joint_projection_component_px=.01),
        followup='One diagnostic candidate only, original stage3/4, bounded 30 iterations each and 3 frames. Reuse the same decoded baseline and optimize VPoser latent only for collars/shoulders/elbows; freeze root, translation, betas, face, hands, neck and legs.',
        hand_2d_weights=[0]*5,old_shape_is_not_independent_truth=True,quality_accepted=False,optimization_admitted=False)
    assert not (OUT/'plan.json').exists();atomic(OUT/'plan.json',plan)
    return plan


def replay():
    plan=read(OUT/'plan.json') if (OUT/'plan.json').exists() else register()
    def validate_inputs():
        planned = plan['source_hashes']
        assert all(official.sha(mounted(k) if WSL and k[1:2]==':' else k)==v for k,v in planned.items())
        for rel in ('smplx/body_models.py','smplx/lbs.py'):
            expected, = {v for k,v in planned.items() if k.replace('\\','/').endswith(rel)}
            assert official.sha(SMPLX_ROOT/rel)==expected

    validate_inputs()
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    sys.path.insert(0,str(SMPLX_ROOT))
    import smplx
    from smplx import body_models,lbs
    assert Path(body_models.__file__).resolve()==(SMPLX_ROOT/'smplx/body_models.py').resolve()
    assert Path(lbs.__file__).resolve()==(SMPLX_ROOT/'smplx/lbs.py').resolve()
    utils=official.module(OFFICIAL/'smplifyx/utils.py','_leaf138_exact_utils')
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(ASSET),batch_size=1,num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
        flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
        joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).eval().requires_grad_(False)
    rows=[]
    with np.load(ARCHIVE,allow_pickle=False) as z:
        saved={k:z[k] for k in (*PARAMS,'camera_translation','body_pose_axis_angle','vertices_camera','joints_camera',
                                 'joints_projected_native','observed_native_keypoints','K_compute_float32','faces')}
    assert np.array_equal(body.faces,saved['faces'])
    K=saved['K_compute_float32'];arm=np.arange(2,8)
    with torch.no_grad():
        for i,name in zip(plan['slots'],plan['frames']):
            body.reset_params(**{k:saved[k] if k=='betas' else saved[k][i:i+1] for k in PARAMS})
            out=body(body_pose=torch.as_tensor(saved['body_pose_axis_angle']),return_verts=True,return_full_pose=True)
            t=saved['camera_translation'][i]
            verts=out.vertices[0].numpy()+t;joints=out.joints[0].numpy()+t
            q=joints@K.T;uv=q[:,:2]/q[:,2:3]
            vm=float(abs(verts-saved['vertices_camera'][i]).max());jm=float(abs(joints-saved['joints_camera'][i]).max())
            pm=float(abs(uv-saved['joints_projected_native'][i]).max())
            assert max(vm,jm)<2e-5 and pm<.01,(name,vm,jm,pm)
            detector=saved['observed_native_keypoints'][i,arm];assert (detector[:,2]>0).all()
            e=np.linalg.norm(uv[arm]-detector[:,:2],axis=1)
            rows.append(dict(frame=name,mesh_max_abs_model_units=vm,joints_max_abs_model_units=jm,
                projection_max_abs_px=pm,baseline_arm_detector_errors_px=e.tolist(),
                baseline_weighted_arm_RMS_px=float(np.sqrt(np.average(e**2,weights=detector[:,2]**2))),
                wrist_minus_shoulder_camera_z_model_units=(joints[[4,7],2]-joints[[2,5],2]).tolist()))
    assert not torch.cuda.is_initialized()
    validate_inputs()
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='pinned_SMPLX_zero_replay_passed',
        plan_sha256=official.sha(OUT/'plan.json'),rows=rows,torch_version=torch.__version__,decoder_loaded=False,
        uses_saved_axis_angles_only=True,optimizer_steps=0,physical_pose_accepted=False,runtime='WSL' if WSL else 'Windows',
        new_camera_written=False,gaussian_started=False,quality_accepted=False)
    atomic(OUT/'zero-replay-result.json',result)
    print('VIDEO22_ARM_EXACT_ZERO_REPLAY',[(r['frame'],r['projection_max_abs_px']) for r in rows],flush=True)


if __name__=='__main__':replay()
