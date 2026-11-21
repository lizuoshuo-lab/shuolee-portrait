"""CPU-only27-frame interface audit; never a training or quality admission."""
import os
import time
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read,save,sha,now,ASSETS

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
RUN=ROOT/'cross-subject-sh0-124/grass-temporal-refinement'
OUT=RUN/'gs-interface-preview'
POLICY=ROOT/'demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit/fit/surface-cache.npz'

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not OUT.exists(),'Never repeat or overwrite this interface audit.'
    review=read(RUN/'review-027/visual-review.json')
    assert review['actual_visual_review'] and review['checkpoint_passed']
    ids=review['reviewed_frames'];assert len(ids)==27
    files=[RUN/'official-fit/frames'/name/'winner-camera-space.npz' for name in ids]
    assert all(p.exists() for p in files)
    inputs={str(p):sha(p) for p in files+[POLICY,Path(__file__),Path(__file__).with_name('frozen_skeleton_surface123.py')]}
    save(OUT/'plan.json',dict(updated=now(),scope='Only completed27 poses; full94 still required before GS.',
        single_read_only_audit=True,max_cpu_seconds=120,CUDA=False,optimizer_updates=0,
        hypothesis='The existing original SMPL-X clothing bridge preserves new free-placement camera/world geometry.',
        tests=['all27 zero-offset world geometry replay','all27 full-K native projection',
            'all27 original LBS independent shared-offset forward with skeleton fixed','protected face displacement zero'],
        thresholds=dict(world_m=2e-6,native_pixel=.004),sources=inputs,
        no_gaussian_init=True,no_pose_or_mesh_modification=True,quality_accepted=False,gaussian_admitted=False))
    started=time.monotonic()
    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    arrays=[]
    for file in files:
        with np.load(file,allow_pickle=False) as z:arrays.append(dict(z))
    keys=['vertices_world','global_orient','expression','jaw_pose','leye_pose','reye_pose',
          'left_hand_pose','right_hand_pose','body_pose_axis_angle','camera_translation']
    fit={k:np.concatenate([a[k] for a in arrays]) for k in keys}
    fit.update(frame_ids=np.asarray(ids),faces=arrays[0]['faces'],shared_betas=arrays[0]['betas'],
        w2c=np.stack([a['w2c'] for a in arrays]))
    assert all(np.array_equal(a['betas'],fit['shared_betas']) for a in arrays)
    with np.load(POLICY,allow_pickle=False) as z:policy=dict(z)
    assert np.array_equal(fit['faces'],policy['faces'])
    clothing=torch.tensor(policy['clothing_vertices'],dtype=torch.bool)
    protected=torch.tensor(policy['protected_face_vertices'],dtype=torch.bool)
    assert not torch.any(clothing&protected)
    bridge=FrozenSkeletonSurface(fit,np.zeros((10475,3),np.float32),ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    delta=torch.sin(torch.arange(10475*3).reshape(10475,3).float())*.0001*clothing[:,None]
    reports=[]
    with torch.no_grad():
        for i,(frame,a) in enumerate(zip(ids,arrays)):
            assert time.monotonic()-started<120,'CPU budget exceeded; do not extend automatically.'
            predicted=bridge.base_vertices[i].numpy();expected=a['vertices_world'][0]
            E=a['w2c'];K=a['K_integer_centers']
            camera=predicted@E[:3,:3].T+E[:3,3]
            expected_camera=a['vertices_camera'][0]
            assert camera[:,2].min()>0 and expected_camera[:,2].min()>0
            project=lambda x:(x@K.T)[:,:2]/x[:,2:3]
            ve=float(np.abs(predicted-expected).max())
            pe=float(np.abs(project(camera)-project(expected_camera)).max())
            original_v,original_j=bridge.independent_original_forward(delta,i)
            dv=float((bridge.vertices(delta,i)-original_v).abs().max())
            dj=float((bridge.joints(i)-original_j).abs().max())
            face=float((bridge.vertices(delta,i)[protected]-bridge.base_vertices[i][protected]).abs().max())
            assert max(ve,dv,dj)<2e-6 and pe<.004 and face==0,(frame,ve,pe,dv,dj,face)
            reports.append(dict(frame=frame,zero_offset_world_max_abs_m=ve,full_K_native_max_abs_px=pe,
                original_LBS_displacement_max_abs_m=dv,skeletal_joint_max_abs_m=dj,protected_face_displacement_m=face))
    assert not torch.cuda.is_initialized()
    for path,digest in inputs.items():assert sha(path)==digest,path
    save(OUT/'result.json',dict(updated=now(),state='interface_passed_quality_unassessed',frames=27,
        full_roster_frames=94,records=reports,seconds=time.monotonic()-started,
        original_clothing_policy=True,all_input_hashes_unchanged=True,CUDA_initialized=False,
        optimizer_updates=0,new_ply=False,training_started=False,quality_accepted=False,gaussian_admitted=False))
    print('27_FRAME_ORIGINAL_LBS_INTERFACE_PASSED; FULL94_POSE_REVIEW_REQUIRED',flush=True)

if __name__=='__main__':main()
