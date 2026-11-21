"""Read-only CPU bridge audit from fitted camera space to a labelled body gauge.

The derived cameras represent fitted relative body/view motion. They are not
background-SfM camera estimates or independent physical calibration.
"""
from pathlib import Path
import argparse,os,time
import numpy as np
from scipy.spatial.transform import Rotation
from run_official_gaussians123 import ASSETS,read,save,sha,now


def main(source,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not out.exists()
    plan=read(source/'plan.json');receipt=read(source/'result.json')
    assert sha(source/'selected-fit.npz')==receipt['fit_sha256']
    rows=plan['rows'];ids=[r['frame'] for r in rows]
    files=[source/'frames'/f/'winner-camera-space.npz' for f in ids]
    helper=Path(__file__).resolve().parent
    root=source.parent
    policy=root.parent/'demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit/fit/surface-cache.npz'
    inputs=files+[source/'plan.json',source/'result.json',Path(__file__),
        helper/'frozen_skeleton_surface123.py',policy,ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    digests={str(p):sha(p) for p in inputs}
    out.mkdir();save(out/'plan.json',dict(updated=now(),source=str(source),frames=ids,
        scope='CPU coordinate and original LBS handoff only; zero optimizer updates, no quality admission.',
        transform='camera=R*world+(J0+t-R*J0), with world the same posed SMPL-X model evaluated at zero global rotation. J0 is the original model pelvis for the same shape/expression.',
        camera_meaning='Relative body-view gauge from fitted body orientation/placement; not measured scene trajectory.',
        budgets=dict(cpu_seconds=120,passes=1),sources=digests,
        thresholds=dict(vertex_m=2e-6,native_projection_px=.004),gaussian_admitted=False))
    import torch,smplx
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    started=time.monotonic()
    model=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,
        num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
        flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
        dtype=torch.float32).eval().requires_grad_(False)
    arrays=[];world=[];cameras=[];pivots=[];checks=[]
    names=('global_orient','expression','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose')
    for row,path in zip(rows,files):
        assert time.monotonic()-started<120
        with np.load(path,allow_pickle=False) as z:a=dict(z)
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        assert np.array_equal(a['K_integer_centers'],row['K_integer_centers'])
        assert np.array_equal(a['faces'],model.faces)
        params={k:torch.as_tensor(a[k]) for k in names}
        params.update(global_orient=torch.zeros(1,3),body_pose=torch.as_tensor(a['body_pose_axis_angle']),betas=torch.as_tensor(a['betas']))
        with torch.no_grad():pred=model(**params,return_verts=True)
        v=pred.vertices[0].numpy();j0=pred.joints[0,0].numpy()
        R=Rotation.from_rotvec(a['global_orient'][0]).as_matrix()
        E=np.eye(4);E[:3,:3]=R;E[:3,3]=j0+a['camera_translation'][0]-R@j0
        vc=v@R.T+E[:3,3];expected=a['vertices_camera'][0]
        K=a['K_integer_centers'];project=lambda x:(x@K.T)[:,:2]/x[:,2:3]
        ve=float(np.max(abs(vc-expected)));pe=float(np.max(abs(project(vc)-project(expected))))
        assert ve<2e-6 and pe<.004,(row['frame'],ve,pe)
        assert abs(np.linalg.det(R)-1)<1e-8
        arrays.append(a);world.append(v);cameras.append(E);pivots.append(j0)
        checks.append(dict(frame=row['frame'],camera_vertex_max_abs_m=ve,native_projection_max_abs_px=pe))
    fit={k:np.concatenate([a[k] for a in arrays]) for k in names+('body_pose_axis_angle','pose_embedding','camera_translation','vertices_camera','joints_camera','joints_projected_native','observed_native_keypoints')}
    assert all(np.array_equal(a['betas'],arrays[0]['betas']) for a in arrays)
    fit.update(frame_ids=np.asarray(ids),faces=arrays[0]['faces'],shared_betas=arrays[0]['betas'],betas=arrays[0]['betas'],
        vertices_world=np.stack(world),w2c=np.stack(cameras),source_w2c=np.asarray([r['w2c'] for r in rows]),
        K_integer_centers=np.asarray([r['K_integer_centers'] for r in rows]),
        root_rotation=np.tile(np.eye(3,dtype=np.float32),(len(ids),1,1)),world_translation=np.zeros((len(ids),3),np.float32),
        root_pivot_model=np.stack(pivots),static_offset=np.zeros((1,10475,3),np.float32),
        image_sha256=np.asarray([r['source_sha256'] for r in rows]),mask_sha256=np.asarray([r['mask_sha256'] for r in rows]))
    bridge=FrozenSkeletonSurface(fit,np.zeros((10475,3),np.float32),ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    with np.load(policy,allow_pickle=False) as z:
        clothing=torch.tensor(z['clothing_vertices'],dtype=torch.bool)
        protected=torch.tensor(z['protected_face_vertices'],dtype=torch.bool)
        assert np.array_equal(z['faces'],fit['faces'])
    assert not torch.any(clothing&protected)
    delta=.0001*torch.sin(torch.arange(10475*3).reshape(10475,3).float())*clothing[:,None]
    with torch.no_grad():
        for i,c in enumerate(checks):
            assert time.monotonic()-started<120
            base=float((bridge.base_vertices[i]-torch.tensor(world[i])).abs().max())
            ov,oj=bridge.independent_original_forward(delta,i)
            dv=float((bridge.vertices(delta,i)-ov).abs().max())
            dj=float((bridge.joints(i)-oj).abs().max())
            face=float((bridge.vertices(delta,i)[protected]-bridge.base_vertices[i][protected]).abs().max())
            assert max(base,dv,dj)<2e-6 and face==0,(ids[i],base,dv,dj,face)
            c.update(base_world_replay_m=base,original_LBS_offset_max_abs_m=dv,
                original_skeleton_max_abs_m=dj,protected_face_offset_m=face)
    for path,digest in digests.items():assert sha(path)==digest,path
    assert not torch.cuda.is_initialized()
    # Store diagnostic arrays only; this folder has no training-ready receipt.
    np.savez_compressed(out/'body-gauge-interface.npz',**fit)
    save(out/'result.json',dict(updated=now(),state='coordinate_interface_passed_quality_unassessed',
        frames=len(ids),records=checks,seconds=time.monotonic()-started,CUDA_initialized=False,
        archive_sha256=sha(out/'body-gauge-interface.npz'),source_files_unchanged=True,
        source_fit=str(source),body_relative_gauge=True,independent_camera_calibration=False,
        fitted_pose_quality_accepted=False,gaussian_admitted=False,optimizer_updates=0,new_PLY=False))
    print('BODY_GAUGE_ORIGINAL_LBS_INTERFACE_PASSED',len(ids),round(time.monotonic()-started,3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    x=p.parse_args();main(x.source,x.out)
