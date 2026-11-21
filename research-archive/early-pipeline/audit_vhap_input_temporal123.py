"""Read-only input and temporal contract audit; no new fitting or cameras."""
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from vhap_demo_adapter123 import ROOT,DATA,REPO,COMMIT
from run_official_gaussians123 import read,save,sha,now


def stat(x):
    x=np.asarray(x);return dict(median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))


def source(name,needle):
    p=REPO/name;lines=p.read_text(encoding='utf-8-sig').splitlines()
    return dict(path=str(p),sha256=sha(p),line=next(i+1 for i,s in enumerate(lines) if needle in s))


def main():
    out=ROOT/'vhap-input-contract.json'
    assert not out.exists(),'Read the existing audit rather than replacing it.'
    rows=read(DATA/'data/inputs.json')['records'];times=np.array([r['time_seconds'] for r in rows]);dt=np.diff(times)
    assert len(rows)==59 and (dt>0).all()
    with np.load(DATA/'data/observations.npz',allow_pickle=False) as z:
        obs=dict(z)
    assert list(obs['frame_ids'])==[r['frame'] for r in rows]
    all_identity=bool(np.allclose(np.asarray([r['w2c'] for r in rows]),np.eye(4)))
    tracked=subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',COMMIT],text=True).splitlines()
    temporal={}
    for name in ['vhap-selected-reference','vhap-selected-reference-white']:
        p=ROOT/name/'global-epoch-30/tracked-smplx-params.npz'
        with np.load(p,allow_pickle=False) as z:
            root=z['global_orient'];trans=z['transl'];latent=z['pose_embedding'];frames=z['frame_ids']
        assert list(frames)==list(obs['frame_ids'])
        rot=Rotation.from_rotvec(root)
        deg=(rot[1:]*rot[:-1].inv()).magnitude()*180/np.pi
        speed=np.linalg.norm(np.diff(trans,axis=0),axis=1)/dt
        latent_step=np.linalg.norm(np.diff(latent,axis=0),axis=1)
        temporal[name]=dict(checkpoint_sha256=sha(p),root_adjacent_degrees=stat(deg),
            root_speed_degrees_per_second=stat(deg/dt),translation_speed_model_m_per_second=stat(speed),
            vposer_latent_adjacent_L2=stat(latent_step),
            largest_root_change_pair=[int(deg.argmax()),int(deg.argmax()+1)],
            raw_root_delta_degrees=deg.tolist(),raw_translation_speed=speed.tolist(),
            limits='Camera-relative motion; neither recovered world motion nor physical motion ground truth. Latent distance is not joint-angle distance.')
    result=dict(updated=now(),state='read_only_input_audit_complete',commit=COMMIT,optimizer_updates=0,
        observed_inputs=dict(frames=59,image_wh=[2704,1520],source_timestamps=True,dt_seconds=stat(dt),
            keypoint_array_shape=list(obs['keypoints'].shape),confidence_array_shape=list(obs['scores'].shape),
            image_derived_3d_points=False,estimated_scene_extrinsics=False,all_w2c_identity=all_identity,
            views_per_timestep=1,timesteps=59,mask_present=True,
            camera_mode='Camera-relative body coordinates with fixed conditional K; identity w2c is a gauge choice, not scene calibration.',
            pose_initialization='Zero VPoser latent and bbox-depth seed; selected frame32 is optimized first, not an image-to-3D network estimate.'),
        contract=[
            dict(item='RGB frames',required='Real time/view identities and correct pixel domain',provided='59 real frames,3fps,native2704x1520',status='present'),
            dict(item='Foreground mask',required='Alpha aligned to each image, consistent background compositing',provided='SAM2 masks; hats/shoes/held-device and occlusion need review',status='present_not_ground_truth'),
            dict(item='2D landmarks',required='Face/body/hands in pixel coordinates with correct ordering and confidence',provided='RTMW133 adapted to BODY25/face68/hands21; innerface51 used',status='adapted_not_original_detector'),
            dict(item='Intrinsics',required='K for the exact resized/cropped image and coordinate convention',provided='Conditional first-YAML K transformed to native video pixels',status='numerically_checked_physical_export_unconfirmed'),
            dict(item='Multi-view extrinsics',required='For the calibrated SMPLX multi-camera recipe: all views in one world coordinate frame',provided='No estimated scene extrinsics; identity camera-relative gauge',status='not_multiview_calibrated'),
            dict(item='View/time grouping',required='Same-time camera views share that timestep pose; different times may have different poses',provided='One image per timestep,59 independently posed timesteps with shared shape/offset',status='monocular_adaptation_not_synchronous_multiview'),
            dict(item='Model and priors',required='SMPL-X topology/UV/weights; selected VPoser and hand-prior assets',provided='Official SMPL-X10475/20908,UV,VPoser32,handPCA12',status='present'),
            dict(item='Initial mesh/3D points',required='Pre-fitted parameters optional; independent 3D landmark/point-cloud file not required by this tracker',provided='No image-derived3D observations; model3D joints are predictions',status='optional_not_provided')],
        verified_sources={
            'fork_multiview_description':source('README.md','full-body SMPLX tracking'),
            'entry':source('vhap/track_nersemble_smplx.py','NersembleV2ColmapTrackingConfig'),
            'camera_reader':source('vhap/data/nersemble_dataset.py','world_2_cam'),
            'same_time_multiview_batch':source('vhap/data/video_dataset.py','def getitem_by_timestep'),
            'landmark_input_contract':source('vhap/model/tracker_smplx.py','All GT landmarks are expected'),
            'optional_existing_parameters':source('vhap/model/tracker_smplx.py','def load_from_tracked_smplx_params'),
            'neutral_seed':source('vhap/model/tracker_smplx.py','def init_params'),
            'temporal_pose':source('vhap/model/tracker_smplx.py','def compute_pose_smooth_energy'),
            'temporal_joints':source('vhap/model/tracker_smplx.py','def compute_joint_smooth_energy')},
        missing_at_pinned_commit=[s for s in ['vhap/data/nersemble_v2_colmap_dataset.py','vhap/data/nerfstudio_dataset.py','smplx/smplx/body_models_modified.py'] if s not in tracked],
        original_vhap_scope='Official ShenhanQian/VHAP supports monocular FLAME head tracking. This does not establish full-body monocular validity of the jmaraval fork.',
        official_reference_urls=['https://github.com/ShenhanQian/VHAP/blob/main/doc/monocular.md','https://github.com/jmaraval/VHAP_smplx'],
        temporal_implementation='Original adjacent predecessor with detach; first differences of axis-angle/translation/VPoser latent/face/hand parameters; no explicit delta-time weighting or joint acceleration term.',
        temporal_read_only=temporal,
        causal_limits=['Missing reliable spatial constraints is not a demonstrated unique cause.',
            'A camera-relative identity transform is legitimate for monocular fitting, but provides no multiview parallax constraint.',
            'Different-time articulated body landmarks cannot be triangulated as static material points without a motion model.',
            'Do not invent 3D observations by recycling the same fitted SMPL-X joints.',
            'A smooth sequence can still be systematically misaligned.'],
        next='Resolve the monocular input/pose-initialization contract in the existing repository before another full fitting or Gaussian run.',
        new_fit_authorized_by_this_audit=False,quality_accepted=False)
    save(out,result)
    print(dict(path=str(out),views_per_timestep=1,image_derived_3d_points=False,scene_extrinsics=False,temporal=temporal))


if __name__=='__main__':main()
