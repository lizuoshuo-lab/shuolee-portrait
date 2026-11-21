"""One bounded eight-view shared-local-pose SMPLify-X experiment."""
from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import time
import traceback
import json
import fcntl
import signal
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import official_smplifyx_entry123 as entry
import smplify_selected_views124 as adapter
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster
from run_official_gaussians123 import read,save,sha,now,PROJECT,ASSETS

H=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=REC/'cross-subject-sh0-124'
SRC=ROOT/'grass-input'
OUT=ROOT/'grass-selected-view-pose124'
OFFICIAL=REC/'smplifyx-reference-control/official'


def data():
    rows=read(SRC/'data/inputs.json')['records']
    selected=read(SRC/'selection.json')['training_frames']
    indices=[[r['frame'] for r in rows].index(n) for n in selected]
    with np.load(SRC/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    c,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    dst=np.r_[entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    target=np.zeros((len(rows),118,3),np.float32)
    target[:,dst,:2]=obs['keypoints'][:,source];target[:,dst,2]=c[:,source]
    target=np.concatenate([mask_outside_raster(target[i:i+1],r['wh'])[0] for i,r in enumerate(rows)])
    return [rows[i] for i in indices],target[indices]


def root_initialization(rows,reference):
    # Only a conditional orientation seed. Every view's root rotation remains
    # free, and no background translation or old optimized body pose is reused.
    R0=np.array(next(r for r in rows if r['frame']==reference)['w2c'])[:3,:3]
    front=Rotation.from_rotvec([0,np.pi,0]).as_matrix()
    return np.stack([Rotation.from_matrix(np.array(r['w2c'])[:3,:3]@R0.T@front).as_rotvec() for r in rows]).astype(np.float32)


def make_models(n):
    import torch,smplx
    utils,_,_,_=entry.load_official(OFFICIAL)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,
        use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=n,
        num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
        flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
        joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cuda()
    adapter.broadcast_body_pose(body,n)
    loader=entry.module(PROJECT/'src/avatar_core/shuolee_vposer.py','_selected124_vposer_loader')
    loader.MODEL=ASSETS/'vposer/V02_05';loader.CODE=PROJECT/'third_party/human_body_prior'
    vposer,_,_=loader.load_vposer('cuda')
    return body,vposer


def prepare():
    assert not OUT.exists(), 'Use the existing result; do not duplicate this candidate.'
    rows,target=data();assert len(rows)==8
    ranking=read(ROOT/'grass/selection.json')
    reference=min((r for r in ranking['records'] if r['frame'] in [x['frame'] for x in rows]),
        key=lambda r:(r['quality_score'],r['frame']))['frame']
    assert reference=='frame_000046'
    for r in rows:
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
        assert r['K_integer_centers']==rows[0]['K_integer_centers'] and r['wh']==[2704,1520]
    source=[Path(__file__),Path(adapter.__file__),Path(entry.__file__),
        H/'smplify_score_contract123.py',H/'smplify_observation_validity124.py',
        SRC/'data/inputs.json',SRC/'data/observations.npz',SRC/'selection.json',
        SRC/'reference-fit/winner-camera-space.npz',ROOT/'grass/selection.json',
        ROOT/'quality-first-eight124/grass-multiview-joint-check/visual-review.json']
    source += [OFFICIAL/name for name in entry.PINNED]
    source += [ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    OUT.mkdir()
    plan=dict(updated=now(),state='registered',candidate='one_selected_eight_shared_local_body_pose',
        frames=[r['frame'] for r in rows],reference_for_root_seed=reference,
        rows=rows,source_hashes={str(p):sha(p) for p in source},
        hypothesis='The same approximately static body gesture observed from different directions can constrain arm depth jointly, without treating failed camera triangulation as3D truth.',
        shared=['fixed reference32 betas','one32D VPoser latent and decoded63D local body pose'],
        per_view_free=['global body orientation','body placement translation','expression','jaw','eyes','left/right finger PCA'],
        root_initialization='Conditional source rotation relative to the existing highest-ranked selected reference46, with the standard front-facing model rotation. No old optimized pose is used. Root rotations remain free.',
        image_observations='Existing native118 joint mapping, relative score/flip reliability, and outside-raster guard; no new detector, confidence ablation or result-based point removal.',
        sources_of_prior='Pinned upstream VPoser2/L2 and bending prior; collision remains disabled as in the existing local implementation.',
        original_optimizer_and_stages=True,maxiters_per_stage=30,stages=5,
        normalization='Mean of the upstream objective across selected views; shared VPoser prior broadcast before averaging. N=1 objective/gradient parity required.',
        temporal='This finite candidate assumes the local body pose is constant at its8selected times. It does not fit or constrain the other86frames and is not a general dynamic-sequence solver.',
        static_gesture_evidence='12 original images reviewed in the previous fixed-camera diagnostic show the raised hand near the chest/hair. Exact rigid-body equality is an assumption and must be checked after fitting.',
        physical_background_camera_calibration=False,original_K_fixed=True,
        forbidden=['3D triangulated targets from the rejected diagnostic','hardcoded wrist chest contact','old optimized pose initialization','mixing per-frame winners from different runs','threshold/weight sweep'],
        risks=['Local pose sharing can erase actual small motions.','Ambiguous/occluded2D joints may pull the common pose incorrectly.','Free per-view root placement is a body gauge, not scene camera recovery.'],
        budget=dict(candidates=1,selected_views=8,GPU_seconds=600,new_gaussian_steps=0),
        quality_accepted=False,gaussian_admitted=False)
    save(OUT/'plan.json',plan)
    code,delta=adapter.source(OFFICIAL);(OUT/'upstream-interface.diff').write_text(delta)
    _,entrydiff=adapter.make_entry(8,root_initialization(rows,reference));(OUT/'entry-interface.diff').write_text(entrydiff)
    print('SELECTED_VIEW_CANDIDATE_REGISTERED',plan['frames'],flush=True)


def preflight():
    assert not (OUT/'preflight.json').exists() and not (OUT/'progress.json').exists()
    p=read(OUT/'plan.json')
    for path,d in p['source_hashes'].items():assert sha(path)==d,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    torch.set_num_threads(4)
    result=adapter.verify(OFFICIAL)
    body,vposer=make_models(8)
    with np.load(SRC/'reference-fit/winner-camera-space.npz') as z:beta=z['betas'].copy()
    body.reset_params(betas=beta,global_orient=root_initialization(p['rows'],p['reference_for_root_seed']))
    body.betas.requires_grad_(False)
    latent=torch.zeros((1,32),device='cuda',requires_grad=True)
    pose=vposer.decode(latent)['pose_body'].reshape(1,63)
    output=body(body_pose=pose,return_verts=True,return_full_pose=True)
    assert output.vertices.shape==(8,10475,3) and output.joints.shape==(8,118,3)
    assert torch.equal(output.full_pose[:,3:66],pose.expand(8,-1))
    assert np.array_equal(body.betas.detach().cpu().numpy(),np.repeat(beta,8,0))
    root_grad,latent_grad=torch.autograd.grad(output.joints[:,4].square().sum(),(body.global_orient,latent))
    assert root_grad.shape==(8,3) and latent_grad.shape==(1,32)
    assert torch.isfinite(root_grad).all() and torch.isfinite(latent_grad).all()
    assert (root_grad.abs().sum(1)>0).all() and latent_grad.abs().max()>0
    result.update(actual_SMPLX_batch_and_joint_mapping=True,shared_local_pose_exact=True,
        all_eight_root_gradients_present=True,shared_latent_gradient_present=True,
        beta_frozen_shared=True,quality_accepted=False,new_fit_calls=0)
    save(OUT/'preflight.json',result);print('SELECTED_VIEWS_GPU_PREFLIGHT_PASSED',flush=True)


def run():
    assert (OUT/'preflight.json').exists() and not (OUT/'progress.json').exists()
    p=read(OUT/'plan.json')
    for path,d in p['source_hashes'].items():assert sha(path)==d,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('registered600s candidate budget')))
    signal.alarm(600)
    save(OUT/'progress.json',dict(updated=now(),state='running_original_five_stages',pid=os.getpid(),frames=p['frames']))
    save(OUT/'launch.json',dict(updated=now(),pid=os.getpid(),command=sys.argv,proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    started=time.monotonic()
    try:
        import torch
        torch.set_num_threads(4)
        body,vposer=make_models(8);rows,target=data()
        with np.load(SRC/'reference-fit/winner-camera-space.npz') as z:beta=z['betas'].copy()
        method,_=adapter.make_entry(8,root_initialization(rows,p['reference_for_root_seed']))
        raw=np.asarray(Image.open(next(r for r in rows if r['frame']==p['reference_for_root_seed'])['image']).convert('RGB'),dtype=np.float32)/255
        arrays,receipt=method(official_root=OFFICIAL,image=raw,keypoints=target,K_integer_centers=rows[0]['K_integer_centers'],
            body_model=body,vposer2=vposer,result_path=OUT/'official-result.pkl',shared_betas=beta,
            freeze_betas=True,device='cuda',diff_path=OUT/'executed-upstream-interface.diff')
        assert np.array_equal(arrays['betas'],np.repeat(beta,8,0))
        assert arrays['vertices_camera'].shape==(8,10475,3)
        uv=arrays['joints_camera']@arrays['K_integer_centers'].T
        projection_error=float(np.max(abs(uv[:,:,:2]/uv[:,:,2:]-arrays['joints_projected_native'])))
        assert projection_error<.001
        assert arrays['pose_embedding'].shape==(1,32) and arrays['body_pose_axis_angle'].shape==(1,63)
        arrays.update(frame_ids=np.array(p['frames']),observed_native_keypoints=target,
            image_sha256=np.array([r['source_sha256'] for r in rows]),mask_sha256=np.array([r['mask_sha256'] for r in rows]),
            source_frames=np.array([r['source_frame_index'] for r in rows]))
        np.savez_compressed(OUT/'selected-fit.npz',**arrays)
        metrics=[]
        for i,r in enumerate(rows):
            f=OUT/'frames'/r['frame'];f.mkdir(parents=True)
            sharedkeys={'faces','K_integer_centers','K_compute_float32','body_pose_axis_angle','pose_embedding'}
            individual={k:(v if k in sharedkeys else v[i:i+1]) for k,v in arrays.items() if k not in ['frame_ids']}
            individual['frame_id']=np.array(r['frame']);np.savez_compressed(f/'winner-camera-space.npz',**individual)
            error=np.linalg.norm(arrays['joints_projected_native'][i]-target[i,:,:2],axis=-1)
            metric=dict(frame=r['frame'],all118_error_px=error.tolist(),arm_error_px=error[[2,3,4,5,6,7]].tolist())
            for name,ix in [('body',entry.COCO),('face',np.arange(67,118)),('feet',np.arange(19,25))]:
                w=target[i,ix,2]**2;metric[name+'_weighted_rms_px']=float(np.sqrt(np.sum(w*error[ix]**2)/w.sum()))
                metric[name+'_supported_max_px']=float(error[ix][w>0].max())
            metrics.append(metric)
        for path,d in p['source_hashes'].items():assert sha(path)==d,path
        receipt.update(updated=now(),state='fitted_eight_views_review_pending',seconds=time.monotonic()-started,
            frames=p['frames'],single_shared_local_body_pose=True,per_view_root_and_placement=True,
            cross_frame_identity_policy_implemented=True,original_zero_body_latent=True,
            conditional_rotation_used_only_as_seed=True,external_world_camera_used=False,
            independent_camera_calibration=False,projection_replay_max_abs_px=projection_error,
            metrics=metrics,fit_sha256=sha(OUT/'selected-fit.npz'),quality_accepted=False,gaussian_admitted=False)
        save(OUT/'result.json',receipt);save(OUT/'progress.json',dict(updated=now(),state='completed_review_pending',pid=None))
        print('SELECTED_VIEWS_FIT_COMPLETED',receipt['seconds'],flush=True)
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None));raise
    finally:signal.alarm(0)


if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'run':run}[sys.argv[1]]()
