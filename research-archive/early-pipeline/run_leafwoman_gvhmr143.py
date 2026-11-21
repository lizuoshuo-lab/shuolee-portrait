"""022 GVHMR keypoint-only camera-space diagnostic with frozen native K."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from official_smplifyx_entry123 import sha
from replay_leafwoman_arm138 import ARCHIVE, FIT, OFFICIAL, OUT as ARM, WSL, atomic, mounted, read
from update_leafwoman129_state import NATIVE


def local(path):
    return mounted(path) if WSL else Path(path)


OUT=ARM.parent/'gvhmr-keypoints143'
REPO=local('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GVHMR-shuolee')
ASSETS=local('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/gvhmr')
WEIGHT=local('D:/GS-SLAM-Data/models/gvhmr/gvhmr/gvhmr_siga24_release.ckpt')
CONTACTS=local(str(NATIVE/'source-pose-review/source-contacts.json'))
FIT_PLAN=ARM.parent/'fit-plan.json'
VIDEO=local('D:/00/工作/数据集/video22.mp4')
SOURCE_OBS=Path('/opt/gs-slam-lab/shuolee-cross-subject-129/leafwoman/data/observations.npz')
COMMIT='ee960bb6e2ea2d381aa97f08e9b71ef320b624b1'


def prepare():
    assert not WSL and not OUT.exists()
    f=read(FIT_PLAN);s=read(CONTACTS);g=read(CONTACTS.with_name('pose-groups.json'))
    assert sha(CONTACTS)==f['source_contacts_sha256'] and sha(VIDEO)==f['source_video_sha256']==s['video_sha256']
    assert sha(ARCHIVE)==f['previous_fit_sha256']
    assert len(s['records'])==57 and len(f['front_arc'])==24 and not f['rear_arc']
    weight_sha=sha(WEIGHT)
    assert weight_sha==read(WEIGHT.with_name(WEIGHT.name+'.receipt.json'))['sha256']
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    with np.load(ARCHIVE,allow_pickle=False) as z: K=z['K_integer_centers'].tolist()
    paths=[REPO/'hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py',REPO/'hmr4d/network/gvhmr/relative_transformer.py',
           REPO/'hmr4d/utils/geo/hmr_cam.py',REPO/'hmr4d/utils/geo_transform.py',
           ASSETS/'inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz']
    inputs=[dict(frame=r['frame'],source_index=r['source_frame_index'],pts=r['time_seconds'],
                 source_rgb_sha256=r['source_sha256'],box=r['initial_detector_box']) for r in s['records']]
    assert [r['frame'] for r in inputs]==[f'frame_{i:06d}' for i in range(1,58)]
    atomic(OUT/'plan.json',dict(source_sha256=f['source_video_sha256'],contacts_sha256=sha(CONTACTS),
        baseline_sha256=sha(ARCHIVE),observations_sha256=g['observations_npz_sha256'],
        checkpoint_sha256=weight_sha,repo_commit=COMMIT,runner_sha256=sha(__file__),
        implementation_pins={p.as_posix():sha(p) for p in paths},K_integer_centers=K,all_source_samples=inputs,
        comparison_roster=[r['frame'] for r in f['front_arc']],window='all 57 original timestamp-nearest 3fps samples',
        model='GVHMR released weights, original transformer and raw camera-space SMPL-X branch',
        observations='Existing native RTMW COCO17, original raw scores; explicitly hide known unsupported hip/knee/ankle locations. '
                     'These image observations condition the model and are not heldout ground truth.',
        camera_assumption='No admitted 022 camera rotations. Use upstream rotation-6D identity input as a no-rotation control. '
                          'Known moving camera; never treat the control as a VO or world-space estimate.',
        image_features='None, upstream optional branch; do not load HMR2 or ViTPose.',postprocess=False,
        no_previous_fit_used_for_inference=True,gaussian_started=False,quality_accepted=False))
    print('GVHMR022_PLAN_READY',sha(OUT/'plan.json'),flush=True)


def predict():
    assert WSL and not (OUT/'launch.json').exists()
    started=time.monotonic()
    plan=read(OUT/'plan.json')
    assert plan['runner_sha256']==sha(__file__) and plan['contacts_sha256']==sha(CONTACTS)
    assert plan['observations_sha256']==sha(SOURCE_OBS) and plan['checkpoint_sha256']==sha(WEIGHT)
    assert all(sha(mounted(p))==v for p,v in plan['implementation_pins'].items())
    records=plan['all_source_samples']; assert len(records)==57
    with np.load(SOURCE_OBS,allow_pickle=False) as z:
        assert z['frame_ids'].tolist()==[r['frame'] for r in records]
        xy=z['keypoints'][:,:17];confidence=z['scores'][:,:17].copy()
        assert xy.shape==(57,17,2) and confidence.shape==(57,17) and np.isfinite(xy).all()
    # Lower-body material is covered by the long skirt or leaves the image.
    confidence[:,11:]=0
    kp=np.concatenate([xy,confidence[...,None]],axis=2).astype(np.float32)
    assert np.isfinite(kp).all() and (kp[:,5:11,2]>.5).any()
    boxes=np.asarray([r['box'] for r in records],np.float32)
    assert boxes.shape==(57,4) and (boxes[:,2:]>boxes[:,:2]).all()
    atomic(OUT/'launch.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json'),stage='loading_GVHMR_only',
                                  frames=len(records),no_image_frontend=True,gaussian_started=False))
    os.chdir(REPO);sys.path.insert(0,str(REPO))
    import torch
    torch.set_num_threads(1)
    import hmr4d
    hmr4d.PROJ_ROOT=ASSETS
    from hydra import compose,initialize_config_module
    from hydra.utils import instantiate
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy,normalize_kp2d
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.smplx_utils import make_smplx
    import hmr4d.model.gvhmr.gvhmr_pl_demo
    import hmr4d.model.gvhmr.utils.endecoder
    import hmr4d.network.gvhmr.relative_transformer
    register_store_gvhmr()
    with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
        config=compose(config_name='demo',overrides=['video_name=video22_gvhmr143'])
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(WEIGHT)
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)['state_dict']
    model=instantiate(config.model,_recursive_=False).eval()
    model.load_state_dict(state,strict=True);del state
    K=torch.tensor(plan['K_integer_centers'],dtype=torch.float32)
    n=len(records)
    b=get_bbx_xys_from_xyxy(torch.from_numpy(boxes),base_enlarge=1.2)
    av=compute_cam_angvel(torch.eye(3).repeat(n,1,1))
    assert torch.equal(av,torch.tensor([1.,0,0,0,1,0]).repeat(n,1))
    batch=dict(length=torch.tensor([n]),obs=normalize_kp2d(torch.from_numpy(kp),b)[None],bbx_xys=b[None],
               K_fullimg=K[None,None].expand(1,n,3,3),cam_angvel=av[None],f_imgseq=None)
    with torch.inference_mode():
        raw=model.pipeline.forward(batch,train=False,postproc=False,static_cam=False)
        params={name:value[0].cpu() for name,value in raw['pred_smpl_params_incam'].items()}
    del model,raw
    body=make_smplx('supermotion').eval()
    with torch.inference_mode():
        chunks=[body(**{k:v[i:i+4] for k,v in params.items()}) for i in range(0,n,4)]
        vertices=torch.cat([x.vertices for x in chunks]).numpy()
        joints=torch.cat([x.joints for x in chunks]).numpy()
    faces=np.asarray(body.faces)
    assert vertices.shape==(n,10475,3) and np.isfinite(vertices).all() and (vertices[:,:,2]>.05).all()
    assert np.isfinite(joints).all() and not torch.cuda.is_initialized()
    arrays={f'incam_{k}':v.numpy() for k,v in params.items()}
    arrays.update(vertices_camera=vertices,joints_camera=joints,faces=faces,observed_keypoints=kp,
        K_integer_centers=np.asarray(plan['K_integer_centers']),frame_ids=np.asarray([r['frame'] for r in records]))
    path=OUT/'prediction-incam.npz';np.savez_compressed(path,**arrays)
    atomic(OUT/'result.json',dict(plan_sha256=sha(OUT/'plan.json'),prediction_sha256=sha(path),source_samples=n,
        inference='strict GVHMR checkpoint, native COCO17 only, no rotation observation, raw incam CPU',
        seconds=time.monotonic()-started,real_prediction=True,source_video_changed=False,
        world_camera_verified=False,face_hand_fitted=False,gaussian_started=False,quality_accepted=False))
    print('GVHMR022_PREDICTED',vertices.shape,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    (prepare if args.prepare else predict)()
