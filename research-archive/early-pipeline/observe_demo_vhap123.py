"""Use the existing native decoder, SAM2 and RTMW observation pipeline for demo."""
from pathlib import Path
import sys,os,time,traceback,fcntl
import numpy as np
import cv2
from PIL import Image
from run_official_gaussians123 import save,read,sha,now,PROJECT,ASSETS
from intake_demo_vhap123 import ROOT,VIDEO

DATA=Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap')
CALIB=Path('/mnt/d/00/工作/数据集/mono_main_calib.yaml')

def main():
    intake=read(ROOT/'intake.json');assert sha(VIDEO)==intake['sha256']
    assert intake['frames']==470 and intake['native_size']==[2704,1520]
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.video_sfm import extract
    from shuolee_portrait.observations import observe
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cfg=dict(frame_stride=5,min_frames=94,max_frames=94,max_image_size=2704)
    inputs={str(VIDEO):sha(VIDEO),str(CALIB):sha(CALIB)}
    sources={str(PROJECT/'src/shuolee_portrait'/f):sha(PROJECT/'src/shuolee_portrait'/f) for f in ['observations.py','video_sfm.py']}
    plan=dict(updated=now(),state='authorized',human_scope='demo high quality; both face and hands fitted; VHAP mesh before bound Gaussian training',
        subject='demo',original_frames=470,selected_source_frames=list(range(0,470,5)),native_wh=[2704,1520],
        stages=['native RGB and SAM2 masks / RTMW body+face+hands','official SMPLify-X coarse body,face,hands and shared identity','VHAP photometric shared mesh offset initialization','mesh-bound Gaussian training with around200k final points'],
        Gaussian_budget=18000,Gaussian_review_step=1000,one_Gaussian_per_triangle_initialization=True,final_point_target=200000,
        reuse_prior_subject_geometry=False,official_camera_fixed=True,K_calibration_status='Conditional first YAML K at4000x3000; resize2704x2028, center crop254top. No physical subpixel guarantee.',
        optimize_new_K=False,COLMAP_used=False,second_undistortion=False,download_weights=False,
        prior_grass_baseline='Preserved at1000 steps; final199011 points; no quality pass.',
        input_hashes=inputs,source_hashes=sources,data_root=str(DATA),preprocess_config=cfg,
        review_required=['Native mask and face/hands landmarks','SMPLify-X and VHAP mesh same-camera face,both hands,body,feet','No new collapsed or reversed faces; preserve failures','Actual Gaussian and points UI before promotion'],
        risk='Single monocular sequence, hands holding an object, unseen back and conditionalK remain limitations; do not fit invisible fingers to arbitrary predictions.',
        quality_accepted=False)
    save(ROOT/'plan.json',plan)
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    DATA.mkdir(exist_ok=False);(DATA/'data').mkdir();(DATA/'logs').mkdir()
    start=time.monotonic();save(ROOT/'progress.json',dict(state='decoding',pid=os.getpid(),updated=now()))
    try:
        rt.configure(ASSETS,DATA);extract(VIDEO,DATA,cfg)
        fs=cv2.FileStorage(str(CALIB),cv2.FILE_STORAGE_READ)
        K0=fs.getNode('camera_matrix').mat();w0=int(fs.getNode('image_width').real());h0=int(fs.getNode('image_height').real());fs.release()
        assert (w0,h0)==(4000,3000)
        sx=2704/4000;sy=2028/3000;A=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2-254],[0,0,1.]])
        Ki=A@K0;K=Ki.copy();K[:2,2]+=.5
        rows=read(DATA/'decoded.json')['records']
        for slot,r in enumerate(rows):
            assert r['wh']==[2704,1520]
            r.update(image=str(DATA/'raw'/r['name']),K=K.tolist(),K_integer_centers=Ki.tolist(),w2c=np.eye(4).tolist(),
                mask=None,mask_sha256=None,observation_weight=1.,official_prediction_slot=slot,
                camera_kind='fixed_factory_K_opencv_research',camera_registration=dict(interpolation_used=False,pose_scope='camera relative body fitting; no measured scene E'))
        save(DATA/'data/inputs.json',dict(records=rows,pixel_domain='already_rectified_MP4_native_integer_centers',
            intrinsic_origin='First YAML matrix with explicit conditional resize/crop',from_scratch=True,formal_calibration_accepted=False))
        save(ROOT/'pixel-contract.json',dict(K_source=K0.tolist(),A_native_integer_centers=A.tolist(),K_integer_centers=Ki.tolist(),K_corner=K.tolist(),input_size=[4000,3000],resized_size=[2704,2028],crop_xy=[0,254],output_size=[2704,1520],conditional=True,second_undistortion=False))
        save(ROOT/'progress.json',dict(state='observations',pid=os.getpid(),updated=now(),data_progress=str(DATA/'progress.json')))
        observe(DATA,cfg)
        manifest=dict(frames=dict(path=str(DATA/'data/inputs.json'),sha256=sha(DATA/'data/inputs.json')),
            observations=dict(path=str(DATA/'data/observations.npz'),sha256=sha(DATA/'data/observations.npz')),
            camera_length_contract=dict(translation_unit='meters',meters_per_camera_unit=1.,basis='model_conditioned',
                evidence='SMPL-X camera relative body coordinates. Identity extrinsics are a coordinate choice; no independently recovered static-scene trajectory.'))
        save(ROOT/'full-sequence-inputs.json',manifest)
        for p,h in inputs.items():assert sha(p)==h
        save(ROOT/'observations-result.json',dict(state='complete_pending_visual_review',updated=now(),frames=94,body=True,face=True,left_hand=True,right_hand=True,data_root=str(DATA),elapsed_seconds=time.monotonic()-start,quality_accepted=False))
        save(ROOT/'progress.json',dict(state='observations_ready_for_review',pid=None,updated=now()))
    except BaseException as e:
        save(ROOT/'observations-failure.json',dict(error=repr(e),traceback=traceback.format_exc(),updated=now()))
        save(ROOT/'progress.json',dict(state='failed_preserved',pid=None,updated=now()));raise

if __name__=='__main__': main()
