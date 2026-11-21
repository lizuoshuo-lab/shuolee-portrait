"""The user's 3 fps / SfM-resolution / GS-resolution preprocessing contract.

Only decoding, resizing and existing observations run here; no COLMAP solver,
intrinsic optimization or second undistortion is performed.
"""
from pathlib import Path
import os, sys, json, time, subprocess, fcntl, traceback
import numpy as np
import cv2
from PIL import Image
from run_official_gaussians123 import save, read, sha, now, PROJECT, ASSETS
from intake_demo_vhap123 import ROOT, VIDEO

DATA = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')
REPORT = ROOT/'preprocess-fps3'
CALIB = Path('/mnt/d/00/工作/数据集/mono_main_calib.yaml')
SAMPLE_FPS = 3
# Exact branches supplied by the user. Values are image-size divisors.
RESOLUTION_RULES = {(1920,1080):(2,2), (3840,2160):(4,2),
                    (2704,1520):(2,1), (1520,2704):(2,1)}


def integer_resize_matrix(source_wh, target_wh):
    sx, sy = np.asarray(target_wh, float)/np.asarray(source_wh, float)
    return np.array([[sx,0,(sx-1)/2], [0,sy,(sy-1)/2], [0,0,1.]])


def sampling_indices(times):
    times = np.asarray(times, float)
    if len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('Require real strictly increasing decoded PTS.')
    grids = times[0]+np.arange(int(np.floor((times[-1]-times[0])*SAMPLE_FPS))+1)/SAMPLE_FPS
    hi = np.searchsorted(times, grids).clip(0,len(times)-1)
    lo = (hi-1).clip(0,len(times)-1)
    ids = np.where(abs(times[lo]-grids)<=abs(times[hi]-grids),lo,hi)
    if len(set(ids.tolist()))!=len(ids):
        raise ValueError('Source cannot supply distinct samples at the requested 3 fps.')
    return grids, ids


def contract_test():
    for wh, (pose_div,gs_div) in RESOLUTION_RULES.items():
        pose_wh = [x//pose_div for x in wh]; gs_wh = [x//gs_div for x in wh]
        K = np.array([[1100.,0.,wh[0]/2+3.27],[0.,1090.,wh[1]/2-2.71],[0,0,1]])
        T = integer_resize_matrix(wh,pose_wh); G=integer_resize_matrix(wh,gs_wh)
        X=np.array([[.2,-.3,2.],[.7,.4,3.],[-.5,.1,1.4]])
        uv=X@K.T; uv=uv[:,:2]/uv[:,2:]
        projected=X@(T@K).T; projected=projected[:,:2]/projected[:,2:]
        mapped=np.c_[uv,np.ones(len(uv))]@T.T
        assert np.max(abs(projected-mapped[:,:2]))<1e-9
        bridge=G@np.linalg.inv(T)
        gsuv=X@(G@K).T; gsuv=gsuv[:,:2]/gsuv[:,2:]
        assert np.max(abs((np.c_[projected,np.ones(len(uv))]@bridge.T)[:,:2]-gsuv))<1e-9
    grid, ids=sampling_indices(np.arange(470)/24.)
    assert len(ids)==59 and ids.tolist()==list(range(0,465,8))


def main():
    contract_test()
    assert not DATA.exists() and not REPORT.exists(), 'Never overwrite/restart a preprocessing run.'
    intake=read(ROOT/'intake.json'); assert sha(VIDEO)==intake['sha256']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_frames','-show_streams','-show_entries',
        'frame=best_effort_timestamp_time:stream',
        '-of','json',str(VIDEO)]))
    stream=probe['streams'][0]; wh=(stream['width'],stream['height'])
    if wh not in RESOLUTION_RULES: raise ValueError(f'No user-specified resize rule for {wh}.')
    if any(float(x.get('rotation',0))!=0 for x in stream.get('side_data_list',[])):
        raise ValueError('Rotation requires an explicit camera transform, not silent autorotation.')
    if stream.get('color_transfer') in ('smpte2084','arib-std-b67'):
        raise ValueError('HDR conversion has not been authorized/reviewed.')
    times=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    grids, ids=sampling_indices(times)
    pose_div,gs_div=RESOLUTION_RULES[wh]
    # This intake's conditional calibration is established only for the demo.
    assert wh==(2704,1520) and gs_div==1
    pose_wh=[x//pose_div for x in wh]
    fs=cv2.FileStorage(str(CALIB),cv2.FILE_STORAGE_READ)
    K0=fs.getNode('camera_matrix').mat()
    assert (int(fs.getNode('image_width').real()),int(fs.getNode('image_height').real()))==(4000,3000)
    fs.release()
    A=integer_resize_matrix((4000,3000),(2704,2028)); A[1,2]-=254
    Ki=A@K0; K=Ki.copy(); K[:2,2]+=.5
    T=integer_resize_matrix(wh,pose_wh); pose_Ki=T@Ki
    pose_K=pose_Ki.copy(); pose_K[:2,2]+=.5
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    REPORT.mkdir();DATA.mkdir();(DATA/'raw').mkdir();(DATA/'pose').mkdir();(DATA/'data').mkdir();(DATA/'logs').mkdir()
    started=time.monotonic()
    plan=dict(updated=now(),sample_fps=3,sampling='Nearest real source PTS to t0+n/3; no duplicated/interpolated frames',
        video_sha256=intake['sha256'],frame_count=len(ids),source_indices=ids.tolist(),
        video_r_for_colmap=pose_div,video_r_for_gs=gs_div,gs_intrinsic_scale=pose_div/gs_div,
        native_wh=list(wh),pose_wh=pose_wh,gaussian_wh=list(wh),
        user_rules=[dict(input_wh=list(k),video_r_for_colmap=v[0],video_r_for_gs=v[1]) for k,v in RESOLUTION_RULES.items()],
        stages=['3fps shared frame identities / masks / native body-face-hands observations',
                'VHAP SMPL-X fitting including face and both hands; no SMPLify-X fitting',
                'Mesh-bound GaussianAvatars_SMPLX; one Gaussian per triangle, around200k final, max18000 steps'],
        source_script_sha256=sha(__file__),old94_observations_preserved=True,second_undistortion=False,
        COLMAP_used=False,camera_pose_estimated=False,physical_calibration_accepted=False,quality_accepted=False)
    save(REPORT/'plan.json',plan);save(REPORT/'ffprobe.json',probe)
    save(REPORT/'pixel-contract.json',dict(K_source=K0.tolist(),A_source_to_native_integer=A.tolist(),
        K_integer_centers=Ki.tolist(),K_corner=K.tolist(),native_to_pose_integer=T.tolist(),
        pose_to_GS_integer=np.linalg.inv(T).tolist(),pose_K_integer_centers=pose_Ki.tolist(),pose_K_corner=pose_K.tolist(),
        conditional=True,calibration_note='Same conditional resize/crop K; changing sample rate does not verify physical calibration.'))
    try:
        save(REPORT/'progress.json',dict(state='decoding',pid=os.getpid(),updated=now()))
        vf='select='+ '+'.join(f'eq(n\\,{int(i)})' for i in ids)
        with (DATA/'logs/decode.log').open('w') as log:
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-v','warning','-n','-noautorotate','-i',str(VIDEO),
                '-vf',vf,'-pix_fmt','rgb24','-vsync','0',str(DATA/'raw/frame_%06d.png')],stdout=log,stderr=subprocess.STDOUT,check=True)
        paths=sorted((DATA/'raw').glob('frame_*.png'));assert len(paths)==len(ids)
        rows=[];pose_rows=[]
        for slot,(path,idx,grid) in enumerate(zip(paths,ids,grids)):
            rgb=np.array(Image.open(path).convert('RGB'));assert list(rgb.shape[1::-1])==list(wh)
            small=DATA/'pose'/path.name;Image.fromarray(cv2.resize(rgb,pose_wh,interpolation=cv2.INTER_AREA)).save(small)
            row=dict(frame=path.stem,name=path.name,source_frame_index=int(idx),time_seconds=float(times[idx]),
                target_time_seconds=float(grid),sampling_error_seconds=float(times[idx]-grid),image=str(path),
                source_sha256=sha(path),wh=list(wh),K=K.tolist(),K_integer_centers=Ki.tolist(),w2c=np.eye(4).tolist(),
                mask=None,mask_sha256=None,observation_weight=1.,official_prediction_slot=slot,
                camera_kind='fixed_conditional_factory_K',camera_registration=dict(interpolation_used=False,
                pose_scope='Camera-relative body fitting coordinate choice; not estimated scene extrinsics.'))
            rows.append(row)
            pose_rows.append(dict(row,image=str(small),source_sha256=sha(small),wh=pose_wh,
                                  K=pose_K.tolist(),K_integer_centers=pose_Ki.tolist()))
        save(DATA/'decoded.json',dict(records=rows,video_sha256=intake['sha256'],sample_fps=3,decoded_total=len(times)))
        manifest=dict(records=rows,pixel_domain='already_rectified_MP4_native_integer_centers',from_scratch=True,
            formal_calibration_accepted=False,intrinsic_origin='Conditional first YAML K; half-pixel resize and center crop')
        save(DATA/'data/inputs.json',manifest)
        sys.path.insert(0,str(PROJECT/'src'))
        from shuolee_portrait import runtime as rt
        from shuolee_portrait.observations import observe
        import torch
        torch.set_num_threads(4);rt.configure(ASSETS,DATA)
        save(REPORT/'progress.json',dict(state='observations',pid=os.getpid(),updated=now(),detail=str(DATA/'progress.json')))
        observe(DATA,{})
        # Pose-domain observations are an explicit coordinate transform of
        # native observations; never redetect low-resolution face/hand points.
        rows=read(DATA/'data/inputs.json')['records'];(DATA/'pose-masks').mkdir()
        for native,small in zip(rows,pose_rows):
            mask=np.array(Image.open(native['mask']).convert('L'));p=DATA/'pose-masks'/small['name']
            Image.fromarray(cv2.resize(mask,pose_wh,interpolation=cv2.INTER_NEAREST)).save(p)
            small.update(mask=str(p),mask_sha256=sha(p),native_image_sha256=native['source_sha256'],
                observation_origin='Native-resolution detector coordinates transformed with pixel contract')
        save(DATA/'data/pose-inputs.json',dict(manifest,records=pose_rows))
        with np.load(DATA/'data/observations.npz',allow_pickle=False) as z: obs=dict(z)
        pobs=dict(obs);pobs['keypoints']=(obs['keypoints']+.5)/pose_div-.5
        pobs['boxes']=(obs['boxes']+.5)/pose_div-.5;pobs['flip_disagreement']=obs['flip_disagreement']/pose_div
        np.savez_compressed(DATA/'data/pose-observations.npz',**pobs)
        assert np.max(abs((pobs['keypoints']+.5)*pose_div-.5-obs['keypoints']))<.001
        assert [r['source_frame_index'] for r in rows]==[r['source_frame_index'] for r in pose_rows]
        assert sha(VIDEO)==intake['sha256']
        inputs=dict(frames=dict(path=str(DATA/'data/inputs.json'),sha256=sha(DATA/'data/inputs.json')),
            observations=dict(path=str(DATA/'data/observations.npz'),sha256=sha(DATA/'data/observations.npz')),
            camera_length_contract=dict(translation_unit='meters',meters_per_camera_unit=1.,basis='model_conditioned',
                evidence='Camera-relative SMPL-X body; identity extrinsics are not a recovered static-scene trajectory.'))
        save(REPORT/'full-sequence-inputs.json',inputs)
        result=dict(state='complete_pending_visual_review',updated=now(),frames=len(rows),sample_fps=3,
            native_wh=list(wh),pose_wh=pose_wh,Gaussian_wh=list(wh),data_root=str(DATA),
            elapsed_seconds=time.monotonic()-started,max_sample_offset_seconds=float(np.max(abs(times[ids]-grids))),
            face=True,left_hand=True,right_hand=True,source_indices=ids.tolist(),quality_accepted=False)
        save(REPORT/'result.json',result);save(REPORT/'progress.json',dict(state='observations_ready_for_review',pid=None,updated=now()))
        plan0=read(ROOT/'plan.json');plan0['superseded_preprocessing']=dict(frame_stride=5,frames=94,reason='User specifies3fps')
        plan0.update(updated=now(),stages=plan['stages'],active_input_report=str(REPORT),active_data_root=str(DATA),
                     selected_source_frames=ids.tolist(),preprocess_config=dict(sample_fps=3,pose_divisor=pose_div,gs_divisor=gs_div),
                     review_required=['Native masks and body/face/both-hand observations','VHAP same-camera mesh face/hands/body','Bound Gaussian plus current mesh and point distribution'])
        save(ROOT/'plan.json',plan0)
        print(json.dumps(result),flush=True)
    except BaseException as e:
        save(REPORT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),updated=now()))
        save(REPORT/'progress.json',dict(state='failed_preserved',pid=None,updated=now()));raise


if __name__=='__main__':main()
