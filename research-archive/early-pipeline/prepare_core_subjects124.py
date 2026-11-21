"""Native input adapter for the remaining named subjects; no geometry reuse.

Reuses the registered 3-fps sampling and pixel-transform functions, and the
existing detector/SAM2/RTMW observation entry. CPU decoding and GPU observation
are separate explicit commands so an active fitter is never interrupted.
"""
import argparse
import os
import sys
import time
import subprocess
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS
from preprocess_demo_fps3_123 import sampling_indices, integer_resize_matrix

REC = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
DATA = Path('/opt/gs-slam-lab/shuolee-cross-subject-124')
CALIB = Path('/mnt/d/00/工作/数据集/mono_main_calib.yaml')
SUBJECTS = {
    'sofa': ('video6.mp4', '006', 'cb11ad325834b399c655371dc5d161e1801a3719b0892da959dc1e57e764572f'),
    'indoor': ('video8.mp4', '008', '871d55a34424c942dab7c51755dc24bc1cf8eb0a649df2b422bbdabc800cd79b'),
}


def paths(subject):
    name, number, digest = SUBJECTS[subject]
    return Path('/mnt/d/00/工作/数据集')/name, DATA/subject, REC/(subject+'-native'), number, digest


def decode(subject):
    import cv2
    video, data, report, number, digest = paths(subject)
    assert not data.exists() and not report.exists(), 'Preserve the existing intake; do not rerun.'
    assert sha(video) == digest
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_frames','-show_streams','-show_entries','frame=best_effort_timestamp_time:stream','-of','json',str(video)]))
    stream = probe['streams'][0]
    wh = (int(stream['width']), int(stream['height']))
    assert all(v > 0 for v in wh)
    assert not stream.get('tags',{}).get('rotate')
    assert all(float(x.get('rotation',0)) == 0 for x in stream.get('side_data_list',[]))
    assert stream.get('color_transfer') not in ('smpte2084','arib-std-b67')
    times = np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    grid, ids = sampling_indices(times)
    f = cv2.FileStorage(str(CALIB),cv2.FILE_STORAGE_READ)
    K0 = f.getNode('camera_matrix').mat()
    assert (int(f.getNode('image_width').real()), int(f.getNode('image_height').real())) == (4000,3000)
    f.release()
    scale = max(wh[0] / 4000, wh[1] / 3000)
    resized = (round(4000 * scale), round(3000 * scale))
    crop = tuple((s - n) // 2 for s, n in zip(resized, wh, strict=True))
    A = integer_resize_matrix((4000,3000), resized)
    A[:2,2] -= crop
    Ki = A @ K0
    K = Ki.copy(); K[:2,2] += .5
    report.mkdir(parents=True)
    data.mkdir(parents=True)
    for name in ('raw','data','logs'):
        (data/name).mkdir()
    started = time.monotonic()
    plan = dict(updated=now(),subject=subject,record_id=number,source_video=str(video),video_sha256=digest,
        user_scope='Current SMPLify-X to SH0 mesh-bound Gaussian multi-person goal; use existing local video and weights.',
        source_frames=len(times),selected_frames=len(ids),sample_fps=3,selected_source_indices=ids.tolist(),
        native_wh=list(wh),gaussian_wh=list(wh),sampling='Existing nearest real PTS to t0+n/3; no duplicate or interpolated frame.',
        reused_sampler=str(Path(__file__).with_name('preprocess_demo_fps3_123.py')),
        sampler_sha256=sha(Path(__file__).with_name('preprocess_demo_fps3_123.py')),
        source_script_sha256=sha(__file__),calibration_sha256=sha(CALIB),
        calibration='First YAML camera matrix with conditional half-pixel resize and integer center crop.',
        conditional_rectified_size=[4000, 3000], conditional_resized_size=list(resized), conditional_crop_xy=list(crop),
        full_K_integer_centers=Ki.tolist(),full_K_corner=K.tolist(),source_to_native_integer=A.tolist(),
        camera_scope='Camera-relative SMPL-X fitting; identity E is not a reconstructed static camera trajectory.',
        original_smaller_inputs_preserved=True,old_individual_geometry_used=False,physical_calibration_accepted=False,
        cold_start_speed_claim=False,comparison_scope='New3fps intake uses the prior user rule. Winterhat temporal comparison retains its frozen stride5 roster; this is not an identical end-to-end timing comparison.',
        observation_source=str(PROJECT/'src/shuolee_portrait/observations.py'),
        observation_source_sha256=sha(PROJECT/'src/shuolee_portrait/observations.py'),
        fitting='Not launched: first native mask/landmark visibility review, then independently fitted shared shape and temporal pose.',
        risks=['Seated/occluded legs are not empty-body negative observations.','Mask scores and detector landmarks are not material3D truth.'],
        no_new_weights=True,no_gaussian_training=True,quality_accepted=False,decode_budget_seconds=240)
    save(report/'plan.json',plan)
    save(report/'ffprobe.json',probe)
    save(report/'progress.json',dict(state='cpu_decoding',pid=os.getpid(),updated=now()))
    vf = 'select=' + '+'.join('eq(n\\,%d)'%int(i) for i in ids)
    with (data/'logs/decode.log').open('w') as log:
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-v','warning','-n','-noautorotate','-i',str(video),
            '-vf',vf,'-pix_fmt','rgb24','-vsync','0',str(data/'raw/frame_%06d.png')],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    files = sorted((data/'raw').glob('frame_*.png'))
    assert len(files) == len(ids)
    rows = []
    for slot,(file,idx,target) in enumerate(zip(files,ids,grid)):
        assert Image.open(file).size == wh
        rows.append(dict(frame=file.stem,name=file.name,source_frame_index=int(idx),time_seconds=float(times[idx]),
            target_time_seconds=float(target),sampling_error_seconds=float(times[idx]-target),source_sha256=sha(file),
            image=str(file),wh=list(wh),K=K.tolist(),K_integer_centers=Ki.tolist(),w2c=np.eye(4).tolist(),
            mask=None,mask_sha256=None,official_prediction_slot=slot,observation_weight=1.,
            camera_kind='camera_relative_fixedK_body_motion',camera_registration=dict(static_trajectory_claim=False,interpolation_used=False)))
    save(data/'data/inputs.json',dict(records=rows,pixel_domain='Already rectified native MP4 pixels',
        video_sha256=digest,sample_fps=3,conditional_K=True,formal_calibration_accepted=False,from_scratch=True))
    save(data/'decoded.json',dict(records=rows,video_sha256=digest,sample_fps=3,decoded_total=len(times)))
    save(report/'result.json',dict(state='native_decode_complete_observations_pending',updated=now(),frames=len(rows),
        seconds=time.monotonic()-started,data_root=str(data),video_sha256=digest,
        maximum_sampling_error_seconds=float(np.max(np.abs(times[ids]-grid))),
        input_manifest_sha256=sha(data/'data/inputs.json'),native_pixels=True,
        observations_ready=False,geometry_fitted=False,gaussian_started=False,quality_accepted=False))
    save(report/'progress.json',dict(state='native_decode_complete_observations_pending',pid=None,updated=now()))
    print('CORE_NATIVE_READY',subject,len(rows),flush=True)


def observe(subject):
    import fcntl
    video, data, report, number, digest = paths(subject)
    plan = read(report/'plan.json')
    assert read(report/'result.json')['state']=='native_decode_complete_observations_pending'
    assert not (data/'data/masks').exists() and not (data/'data/observations.npz').exists()
    assert sha(video)==digest and sha(PROJECT/'src/shuolee_portrait/observations.py')==plan['observation_source_sha256']
    # These are existing frozen assets, not a download path.
    manifest = read(PROJECT/'configs/server-assets.json')['files']
    required = ['detection/yolov8n.pt','sam2/sam2.1_hiera_tiny.pt','rtmw/rtmw-l-384x288.onnx','rtmw/normalization.json']
    for rel in required:
        entry = next(x for x in manifest if x['path']==rel)
        assert sha(ASSETS/rel)==entry['sha256'],rel
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(report/'progress.json',dict(state='native_observations',pid=os.getpid(),updated=now()))
        sys.path.insert(0,str(PROJECT/'src'))
        from shuolee_portrait import runtime as rt
        from shuolee_portrait.observations import observe as original_observe
        rt.configure(ASSETS,data)
        started = time.monotonic()
        original_observe(data,{})
        result = read(report/'result.json')
        result.update(state='observations_complete_review_pending',updated=now(),observations_ready=True,
            observation_seconds=time.monotonic()-started,observation_manifest_sha256=sha(data/'data/inputs.json'),
            observations_sha256=sha(data/'data/observations.npz'))
        save(report/'result.json',result)
        save(report/'progress.json',dict(state='observations_complete_review_pending',pid=None,updated=now()))
        print('CORE_OBSERVATIONS_READY',subject,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['decode','observe']);p.add_argument('subject',choices=SUBJECTS)
    args=p.parse_args()
    try:
        (decode if args.mode=='decode' else observe)(args.subject)
    except Exception as error:
        import traceback
        report=paths(args.subject)[2]
        if report.exists():
            save(report/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),mode=args.mode))
        raise
