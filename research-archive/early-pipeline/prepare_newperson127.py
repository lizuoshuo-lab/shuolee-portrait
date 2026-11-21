"""New officewoman intake; reuse native decode/observe without modifying them.

The existing decode entry is horizontal-only. This wrapper changes exactly
its dimension guard, conditional affine and explanatory calibration string.
The vertical affine is the previously registered H0 formula, not a new fit.
"""
import argparse
import ast
import difflib
import inspect
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import prepare_core_subjects124 as core
import audit_output_intrinsics122 as intrinsic
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS

SUBJECT = 'officewoman'
VIDEO_SHA256 = '71d993f2f1e2c1e2ad1f5e3b58c4f351cd375e172f92c3526818dfb4ab380edb'
SOURCE = core.REC/'new-person-intake126/video19-arc/arc-intake.json'
NATIVE_WH = (1520,2704)
PINS = {
    Path(core.__file__): 'e388e65e945f56c2fad0ee355fdb31f05ff85878bc5fc8280cf17b49360b14a8',
    Path(__file__).with_name('preprocess_demo_fps3_123.py'): '9d99f91032cd82ad36eac1b13ddab7db3e7b84cd4fa2ca35cd2328eee4566ac6',
    Path(intrinsic.__file__): 'ce01e6363ed131ca616e2be366fde3ccf2d62ee158f34478aebf3f045ef8c61f',
    PROJECT/'src/shuolee_portrait/observations.py': 'ea7e6998d92583bd44e0e447ee26f91c806350e58df3d56ff4c9449f5caa35b7',
}
core.DATA = Path('/opt/gs-slam-lab/shuolee-cross-subject-127')
core.SUBJECTS = {SUBJECT: ('video19.mp4','019',VIDEO_SHA256)}
original_paths = core.paths


def paths(subject):
    video,data,_,number,digest = original_paths(subject)
    return video,data,core.REC/'officewoman-native127',number,digest


core.paths = paths


def vertical_affine():
    # Existing N4 H0: first YAML K is conditionally treated as rectified
    #4000x3000, height-resized to2704 with half-pixels and ideal centered crop.
    s = NATIVE_WH[1]/3000.
    return intrinsic.crop((4000*s-NATIVE_WH[0])/2,0)@intrinsic.resize(s,s)


def decoder():
    before = inspect.getsource(core.decode)
    code = before
    edits = [
        ('assert wh == (2704,1520)','assert wh == (1520,2704)'),
        ('A = integer_resize_matrix((4000,3000),(2704,2028)); A[1,2] -= 254','A = vertical_affine()'),
        ('Same first YAML camera_matrix, conditional4000x3000 ->2704x2028 half-pixel resize -> centered y crop254. No K fitting or second undistortion.',
         'Same first YAML camera_matrix, conditional4000x3000 -> height2704 half-pixel isotropic resize -> ideal centered1520 horizontal crop. Fractional intermediate width/crop are assumptions, not measured export integers. No K fitting or second undistortion.'),
    ]
    for old,new in edits:
        assert code.count(old)==1,(old,code.count(old))
        code = code.replace(old,new)
    ast.parse(code)
    scope = dict(core.__dict__,vertical_affine=vertical_affine,__file__=__file__)
    exec(compile(code,__file__+':three-exact-vertical-interface-edits','exec'),scope)
    return scope['decode'],before,code


def verify():
    for path,digest in PINS.items():
        assert sha(path)==digest,str(path)
    source = read(SOURCE)
    assert source['source_sha256_before']==source['source_sha256_after']==VIDEO_SHA256
    assert source['source_wh']==list(NATIVE_WH)
    assert source['actual_source_visual_review'] and source['full_circle_visual_coverage_supported']
    assert not source['exact360_degree_or_physical_calibration_accepted']
    return source


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source = verify()
    video,data,report,number,digest = paths(SUBJECT)
    assert sha(video)==digest
    assert not data.exists() and not report.exists(),'Existing intake: inspect, do not repeat.'
    manifest = read(PROJECT/'configs/server-assets.json')['files']
    assets = {}
    for name in ['detection/yolov8n.pt','sam2/sam2.1_hiera_tiny.pt',
                 'rtmw/rtmw-l-384x288.onnx','rtmw/normalization.json']:
        entry = next(x for x in manifest if x['path']==name)
        assert sha(ASSETS/name)==entry['sha256'],name
        assets[name]=entry['sha256']
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_frames','-show_streams','-show_entries','frame=best_effort_timestamp_time:stream','-of','json',str(video)]))
    stream = probe['streams'][0]
    assert (stream['width'],stream['height'])==NATIVE_WH
    assert not stream.get('tags',{}).get('rotate')
    assert all(float(x.get('rotation',0))==0 for x in stream.get('side_data_list',[]))
    assert stream.get('color_transfer') not in ('smpte2084','arib-std-b67')
    times=np.asarray([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    grid,ids=core.sampling_indices(times)
    import cv2
    f=cv2.FileStorage(str(core.CALIB),cv2.FILE_STORAGE_READ)
    K0=f.getNode('camera_matrix').mat()
    assert (int(f.getNode('image_width').real()),int(f.getNode('image_height').real()))==(4000,3000)
    f.release()
    A=vertical_affine();Ki=A@K0;K=Ki.copy();K[:2,2]+=.5
    # Independent direct half-pixel formula at the same conditional scale.
    s=2704/3000.;left=(4000*s-1520)/2
    direct=K0.copy();direct[0,0]*=s;direct[1,1]*=s
    direct[0,2]=s*(K0[0,2]+.5)-.5-left;direct[1,2]=s*(K0[1,2]+.5)-.5
    error=float(np.max(abs(Ki-direct)));assert error<1e-10
    _,before,code=decoder()
    diff_path=Path(__file__).with_suffix('.decode.diff')
    diff_path.write_text(''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),
        fromfile='unchanged-core-horizontal-interface',tofile='three-exact-vertical-interface-edits')))
    doc=dict(updated=now(),state='cpu_configuration_audit_passed',subject=SUBJECT,record_id=number,
        helper_sha256=sha(__file__),video=str(video),video_sha256=digest,source_arc_receipt=str(SOURCE),
        source_arc_receipt_sha256=sha(SOURCE),decoded_frame_count=len(times),sample_fps=3,samples=len(ids),
        source_indices=ids.tolist(),source_times_seconds=times[ids].tolist(),
        maximum_sampling_error_seconds=float(np.max(abs(times[ids]-grid))),native_wh=list(NATIVE_WH),
        source_pixel_format=stream.get('pix_fmt'),source_color_transfer=stream.get('color_transfer'),decoded_pixel_format='rgb24_8bit',
        output_data_root=str(data),output_report_root=str(report),source_pins={str(p):d for p,d in PINS.items()},
        asset_sha256=assets,decode_interface_diff_sha256=sha(diff_path),
        decoder='Existing core.decode with exactly native-size guard, old registered vertical H0 affine, and description adapted; original file unchanged.',
        detector='Unchanged core.observe: YOLOv8n/SAM2 tiny CUDA; RTMW133 ONNX CPU flip-average.',
        K_source=K0.tolist(),source_to_native_integer=A.tolist(),K_integer_centers=Ki.tolist(),K_corner=K.tolist(),
        conditional_affine_equivalence_max_abs=error,
        conditional_crop=dict(ideal_resized_width=4000*s,ideal_left=left,integer_rounding_verified=False),
        conditional_K=True,physical_calibration_accepted=False,second_undistortion=False,camera_estimation=False,
        capture_scope=source['observed_coverage'],full_circle_source_contact_only=True,physical_angles_accepted=False,
        half_circle_training_target=12,full_circle_training_target=24,
        reference_selection='Later source clarity/open-eye and fit-quality selection, not automatically frame1.',
        prior_or_GVHMR_called=False,old_individual_geometry_used=False,geometry_fitted=False,Gaussian_started=False,
        GPU_started=False,quality_accepted=False)
    save(Path(__file__).with_suffix('.audit.json'),doc)
    print(json.dumps(doc,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit','decode','observe'])
    args=parser.parse_args()
    if args.mode=='audit':
        audit()
    else:
        verify()
        if args.mode=='decode':
            decoder()[0](SUBJECT)
        else:
            core.observe(SUBJECT)
