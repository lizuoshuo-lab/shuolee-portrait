"""Freeze the first new-video camera diagnostic before running any solver."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
TOP = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O = TOP / 'current/new-capture-118'
LOG = Path('D:/00/工作日志/2026-10-26/新视频118')
intake = json.loads(Path('D:/00/工作日志/2026-10-26/新视频标定核验/intake.json').read_text(encoding='utf-8'))
video = next(x for x in intake['videos'] if 'df776c03' in x['path'])
cal = intake['calibration']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

assert not O.exists(), 'A prior 118 exists; inspect instead of overwriting.'
assert sha(video['path']) == video['sha256'] and sha(cal['path']) == cal['sha256']
src_w, src_h = cal['image_wh']
video_w, video_h = video['encoded_wh']
decoded_w, decoded_h = 1600, round(video_h * 1600 / video_w)
crop_h = src_w * video_h / video_w
offset_y = (src_h - crop_h) / 2
sx, sy = decoded_w / src_w, decoded_h / crop_h
K = cal['K']
fx, fy = K[0][0] * sx, K[1][1] * sy
cx, cy = K[0][2] * sx, (K[1][2] - offset_y) * sy
initial = dict(kind='center_crop_unverified_focal_prior', decoded_wh=[decoded_w, decoded_h],
    sensor_wh=cal['image_wh'], encoded_wh=video['encoded_wh'],
    center_crop_xywh=[0., offset_y, float(src_w), crop_h],
    derived_pinhole_K=[[fx,0,cx],[0,fy,cy],[0,0,1]],
    simple_radial_params=[(fx+fy)/2,cx,cy,0.],
    calibration_sha256=cal['sha256'], physical_calibration_validated=False,
    assumption='Centered landscape crop and resize are only an initialization hypothesis. FISHEYE_8 semantics and export stabilization remain unknown.',
    raw_fisheye_coefficients_applied=False)
cfg=json.loads((P/'configs/server.json').read_text(encoding='utf-8'))
cfg.update(frame_stride=10,max_image_size=1600,max_frames=100,
    camera_initialization=initial,
    sfm_holdout=dict(max_features=2000,temporal_hops=[1,4],ratio=.75,fraction=.2,max_tracks=256,min_tracks=12,alias_radius_px=8.))
save(P/'configs/new-capture-118.json',cfg)
now=datetime.now(timezone.utc).isoformat()
plan=dict(created=now,round='new-capture-118',input=video,calibration=cal,
    source=str(P),source_commit=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),
    runtime='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python',
    assets='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',
    run_root='/opt/gs-slam-lab/shuolee-new-capture-118',
    current_stage='camera_and_material_preflight',
    raw_frames=video['frames'],preflight_stride=10,preflight_frames=(video['frames']+9)//10,
    preflight_not_final_training_frame_selection=True,
    goal='Fresh static clothed portrait reconstruction from the newly supplied video; no historical subject, fitted mesh, camera or Gaussian reuse.',
    competing_hypotheses=[
      'The exported landscape video can be represented by one shared camera model after estimating its apparent distortion.',
      'Per-frame stabilization or lens-model mismatch violates the shared-intrinsic assumption.',
      'Subject motion or material ambiguity conflicts with one shared static surface even if background cameras are adequate.'],
    first_variable='The crop-derived focal prior initializes a fresh full-scene SIMPLE_RADIAL solve. This first new-video run is not a causal A/B comparison.',
    frozen_conditions=['All preflight frames spread over the complete 36.79 second orbit; exact frame IDs and timestamps.',
      'Reserve measured appearance tracks before geometric camera estimation; never fit to validation rows.',
      'No unknown FISHEYE_8 coefficients interpreted as another camera model.',
      'No interpolated missing poses; no historical camera, body fit, Gaussian or optimizer.'],
    stop_conditions=['Any stage error, unregistered frame or nonfinite transform.',
      'Camera-fit mean reprojection error above 2 px or fewer than 50 registered observations in any view.',
      'No measurable held tracks or inadequate face/body/background support.',
      '30 minute preprocessing wall-clock budget. No retry with weaker thresholds.',
      'Stop for actual review after alignment-diagnostics; do not turn coarse fit or a boolean into physical acceptance.'],
    subsequent_training=dict(initial_points=200000,iterations=18000,from_scratch=True,
      allowed_now=False,condition='Review camera, material and staticity evidence, complete fresh person fit, then implement measured admission for this input.',
      all_frame_registration_required_before_full_frame_claim=True),
    quality_accepted=False,training_started=False,company00_verified=False,
    source_hashes={str(p.relative_to(P)):sha(p) for p in (P/'src/shuolee_portrait').glob('*.py')})
save(O/'protocol.json',plan)
save(O/'queue-state.json',dict(updated=now,state='prepared',stage='camera_and_material_preflight',run_root=plan['run_root'],training_started=False,quality_accepted=False))
save(LOG/'protocol.json',plan)
save(LOG/'内参初值假设.json',initial)
save(P/'provenance/new-capture-118-plan.json',plan)
print(json.dumps(dict(plan=str(O/'protocol.json'),config=str(P/'configs/new-capture-118.json'),decoded_wh=initial['decoded_wh'],initial=initial['simple_radial_params']),ensure_ascii=False))
