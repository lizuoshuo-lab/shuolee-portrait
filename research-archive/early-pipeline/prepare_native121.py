from pathlib import Path
from datetime import datetime,timezone
import json, hashlib
H=Path(__file__).parent
TOP=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=TOP/'shuolee/current/raw-domain-121';R.mkdir(parents=True,exist_ok=True)
now=datetime.now(timezone.utc).isoformat()
def put(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
protocol=dict(updated=now,objective='Detect and refine material features in source pixels before geometric image resampling; solve calibrated camera poses and freshly train all four static clothed people when admitted.',
    domain_rule='A source MP4 is not automatically raw fisheye. Never invent or recover raw samples by inverse-warping detections from stretched rectified images.',
    source_dataset='D:/00/工作/数据集',calibration='mono_main_calib.yaml first camera_matrix',
    input_lens_model='FISHEYE_8; vendor formula/order/output-K semantics unverified',
    source_processing_sequence=['first_camera_undistort','downsample','center_crop'],diagram_dimensions_authoritative=False,
    phase_A=dict(kind='native_encoded_domain_observation_preparation',max_frames_per_video=24,videos=4,
        detector='SIFT subpixel + separate Shi-Tomasi/native cornerSubPix audit',native_resolution=True,
        resample_before_detection=False,ratio=.75,track_cycles='exact 3-view identity cycles with conflict rejection',
        heldout='whole-track hash mod5 ==0, frozen before geometric fit',camera_K_used=False,
        raw_fisheye_confirmed=False,worker=str(H/'run_native_features121.py')),
    phases=[
        'Audit native MP4 frame identity, actual dimensions and source preprocessing. Match directly at native encoded resolution; no added undistortion/resize before feature detection.',
        'Visually review native images and tracks: face/nose/glasses, clothing/feet, static-scene candidates, reflections/other people. Appearance cycles alone are not physical truth.',
        'On verified raw fisheye: original-domain detection/refinement -> vendor inverse projection to unit bearings -> calibrated pose solver. Use forward projection to raw pixels for errors; propagate local Jacobians rather than a constant remapped-pixel noise.',
        'If only processed export is available, retain native-export features as such; do not claim raw-fisheye recovery. A verified output pinhole K plus recorded resize/crop is required for physical calibration.',
        'Fixed verified K, PoseLib relative/PnP and fixed-K robust bundle adjustment. No old optimized camera/3D initialization, no interpolation or 2D warping to fill real frames.',
        'Hold whole tracks out; protect face, body/feet and background separately, and isolate human motion after static-scene pose credibility.',
        'Each admitted person: refit geometry anew, initialize 200000 free surface/clothed Gaussians and an empty optimizer, maximum18000 steps; review at1000. Never bind final centers to naked SMPL surface.',
        'Review same actual cameras and continuous unseen views. Export complete world/+Y covariance/SH; native reread, actual OpenFile Gaussian/points, then publish current models and quality status.'
    ],
    budgets=dict(native_feature_preparation_runs=1,max_new_camera_candidates_per_person=1,max_fresh_training_per_person=1,max_steps=18000),
    stop_conditions=['unknown input/output lens domain or absent inverse projection prevents calibrated solve','disconnected camera graph/insufficient parallax','ambiguous material correspondences','face/body/background validation fails','1000-step obvious ghosting or geometry collapse'],
    constraints=['one GPU job','no new large downloads','no company training or personal-data upload','do not alter running code','no cross-run learned initialization','preserve historical models and tab15'],
    missing_material_policy='Continue bounded independent matching and code checks; do not repeatedly rescan missing files. After those complete, remain quiet awaiting new evidence; pause followup if no actionable work remains.',
    acceptance_not_equal_to=['feature extraction completion','synthetic checks','camera-format compatibility','training completion'])
if (R/'protocol.json').exists():raise RuntimeError('Protocol already exists; inspect before changing')
put('protocol.json',protocol)
put('queue-state.json',dict(updated=now,state='preregistered_native_features',gpu_job=None,training_started=False,
    next='Launch one bounded CPU worker for four videos; calibration/training admission remain closed.',automation_id='118'))
put('sources.json',dict(updated=now,sources=[
    dict(url='https://github.com/PoseLib/PoseLib',role='calibrated pose solvers'),
    dict(url='https://colmap.github.io/format.html',role='export format only, not pose estimator'),
    dict(url='https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/camera_utils.py',role='cameras.json array fields'),
    dict(query='"FISHEYE_8" "camera_matrix"; "MONO_MAIN_MFR"',result='No official semantics found in bounded exact-term search; unknown retained')]))
print(str(R))
