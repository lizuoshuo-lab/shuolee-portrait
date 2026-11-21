from pathlib import Path
from datetime import datetime,timezone
import json
H=Path(__file__).parent
R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\raw-domain-121')
now=datetime.now(timezone.utc).isoformat()
def put(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=json.loads((R/'protocol.json').read_text())
if not (R/'protocol-before-user-domain-confirmation.json').exists():put('protocol-before-user-domain-confirmation.json',p)
p.update(updated=now,user_confirmed_input_domain='already_undistorted_downsampled_cropped_export',user_quote='这些视频就是去畸变然后裁剪完成的',
    input_lens_model='Pinhole export per user; do not apply raw FISHEYE_8 distortion a second time. Precise output K and EIS remain to be tested.',
    domain_rule='All four current MP4s are processed exports, not raw fisheye. Detect and refine in native encoded pixels without additional undistortion/resampling. Do not inverse-warp detections to fabricated raw samples.')
p['phases'][2]='Current four inputs are user-confirmed processed exports. Raw fisheye inverse projection is inapplicable; do not wait for FISHEYE_8 semantics to analyze these native images.'
p['phases'][3]='Use the first YAML K and explicit measured-size, uniform-resize/center-crop transform as one declared conditional output-K candidate. Keep K fixed during material/pose diagnosis; do not call it physically validated or scan K variants.'
p['stop_conditions'][0]='Conditional output K is inconsistent with independent observations, or evidence indicates non-pinhole/time-varying export mapping.'
p['phase_B_probe']=dict(kind='fixed_conditional_K_pair_geometry_diagnostic',candidate_count_per_person=1,
    pairs='Reuse the 45 consecutive/one-gap native pairs; no new feature extraction, no old poses/3D.',
    fitting='Only fit-split cycle tracks; heldout track IDs excluded from all relative pose estimation/refinement.',
    estimator='PoseLib 2.0.5 calibrated relative pose, fixed PINHOLE K, seed121',
    thresholds=dict(min_fit_tracks=20,max_epipolar_error_px=1.5,min_iterations=1000,max_iterations=10000,min_pair_inliers=20),
    evaluation='Whole-track-heldout Sampson pixel errors, positive-depth/parallax, regional support and rotation cycles. Pairwise consistency is not a complete global camera solution.',
    time_budget_seconds_per_subject=300,retry_or_parameter_scan=False,
    next_gate='Inspect physical material patches and face/body/scene coverage before global SfM/BA; never train directly from pair poses.')
put('protocol.json',p)
put('input-domain-confirmation.json',dict(updated=now,source='direct_user_message',quote=p['user_quote'],scope='existing four MP4s',
    new_undistortion_applied=False,raw_samples_recovered=False,output_K_physically_verified=False,
    unknowns=['exact rectified output K','fixed versus frame-dependent stabilization/crop','exact source crop integer offsets']))
audit=json.loads((R.parent/'camera-contract-120/actual-video-size-audit120.json').read_text())
ks={e['orientation']:e['K_conditional'] for e in audit['conditional_examples']}
plan=dict(updated=now,subjects={s:dict(K_integer_centers=ks['portrait' if s=='portrait' else 'landscape']) for s in ['grass','sofa','portrait','indoor']},
    assumptions='First YAML K is the rectified output K at4000x3000; uniform resize plus centered crop; no extra EIS/zoom. Measured native video sizes used.',
    source_K_physical_accuracy_verified=False,thresholds=p['phase_B_probe']['thresholds'],
    max_seconds_per_subject=300,do_not_use_for_training=True,script=str(H/'probe_native_calibrated121.py'),
    reviewed_input_findings=dict(grass='Person small in frame; grass/foliage repetitive and may move.',
        sofa='Seated person; back side not captured; fine fabric and curtain repetitions need physical matching review.',
        portrait='Half orbit; limbs/clothes/hair can move; distant foliage/water not automatically static.',
        indoor='Full orbit with sofa occluding lower body and feet in sampled frames; occlusion is missing evidence, not empty-body supervision.'))
if (R/'pair-probe-result.json').exists():raise RuntimeError('Probe already completed; do not reset its plan')
put('pair-probe-plan.json',plan)
put('queue-state.json',dict(updated=now,state='native_features_complete_pair_geometry_preregistered',gpu_job=None,training_started=False,
    next='Run one CPU fixed-K PoseLib pair diagnostic on existing frozen tracks, then review regional physical matches.',automation_id='118'))
print('PROBE_PREREGISTERED_USER_DOMAIN_UPDATED')
