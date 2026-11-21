from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-camera'
O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sources=['native_residual_camera122.py','native_portrait_renderer122_v3.py','native_scene_package122_v4.py',
         'native_training122_v3.py','probe_residual_camera122.py','native_linear_archive122.py',
         'native_portrait_renderer122_v2.py','native_training122_v2.py','native_training122.py',
         'native_scene_package122.py','native_scene_package122_v2.py','native_gaussian_ply122.py']
stamp=datetime.now(timezone.utc).isoformat()
plan=dict(stage='R2 output residual camera integration',created=stamp,
    hypotheses=['Missing adapter support can cause explicitly supplied Brown output-residual parameters to be discarded or mismatched between feature projection and ray evaluation.',
                'One explicit per-camera K/E/D record can make projection, training and exact native package replay consistent. This is an implementation hypothesis, not attribution of real double noses.'],
    variables='Only new explicit output-residual6+2+4 path; synthetic given cameras/coefficients. Preserve geometry-only bounds and pinned backend.',
    forbidden=['No real coefficients fitted','No video pixels resampled','No fisheye_8 reinterpreted','No real K/poses/observations changed','No persontraining','No rerun D1-D4,T8,T9'],
    source_hashes={str(H/n):sha(H/n) for n in sources},
    preserved_live_sources={str(H/n):sha(H/n) for n in ['native_portrait_renderer122.py','native_portrait_renderer122_v2.py','native_scene_package122_v3.py','native_interactive122_v3.py']},
    budget=dict(cpu_runs=1,cuda_runs=1,seconds=180,synthetic_steps=20,points=96,images=[96,80]),
    acceptance=dict(cpu_projection_max_px=1e-9,inverse_ray_max=1e-9,jacobian_max=1e-8,
        independent_ray_rgb_alpha_max=3e-5,zero_coefficient_v2_parity=0.,exact_array_and_camera_roundtrip=True,
        exact_native_response_max=0.,training_loss_must_decrease=True,physical_quality_accepted=False),
    stop='Preserve first failure and logs. Fix only an evidenced implementation error under the same registered fixture; no threshold scan or physical calibration promotion.',
    fixture=dict(seed=122203,known_radial=[.10,-.025,.003,.008,-.001,.0002],known_tangential=[.003,-.002],
                 known_prism=[.001,-.0003,.0005,.0002],principal_point=[43.2,37.1],focal=[68.,70.]))
save(O/'plan.json',plan)
q=json.loads((R/'queue-state.json').read_text(encoding='utf-8-sig'))
q.update(updated=stamp,phase='R2_residual_camera_registered',running=None)
q['last_goal_turn']=dict(classification='progress',previous='gsplat source andCUDA evidence completed, not a wait',
    completed='Registered same-camera residual projection/training/package integration and a bounded synthetic oracle suite.',
    remaining='Run registeredR2 once; inspect failures and actual native controls, then publish. No human admission.',publication_complete=False)
save(R/'queue-state.json',q)
print(str(O/'plan.json'))
