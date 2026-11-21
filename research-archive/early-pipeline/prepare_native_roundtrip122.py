"""Register the finite R1b fixture before running any CUDA work."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib

H = Path(__file__).parent
R = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\robust-portrait-122')
O = R / 'covariance-sh-ply'
O.mkdir(exist_ok=False)
def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
stamp = datetime.now(timezone.utc).isoformat()
plan = {
    'created': stamp, 'stage': 'R1b', 'scope': 'Known synthetic world/+Y-up Gaussian serialization and same-native-backend image control only',
    'hypothesis': 'The standard 3DGS log-scale, wxyz quaternion, logit-opacity and channel-major SH fields preserve full covariance, directional color and native images without a coordinate change.',
    'seed': 122, 'width': 320, 'height': 240,
    'K': [[210.25, 0, 151.25], [0, 198.5, 126.75], [0, 0, 1]],
    'camera_eyes': [[0, .1, 3.2], [1, .25, 3.1], [-.7, -.1, 2.9]],
    'camera_target': [0, 0, 0], 'world_up': [0, 1, 0],
    'means': [[-.55,.38,.08],[0,.4,-.12],[.53,.32,.1],[-.50,-.32,-.1],[.02,-.34,.16],[.51,-.35,-.12]],
    'scales': [[.12,.028,.045],[.036,.11,.025],[.1,.024,.055],[.035,.035,.14],[.13,.04,.025],[.04,.1,.065]],
    'euler_xyz_radians': [[.3,.2,.7],[-.4,.8,-.2],[.6,-.5,1.0],[.5,.3,-.6],[.1,.9,.3],[-.7,.1,-.3]],
    'opacities': [.72,.84,.65,.91,.78,.87],
    'base_rgb': [[.70,.40,.30],[.30,.65,.45],[.40,.35,.7],[.65,.60,.30],[.35,.70,.65],[.70,.35,.60]],
    'sh_degree': 3, 'rest_coefficients': 'seeded standard normal times 0.14; all RGB and degrees populated',
    'projections': ['pinhole', 'opencv_fisheye4'], 'fisheye_coeffs': [.015,-.001,.0002,-.00003],
    'fixed_limits': {'parameter_max_abs': 3e-7, 'covariance_max_abs': 3e-8, 'sh_color_max_abs': 2e-6,
                     'native_rgb_max_abs': 3e-6, 'native_alpha_max_abs': 3e-6,
                     'negative_rgb_min_abs': 1e-3},
    'controls': ['CPU SciPy covariance versus CUDA quaternion covariance',
                 'Independent polynomial SH3 evaluation versus CUDA SH',
                 'Attribute-level comparison with existing gsplat standard PLY writer',
                 'Native RGB and alpha before/after standard PLY readback for 3 cameras and both projections',
                 'All means/quats/scales/opacity/SH gradients finite and nonzero; no claim of numerical derivative equivalence',
                 'Wrong SH layout and xyzw/wxyz swap must measurably alter native RGB',
                 'Missing SH field must be rejected, no silent zero-fill'],
    'budget': 'One scene, two camera projections, three views; one CUDA suite. Fix only demonstrated implementation defects; no sweep or human training.',
    'stop_if': 'Any fixed control fails: save all evidence and diagnose before further integration.',
    'limitations': ['Synthetic numeric tests cannot establish real camera calibration or face<1px',
                    'No person/model/optimizer from old experiments', 'No change of world basis in this fixture',
                    'No Spark/SIBR/00/native interactive GUI acceptance', 'No new weights or downloads']
}
plan['code_sha256'] = {n: hashlib.sha256((H/n).read_bytes()).hexdigest() for n in
                      ['native_gaussian_ply122.py','native_portrait_renderer122.py','probe_native_roundtrip122.py']}
save(O/'plan.json', plan)
q = json.loads((R/'queue-state.json').read_text(encoding='utf-8-sig'))
q.update(updated=stamp, phase='R1b_native_covariance_sh_ply_roundtrip', running=None,
         r1b_plan='covariance-sh-ply/plan.json', next='Run the pre-registered R1b fixture once, then inspect native images and full results.')
q['native_command'] = '/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python /mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_native_renderer122.py --attempt projection-observability-fix'
save(R/'queue-state.json', q)
print(json.dumps({'registered':str(O/'plan.json'),'gpu_started':False}))
