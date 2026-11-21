from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

H = Path(__file__).parent
R = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\robust-portrait-122')
O = R / 'native-package-loader'
U = R / 'native-interactive'
U.mkdir(exist_ok=True)
plan = U / 'plan.json'
assert not plan.exists(), 'Existing registration must be reviewed, not overwritten'
files = [O/'fixture-package'/n for n in ['scene.json', 'cameras.json', 'point_cloud.ply']]
files += [H/n for n in ['native_scene_package122.py', 'native_portrait_renderer122.py', 'native_gaussian_ply122.py', 'render_native_scene122.sh']]
files += [H/n for n in ['native_interactive122.py', 'native_interactive122.html']]
stamp = datetime.now(timezone.utc).isoformat()
p = dict(stage='R1e', registered=stamp, scope='Synthetic recorded-camera native UI integration; not real-person calibration or an orbit renderer',
    hypothesis='The UI-selected recorded camera and projection reach the existing native UT/eval3d renderer unchanged, with no cached image or Spark fallback.',
    scene=str(files[0]), immutable_inputs={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    controls=[dict(camera_id=c, projection=m) for c,m in [('0','pinhole'),('2','pinhole'),('0','opencv_fisheye4'),('2','opencv_fisheye4')]],
    required=['Four real UI button renders, one at a time, each new output directory and process receipt',
              'Exact scene hash, camera ID, full K/w2c and selected projection in result',
              'Compare raw RGB/alpha against frozen R1c result at the same camera/projection, max abs <=3e-6',
              'Actual raw-array download from interface, SHA equals rendered output',
              'Reject client K/w2c or path overrides, unknown camera/projection, duplicate/busy and exhausted budget before GPU',
              'Changing selectors invalidates displayed previous image; no stale frame under a new camera label',
              'Core/backend/scene/cameras/PLY unchanged; main119 viewer and tab15 untouched'],
    budget=dict(GPU_renders=4, concurrent_GPU_jobs=1, per_render_seconds=60, threshold_search=False, training=False),
    acceptance=dict(rgb_max_abs=3e-6, alpha_max_abs=3e-6, camera_identity_exact=True, original_native_pixels=True),
    stop='Preserve failures. Only repair a demonstrated interface implementation defect; no new parameters, model candidates, or extra renders without a new causal justification.',
    limitations=['Synthetic known cameras only', 'This interface is recorded-camera rendering, not continuous free-flight', 'Physical outputK/material identity and <1px face admission remain unresolved', 'Official SIBR, independent Spark and company00 are not validated here'])
plan.write_text(json.dumps(p, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
qpath = R/'queue-state.json'; q=json.loads(qpath.read_text(encoding='utf-8-sig'))
q.update(updated=stamp, phase='native_interactive_plan_registered', next='Implement and verify R1e once under native-interactive/plan.json; no person training.', r1e_plan='native-interactive/plan.json')
qpath.write_text(json.dumps(q, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(plan=str(plan), controls=4, GPU_started=False)))
