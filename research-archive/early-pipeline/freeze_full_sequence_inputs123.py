"""Freeze the declared common gauge and decode the existing full94 input files.

CPU file IO only. Does not import torch or perform any model inference/fit.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
D = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport')
sys.path.insert(0, str(P / 'src'))
from shuolee_portrait.sequence_input_adapter import load_sequence_inputs

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def resolve(path):
    value = str(path).replace('\\', '/')
    if value.startswith('/opt/'):
        return '//wsl.localhost/Ubuntu-20.04' + value
    if value.startswith('/mnt/c/'):
        return 'C:/' + value[7:]
    if value.startswith('/mnt/d/'):
        return 'D:/' + value[7:]
    return value

started = time.monotonic()
draft = D / 'full-sequence-inputs.draft.json'
manifest = D / 'full-sequence-inputs.json'
m = json.loads(draft.read_text(encoding='utf-8'))
m['camera_length_contract'] = dict(
    translation_unit='sfm_units', meters_per_camera_unit=1.0, basis='model_conditioned',
    evidence='Explicit arbitrary common scene-length gauge for camera-domain SMPL-X fitting; factor1 is a declared convention, not a measured/estimated scene scale. Per-frame body fits and projections are in model camera units, world roots are derived using the same declared E; no static-scene placement claim. Not copied from prior optimized fits.')
text = json.dumps(m, ensure_ascii=False, indent=2) + '\n'
if manifest.exists():
    assert manifest.read_text(encoding='utf-8') == text, 'Existing frozen manifest differs; do not overwrite.'
else:
    manifest.write_text(text, encoding='utf-8')
bundle = load_sequence_inputs(manifest, source_path_resolver=resolve, decode_images=True)
bundle.assert_unchanged()
report = bundle.report()
assert len(bundle.frame_ids) == 94
assert report['complete_decode_count'] == 188
assert report['active_face_fit'] == 27 and report['active_face_held'] == 7
assert report['face_records_received'] == 153 and report['face_records_missing_inactive'] == 4641
assert 'torch' not in sys.modules
report.update(state='input_files_verified_not_real_fitting', seconds=time.monotonic()-started,
    manifest=str(manifest), manifest_sha256=sha(manifest), draft_sha256=sha(draft),
    adapter_sha256=sha(P / 'src/shuolee_portrait/sequence_input_adapter.py'),
    torch_imported=False, camera_length_factor_is_declared_convention=True,
    static_scene_world_placement_validated=False, real_model_forwards=0, new_person_fits=0,
    source_path_relocation='Linux absolute files read through WSL share or mounted Windows drive; original hashes and pixel coordinates unchanged.')
output = D / 'full-sequence-inputs-readback.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ['state','seconds','manifest_sha256','frame_count',
    'complete_decode_count','active_face_fit','active_face_held','face_records_missing_inactive','torch_imported']}))
