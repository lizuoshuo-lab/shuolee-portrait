"""Explicit original-to-derived mapping repair; no source fit or global edits."""
from pathlib import Path
import shutil
import train_indoor_person_zoom124 as trial
gs=trial.gs;out=trial.OUT
assert not (out/'launch.json').exists() and not (out/'preflight-v2-launch.json').exists()
assert (out/'preflight-failure.json').exists()
assert not (out/'plan-before-input-contract-repair.json').exists()
shutil.copyfile(out/'plan.json',out/'plan-before-input-contract-repair.json')
shutil.copyfile(out/'adapter.py',out/'adapter-before-input-contract-repair.py')
source=(out/'adapter.py').read_text()
old='    layout = named_pose_layout(data, rows)'
new="""    from derived_camera_contract124 import layout_for_derived_pixels
    source_rows = read(OUT/'source-native-inputs.json')['records']
    pixel_mappings = read(OUT/'pixel-mapping.json')['records']
    layout = layout_for_derived_pixels(data, rows, source_rows, pixel_mappings, named_pose_layout, sha)"""
assert source.count(old)==1
(out/'adapter.py').write_text(source.replace(old,new))
plan=gs.read(out/'plan.json')
for path in [Path(__file__),Path(trial.__file__),gs.HERE/'derived_camera_contract124.py',out/'source-native-inputs.json',out/'adapter.py']:
    plan['source_hashes'][str(path)]=gs.sha(path)
plan.update(updated=gs.now(),input_contract_repair=dict(
    cause='Original named-pose check correctly rejects new image hashes and K as if they were the fitted raw pixels.',
    fix='Original check still runs unchanged against immutable source rows. Explicit per-frame affine validates derived K, dimensions, w2c and frame identity; real source image/mask hashes rechecked.',
    original_fit_sha256=gs.sha(out/'fit/fitted.npz'),old_failure_preserved=True,global_runtime_modified=False,
    preflight_retry_limit=1,training_started=False))
gs.save(out/'plan.json',plan)
print('ZOOM_DERIVED_INTERFACE_REPAIRED_NO_FIT_CHANGE')
