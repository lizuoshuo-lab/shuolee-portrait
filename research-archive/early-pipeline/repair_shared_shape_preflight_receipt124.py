"""Log the confirmed JSON-only preflight failure before retrying the check."""
from pathlib import Path
import audit_gaussianavatars_params123 as a
from datetime import datetime,timezone
OUT=a.REC/'cross-subject-sh0-124/grass-selected-shape124'
path=Path(__file__).with_name('run_shared_shape_views124.py')
plan=a.read(OUT/'plan.json');key=next(p for p in plan['source_hashes'] if p.endswith('/run_shared_shape_views124.py'))
assert not a.lp(OUT/'preflight.json').exists() and not a.lp(OUT/'progress.json').exists()
assert not a.lp(OUT/'preflight-failure.json').exists()
a.save(OUT/'preflight-failure.json',dict(updated=datetime.now(timezone.utc).isoformat(),
    error='TypeError: Object of type bool_ is not JSON serializable',
    location='preflight checks passed; saving preflight.json failed at JSON encoding',
    optimizer_steps=0,fit_calls=0,confirmed_fix='Convert check booleans to built-in bool before save.',
    old_source_sha256=plan['source_hashes'][key],new_source_sha256=a.sha(path)))
plan['source_hashes'][key]=a.sha(path)
plan['preflight_serialization_repair']='preflight-failure.json; no numerical threshold, model or optimizer change.'
a.save(OUT/'plan.json',plan)
print('PREFLIGHT_JSON_FIX_RECORDED')
