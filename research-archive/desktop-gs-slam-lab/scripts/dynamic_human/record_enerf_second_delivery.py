"""Persist the verified local-site delivery without completing the research goal."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';M=L/'outputs/dynamic-human-enerf-matte-diagnosis-080'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert read(R/'website-qa.json')['status']=='two_pages_and_ten_new_assets_HTTP_verified'
out=dict(status='new_native_results_and_supervision_diagnosis_visible_on_existing_site',
 url='http://localhost:8799/gs-lab/read/dynamic-human-scene-reconstruction#当前可见的重建效果',
 actual_browser_review='Reloaded existing tab15 after production build. Screenshot shows updated full-person731comparison, scope paragraph and following OmniRe figure; no clipped images or overlapping text in visible viewport. Read-only DOM reports zero KaTeX errors.',
 build='Succeeded after stopping only the verified previous8799preview process tree that locked dist. Preview restarted hidden at127.0.0.1:8799.',
 preview_parent_pid=9932,goal_complete=False,updated_at=datetime.now(timezone.utc).isoformat())
save(R/'website-visual-review.json',out)
d=read(M/'summary.json');d['detail_prepared']['visually_reviewed']=True;d['detail_findings']='Actual04/732detail confirms a recognizable open palm and several curved finger outlines in both original and composited target, while heldout04/731renders remain smeared.';save(M/'local-detail-visual-review.json',d)
m=read(L/'manifests/dynamic-human-work.json');m['latest_verified_site']='outputs/'+R.name+'/website-qa.json';m['latest_visible_result']=out['url'];m['current_runs']=[]
m['goal_turn_classification']='Progress: completed new motion/initialization plus2x15000training,240native predictions,CPU recomputation and52actual review panels; confirmed false foreground in six training targets and clear openhand in source. Updated existing website and verified bothpages/10assets and actual rendered page. Clarity remains unaccepted; no external blocker.'
m['updated_at']=out['updated_at'];save(L/'manifests/dynamic-human-work.json',m)
print(out['status'])
