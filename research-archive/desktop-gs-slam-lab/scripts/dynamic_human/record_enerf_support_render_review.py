from pathlib import Path
import sys,json,hashlib
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-support-correction-081';p=O/'render-visual-review.json'
d=json.loads(p.read_text()) if p.exists() else dict(status='actual_native_panels_review_in_progress',panels=[],clarity_accepted=False,goal_complete=False)
by={r['file']:r for r in d['panels']}
for name in sys.argv[1:]:
 f=O/'comparison'/(name+'.png');assert f.exists();by[f.name]=dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),visually_reviewed=True)
d['panels']=list(by.values());d['actual_panels_reviewed']=len(by);p.write_text(json.dumps(d,indent=2)+'\n');print('Actual panels reviewed:',len(by))
