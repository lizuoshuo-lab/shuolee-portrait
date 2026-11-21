from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'indoor-preflight'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (PRE/'topology-review-plan.json').exists()
assert not (PRE/'native-material/topology-review').exists()
old=H/'field_material_topology123.py';new=H/'indoor_material_topology123.py';assert not new.exists()
s=old.read_text(encoding='utf-8')
assert s.count('/field-preflight')==1
s=s.replace('/field-preflight','/indoor-preflight').replace('field candidates','indoor candidates').replace('Existing field topology review','Existing indoor topology review')
compile(s,str(new),'exec');new.write_text(s,encoding='utf-8')
save(PRE/'topology-review-plan.json',dict(registered=datetime.now(timezone.utc).isoformat(),
    before_completed_inventory=not (PRE/'native-material/result.json').exists(),
    parent_plan_sha256=sha(PRE/'native-material-plan.json'),helper_sha256=sha(new),
    reference_sha256=sha(H/'pattern_bridge_control123.py'),passes=1,wall_seconds=600,
    known_method='Same Tarjan bridge isolation independently checked by NetworkX per eligible component; unchanged whole-family identity and FIT/HELD.',
    source_adaptation='Input/output paths only; no thresholds, selection, budgets or geometric rules changed.',
    review_selection='Same30 slot/region/split queries preregistered before material extraction, no error-based exclusion.',
    no_camera=True,no_fitting=True,no_training=True))
print(str(PRE/'topology-review-plan.json'))
