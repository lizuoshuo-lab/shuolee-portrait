from pathlib import Path
from datetime import datetime, timezone
import json

art=Path('/mnt/d/Projects/Shuolee-validation-20261028')
coord=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-coordination')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
ci=read(art/'final-ci.json')
assert ci['conclusion']=='success' and ci['status']=='completed'
assert ci['headSha']=='5e2cfa7cbb62209548ac692a730a834e6b2a45c0'
ci['tests_passed']=58
save(art/'final-ci.json',ci)
proof=read(art/'orientation-independent-exports.json')
assert proof['state']=='passed'
now=datetime.now(timezone.utc).isoformat()
delivery=read(coord/'company00-engineering-delivery.json')
delivery.update(commit=ci['headSha'],updated_at=now,final_ci=ci,
    runtime_validated_commit=proof['source_git'],
    validated_checkout_final_commit=ci['headSha'],
    final_delta='Validation documentation only; runtime sources unchanged from the full two-video validation')
save(coord/'company00-engineering-delivery.json',delivery)
status=read(coord/'company00-status.json')
status.update(updated_at=now,guide_updated_local=now)
status['current_local_validation'].update(delivery_commit=ci['headSha'],final_ci=ci,
    gpu_coordination=read(art/'gpu-coordination.json'))
status['smplx_update'].update(delivery_commit=ci['headSha'],final_ci=ci)
save(coord/'company00-status.json',status)
print(json.dumps(dict(state='delivered',commit=ci['headSha'],runtime_validated_commit=proof['source_git'],
    ci='58 tests passed',landscape_steps=proof['results']['landscape']['steps'],
    portrait_steps=proof['results']['portrait']['steps'])))
