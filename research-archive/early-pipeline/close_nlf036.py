from pathlib import Path
from datetime import datetime, timezone, timedelta
import json,hashlib
BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
BUILD=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt=read(BUILD/'nlf036-deck-receipt.json')
assert receipt['canonical_replaced']
assert hashlib.sha256(Path(receipt['source']).read_bytes()).hexdigest()==receipt['final_sha256']
now=datetime.now(timezone(timedelta(hours=8))).isoformat()
s=read(BASE/'work-status.json')
s['updated']=now
s['latest_report']=dict(path=receipt['source'],sha256=receipt['final_sha256'],changed_slides=[55],
    changed_notes_slides=[55],slides=55,updated=now)
s['body_sam_input_028'].update(state='prepared_deferred_user_prefers_existing_priors',
    user_instruction='先使用现有身体先验',download_requested_again=False)
s['existing_prior_scope']=dict(updated=now,use=['VPoser V02_05','GVHMR','NLF-L v0.3.2 patch6'],
    SAM3DBody='deferred by user; no new download or repeated access question')
s['nlf_shared_fit_036']['presentation']=s['latest_report']
s['nlf_shared_fit_036']['web_evidence']='http://127.0.0.1:8870/body-smplx.html#nlf-shared-fit-036'
s['nlf_shared_fit_036']['browser_proof']=str(BUILD/'nlf036-page-proof.png')
s['nlf_shared_fit_036']['new_duplicate_cleanup']=str(BUILD/'nlf036-duplicate-cleanup.json')
save(BASE/'work-status.json',s)
p=BASE/'long-term-plan.md'
text=p.read_text(encoding='utf-8')
old='<!-- NLF_SEQUENCE_036 -->'
assert text.count(old)==1
text=text.replace(old,old+'\n用户最新指示：先使用现有身体先验。VPoser、GVHMR、NLF已可用，SAM3DBody暂缓，不再请求下载。',1)
p.write_text(text,encoding='utf-8')
print(json.dumps(dict(current_phase=s['current_phase'],goal_status=s['long_term_goal_status'],
    geometry_accepted=s['goal_accepted'],sam_route=s['body_sam_input_028']['state'],report_sha=receipt['final_sha256']),ensure_ascii=False))
