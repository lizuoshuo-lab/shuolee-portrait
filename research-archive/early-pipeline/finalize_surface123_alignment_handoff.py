from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/local-clothed-surface-123';C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();d=read(R/'whiteT-staged-delivery.json');p=read(H/'surface123-alignment-deck-receipt.json')
assert d['local_UI_verified'] and p['canonical_replaced']
rec=dict(updated=stamp,whiteT=d,quality_accepted=False,grayT='No Gaussian training; oldsupervisor361 stopped at idle review boundary after alignment failure.',
 source_diagnosis=str(R/'smplx-alignment-conclusion.json'),diagnostic_UI_screenshot=str(R/'smplx-alignment-diagnostic-ui.png'),
 same52slide_PPT=dict(path=p['canonical_path'],sha256=p['final_sha256'],changed_slides=[52],actual_visual_review=True),
 current_training_worker=None,current_publication_worker=None,goal_active=True,company00_verified=False,
 next='Inspect frozen grayT solver input support and independently verified static background points before any new fit or camera intervention. No oldrun resume or unchanged queue restart.')
save(R/'alignment-handoff.json',rec);save(C/'local-surface123-delivery.json',rec);save(L/'alignment-handoff.json',rec)
save(C/'local-surface123-status.json',dict(updated=stamp,state='alignment_diagnosis_active_old_queue_stopped',training_running=False,
 summary='SMPL-X灰衣脸部中位44px，独立静态3D点仍47px。投影数值映射通过，物理相机/对应/人物变化未唯一分离。白T18k研究模型已实测配套ZIP并上架，质量未通过。',
 evidence=rec,company_server_used=False))
n=read(R/'next-action.json');n.update(updated=stamp,publication_worker_pid_windows=None,
 second_person='whiteT18k_completed_published_quality_failed',
 next=[rec['next'],'Original52slide PPT now updated to SMPL-X failure evidence; currentSHA='+p['final_sha256']+'. Do not run oldwhiteTdeck preparation against the formercanonical source.',
 'WhiteT completed package actualOpenFileGaussian/points and93cameras passed; latestpublished. Preserve previouswinterhat and119results, qualityfalse.'])
save(R/'next-action.json',n)
con=read(R/'smplx-alignment-conclusion.json');con['completed_whiteT'].update(actual_package_UI_pending=False,local_UI_verified=True,published=True);save(R/'smplx-alignment-conclusion.json',con);save(L/'smplx-alignment-conclusion.json',con)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,current_local_diagnosis=con,local_GPU_task_active=False,training_running=False,next=rec['next']);save(C/'local-research-status.json',s)
ledger=read(R/'evidence-ledger.json');ledger['entries'].append(dict(kind='alignment_diagnosis_delivery_and_whiteT18k_UI',date=stamp,evidence=rec));ledger['updated']=stamp;save(R/'evidence-ledger.json',ledger)
print(json.dumps(rec,ensure_ascii=False))
