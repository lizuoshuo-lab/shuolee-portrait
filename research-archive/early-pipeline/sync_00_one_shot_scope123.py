from pathlib import Path
from datetime import datetime,timezone
import json
C=Path(__file__).parent.parent/'shuolee-coordination'
stamp=datetime.now(timezone.utc).isoformat()
scope={
 'updated':stamp,'source':'direct user steering 2026-10-29',
 'company00_delivery':'one integrated video-to-static-portrait reconstruction project',
 'company00_scope':['single documented command taking a new video and camera configuration','automatic ordered preprocessing, per-frame SMPL-X/shared identity fitting, free-clothing Gaussian training, named static reference export','standard PLY plus same-coordinate cameras and concise run status/failure reason','Git source deployment and one end-to-end execution feedback'],
 'research_owner':'01a0e316-7534-7693-a60c-ded52094785d',
 'research_continuation':'Continue current SMPL-X framewise shared free-clothing Gaussian native training, preserve implemented transport/trainer; repair confirmed surface-boundary inconsistency and complete real sequence fitting. Do not restart prior baselines.',
 'exclude_from_user_delivery':['separate contract-only projects','repeated stand-alone synthetic experiments or regression reports as deliverables','research notebooks/checkpoints/weights/images committed to public Git'],
 'existing_tests':'Keep necessary automated tests inside the one repository; do not delete historical evidence or weaken code validation.',
 'latest_research':'Synthetic two-pose native16steps passed; actual200k initialization and65x65single step ran. Surface recheck168495/200000; real94fit remains incomplete. Not a completed real reconstruction.',
 'new_fragment_handoff':'cancelled before directory/package creation',
 'no_restart_of_old_training':True,
 'feedback_contract':['exact Git branch and commit','one command and required input paths','actual environment/start/end state','result PLY and cameras or first actionable failure','quality and deployment status separately stated'],
 'target_thread_id':'01a17c8c-e70f-7822-ad39-440290b3d1da',
 'message_state':'prepared'
}
p=C/'one-shot-reconstruction-handoff.json';p.write_text(json.dumps(scope,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=C/'local-research-status.json';j=json.loads(p.read_text(encoding='utf-8-sig'));j.update(updated_at=stamp,company00_delivery_scope=scope,research_continues_from_current_native_training=True);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'handoff':str(C/'one-shot-reconstruction-handoff.json'),'new_fragment_cancelled':True}))
