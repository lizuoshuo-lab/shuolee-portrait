from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PARENT=T/'shuolee/current/robust-portrait-122'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
stamp=datetime.now(timezone.utc).isoformat();folder=REC/'winterhat-train-012000-review';idx=read(folder/'index.json')
review=dict(updated=stamp,actual_visual_review=True,index_sha256=sha(folder/'index.json'),steps=12000,
    summary='实际看完六个同相机全身与中心分布，并放大frame_45/67/112人脸。衣服和裙面仍有中心支持，靴子轮廓大体存在。相比6000步，多个视角脸部出现更明显的多重轮廓与杂色块，肩袖、裙边和包的细长溅射增加，靴底复影仍在。质量未通过，不能把步数增多描述成质量改善。此观察不会改动正在执行的训练源码或相机，本次有限18k运行与最终交付仍需保留该退化事实。',
    actually_reviewed_body_frames=[x['frame'] for x in idx['images'] if x['region']=='body'],
    detailed_face_frames=['frame_000045','frame_000067','frame_000112'],
    quality_accepted=False,physical_calibration_accepted=False,training_parameters_changed=False,
    native_ply_parity=read(ROOT/'winterhat/gaussian-run/step-012000/native-ply-parity.json'))
save(folder/'actual-review.json',review)
q=read(REC/'queue-state.json');state=read(REC/'winterhat-state.json');progress=read(ROOT/'winterhat/progress.json')
if q['active_subject']=='winterhat' and progress['step']>1000 and q['current_stage']=='gaussian-step-001000_review_pending':
    evidence=dict(updated=stamp,kind='state-only reconciliation',live_worker_pid_checked=361,
        exact_process_command='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python /mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/local_surface123.py --run',
        before=q['current_stage'],after='training',authoritative_progress=progress,
        reason='The callback records the pause but does not record the return to the loop. Progress above1000 proves the review was consumed; current process was checked live.',
        runtime_source_modified=False,optimizer_or_parameters_changed=False)
    save(REC/'state-reconciliation.json',evidence)
    state.update(updated=stamp,stage='training',optimizer_waiting=False,training_running=True,step=progress['step'],review_consumed=True)
    save(REC/'winterhat-state.json',state)
    q.update(updated=stamp,current_stage='training')
    for s in q['subjects']:
        if s['id']=='winterhat':s.update(stage='training',updated=stamp)
    save(REC/'queue-state.json',q)
p=PARENT/'protocol.json';d=read(p)
if 'prior_next_before_surface123' not in d:d['prior_next_before_surface123']=d['next']
d['next']='Latest user priority: finish the existing local surface123 sequential worker. Do not start another run or change its frozen sources. Inspect real process, native progress and reviews. Export each completed18k subject with same-coordinate camera, perform actual quality and local OpenFile Gaussian/points review, then publish truthful quality status. Seven new subject runs are finite; sofa119 remains reference. R7 source drafts remain deferred, subpixel robustness target remains unpassed.'
save(p,d)
ledger=read(REC/'evidence-ledger.json');kind='winterhat_actual_12000step_review'
if not any(x.get('kind')==kind for x in ledger['entries']):
    ledger['entries'].append(dict(kind=kind,date=stamp,summary=review['summary'],quality_accepted=False,
        same52page_report=dict(path='D:/00/报告/02_有限视角重建优化路线.pptx',sha256='e7db76800fbba00d03b29a03fabda28f8a41409c1da760be45ff785de46c7cca',
            evidence_stage=6000,first51_byte_and_render_unchanged=True,actual_last_page_review=True)))
ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
g=read(PARENT/'queue-state.json');g.update(updated=stamp,phase='LOCAL_SURFACE123_actual_training',
    last_goal_turn=dict(classification='progress',current='Actual6000/12000 images reviewed, frozen regional support and six material triplets inspected; existing52page report updated only last. Live local worker continues.',publication_complete=True));save(PARENT/'queue-state.json',g)
print(json.dumps(review,ensure_ascii=False))
