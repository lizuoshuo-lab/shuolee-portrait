"""Record actually inspected native correspondence evidence; preserve old runs."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
from audit_grayT123_camera_support import REC,ROOT,OUT,REVIEW,read,save,sha

stamp=datetime.now(timezone.utc).isoformat()
stats=read(OUT/'result.json');attr=read(REC/'grayT-track-attrition/result.json')
summary=('灰衣相机优化最终使用53761个观测，其中52179个（97.06%）位于人物遮罩外。'
    '面部区域内只有9个观测，属于7条轨迹、覆盖4/95帧；逐个原生裁片与精确点位复核，9个全在墙边或栏杆等背景，脸区与人物mask交集的最终BA观测为0。'
    '原始SIFT有556个脸区与mask交集特征；拟合分组有25条全部节点落在该区域的候选轨迹，最终相机BA一条也未保留。'
    '抽检3条未入选候选，点位分别在鼻部和眼周，仍未完成全部物理身份或亚像素测量验收。'
    '保留的匹配日志没有逐轨迹剔除原因；25条中10条原长度少于最终入选的最短6次观测，但另15条更长，不能把全部丢失归因于数量截断。'
    '此外，留出轨迹ee91c8b1f0c2从眼周连接到背景栏杆，已确认跨物体错配；另外两条抽检为嘴角区域及重复栏杆，分别有形变/身份歧义。'
    '这解释了为何全图拟合中位0.393px不能作为人脸准确性的证据。它证明流程中存在人体支持丢失与验收缺口，尚不能唯一归因于K、外参或人物微动。'
    '灰衣SMPL-X面部检测残差中位43.99px，固定相机并放开模板约束后47.05px；当前人体拟合未通过，灰衣高斯尚未开始。')
review=dict(updated=stamp,actual_visual_review=True,all_initial12_native_triplets_reviewed=True,
    additional4_FIT_face_track_triplets_reviewed=True,additional3_background_anchor_check_anchor_reviewed=True,
    all9_actual_saved_FIT_face_ROI_observations_reviewed=True,all7_FIT_face_ROI_tracks_are_background=True,
    source_index_sha256=sha(REVIEW/'index.json'),summary=summary,quality_accepted=False,
    physical_material_ground_truth_certified=False,
    sample_notes={
        '000252c0037f':'Leaves/branches in front of wall; not certified stationary.',
        '7ff6f673633e':'Fence and foliage; region is outside person, physical identity/subpixel location not calibrated.',
        'fff17749b443':'Tree/foliage against building; last shown node is not a retained BA observation.',
        '2bcbdb4e35ae':'Exact point remains on rail connection beside face in all shown views, not face.',
        '5808dda4ec15':'Exact point remains on rail connection beside face, not face.',
        'c427bb2ac9be':'Exact point lies on background wall/plant boundary, not lips.',
        '3c36a4d1c045':'Background wall/plant edge, not face.',
        '4bc16e4ed7bf':'Rail joint beside face, not face.',
        '7a0970799272':'Rail connection beside nose, not nose.',
        '9f4773381044':'Background plant/wall boundary, including both actual BA face-ROI frames1/2.',
        '0006b9fe08c4':'Foliage point; first actual held check1.407px. Staticity not established.',
        '8036d2b7f840':'Asphalt patch; first held check0.962px. Looks consistent, not subpixel physical truth.',
        'ffef1195aa16':'Road/curb boundary; first held check0.827px. Looks consistent, no physical survey truth.',
        '344fc01b6ef5':'Point lies at mouth/corner region across views; smile changes. Saved check frame76=27.884px. Neither camera nor motion uniquely identified.',
        'c0ddb67d7eb2':'Repeating background rail edge in all three marked crops. Face box overlap is not a face track; same physical rail identity remains uncertain.',
        'ee91c8b1f0c2':'Eye/brow in frames4/11 but background rail in frame94: confirmed material-identity switch.'},
    no_relabeling_of_fit_or_held=True,no_observation_deletion=True,no_camera_update=True)
save(REVIEW/'actual-review.json',review)
a_review=REC/'grayT-track-attrition-review'
save(a_review/'actual-review.json',dict(updated=stamp,actual_visual_review=True,summary='三条预先按ID均匀选择的拟合分组原始轨迹均已查看：083ced鼻部，6723b2眼眉/额侧，f300f4眼周。三条都未进入最终相机BA。只有区域与可见点位得到核实，跨全部帧物理身份和误差未验收；不能据此把全部25条当真值或直接加入新拟合。',
    selected_ids=attr['selected_samples'],quality_accepted=False,all25_physical_tracks_verified=False))
con=dict(updated=stamp,subject='grayT',summary=summary,evidence=str(OUT/'result.json'),attrition=str(REC/'grayT-track-attrition/result.json'),
    raw_observation_mapping_unchanged=True,fit_background_observation_fraction=52179/53761,
    actual_face_region_BA_observations=0,face_region_candidate_tracks_lost_before_final_BA=25,
    confirmed_cross_object_track='ee91c8b1f0c212b638c5f4558bded44093bcaca3ecc598f49607916ee9c0fce1',
    no_unique_physical_cause_claim=True,training_started=False,quality_accepted=False,
    next='Audit fixed-camera static consistency of the already saved material candidates and the negative controls, after identity/visibility review. Keep K/E frozen and all data splits unchanged; do not restart old camera solver or training. Need distinguish correspondence failures, residual camera errors and deforming mouth/head.',
    sources=[dict(path=str(OUT/'result.json'),sha256=sha(OUT/'result.json')),dict(path=str(REC/'grayT-track-attrition/result.json'),sha256=sha(REC/'grayT-track-attrition/result.json'))])
save(REC/'camera-support-conclusion.json',con)
ledger=read(REC/'evidence-ledger.json');kind='grayT_camera_support_and_face_track_attrition'
assert not any(e.get('kind')==kind for e in ledger['entries'])
ledger['entries'].append(dict(kind=kind,date=stamp,summary=summary,evidence=con));ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
n=read(REC/'next-action.json');n.update(updated=stamp,current_worker_pid_wsl=None,publication_worker_pid_windows=None,
    latest_actual_checkpoint_review=str(REVIEW/'actual-review.json'),latest_diagnosis=str(REC/'camera-support-conclusion.json'),
    next=[con['next'],'Preserve WhiteT/winterhat completed research results and all original123 camera/fit files; no grayT Gaussian or unchanged old queue restart.'])
save(REC/'next-action.json',n)
q=read(REC/'queue-state.json');q.update(updated=stamp,state='active_alignment_diagnosis',training_running=False,worker_pid=None,
    reason='灰衣相机最终BA没有真实脸部区域观测，原25条脸区候选未入选；存在跨物体留出错配。保留原相机，继续分离物理原因；灰衣高斯未开始。')
save(REC/'queue-state.json',q)
T=REC.parents[2];C=Path(__file__).parent.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
for folder in [OUT,REC/'grayT-track-attrition']:
    for dest in [L/folder.name,T/'viewer/local-surface-123'/folder.name,Path('D:/00/工作/查看器/local-surface-123')/folder.name]:
        dest.mkdir(parents=True,exist_ok=True)
        for p in folder.iterdir():
            if p.is_file():shutil.copy2(p,dest/p.name)
save(L/'camera-support-conclusion.json',con)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,current_local_camera_support_diagnosis=con,local_GPU_task_active=False,training_running=False,next=con['next']);save(C/'local-research-status.json',s)
goal=read(REC.parent/'robust-portrait-122/queue-state.json');goal.update(updated=stamp,phase='LOCAL_SURFACE123_camera_face_support_missing_and_material_mismatch_verified',running=None,
    last_goal_turn=dict(classification='progress',current=summary,quality_accepted=False),next=con['next']);save(REC.parent/'robust-portrait-122/queue-state.json',goal)
plan=T/'shuolee/long-term-plan.md'
with plan.open('a',encoding='utf-8') as f:f.write('\n\n2026-10-28：灰衣相机支持与对应审阅\n\n'+summary+'\n\n下一步：'+con['next']+'\n')
print(json.dumps(con,ensure_ascii=False))
