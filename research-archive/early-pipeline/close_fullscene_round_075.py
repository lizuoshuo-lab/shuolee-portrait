from pathlib import Path
import datetime,hashlib,json,shutil

W=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V=B.parent/'viewer';R=B/'current/person-static-fullscene-camera-fresh-075'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save(p,d):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)

deck=read(W/'fullscene075-deck-receipt.json')
validation=read(W/'fullscene075-validation.json')
src,dst=Path(deck['finalPath']),Path(deck['source'])
assert sha(src)==deck['final_sha256']==validation['finalSha256']
assert validation['packageIntegrity']['status']=='pass' and validation['firstPartyImport']['passed']
assert sha(dst)==deck['source_sha256'],'Canonical was edited concurrently; preserve it'
assert sum(not r['changed'] for r in deck['renders'])==49
proof=[W/f'fullscene075-final-{s}.png' for s in [20,51,52]]
assert all(p.is_file() for p in proof)
visual=dict(reviewed_at=now,final_sha256=sha(src),actual_render_inspection=True,
    inspected_changed_slides=[20,51,52],render_files=[str(p) for p in proof],
    unchanged_slide_render_hashes=49,slides=52,passed=True,
    native_powerpoint_verified=False,pc00_deployed=False)
save(W/'fullscene075-deck-visual-review.json',visual)
shutil.copy2(src,dst);assert sha(dst)==deck['final_sha256']
deck.update(canonical_replaced=True,canonical_replaced_at=now)
save(W/'fullscene075-deck-receipt.json',deck)
manifest_path=B/'reports/report-manifest.json';manifest=read(manifest_path)
report=next(x for x in manifest['reports'] if x['key']=='reconstruction-optimization')
report.update(sha256=sha(dst),bytes=dst.stat().st_size,slides=52,
    local_validation=str(W/'fullscene075-validation.json'),
    local_visual_review=str(W/'fullscene075-deck-visual-review.json'),
    latest_scope='075整幅图相机、重新人体和衣物初始化、18k训练、相同相机对照与残余重影',
    current_local_report_deployed_to_00=False)
manifest.update(updated=now,last_local_revision=dict(path=str(dst),sha256=sha(dst),
    bytes=dst.stat().st_size,slides=52,remote_delivered=False,transfer_performed=False,
    local_review=str(W/'fullscene075-deck-visual-review.json'),updated=now))
save(manifest_path,manifest)

delivery=read(V/'models/whole-body-static-fullscene/delivery.json')
decision=read(R/'review/visual-decision.json')
note=('075已按要求完成整图相机与人体初始化的完整流程：074的175个整图SfM+BA相机，'
      '重新统一去畸变图像/mask/特征，新的1200步人体粗拟合，175帧轮廓和765条衣物纹理初始化20万点，'
      '新优化器训练18,000步，最终290,458个高斯。未复用旧个人拟合、高斯或优化器。'
      '同075相机评测072/075：内部RGB误差0.011324→0.006286（降低44.49%），'
      '内部欠覆盖1.862%→0.711%，轮廓2.138664→2.310662像素（增加8.04%），'
      '边界alpha0.191336→0.194389；鼻口、眼镜、肩部及仰视破碎仍在，质量未通过。'
      '标准+Y世界PLY、原生回读、175相机导出及本机打开文件的两种模式均完成核验；'
      'latest已指向075并明确质量状态，未部署00。相机误差未被排除，未证明微动是唯一原因。'
      '下一步冻结模型追踪异常鼻口/眼镜点的原始观测帧、邻域图像块与深度冲突，取得可区分证据后再定义下一轮从头训练；不自动追加步数或扫权重。')
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8')
start=text.index('075用户明确接续：');end=text.index('\n',start)
text=text[:start]+note+text[end:]
text=text.replace('074已完成整图从头SfM：','074历史诊断（后续端到端试验见075）：已完成整图从头SfM：',1)
text=text.replace('## 当前执行：先归因边界失效，再选择静态训练改动','## 当前执行：整图相机流程已完成，追踪残余重影',1)
text=text.replace('当前071已补齐175帧观测配准。','当前075使用重新计算的175帧整图SfM与全局BA相机，没有插值。',1)
plan.write_text(text,encoding='utf8')
s=read(B/'work-status.json')
s.update(updated=now,current_decision=note,current_phase='075 local pipeline and delivery complete; geometry quality failed',
    local_training_running=False)
s['iteration_policy'].update(new_training_ready=False,completed_protocol_round='075',
    required_next_protocol='Attribute residual nose/mouth/glasses Gaussians to source-frame and neighbor-patch observations and texture-depth conflicts before a bounded fresh change; user authorization remains active')
s['latest_completed_static_delivery']=dict(round='075',steps=18000,points=290458,
    quality_accepted=False,compatibility_passed=True,local_viewer_passed=True,pc00_opened_this_round=False,
    viewer_url=delivery['viewer_url'],download_url=delivery['download_url'],local_ply=delivery['download_path'],
    training_cameras='175 fresh full-image SfM+global BA, 0 interpolated',review=str(R/'review/visual-decision.json'),
    camera_export_check=str(R/'gaussian-run/viewer-camera-roundtrip.json'))
s['full_scene_localization'].update(experimental_use_round='075',experimental_use_completed=True,
    new_gaussian_training=True,camera_accuracy_accepted=False,experimental_use_accepted_as_quality=False)
s['latest_report']=dict(path=str(dst),sha256=sha(dst),changed_slides=[20,51,52],slides=52,updated=now,remote_deployed=False)
s['pending_report_delivery']=dict(status='local_updated_remote_deferred',source=str(dst),file=dst.name,
    sha256=sha(dst),bytes=dst.stat().st_size,remote_delivered=False)
save(B/'work-status.json',s)
completion=dict(updated=now,workflow_completed=True,steps=18000,initial_points=200000,final_points=290458,
    frames=175,from_scratch=True,camera_source='074 full-image SfM and global BA',
    quality_accepted=False,geometry_accepted=False,root_cause_established=False,
    local_standalone_viewer_passed=True,remote_00_deployed=False,
    decision=str(R/'review/visual-decision.json'),download_url=delivery['download_url'],
    report='http://127.0.0.1:8870/body-smplx.html#full-scene-person',canonical_deck=str(dst))
save(R/'completion.json',completion)
for name in ['check_fullscene_person_075.py','train_fullscene_person_075.sh','review_fullscene_person_075.sh',
             'decide_fullscene_person_075.py','verify_fullscene_person_delivery_075.py']:
    shutil.copy2(W/name,R/'source-final'/name)
save(R/'source-final/manifest.json',dict(updated=now,files=[dict(file=p.name,sha256=sha(p))
    for p in sorted((R/'source-final').iterdir()) if p.is_file() and p.name!='manifest.json']))
print(json.dumps(completion,ensure_ascii=False,indent=2))
