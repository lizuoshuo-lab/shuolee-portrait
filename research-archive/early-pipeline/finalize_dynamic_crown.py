"""Publish reviewed read-only diagnostics and the same evolving report locally."""
from pathlib import Path
import json,hashlib,shutil,datetime
from PIL import Image,ImageChops

BASE=Path(r'\\?\C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
PHOTO=BASE/'current/novel-view-face/head-structure/flame-photometric-registration'
NEW=PHOTO/'whole-body-gaussianavatar/all-views-background'
ROOT=NEW/'crown-attribution'
VIEWER=BASE.parent/'viewer'
BUILD=Path(r'C:\Users\Administrator\Documents\New project\shuolee-slide-update')
FINAL=BUILD.parent/'shuolee-slide-output/02_有限视角重建优化路线_20261022_实拍约束修订.pptx'
CANON=BASE/'reports/02_有限视角重建优化路线.pptx'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
now=datetime.datetime.now().astimezone().isoformat()
old='ae5b4c6cb5c918c6ee992727b08cca44c069688c04bdcc2214a0f7d4a54cd3fd'
assert sha(CANON)==old,'Preserve any new report edits'
main=NEW/'run18000/current.pt';candidate=VIEWER/'models/whole-body-clothing-candidate/point_cloud.ply'
protected={'main':sha(main),'published_candidate':sha(candidate)}
assert protected=={'main':'676675cc2f38913ee95e00f4ceb2ba1fb7d488bf6a1fbaa711516d0e69411ec2',
 'published_candidate':'338b0816e1bde4a064a49b18d55473fe36b369d89646ae92cc2dc66051e1f844'}
receipt=read(BUILD/'crown-final-validation.json')
assert receipt['packageIntegrity']['status']=='pass' and receipt['presentationLayout']['findingCount']==0
assert receipt['firstPartyImport']['passed'] and sha(FINAL)==receipt['finalSha256']
unchanged=[]
for i in range(1,55):
    aa=Image.open(BUILD/'hair-final-import-renders'/f'{i:02d}.png').convert('RGB')
    bb=Image.open(BUILD/'crown-final-renders'/f'{i:02d}.png').convert('RGB')
    assert aa.size==bb.size and ImageChops.difference(aa,bb).getbbox() is None,i
    unchanged.append(i)
assert ImageChops.difference(Image.open(BUILD/'crown-after-55.png').convert('RGB'),Image.open(BUILD/'crown-final-renders/55.png').convert('RGB')).getbbox() is None
report=dict(updated=now,slides=55,changed_slides=[55],changed_slide_full_size_reviewed=True,
 all_final_slides_rendered=True,earlier_slides_pixel_identical=unchanged,nativePowerPointVerified=False,
 source_sha256=old,final_sha256=receipt['finalSha256'],final_bytes=FINAL.stat().st_size)
save(BUILD/'crown-visual-review.json',report)
shutil.copyfile(FINAL,CANON);assert sha(CANON)==receipt['finalSha256']

trace=read(ROOT/'result.json');rad=read(ROOT/'radiance-probe/result.json');corr=read(ROOT/'real-correspondence/result.json')
assert corr['descriptor_support_interior'] and corr['summary']['dynamic']['consistent_tracks']==8
observations=[
 '俯视头顶亮像素约94.09%的实际合成贡献来自实拍头发高斯，未观察到脸部点大量混入。上仰的启发式头顶框包括额头，不能从框内脸部贡献推断串色。',
 '俯视亮头顶相关点有效观测数量加权中位数35，最近真实方向间隔中位数21.07度。多次可见不等于覆盖了俯仰方向。',
 '688个偏白头发高斯向关联实拍颜色做50%和100%插值后，头发L1由0.015076升至0.017044和0.020843。实际未知俯仰仍有碎片，本方案不采用。混合像素颜色不是单点辐射真值。',
 '初始SIFT描述窗口跨越头发轮廓混入背景，中心落在掩膜内不足以保证对应可靠。中心掩膜版本不用于几何监督。',
 '限制描述窗口在头发内部后，38帧有414个特征和17条至少4帧轨迹。静态通过6条，现有动态刚性近似通过8条。该数量不足以监督整个头顶。',
 '8条候选是同一训练视频内的一致性检查，不是独立深度真值。动态头皮刚性近似本身仍有误差，不能据此直接移动头发点。']
next_step='沿用原动态人体。先保存并扩展真实头发内部的对应坐标，对比每条轨迹附近的三角形局部形变与整头刚体补偿，核验重投影、运动残差及覆盖范围。证据通过后才做有界点位置对照；不根据8条稀疏轨迹直接修改整头，不再调暗高斯、整块换发或盲增点数。'
review=dict(updated=now,state='reviewed_original_retained_sparse_correspondence_pending',optimizer_updates=0,
 successful_recorded_native_forwards=trace['render_forwards']+trace['support_rebuild_forwards']+rad['render_forwards'],
 source_frames=59,semantic_frames=38,main_training_steps=18000,observations=observations,
 metric_summary={k:v['summary'] for k,v in rad['metrics'].items()},real_correspondence=corr['summary'],
 reviewed_images=['pitch-45-attribution.png','pitch-15-attribution.png','radiance-probe/pitch-45-comparison.png',
 'radiance-probe/000023-comparison.png','real-correspondence/track-0.png','real-correspondence/track-3.png',
 'real-correspondence/interior-track-0.png','real-correspondence/interior-track-1.png'],
 all59_are_training=True,independent_test=False,novel_pitch_GT=False,quality_accepted=False,
 model_replaced=False,main_preserved=True,motion_preserved=True,background_preserved=True,
 eyewear_optimization=False,deployed_to_00=False,protected_sha256=protected,next_experiment=next_step,report=report,
 execution_repairs=['Earlier support intermediate had been cleaned; rebuilt59weights only, without retraining.',
 'Read-only triangulation result serialization converted NumPy boolean to plain boolean; no optimizer updates.',
 'Visual review rejected center-only SIFT masking before refining full descriptor support.'])
save(ROOT/'root-review.json',review)

assets={
 'crown-attribution.png':ROOT/'pitch-45-attribution.png',
 'crown-radiance-probe.png':ROOT/'radiance-probe/pitch-45-comparison.png',
 'crown-real-correspondence.png':ROOT/'real-correspondence/interior-track-0.png'}
for name,src in assets.items():
    shutil.copyfile(src,VIEWER/'body-dynamic'/name)
    assert sha(src)==sha(VIEWER/'body-dynamic'/name)
html=VIEWER/'body-dynamic.html';content=html.read_text(encoding='utf-8')
assert 'id="crown-attribution"' not in content
anchor='<p>当前仍使用已有动态人体、独立衣物和静态背景。全部59个顺序视频观测已加入主轮，人体训练保持18,000步。发束试验的根随同一套头皮运动变化，未重启眼镜专项。</p>'
assert content.count(anchor)==1
block='''
<div id="crown-attribution"><h4>原动态头部：俯仰伪影与真实观测</h4>
<p>沿用18,000步动态模型。俯视头顶亮像素的合成贡献约94%来自头发高斯，它们虽然在多帧可见，俯仰方向仍缺少直接观测。少观测点的贡献很小，单纯调色的实际对照也未改善碎片。</p>
<figure><a href="body-dynamic/crown-attribution.png"><img src="body-dynamic/crown-attribution.png" alt="完整四列实际诊断：原动态头部、头顶和亮像素范围、实拍语义来源、与真实观测方向的间隔"></a><figcaption>固定时刻23的原生渲染。后3列为原模型的测量着色，不是新的效果图；亮像素阈值只用于归因，不能自动标记错误几何。该俯仰没有实拍真值。</figcaption></figure>
<details><summary>真实特征对应与被排除的调色方案</summary>
<figure><a href="body-dynamic/crown-real-correspondence.png"><img src="body-dynamic/crown-real-correspondence.png" alt="四个真实视频时刻中的同一头发内部特征与三角化重投影"></a><figcaption>绿色为实拍特征位置，紫色为现有运动补偿后的三维点重投影。匹配窗口限制在头发内部，排除轮廓外背景。</figcaption></figure>
<p>38帧获得414个内部特征，形成17条至少4帧的轨迹。相同轨迹的静态三角化通过6条，现有动态补偿通过8条。它们只通过本视频内的一致性检查，尚不足以监督整个头顶，也不是真实深度或独立测试结果。下一步核对局部运动误差并扩展实拍对应。</p>
<figure><a href="body-dynamic/crown-radiance-probe.png"><img src="body-dynamic/crown-radiance-probe.png" alt="完整三列同俯仰对照：原模型、688个偏白点颜色向真实像素均值插值50%与100%"></a><figcaption>只读可逆诊断。几何和不透明度固定，头顶碎片仍在，没有采用右侧结果。</figcaption></figure>
<p>直接调暗688个偏白点后，真实头发训练L1从0.01508升至0.01704和0.02084。混合像素颜色不能当作单个高斯的真实颜色。原动态人体、衣物和地面继续保留，眼镜专项停止，本次没有新增训练或部署00。</p>
</details></div>
'''
content=content.replace(anchor,anchor+block)
old_next='<p>后续回到原高斯，用实际合成贡献追踪未知俯仰的头顶伪影，再核对相关点的真实视角支持、协方差与动态一致性。先定位后修正，保留现有纹理。原模型、衣物与地面改善继续保留，整体验收未通过，未部署00。</p>'
assert old_next in content
content=content.replace(old_next,'<p>上方已补充原高斯的合成贡献归因与实拍特征检查。继续保留原动态模型，下一步在真实对应上核对局部运动和几何，不再整块替换头发。整体验收未通过，未部署00。</p>')
html.write_text(content,encoding='utf-8')

plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
end='<!-- END_CURRENT_HEAD_HAIR_REVIEW -->';assert text.count(end)==1
addition='''本次增量：已完成原高斯的实际合成贡献回溯和688个偏白头发点的可逆颜色对照，没有新增训练。俯视亮头顶约94%来自实拍头发高斯；调色使实拍误差升高且碎片仍在，不采用。保守排除描述窗口跨越背景后，38帧414个SIFT内部特征形成17条至少4帧轨迹，动态补偿下仅8条通过当前一致性门槛。它们不是真值，覆盖不足，不能直接驱动整个头顶。下一步核对局部三角形运动而非只依赖整头刚体近似，并扩展真实对应。原主模型和查看器候选哈希不变，眼镜专项停止。

[原高斯与实拍对应](http://127.0.0.1:8870/body-dynamic.html#crown-attribution) · 同一55页PPT已修订总结与备注。
'''
text=text.replace(end,addition+end);plan.write_text(text,encoding='utf-8')
ws=read(BASE/'work-status.json');ws.update(updated=now,local_training_running=False,current_phase='dynamic_head_real_correspondence_preflight',goal_accepted=False,
 next_experiment=next_step,latest_report=dict(path=str(CANON),**report))
ws['dynamic_crown_attribution']=review
ws['latest_local_execution']=dict(kind='readonly_attribution_and_real_correspondence',optimizer_updates=0,
 successful_recorded_native_forwards=review['successful_recorded_native_forwards'],evidence=str(ROOT/'root-review.json'),main_preserved=True)
ws['goal_turn_audit']=dict(updated=now,outcome='progress',reason='Completed source-contribution attribution, reversible radiance controls, and background-safe real feature triangulation. No accepted model change.',consecutive_blocked=0)
save(BASE/'work-status.json',ws)

# Only exact disposable duplicates created by this turn, within checked roots.
targets=[ROOT/'sampling-private.npz',ROOT/'radiance-probe/sampling-private.npz',ROOT/'real-correspondence/sampling-private.npz']
draft=BUILD/'crown-candidate.pptx'
if draft.exists():
    assert sha(draft)==sha(FINAL)==sha(CANON)
    targets.append(draft)
cleanup=[]
for p in targets:
    assert p.is_file() and (p.is_relative_to(ROOT) or p==draft)
    cleanup.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
for row in cleanup:Path(row['path']).unlink()
assert not any(Path(row['path']).exists() for row in cleanup)
save(ROOT/'cleanup.json',dict(updated=now,deleted_files=len(cleanup),deleted_bytes=sum(r['bytes'] for r in cleanup),records=cleanup,
 original_sources_preserved=True,protected_checkpoints_preserved=True,previous_auto_review_blocked_paths_untouched=True))
print(json.dumps(dict(report_sha256=sha(CANON),main_unchanged=sha(main)==protected['main'],candidate_unchanged=sha(candidate)==protected['published_candidate'],
 deleted_files=len(cleanup),deleted_bytes=sum(r['bytes'] for r in cleanup),web='http://127.0.0.1:8870/body-dynamic.html#crown-attribution'),ensure_ascii=False))
