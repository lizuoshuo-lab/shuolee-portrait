"""Record the completed, read-only scene-support audit without changing cameras."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, html, json, shutil

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
O=REC/'grayT-scene-support-cross-run-review'
R=T/'shuolee/current/robust-portrait-122'
C=H.parent/'shuolee-coordination'
L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    q.replace(p)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
result=read(O/'result.json'); idx=read(O/'index.json'); membership=read(O/'BA-membership-review.json')
assert result['source_hashes_unchanged'] and result['no_new_features'] and result['no_camera_or_point_optimization']
assert len(idx['images'])==14
notes={
 'd81d70b0e8':'原始末端与前两幅树木区域不一致。成员清单显示末端57帧不在BA中；实际36/41/45帧均为树叶区域，仍未证明叶片静止或同一材料身份。',
 '7fdd85d809':'原始轨迹从路面错接到屋顶/栏杆区域，但54帧末端未进入BA。实际37/42/48帧均是路面纹理，不能用原始错端指控实际BA受污染。',
 'd0326813da':'实际54/58/61帧为栏杆支架区域，全部7节点参与BA。固定物体看似一致，但平滑金属/边缘和阴影上的亚像素材料身份未确认。',
 '0448f0272c':'重复栏杆竖条，无法仅凭三幅局部裁片确认是同一根、同一物理位置。',
 '4f1c7c45b1':'远处道路、标牌与建筑结构，存在图像边缘黑色填充；固定结构候选，但精确材料身份及深度未验证。',
 'ae29b3bac6':'三视角中蓝色小路牌图案大致一致，属于固定结构候选；亚像素中心定位和物理深度仍未验证。',
 'fbcd0c1b5b':'重复围栏纹理，条纹之间的对应歧义未消除。',
 '04b37e569b':'墙沿、黑色栏杆支柱交界处，固定结构候选；角点测量与同物理点精度未验证。',
 '5729b51fb8':'灰色围栏竖条高度重复，局部图像不能证明材料身份唯一。',
 'a0a42f181c':'栏杆支架和上沿区域大致一致，但光照边缘与实际材料点仍未区分。',
 'fbdf64b04a':'栏杆上沿与支柱有较强透视变化，相邻支柱可能混淆，材料身份保留未知。',
}
members={x['track_id']:x for x in membership['records']}
reviews=[]
for im in idx['images']:
    key=im['frame'][:10]; note=notes[key]
    actual='_actually_used_BA' in im['file']
    if im['frame'] in members:
        m=members[im['frame']]
        original_label='原始FIT轨迹族，全部节点进入BA' if set(m['original_frames'])==set(m['BA_frames']) else '原始FIT轨迹族，部分节点未进入BA'
        im['region']=('实际进入BA的观测' if actual else original_label)+'；'+note
        for source in im.get('sources',[]):
            f=int(source['frame'].split('_')[-1])
            source['used_in_saved_BA']=f in m['BA_frames']
            source['role']='SAVED_BA_OBSERVATION' if source['used_in_saved_BA'] else 'ORIGINAL_FIT_FAMILY_REJECTED_BEFORE_BA'
    else:
        im['region']='原有留出候选，尚未认证物理身份；'+note
    reviews.append(dict(file=im['file'],track_id=im['frame'],columns=im['columns'],actual_visual_review=True,
                        used_BA_nodes_only=actual,observation=note,physical_identity_certified=False,
                        file_sha256=sha(O/im['file'])))
save(O/'index.json',idx)
summary=('两组灰衣人物相机都存在尚未通过的局部验证：预登记的头部附近图像区域、40至80帧中，研究侧实际BA支持为0条，00复现仅1条、7个观测。'
 '该区域原有留出残差中位分别为1.19和1.50原生像素，22与50个重建观测出现非正深度。实际查看14组三视角裁片后，重复栏杆、树叶和平滑金属的材料身份仍有歧义。'
 '两条原始FIT族的错误末端在BA前已剔除，不能把它们称为已参与优化的错误点。00的特征确实位于人物排除遮罩之外。'
 '当前发现的是头部附近独立场景验证不足，不是已经证明外参错误；远处或其他区域的特征也能约束相机。相机误差与人物微动仍未唯一分离。')
next_text=('先检查现有原图中固定栏杆/建筑的直线方向是否能提供独立旋转检验；只做材料与可辨识性检查，不重新提特征、不拟合新相机。'
 '这一检验只可能约束部分旋转，不能单独确定平移或证明内参正确。若不存在可确认的固定方向，保留缺口。灰衣正式静态训练继续保持未准入。')
metrics={}
for name,run in result['runs'].items():
    metrics[name]={role:run['groups'][role]['near_head_band_during_M0'] for role in ['FIT','HELD']}
review=dict(updated=stamp,actual_visual_review=True,quality_accepted=False,summary=summary,records=reviews,
    original_component_vs_saved_BA_membership_checked=True,reviewed_triplets=14,reviewed_native_crops=42,
    removed_observations=0,held_to_fit_transfers=0,physical_material_identity_verified=False)
save(O/'actual-review.json',review)
con=dict(updated=stamp,summary=summary,next=next_text,metrics=metrics,
    image_roi_is_not_physical_depth=True,head_height_static_3D_truth_available=False,
    camera_error_uniquely_established=False,person_motion_uniquely_established=False,
    conditional_K_physical_calibration_verified=False,quality_accepted=False,training_started=False,
    bounds={'camera_updates':0,'point_updates':0,'new_features':0,'new_training':0,'error_based_point_removal':0},
    native_threshold_caveat='00以899×1600工作，研究侧为1520×2704。同样3工作像素门槛在00约为5.07原生像素；分辨率、特征、遮罩和筛选容差共同变化，不能作单变量归因。',
    memberships=membership['records'],evidence=[str(O/x) for x in ['plan.json','result.json','index.json','actual-review.json','BA-membership-review.json']])
save(REC/'scene-support-cross-run-conclusion.json',con)
q=read(REC/'queue-state.json');q.update(updated=stamp,current_stage='scene-support-cross-run-reviewed',
    reason=summary,latest_diagnosis=str(REC/'scene-support-cross-run-conclusion.json'),
    training_allowed=False,training_started=False,training_running=False,goal_stays_active=True)
save(REC/'queue-state.json',q)
n=read(REC/'next-action.json');n.update(updated=stamp,next=next_text,
    latest_diagnosis=str(REC/'scene-support-cross-run-conclusion.json'),latest_actual_checkpoint_review=str(O/'actual-review.json'),
    completed_this_turn=['Read-only two-run background support audit, all95frames checked.','14native triplets reviewed; original vs actual BA membership distinguished.'],
    next_plan=None,diagnostic_DOM_verified=False,actual_UI_verified=False,training_allowed=False)
save(REC/'next-action.json',n)
ledger=read(REC/'evidence-ledger.json');entryid='grayT123-two-run-scene-support-membership'
assert all(x.get('id')!=entryid for x in ledger['entries']), 'Do not append a duplicate audit'
ledger['entries'].append(dict(id=entryid,updated=stamp,summary=summary,evidence=str(REC/'scene-support-cross-run-conclusion.json'),quality_accepted=False))
ledger['updated']=stamp;save(REC/'evidence-ledger.json',ledger)
r=read(R/'evidence-ledger.json');r['updated']=stamp;r.setdefault('confirmed',[]).append(summary)
r.setdefault('artifacts',[]).append(str(REC/'scene-support-cross-run-conclusion.json'));save(R/'evidence-ledger.json',r)
rq=read(R/'queue-state.json');rq.update(updated=stamp,current_stage='local123_scene_support_cross_run_reviewed',
    latest_evidence=str(REC/'scene-support-cross-run-conclusion.json'),quality_accepted=False);save(R/'queue-state.json',rq)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,next=next_text,
    current_local_scene_support_control=con,company00_independent_training_not_local_admission=True)
save(C/'local-research-status.json',c)
lp=T/'shuolee/long-term-plan.md'
text=lp.read_text(encoding='utf-8')
marker='### 2026-10-29 灰衣人物两轮场景支持核查'
assert marker not in text
lp.write_text(text+'\n\n'+marker+'\n\n'+summary+'\n\n下一步：'+next_text+'\n',encoding='utf-8')
L.mkdir(parents=True,exist_ok=True)
(L/O.name).mkdir(exist_ok=True)
for p in O.iterdir():
    if p.is_file():shutil.copy2(p,L/O.name/p.name)
shutil.copy2(REC/'scene-support-cross-run-conclusion.json',L/'scene-support-cross-run-conclusion.json')
tables=[]
for name,label in [('research','研究侧'),('company00','00对话本机复现')]:
    m=metrics[name];f=m['FIT'];h=m['HELD'];e=h['native_reprojection']
    tables.append(f'<tr><td>{label}</td><td>{f["tracks"]}条／{f["native_reprojection"]["n"]}观测</td><td>{h["tracks"]}条／{e["n"]}观测</td><td>{e["median"]:.3f}px</td><td>{e["ge1"]}/{e["n"]}</td><td>{h["nonpositive_depth"]}</td></tr>')
figures=[]
for im in idx['images']:
    figures.append('<figure><figcaption>'+html.escape(im['region'])+'<br>'+html.escape(' / '.join(im['columns']))+'</figcaption><img src="'+O.name+'/'+im.get('locator',im['file'])+'"><p><a href="'+O.name+'/'+im.get('raw',im['file'])+'">无标记原生裁片</a></p></figure>')
doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>2026-10-29 人物对齐：场景支持核查</title><style>body{font:18px/1.7 system-ui;max-width:1200px;margin:auto;padding:24px;background:#10202c;color:#eef4f7}a{color:#9bd6ff}img{max-width:100%;height:auto}td,th{padding:10px;border-bottom:1px solid #536977;text-align:left}figure{margin:36px 0}</style><h1>SMPL-X对齐：头部附近场景验证仍不足</h1>'''
doc+='<p>'+html.escape(summary)+'</p><p>统计区域由拟合头部附近的原生图像ROI预先定义，不代表相同物理深度。以下留出是既有轨迹的次级核验，尚未认证每条材料身份；未按误差删除观测。</p>'
doc+='<table><thead><tr><th>运行</th><th>实际BA支持</th><th>已有留出</th><th>中位误差</th><th>≥1px</th><th>非正深度</th></tr></thead><tbody>'+''.join(tables)+'</tbody></table>'
doc+='<p>'+html.escape(con['native_threshold_caveat'])+'</p><p>本轮未修改相机、点坐标、人物模型或训练状态。未发布新点云，历史研究模型保留；当前结果没有达到质量验收。</p><p>下一步：'+html.escape(next_text)+'</p>'
doc+='<p><a href="http://127.0.0.1:8870/body-smplx.html#scene-support-cross-run">查看器中的本轮说明</a> · <a href="../../2026-10-28/人物自由表面训练123/index.html">前一日工作记录</a></p>'+''.join(figures)+'</html>'
(L/'index.html').write_text(doc,encoding='utf-8')
save(REC/'scene-support-cross-run-completion.json',dict(updated=stamp,state='evidence_recorded_publication_pending',
    daily_log=str(L/'index.html'),actual_triplet_review=True,training_started=False,quality_accepted=False,
    diagnostic_DOM_verified=False,browser_pixel_review=False,PPT_updated=False))
print(json.dumps(dict(conclusion=str(REC/'scene-support-cross-run-conclusion.json'),daily_log=str(L/'index.html'),triplets=len(reviews)),ensure_ascii=False))
