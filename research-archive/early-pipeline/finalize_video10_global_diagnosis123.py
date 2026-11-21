"""Read-only FIT-cycle explanation, material review notes and unified diagnostic publication."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,html,shutil,itertools
import numpy as np
from PIL import Image
import match_video10_temporal_bridge123 as base
from global_track_initializer123 import angle_between_rotations
H=Path(__file__).parent;T=base.W if hasattr(base,'W') else base.base.W;R=base.R;G=R/'video10-global-initializer';REC=R.parent/'local-clothed-surface-123'
U=R/'video10-remaining-material-review/global-input-preflight';D=G/'relative-graph/repair-invalid-pose'
read=base.read;save=base.save;sha=base.sha;stamp=base.now()
pairrows=read(D/'pairs.json');result=read(D/'result.json');roots=read(G/'minimal-root-diagnosis/result.json');synthetic=read(G/'result-completed.json')
assert not (G/'decision.json').exists()
# Freeze this explanatory check before calculating its output. No pose is re-estimated.
cycleplan=dict(registered=stamp,frames=[698,725,752],selection='Post-hoc explanation of the previously enumerated pair. Middle-frame FITposes already computed; no HELD input to root ranking.',
 method='Rank three saved five-point roots only by distance from R_725_to_752 times R_698_to_725. Do not publish as camera; the10point adjacent edge remains below12support admission.',
 thresholds='No new acceptance or parameter threshold; report all three closure values. This is descriptive, not a global solve.',
 inputs={str(D/'pairs.json'):sha(D/'pairs.json'),str(G/'minimal-root-diagnosis/result.json'):sha(G/'minimal-root-diagnosis/result.json')},training_started=False)
save(G/'minimal-root-diagnosis/cycle-plan.json',cycleplan)
eab=next(p for p in pairrows if (p['a'],p['b'])==(698,725));ebc=next(p for p in pairrows if (p['a'],p['b'])==(725,752))
reference=np.asarray(ebc['R_b_from_a'])@np.asarray(eab['R_b_from_a'])
angles=[float(np.degrees(angle_between_rotations(np.asarray(p['R']),reference))) for p in roots['roots']]
best=int(np.argmin(angles));cycle=dict(updated=stamp,plan_sha256=sha(G/'minimal-root-diagnosis/cycle-plan.json'),closure_deg=angles,smallest_closure_root=best,
    ranking_uses='Only saved FIT relative rotations',heldout_used_for_ranking=False,
    original_root=2,original_heldout=roots['roots'][2]['heldout_sampson_px'],ranked_root_heldout=roots['roots'][best]['heldout_sampson_px'],
    admitted=False,reason='At least one supporting edge has10<12FIT; local closure and post-hoc root ranking do not certify global camera or physical identity.')
save(G/'minimal-root-diagnosis/cycle-result.json',cycle)
idx=read(G/'visual-review/index.json')
notes={1229:'Both centers lie at the air-conditioner vent/mount area. Same fixture recognizable; blur and foreshortening prevent subpixel physical certification.',
1249:'Both centers lie on the cap/logo by the bag. Coarse object identity recognizable; flexible material and edge blur remain.',
1291:'Both centers lie within the framed display/poster. It has visible glass highlights and blur; the exact center is not certified as a static material point.',
1295:'Both centers lie on/near the toaster top and back wall boundary. First crop noticeably blurred; exact feature location uncertain.',
1298:'Both centers lie near the rug boundary/floor joint. Border shape, perspective and blur keep exact material identity uncertain.',
244:'HELD on the air-conditioner fixture at a shifted location relative toFIT1229. Coarse fixture overlap, physical independence not established.',
1297:'HELD near a floor tile junction; both source views show the same local layout. Some person pixels at left border of second crop, while center remains on floor. Blur/repeated tile pattern prevent subpixel certification.'}
save(G/'visual-review/review.json',dict(updated=stamp,actual_saved_image_review=True,pages=['page_1.png','page_2.png','page_3.png'],
    families=[dict(family=r['family'],split=r['split'],note=notes[r['family']],certified_physical_subpixel=False) for r in idx['rows']],
    no_filter_change=True,heldout_retained=True,summary='All5FIT and2HELD crops viewed. Coarse fixed-scene context, blur, glass and flexible materials are visible. No new incorrect identity is declared solely from the high residual.'))
fits={t['family']:t for t in read(U/'heldout-observations.json')['tracks']};obs={o['frame']:o for o in fits[1297]['observations']}
assert all(f in obs for f in [698,725,752]);strip=Image.new('RGB',(1152,384),(235,235,235))
for j,f in enumerate([698,725,752]):
    x,y=np.rint(obs[f]['uv']).astype(int);im=Image.open(base.imgpath(f)).convert('RGB');strip.paste(im.crop((x-192,y-192,x+192,y+192)),(j*384,0))
figure=G/'visual-review/heldout_floor_1297_frames_698_725_752.png';strip.save(figure)
summary=f'视频10固定条件K的58对视角已检验。57对返回可用相对姿态，仅16条边满足预登记支持/视差保护，33视角仍分成{len(result["robust_components"])}组。32个三视角闭环最大差67.40°。第698/752帧的5个FIT存在3组全部正深度、近零拟合误差的解，旋转相差约42°；原选解的一个HELD极线误差达86.08px。'
interpretation='这确认了一个具体失败机制：少量匹配的两视图求解可以选到错误分支，单对拟合误差不能证明外参正确。背景重复纹理/模糊、点数不足和条件K误差仍须区分；结果不证明鱼眼残差或SMPL-X骨架映射是唯一根因。'
nexttext='在现有33视角、冻结FIT与固定K上，检查共享三视角材料能否联合约束多解及尺度；先评估全局分支/三视图支持，不重提特征、不降低12点保护、不靠HELD挑解。不满足时保留具体缺口，不启动SMPL-X或高斯训练。'
decision=dict(updated=stamp,state='global_initializer_controls_and_relative_ambiguity_diagnosed',summary=summary,interpretation=interpretation,
    synthetic_controls=4,synthetic_pass=True,relative_result=str(D/'result.json'),minimal_roots=str(G/'minimal-root-diagnosis/result.json'),cycle_result=str(G/'minimal-root-diagnosis/cycle-result.json'),
    exact_facial_heldout=0,new_global_cameras=0,new_body_fit=False,training_started=False,quality_accepted=False,
    correction='Synthetic final progress writer failed after all controls; recovered without rerun. Relative probe immutable-pose and invalid-zero-pose handling corrected; original attempts retained, estimator/seed/data unchanged.',
    next=nexttext)
save(G/'decision.json',decision)
rootlines=''.join(f'<tr><td>{i}</td><td>{r["fit_sampson_max_px"]:.2e}</td><td>{r["positive_depth_count"]}/5</td><td>{angles[i]:.3f}°</td><td>{r["heldout_sampson_px"][0]:.3f} / {r["heldout_sampson_px"][1]:.3f}</td></tr>' for i,r in enumerate(roots['roots']))
tables=''.join(f'<tr><td>{r["a"]} → {r["b"]}</td><td>{r["count"]}</td><td>{r.get("inlier_count",0)}</td><td>{html.escape(", ".join(r.get("rejection_reasons",[])) or "local checks passed")}</td></tr>' for r in pairrows)
photos=''.join(f'<figure><img src="visual-review/{r["file"]}"><figcaption>{r["split"].upper()}族{r["family"]}，左698／右752；各384px原生裁片，中心为查询坐标。{html.escape(notes[r["family"]])}</figcaption></figure>' for r in idx['rows'])
body=f'''<h1>相机多解与人物对齐</h1><p class="status">2026-10-29 · 实拍诊断完成 · 人物质量未通过</p><p>{summary}</p><p>{interpretation}</p>
<h2>为什么拟合误差几乎为零，人物仍对不齐</h2><p>对第698、752帧相同5组FIT，固定相同内参，五点方程返回3组正深度解。下面全部保留；没有用HELD挑选相机。</p>
<table><tr><th>解</th><th>FIT最大Sampson距离 / px</th><th>正深度</th><th>经725帧旋转闭环</th><th>两组HELD距离 / px</th></tr>{rootlines}</table>
<p>仅使用FIT相邻旋转的局部闭环，更支持解{best}。但一条支持边只有10组FIT，未达到预登记12组门槛；独立点仍未全部低于1px。这里的数值是极线Sampson距离，不是三维重投影误差，也不是面部亚像素验收。</p>
<h2>已经确认与仍未知</h2><ul><li>4项合成全局初始化控制通过，包含纯旋转和断图拒绝；只证明数学实现。</li><li>33个真实诊断视角的可靠图未连通；没有产生完整新相机或人体。</li><li>同一物体的重复图案可能不是同一物理点，前轮两处飞镖盘错配保留隔离。</li><li>K仍以4000×3000、缩放、居中裁剪为条件推导；实际导出K未独立测量。</li><li>已去畸变MP4不做二次鱼眼去畸变；本轮固定K和D=0，不能归因于K改善。</li><li>同一批特征包含模糊、玻璃高光和软物体；粗目视一致不是1px真值。</li></ul>
<h2>原生裁片检查</h2><p>仅做原生整数裁剪，不缩放、不二维扭曲、不在图上写字。事后解释案例；原有FIT/HELD划分和像素保留不变。</p>{photos}
<h2>全部58对记录</h2><table><tr><th>帧</th><th>共享FIT</th><th>拟合内点</th><th>局部保护结果</th></tr>{tables}</table>
<h2>实现错误保留</h2><p>{html.escape(decision['correction'])}</p><h2>下一步</h2><p>{nexttext}</p>'''
css='body{font:17px/1.65 system-ui,sans-serif;max-width:1160px;margin:36px auto;padding:0 24px;color:#17212c;background:#f7f9fc}h1,h2{color:#16335a}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{padding:8px 12px;border:1px solid #cfd7e3;text-align:left}.status{color:#a34216}img{max-width:100%;height:auto}figure{margin:28px 0;padding:16px;background:white}figcaption{font-size:14px;color:#485766}'
(G/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>相机多解与人物对齐</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
article=f'<article id="smplx123-video10-global-ambiguity"><h2>相机多解：5个点也能拟合出3组不同姿态</h2><p>{summary}</p><p>{interpretation}</p><p><a href="diagnostics/video10-global-initializer/index.html">完整记录、3组解与原生裁片</a></p><p>没有更新人物点云，当前历史模型仍为质量未通过的对照。下一步：{nexttext}</p></article>'
(G/'viewer-article.html').write_text(article,encoding='utf-8')
for folder,name in [(T/'viewer','body-smplx.html'),(Path('D:/00/工作/查看器'),'progress.html')]:
    p=folder/name;s=p.read_text(encoding='utf-8');assert 'id="smplx123-video10-global-ambiguity"' not in s
    anchor='<article id="smplx123-video10-remaining-support"';assert anchor in s;s=s.replace(anchor,article+'\n'+anchor,1);p.write_text(s,encoding='utf-8')
    p=folder/'latest-result.json';x=read(p);unchanged={k:v for k,v in x.items() if k!='latestDiagnostic'}
    x['latestDiagnostic']=dict(id='smplx123-video10-global-ambiguity',round='123',updated=stamp,title='相机多解与人物对齐',summary=summary,reportUrl=name+'#smplx123-video10-global-ambiguity',qualityAccepted=False)
    save(p,x);assert unchanged=={k:v for k,v in read(p).items() if k!='latestDiagnostic'}
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    dest.mkdir(exist_ok=True);shutil.copy2(G/'index.html',dest/'index.html');shutil.copy2(G/'decision.json',dest/'decision.json')
    shutil.copytree(G/'visual-review',dest/'visual-review',dirs_exist_ok=True)
    for src,n in [(D/'result.json','relative-result.json'),(D/'pairs.json','pairs.json'),(G/'minimal-root-diagnosis/result.json','minimal-roots.json'),(G/'minimal-root-diagnosis/cycle-result.json','cycle-result.json')]:shutil.copy2(src,dest/n)
for root in [R,REC]:
    p=root/'queue-state.json';x=read(p);x.update(updated=stamp,current_stage='video10_relative_graph_ambiguity_diagnosed',phase='shared_multiview_branch_support',running=None,next=nexttext,next_action=nexttext,reason=summary,latest_diagnosis=str(G/'decision.json'))
    x['last_goal_turn']=dict(updated=stamp,classification='progress',result=str(G/'decision.json'),summary=summary,next=nexttext);save(p,x)
    p=root/'protocol.json'
    if p.exists():x=read(p);x.update(updated=stamp,current_stage='video10_relative_graph_ambiguity_diagnosed',next=nexttext);save(p,x)
p=REC/'next-action.json';x=read(p);x.update(updated=stamp,next=nexttext,next_action=nexttext,latest_result=str(G/'decision.json'));save(p,x)
p=R/'evidence-ledger.json';x=read(p);x['updated']=stamp;x['confirmed'].append(summary+' '+interpretation);x['artifacts'] += [str(G/'decision.json'),str(G/'result-completed.json'),str(D/'result.json')];save(p,x)
p=REC/'evidence-ledger.json';x=read(p);x['entries'].append(dict(id='video10_global_relative_ambiguity',updated=stamp,summary=summary,evidence=str(G/'decision.json'),quality_accepted=False));save(p,x)
p=H.parent/'shuolee-coordination/local-research-status.json';x=read(p);x.update(updated_at=stamp,source_queue_updated=stamp,next=nexttext,current_video10_global_initializer=decision);save(p,x)
p=T/'shuolee/long-term-plan.md';s=p.read_text(encoding='utf-8');p.write_text(s+'\n\n'+summary+'\n\n'+interpretation+'\n\n'+nexttext+'\n',encoding='utf-8')
p=Path('D:/00/工作日志/2026-10-29/人物对齐核验123/index.html');s=p.read_text(encoding='utf-8');s=s.replace('</body>','<section id="video10-global-ambiguity"><h2>相机多解与人物对齐</h2><p>'+summary+'</p><p>'+interpretation+'</p><p><a href="相机多解与人物对齐/index.html">原生裁片与全部数值</a></p></section></body>',1);p.write_text(s,encoding='utf-8')
save(G/'publication.json',dict(updated=stamp,documents_saved=True,viewer_UI_checked=False,PPT_updated=False,model_replaced=False,actual_new_model_quality_review=False))
print(json.dumps(dict(summary=summary,cycle=cycle,figure=str(figure)),ensure_ascii=False))
