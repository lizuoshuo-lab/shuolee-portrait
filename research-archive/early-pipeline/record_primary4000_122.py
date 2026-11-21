from pathlib import Path
from datetime import datetime, timezone
import json, shutil

T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';D=R/'camera-document-review-20261028'
V=T/'viewer';S=Path(r'D:\00\工作\查看器');L=Path(r'D:\00\工作日志\2026-10-27\内参文件解析')
C=Path(__file__).parent.parent/'shuolee-coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
stamp=datetime.now(timezone.utc).isoformat()
change=dict(updated=stamp,user_steering='User judges undistorted output4000×3000 from YAML image_width/height, then downsample and center crop; diagram values may be erroneous.',
    primary_pipeline='First YAML K at4000×3000 -> resize -> center crop to actual MP4 dimensions',
    evidence_level='YAML dimensions directly verified; use as primary rectified-output working assumption per user, not independently observed exportlog.',
    old_N5_N6_already_used_4000_basis=True,diagram4096='historical alternative only, not currentprimary',
    no_fovK_switch=True,no_new_undistortion=True,no_new_fitting=True,
    rounding='Existing N4 has actualinteger3605resize andfloor/ceil center-crop algebra; ~0.5pxchange vsidealcenter. Preserve bracket, do not claim its actual implementation established.',
    next='Use4000basis andactualnativeMP4 sizes; inspect residualstructure/depth/material evidence. Do not wait or repeatedlyask for diagramcorrections.')
save(D/'primary4000-update.json',change)
v=read(D/'result.json');v['latest_user_primary_assumption']=change;save(D/'result.json',v)
paragraph='<p id="robust122-primary4000"><strong>最新主假设：第一组K对应4000×3000去畸变输出 → 下采样 → 按实际视频尺寸居中裁剪。</strong>用户依据YAML的image_width/image_height作出这一判断，4096只保留为图示中的历史替代假设。当前N5/N6本来已使用4000基准，没有把原K不缩放地套到4096上，也没有换用_fov矩阵。YAML尺寸本身直接可读；它是否恰为实际导出输出尺寸仍属于工作假设，但不为此停下独立诊断。整数缩放／裁剪的约0.5像素差异已在N4中量化，不能把未知舍入规则写成精确实测。</p>'
old='<h4 id="robust122-camera-document">2026-10-28：相机方说明、两份PDF与实际输出K</h4>'
reports=[V/'body-smplx.html',S/'progress.html',L/'01_内参文件解析与待核实项.html']
for p in reports:
    text=p.read_text(encoding='utf-8-sig');assert text.count(old)==1 and 'id="robust122-primary4000"' not in text
    text=text.replace(old,old+paragraph)
    text=text.replace('图示4096×3072与YAML4000×3000之间的K处理、实际整数裁剪偏移是剩余关键。1532按不可靠标注处理，不反复等待数字勘误。','当前以4000×3000为主假设推进；如相机方能补充，优先要最终MP4的K即可。1532和4096按历史图示处理，不反复等待勘误。')
    p.write_text(text,encoding='utf-8',newline='\n')
for root in [V/'new-capture-118/calibration-note',S/'new-capture-118/calibration-note']:shutil.copy2(reports[-1],root/reports[-1].name)
for root in [V/'122-camera-document-review',S/'122-camera-document-review',L/'122-camera-document-review',V/'new-capture-118/calibration-note/122-camera-document-review',S/'new-capture-118/calibration-note/122-camera-document-review']:
    for name in ['result.json','primary4000-update.json']:shutil.copy2(D/name,root/name)
for root in [V,S]:
    p=root/'latest-result.json';v=read(p);before={k:t for k,t in v.items() if k!='latestDiagnostic'}
    v['latestDiagnostic'].update(summary='按第一组K对应4000×3000输出、下采样、居中裁剪推进；使用MP4实际尺寸。现有N5/N6已用此基准，残余错位仍待定位。119模型保留。',reportUrl='body-smplx.html#robust122-camera-document')
    save(p,v);assert before=={k:t for k,t in read(p).items() if k!='latestDiagnostic'}
q=read(R/'queue-state.json');q.update(updated=stamp,next=change['next']);q['primary_output_assumption']=change;save(R/'queue-state.json',q)
p=read(R/'protocol.json');p['camera_source_update_20261028']['primary_output_assumption']=change;p['next']=change['next'];save(R/'protocol.json',p)
e=read(R/'evidence-ledger.json');e['updated']=stamp;e['confirmed'].append('User selects4000×3000 asprimary output workingassumption fromYAMLdimensions; existingN5/N6 already uses4000basis. Diagram4096/1532 no longer treated as required literalparameters; no rerun or silentKchange.');e['artifacts'].append('camera-document-review-20261028/primary4000-update.json');save(R/'evidence-ledger.json',e)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,next=change['next'],primary_output_assumption=change);save(C/'local-research-status.json',s)
p=T/'shuolee/long-term-plan.md';text=p.read_text(encoding='utf-8-sig');text=text.replace('<!-- ROBUST122_END -->','\n最新补充：用户判断第一组K对应4000×3000去畸变输出，再下采样居中裁剪。以此为主假设继续；4096只留历史对照。N5/N6已用4000基准，不重复运行。\n<!-- ROBUST122_END -->');p.write_text(text,encoding='utf-8',newline='\n')
print(json.dumps(dict(primary4000_recorded=True,currentN5N6_already4000=True,no_training=True),ensure_ascii=False))
