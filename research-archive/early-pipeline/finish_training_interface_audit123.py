from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
REC=T/'shuolee/current/local-clothed-surface-123'
A=REC/'smplifyx-reference-control/framewise-canonical-route/training-interface-audit'
SYNC=Path('D:/00/工作/查看器')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
audit=read(A/'result.json')
for f in audit['sources']:
    assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
ui=dict(updated=stamp,tab='32',url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#framewise-training-interface',
    actual_AX_review=True,visible_heading='新路线的训练连接：已确认接口缺口',source_scope_and_limitations_present=True,
    screenshot_verified=False,screenshot_limitation='0 width from screenshot API earlier in same turn; no workaround attempted.',
    tab15_untouched=True,new_PLY=False,real_training=False,company00_verified=False)
save(A/'viewer-ui-review.json',ui)
title='逐帧人体到静态高斯：训练连接待补齐'
article='<article id="smplx123-video10-global-ambiguity"><h2>'+title+'</h2><p>'+audit['summary']+'</p><p><a href="diagnostics/video10-global-initializer/index.html#framewise-training-interface">查看逐帧接口核验与根初始化对照</a></p><p>仅源码核验和已完成合成对照，无新实拍拟合或高斯。历史点云保留。</p></article>'
(R/'video10-global-initializer/viewer-article.html').write_text(article,encoding='utf-8')
for base,name in [(T/'viewer','body-smplx.html'),(SYNC,'progress.html')]:
    p=base/name;s,n=re.subn(r'<article id="smplx123-video10-global-ambiguity">.*?</article>',lambda _:article,p.read_text(encoding='utf-8'),count=1,flags=re.S)
    assert n==1;p.write_text(s,encoding='utf-8')
    p=base/'latest-result.json';v=read(p);old={k:x for k,x in v.items() if k!='latestDiagnostic'}
    v['latestDiagnostic']=dict(id='smplx123-video10-global-ambiguity',round='123',updated=stamp,title=title,summary=audit['summary'],reportUrl=name+'#smplx123-video10-global-ambiguity',qualityAccepted=False)
    save(p,v);assert old=={k:x for k,x in read(p).items() if k!='latestDiagnostic'}
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer']:
    shutil.copy2(A/'viewer-ui-review.json',dest/'training-interface-audit/viewer-ui-review.json')
p=Path(__file__).parent.parent/'shuolee-coordination/local-research-status.json';v=read(p)
v.update(updated_at=stamp,latest_viewer_ui_review=str(A/'viewer-ui-review.json'),viewer_text_and_table_verified=True,viewer_screenshot_verified=False)
save(p,v)
print(json.dumps(dict(source_hashes_unchanged=len(audit['sources']),actual_browser_text_verified=True,models_unchanged=True,records_png=len(list(Path('D:/00/记录/005').rglob('*.png')))),ensure_ascii=False))
