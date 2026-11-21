"""Resume the interrupted path-only publication, without repeating experiments or ledger entries."""
from pathlib import Path
from datetime import datetime,timezone
import json,re
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/robust-portrait-122/native-residual-pipeline';V=T/'viewer';S=Path('D:/00/工作/查看器')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
text=(V/'body-smplx.html').read_text(encoding='utf-8')
article=re.search(r'<!-- R7_NATIVE_PIPELINE_BEGIN -->.*?<!-- R7_NATIVE_PIPELINE_END -->',text,re.S).group()
p=S/'progress.html';s=p.read_text(encoding='utf-8');assert '<!-- R7_NATIVE_PIPELINE_BEGIN -->' not in s
s=s.replace('<article id="face-observation">',article+'<article id="face-observation">',1);p.write_text(s,encoding='utf-8')
d=read(S/'latest-result.json');preserved={k:v for k,v in d.items() if k!='latestDiagnostic'}
d['latestDiagnostic']=read(V/'latest-result.json')['latestDiagnostic'];d['latestDiagnostic']['reportUrl']='progress.html#robust122-r7';save(S/'latest-result.json',d)
assert {k:v for k,v in read(S/'latest-result.json').items() if k!='latestDiagnostic'}==preserved
save(O/'publication-path-recovery.json',dict(updated=stamp,original_error='D:/00/工作/查看器/body-smplx.html does not exist; sync viewer uses progress.html.',research_viewer_already_updated=True,sync_progress_updated=True,no_experiment_repeated=True))
save(O/'publication.json',dict(updated=stamp,report='http://127.0.0.1:8870/body-smplx.html#robust122-r7',daily_log='D:/00/工作日志/2026-10-29/人物对齐核验123/index.html',historical_model_preserved=True,browser_DOM_verified=False,actual_browser_pixel_review=False,PPT_updated=False,physical_quality_accepted=False))
print('Resumed sync progress publication; model unchanged.')
