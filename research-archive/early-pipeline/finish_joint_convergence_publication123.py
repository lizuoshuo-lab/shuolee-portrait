"""Resume after the worklog's previously missing section, without repeating experiments or ledger appends."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';G=R/'video10-global-initializer';P=G/'joint-track-candidate';CR=P/'convergence-repair';D=R/'video10-correspondence-contrast/head-roi-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (CR/'publication.json').exists()
d=read(CR/'decision.json');stamp=datetime.now(timezone.utc).isoformat()
save(CR/'publication-initial-failure.json',dict(updated=stamp,error='Worklog section video10-global-ambiguity absent. First publisher stopped on assertion; existing reports and queue already updated. No experiment repeated.',recovery='Insert the missing section in the existing daily worklog, then perform remaining copies.'))
path=Path('D:/00/工作日志/2026-10-29/人物对齐核验123/index.html');txt=path.read_text(encoding='utf-8');assert 'id="video10-global-ambiguity"' not in txt
section='<section id="video10-global-ambiguity"><h1>SMPL-X对齐与相机数值求解</h1><p>'+d['summary']+'</p><p>'+d['roi_conclusion']+'</p><p><a href="相机多解与人物对齐/index.html#convergence-repair">全部原生裁片及证据</a></p></section>'
assert '</style>' in txt;path.write_text(txt.replace('</style>','</style>'+section,1),encoding='utf-8')
pub=dict(updated=stamp,documents_saved=True,viewer_UI_checked=False,PPT_updated=False,model_replaced=False,goal_complete=False,training_started=False,quality_accepted=False)
save(CR/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(CR/'publication.json')))
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    for name in ['index.html','decision.json','publication.json']:shutil.copy2(G/name,dest/name)
    shutil.copytree(CR,dest/'joint-track-candidate/convergence-repair',dirs_exist_ok=True)
    shutil.copytree(P/'joint-synthetic-control',dest/'joint-track-candidate/joint-synthetic-control',dirs_exist_ok=True)
    out=dest/'head-roi-contrast';out.mkdir(exist_ok=True)
    for name in ['plan.json','result.json','native-patch-control-plan.json']:shutil.copy2(D/name,out/name)
    shutil.copytree(D/'visual-review',out/'visual-review',dirs_exist_ok=True)
print(json.dumps(dict(publication=str(CR/'publication.json'),missing_daily_section_added=True,experiments_repeated=False),ensure_ascii=False))
