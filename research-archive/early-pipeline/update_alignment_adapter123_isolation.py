from pathlib import Path
from datetime import datetime,timezone
import hashlib,html,json,re,subprocess,time,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'generic-2d3d-research/adapter-a0';G=R/'video10-global-initializer'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118');SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def linux(p):return '/mnt/'+p.drive[0].lower()+p.as_posix()[2:]
log=(O/'cpu-tests-source-isolation.log').read_text(encoding='utf-8');assert 'Ran 17 tests' in log and log.rstrip().endswith('OK')
start=time.monotonic()
run=subprocess.run(['wsl','-e','bash','-lc',
    'PYTHONNOUSERSITE=1 /opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python '+linux(P/'run.py')+' fit-inputs --output '+linux(O/'real-native-intake')],capture_output=True)
assert run.returncode==0,run.stderr.decode('utf-8',errors='replace')
real=json.loads(run.stdout.decode('utf-8'));assert real['frames']==177 and not real['fitting_allowed']
save(O/'real-cli-preflight.json',real)
stamp=datetime.now(timezone.utc).isoformat();d=read(O/'result.json');save(O/'result-before-source-isolation.json',d)
d.update(updated=stamp,unit_and_CLI_tests=17,coarse_face_supervision_isolated=True,
    observation_objective_change='The candidate uses one fixed51 facial source; coarse COCO facial weights0..4 are diagnostic-only. This is a generic source-ownership rule, not subject-specific pruning.',
    next_comparison_scope='Old versus complete new objective; cannot attribute a future improvement to one component.',
    cumulative_CPU_seconds=d['cumulative_CPU_seconds']+6.1404969+time.monotonic()-start,
    test_script_sha256=sha(P/'scripts/check_alignment_inputs.py'))
for name in d['actual_modules']:d['actual_modules'][name]=sha(P/'src/shuolee_portrait'/name)
assert d['cumulative_CPU_seconds']<120
save(O/'result.json',d)
for root in [R,REC]:
    p=root/'queue-state.json';q=read(p)
    for key in ['reason']:q[key]=q.get(key,'').replace('16项','17项')
    q['last_goal_turn']['summary']=q['last_goal_turn']['summary'].replace('16项','17项')+' 统一关闭重复的粗面部监督，防止跨来源绕过留出。'
    q['updated']=stamp;save(p,q)
    p=root/'evidence-ledger.json';q=read(p)
    for e in q.get('entries',[]):
        if e.get('id')=='generic_2d3d_A0_adapter':e.update(updated=stamp,summary=e['summary'].replace('16项','17项'),source_isolation_addendum=str(O/'plan.json'))
    save(p,q)
note='精细51点是唯一面部拟合来源。粗COCO17中的鼻、眼、耳五点只保留诊断，防止另一检测器绕过面部留出；其余身体点不变。这是所有人物共享的规则。'
p=G/'index.html';s=p.read_text(encoding='utf-8');start=s.index('<section id="generic-2d3d-adapter">');end=s.index('<section id="generic-2d3d-research">',start)
section=s[start:end].replace('16项','17项')
section=re.sub(r'本轮累计CPU核验约[0-9.]+秒','本轮CPU核验在预登记120秒预算内',section)
section=section.replace('<h3>通用工程规则</h3>','<h3>通用工程规则</h3><p>'+note+'</p>')
s=s[:start]+section+s[end:];p.write_text(s,encoding='utf-8')
doc=(P/'docs/通用2D到SMPL-X接口.md').read_text(encoding='utf-8')
(O/'接口说明.html').write_text('<!doctype html><meta charset="utf-8"><title>通用二维到SMPL-X接口</title><style>body{font:16px/1.8 system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><pre>'+html.escape(doc)+'</pre>',encoding='utf-8')
for folder,name in [(T/'viewer','body-smplx.html'),(SYNC,'progress.html')]:
    p=folder/name;s=p.read_text(encoding='utf-8');m=re.search(r'<article id="smplx123-video10-global-ambiguity">.*?</article>',s,re.S);assert m
    s=s[:m.start()]+m.group().replace('16项','17项')+s[m.end():];p.write_text(s,encoding='utf-8')
    p=folder/'latest-result.json';q=read(p);q['latestDiagnostic']['summary']=q['latestDiagnostic']['summary'].replace('16项','17项');q['latestDiagnostic']['updated']=stamp;save(p,q)
p=LOG/'index.html';s=p.read_text(encoding='utf-8');m=re.search(r'<section id="video10-global-ambiguity">.*?</section>',s,re.S);assert m;s=s[:m.start()]+m.group().replace('16项','17项')+s[m.end():];p.write_text(s,encoding='utf-8')
p=T/'shuolee/long-term-plan.md';s=p.read_text(encoding='utf-8');head,tail=s.rsplit('### 2026-10-29 通用二维与SMPL-X接口A0',1);p.write_text(head+'### 2026-10-29 通用二维与SMPL-X接口A0'+tail.replace('16项','17项')+'\n'+note+'\n',encoding='utf-8')
p=O/'slide-review.json';q=read(p);q['result']=q['result'].replace('16项','17项');q['subtitle']='固定51点面部约束与跨来源监督去重';q['notes']=[x.replace('16项','17项') for x in q['notes']]+[note];save(p,q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q['generic_2d3d_adapter'].update(coarse_face_source_isolated=True,unit_and_CLI_tests=17);q['updated_at']=stamp;save(p,q)
for dest in [G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer',LOG/'相机多解与人物对齐']:
    if dest!=G:shutil.copy2(G/'index.html',dest/'index.html')
    folder=dest/'generic-2d3d-research/adapter-a0'
    for name in ['result.json','real-cli-preflight.json','plan.json','接口说明.html']:shutil.copy2(O/name,folder/name)
print(json.dumps(dict(tests=17,coarse_source_isolated=True,actual_cli_intake_frames=177,real_fit=False),ensure_ascii=False))
