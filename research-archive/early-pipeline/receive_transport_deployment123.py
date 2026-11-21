from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;COORD=H.parent/'shuolee-coordination'
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=T/'shuolee/current/local-clothed-surface-123'
D=REC/'smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
E=Path('D:/Projects/Shuolee-validation-20261029/free-clothing-transport')
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def read(p):return json.loads(lp(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(lp(p).read_bytes()).hexdigest()
def save(p,v):lp(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(E/'deployment-receipt.json');reg=read(D/'company00-return/regression.json')
assert receipt['commit']=='d4a3d73fdf8dbec8aed4f661eccd3bbf95f9210b' and reg['passed'] and reg['tests_run']==25
old='00的单位接口已发布8d54497，25项专项/139项独立CPU CI通过；自由衣物接口的独立部署反馈正在核对，未声称服务器或实拍训练完成。'
new='00已发布 <a href="https://github.com/lizuoshuo-lab/shuolee-portrait/tree/smplx/free-clothing-transport-123">smplx/free-clothing-transport-123</a>（d4a3d73）：全新Git检出25项专项通过，WSL与云端CPU CI各164项通过。返回的协方差/损坏文件/默认设备/混合精度保护已合入研究源，本地25项再验通过。服务器和实拍训练尚未执行。原生14项结果对应合入前的b0e271a3源码，修补后版本为c1a3b988；不追认新版本已跑过同一GPU试验。'
current=dict(receipt=receipt,receipt_sha256=sha(E/'deployment-receipt.json'),returned_patch_merged=True,
    local_returned_patch_tests=25,local_returned_patch_result=str(D/'company00-return/regression.json'),
    remote_ref_verified=True,server_verified=False,physical_quality_passed=False)
v=read(D/'result.json');v.update(updated=stamp,company00=current,cpu_core_current=dict(result=str(D/'company00-return/regression.json'),tests_run=25,
    source_sha256='c1a3b98842bf1b3087222ec4723a18e6ae9edd7bd3ba45ee74a1b84f5ae4b492',native14_source_sha256='b0e271a3f66c15551e9aa0d0203f9fb31f8b46fced890eb4efed063f0e596c81'))
save(D/'result.json',v)
merge=read(D/'company00-return/merge.json');merge.update(regression_pending=False,regression_passed=True,tests_run=25);save(D/'company00-return/merge.json',merge)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]:
    p=dest/'index.html';text=p.read_text(encoding='utf-8');assert old in text;p.write_text(text.replace(old,new,1),encoding='utf-8')
    target=dest/'free-clothing-transport';shutil.copy2(lp(D/'result.json'),lp(target/'result.json'))
    lp(target/'company00-return').mkdir(exist_ok=True)
    for name in ['merge.json','regression.json']:shutil.copy2(lp(D/'company00-return'/name),lp(target/'company00-return'/name))
p=COORD/'local-research-status.json';v=read(p);v.update(updated_at=stamp,company00_latest_deployment_received=current,free_clothing_transport_current=read(D/'result.json'));save(p,v)
for base in [R,REC]:
    p=base/'evidence-ledger.json';v=read(p);summary='00自由衣物接口d4a3d73完成25专项/164全仓及CI；保护补丁合入本地并25项通过。服务器实拍未验。'
    if base==REC:
        if not any(x.get('id')=='company00_free_transport_received_20261029' for x in v['entries']):v['entries'].append(dict(id='company00_free_transport_received_20261029',summary=summary,evidence=str(D/'company00-return/merge.json'),quality_passed=False))
    else:
        if summary not in v['confirmed']:v['confirmed'].append(summary)
        if str(D/'company00-return/merge.json') not in v['artifacts']:v['artifacts'].append(str(D/'company00-return/merge.json'))
    save(p,v)
print(json.dumps(dict(received_commit=receipt['commit'],root_regression_tests=25,CI_tests=164,server_verified=False)))
