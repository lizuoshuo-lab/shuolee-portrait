"""Freeze a15-model portable viewer plus completed repeat/spread reports."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
DEST=Path('C:/Users/Administrator/Downloads/05_mini_splatting2_viewer')


def main():
    assert json.loads((ROOT/'native-spread-acceptance-queue.json').read_text())['status']=='complete'
    stage=ROOT/'portable-spread-005';stage.mkdir(exist_ok=False);view=stage/'MiniGS2_Spread'
    shutil.copytree(ROOT/'viewer',view,ignore=shutil.ignore_patterns('*.exe','viewer-address.txt','viewer-errors.txt','viewer-host.log'))
    source=view/'ViewerHost.cs';text=source.read_text(encoding='utf-8')
    text=text.replace('MiniSplatting2-MipDepth-20261018','MiniSplatting2-Spread-20261019-v5')
    text=text.replace('port=8820;port<=8829','port=8850;port<=8859').replace('8820–8829','8850–8859')
    source.write_text(text,encoding='utf-8')
    js=view/'viewer.js';text=js.read_text(encoding='utf-8').replace("?query.get('model'):'original'","?query.get('model'):'native-motion-mip-depthmask'")
    js.write_text(text,encoding='utf-8')
    (view/'使用说明.txt').write_text('双击“打开训练效果对照.exe”。本包含15个可交互模型，保留前12个并增加第三组深度分散实验。\n'
        '侧栏“深度分散约束”可查看真实照片、头肩细节、误差、深度分布及相同远近/俯仰路径。\n'
        '本轮固定平均深度指导0.003，比较分散权重0、0.003、0.010，各完成18000步。全部为种子0。\n'
        '结果：深度分布更集中，但头肩PSNR相对零权重下降约0.16和0.31dB，尚未证明人物清晰度改善。\n'
        '请勿将中心深度方差等同实测表面厚度；共同有效像素以外的透明度覆盖也有下降。大幅俯仰仍没有真实对照。\n'
        '下一轮同源低/原分辨率训练在本机进行，不包含在此冻结快照内。\n'
        '新服务从8850端口开始，旧8840查看器、旧包和原始PLY均保留。本包不会自动更新。\n'
        '所有素材来自该工作视频；未使用扩散补画。红衣主体基本静止，不能视为动态人体重建验收。\n',encoding='utf-8')
    exe=view/'打开训练效果对照.exe'
    subprocess.run(['C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe','/nologo','/target:winexe',f'/out:{exe}',
        '/r:System.Windows.Forms.dll','/r:System.Drawing.dll',str(source)],check=True)
    models={}
    for path in view.glob('model-*.json'):
        m=json.loads(path.read_text(encoding='utf-8'));p=view/m['model'];sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==m['sha256']
        models[path.stem]=dict(sha256=sha,gaussians=m['gaussianCount'])
    assert len(models)==15
    assert models['model-original']['sha256']=='0751327e63353b2c1cf9db302f8f5861d72571879aaade8e791a32abfc823eb8'
    package=DEST/'MiniGS2_Spread_20261019.zip';assert not package.exists()
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for p in view.rglob('*'):
            if p.is_file():archive.write(p,Path(view.name)/p.relative_to(view))
    with zipfile.ZipFile(package) as archive:assert archive.testzip() is None;count=len(archive.namelist())
    receipt=dict(status='packaged_not_transferred',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        package=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        files=count,crc='passed',model_count=15,models=models,local_stage=str(view),default_port=8850,
        old_packages_preserved=True,experimental_snapshot=True,accepted_as_human_clarity_solution=False)
    (ROOT/'spread-package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='models'}))


if __name__=='__main__':main()
