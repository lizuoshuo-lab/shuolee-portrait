"""Build a new portable snapshot; keep the earlier computer00 releases intact."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
DEST=Path('C:/Users/Administrator/Downloads/05_mini_splatting2_viewer')

README='''双击“打开深度训练对照.exe”。本包共12个模型，默认仍是已有的“运动遮罩＋Mip＋补点遮罩”。
新增两个完整18000步深度指导候选（权重0.003和0.01），可在“切换模型”中选择。它们尚未被认定为稳定优于原组合。
侧栏“深度指导：原图、几何与扩散对照”提供真实照片、人物局部、额外24帧、几何检查和失败视角。先查看效果图，再查看末尾实验口径。
原组合已完成三个随机种子；打包时新增深度模型各一个种子，匹配重复在本机继续运行。本离线包是该阶段的冻结快照，不会自动更新。
拍摄视角按原照片比例显示；整体场景、拖动旋转时进入自由浏览。两种模式的显示范围不同。
Mip已改善缩小渲染，但不能据此宣布真实远近运动问题解决。大幅俯仰、未观测的遮挡面和漂浮物仍待验收。
扩散对照是上一阶段模型的二维修正实验，没有把生成图像用于高斯训练，没有修改三维几何。
本视频红衣主体基本静止，走动背景人物被遮罩排除，不能视为动态人体重建结果。
保留整个文件夹。新服务优先使用8840端口，与旧包8830分开；旧包、原始PLY和全部实验记录均保留。
'''


def main():
    assert (ROOT/'viewer/native-depth.html').exists()
    assert json.loads((ROOT/'native-depth-summary.json').read_text(encoding='utf-8'))['guided_repeat_count']==1
    assert json.loads((ROOT/'native-depth-paths-001/manifest.json').read_text())['status']=='complete'
    stage=ROOT/'portable-depth-003';stage.mkdir(exist_ok=False)
    view=stage/'MiniGS2_Depth';shutil.copytree(ROOT/'viewer',view,
        ignore=shutil.ignore_patterns('*.exe','viewer-address.txt','viewer-errors.txt','viewer-host.log'))
    source=view/'ViewerHost.cs';text=source.read_text(encoding='utf-8')
    text=text.replace('MiniSplatting2-MipDepth-20261018','MiniSplatting2-Native-Depth-20261019-v3')
    text=text.replace('port=8820;port<=8829','port=8840;port<=8849').replace('8820–8829','8840–8849')
    source.write_text(text,encoding='utf-8')
    js=view/'viewer.js';text=js.read_text(encoding='utf-8').replace("?query.get('model'):'original'", "?query.get('model'):'native-motion-mip-depthmask'")
    assert 'calibrated-capture-frame-v1' in text
    js.write_text(text,encoding='utf-8');(view/'使用说明.txt').write_text(README,encoding='utf-8')
    exe=view/'打开深度训练对照.exe'
    subprocess.run(['C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe','/nologo','/target:winexe',f'/out:{exe}',
        '/r:System.Windows.Forms.dll','/r:System.Drawing.dll',str(source)],check=True)
    models={}
    for meta in view.glob('model-*.json'):
        m=json.loads(meta.read_text(encoding='utf-8'));p=view/m['model']
        sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==m['sha256']
        models[meta.stem]=dict(sha256=sha,gaussians=m['gaussianCount'])
    assert len(models)==12
    assert models['model-original']['sha256']=='0751327e63353b2c1cf9db302f8f5861d72571879aaade8e791a32abfc823eb8'
    package=DEST/'MiniGS2_Depth_20261019.zip';assert not package.exists()
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in view.rglob('*'):
            if p.is_file():z.write(p,Path('MiniGS2_Depth')/p.relative_to(view))
    with zipfile.ZipFile(package) as z:assert z.testzip() is None;count=len(z.namelist())
    receipt=dict(status='packaged_not_transferred',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        package=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        crc='passed',files=count,model_count=12,models=models,default_port=8840,local_stage=str(stage),
        older_packages_preserved=True,development_snapshot=True,guided_seeds_in_snapshot=[0])
    (ROOT/'depth-package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
