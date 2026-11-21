"""Freeze the reviewed17-model local resolution comparison for computer00."""
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
    review=json.loads((ROOT/'native-resolution-final-review-001.json').read_text(encoding='utf-8'));assert review['status']=='reviewed'
    registration=json.loads((ROOT/'viewer-resolution-registration-001/summary.json').read_text(encoding='utf-8'));assert registration['status']=='browser_verified'
    assert json.loads((ROOT/'native-resolution-cache-queue.json').read_text())['status']=='complete'
    summary=json.loads((ROOT/'native-resolution-cache-review-001/summary.json').read_text())
    stage=ROOT/'portable-resolution-006';stage.mkdir(exist_ok=False);view=stage/'MiniGS2_Resolution'
    shutil.copytree(ROOT/'viewer',view,ignore=shutil.ignore_patterns('*.exe','viewer-address.txt','viewer-errors.txt','viewer-host.log'))
    source=view/'ViewerHost.cs';code=source.read_text(encoding='utf-8')
    assert 'MiniSplatting2-MipDepth-20261018' in code and 'port=8820;port<=8829' in code
    code=code.replace('MiniSplatting2-MipDepth-20261018','MiniSplatting2-Resolution-20261019-v6')
    code=code.replace('port=8820;port<=8829','port=8860;port<=8869').replace('8820–8829','8860–8869')
    source.write_text(code,encoding='utf-8')
    js=view/'viewer.js';code=js.read_text(encoding='utf-8').replace("?query.get('model'):'original'","?query.get('model'):'native-mip-resolution-full'");js.write_text(code,encoding='utf-8')
    delta=summary['compared_to_low']['full']['delta']['head'];small=summary['compared_to_low']['low']['delta']['head']
    readme=('双击“打开训练效果对照.exe”，打开后可选第四组分辨率实验模型。\n'
        '本包17个模型：保留前15个，新增同源757×1347和1514×2694训练结果，各为种子0、18,000步。\n'
        '侧栏“最新对照：训练分辨率与人物细节”提供真实照片、人物/头肩区域、固定误差图、远近与俯仰视频以及另外24帧的分组对照。\n'
        f'共同1514×2694输出时，头肩PSNR变化{delta["psnr"]:+.3f}dB，梯度保真误差变化{delta["normalized_gradient_error"]:+.3f}（越低越好）。\n'
        f'共同757×1347输出时，头肩PSNR变化{small["psnr"]:+.3f}dB。\n'
        '这是单种子开发对照，尚未完成重复验收；不能据此宣布大幅俯仰或漂浮物问题已解决。\n'
        '两组相机、稀疏点、运动遮罩、平均深度指导0.003和训练步数一致。没有加入扩散生成图像、深度分散约束或头部加权。\n'
        '原视频为1520×2704，镜头校正后的训练尺寸为1514×2694；低分辨率图像从同一校正图像缩小得到。\n'
        '当前视频中红衣主体基本静止，不能视为动态人体或独立大场景验收。\n'
        '服务从8860端口启动；旧8850和8840包、原模型均保留。本包为冻结快照，不会自动更新。\n')
    (view/'使用说明.txt').write_text(readme,encoding='utf-8')
    exe=view/'打开训练效果对照.exe'
    subprocess.run(['C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe','/nologo','/target:winexe',f'/out:{exe}',
        '/r:System.Windows.Forms.dll','/r:System.Drawing.dll',str(source)],check=True)
    models={}
    for path in view.glob('model-*.json'):
        metadata=json.loads(path.read_text(encoding='utf-8'));model=view/metadata['model'];digest=hashlib.sha256(model.read_bytes()).hexdigest();assert digest==metadata['sha256']
        models[path.stem]=dict(sha256=digest,gaussians=metadata['gaussianCount'])
    assert len(models)==17 and models['model-original']['sha256']=='0751327e63353b2c1cf9db302f8f5861d72571879aaade8e791a32abfc823eb8'
    package=DEST/'MiniGS2_Resolution_20261019.zip';assert not package.exists()
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in view.rglob('*'):
            if path.is_file():archive.write(path,Path(view.name)/path.relative_to(view))
    with zipfile.ZipFile(package) as archive:assert archive.testzip() is None;count=len(archive.namelist())
    receipt=dict(status='packaged_not_transferred',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        package=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        files=count,crc='passed',model_count=17,models=models,local_stage=str(view),default_port=8860,
        old_packages_preserved=True,experimental_snapshot=True,accepted_as_human_clarity_solution=False,
        review='native-resolution-final-review-001.json')
    (ROOT/'resolution-package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({key:value for key,value in receipt.items() if key!='models'}))


if __name__=='__main__':main()
