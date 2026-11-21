"""Create a separate portable release, leaving the first delivery unchanged."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
DEST=Path('C:/Users/Administrator/Downloads/05_mini_splatting2_viewer')

README='''双击“打开完整训练对照.exe”。默认展示“运动遮罩＋Mip＋补点遮罩”组合模型。
查看器可切换五组完整18000步模型：原版、运动遮罩、Mip、运动遮罩＋Mip、运动遮罩＋Mip＋补点遮罩；另保留原始模型及四组2000步微调对照，共10个模型。
点击侧栏“完整训练：运动遮罩与Mip组合”，查看原始照片、完整训练渲染和多分辨率结果。页面还包含24张未用于训练的额外视频帧；它们来自同一视频，只能验证时间插值，不能替代独立拍摄测试。
原版与运动遮罩完成三个匹配随机种子；其他三组目前各一个种子。
“补点遮罩”指在原生深度重初始化时排除已标记的背景运动区域，本轮没有新增单目深度损失。被排除的移动背景人物不属于已重建的动态人物。
Mip改善同一相机的缩小渲染；真实远近运动和大幅俯仰仍需独立验证。该视频主要是相机绕静态主体运动，不能代表动态人体重建完成。
请保留整个文件夹。第一轮五模型压缩包与原始MiniGS2模型均未覆盖。
'''

def main():
    assert json.loads((ROOT/'native-mip-summary.json').read_text())['status']=='five_full_native_variants_evaluated'
    stage=ROOT/'portable-native-002';stage.mkdir(exist_ok=False)
    view=stage/'MiniGS2_Native';shutil.copytree(ROOT/'viewer',view,
        ignore=shutil.ignore_patterns('*.exe','viewer-address.txt','viewer-errors.txt','viewer-host.log'))
    source=view/'ViewerHost.cs';text=source.read_text(encoding='utf-8')
    text=text.replace('MiniSplatting2-MipDepth-20261018','MiniSplatting2-Native-Mip-20261019-v2')
    text=text.replace('port=8820;port<=8829','port=8830;port<=8839').replace('8820–8829','8830–8839')
    # A distinct identity prevents reconnecting to the older package's server.
    source.write_text(text,encoding='utf-8')
    js=view/'viewer.js';text=js.read_text(encoding='utf-8')
    text=text.replace("?query.get('model'):'original'", "?query.get('model'):'native-motion-mip-depthmask'")
    js.write_text(text,encoding='utf-8')
    (view/'使用说明.txt').write_text(README,encoding='utf-8')
    exe=view/'打开完整训练对照.exe'
    compiler=Path('C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe')
    subprocess.run([str(compiler),'/nologo','/target:winexe',f'/out:{exe}','/r:System.Windows.Forms.dll','/r:System.Drawing.dll',str(source)],check=True)
    package=DEST/'MiniGS2_Native_20261019.zip'
    if package.exists():raise FileExistsError(package)
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in view.rglob('*'):
            if p.is_file():z.write(p,Path('MiniGS2_Native')/p.relative_to(view))
    with zipfile.ZipFile(package) as z:crc=z.testzip();count=len(z.namelist())
    assert crc is None
    receipt=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),package=str(package),bytes=package.stat().st_size,
        sha256=hashlib.sha256(package.read_bytes()).hexdigest(),crc='passed',files=count,model_count=len(list(view.glob('model-*.json'))),
        status='packaged_not_transferred',default_port=8830,local_stage=str(stage),first_delivery_preserved=True)
    (ROOT/'native-package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
