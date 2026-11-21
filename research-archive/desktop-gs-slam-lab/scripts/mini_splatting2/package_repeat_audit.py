"""A report-only offline supplement; never replaces the12-model viewer package."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
DEST=Path('C:/Users/Administrator/Downloads/05_mini_splatting2_viewer')


def main():
    assert json.loads((ROOT/'depth-repeat-acceptance-queue.json').read_text())['status']=='complete'
    stage=ROOT/'portable-repeat-audit-004';stage.mkdir(exist_ok=False);folder=stage/'MiniGS2_RepeatAudit';folder.mkdir()
    html=(ROOT/'viewer/native-repeats.html').read_text(encoding='utf-8')
    html=html.replace('href="index.html"','href="http://127.0.0.1:8840/"').replace('href="native-depth.html"','href="http://127.0.0.1:8840/native-depth.html"')
    (folder/'查看人物清晰度.html').write_text(html,encoding='utf-8')
    shutil.copytree(ROOT/'viewer/native-repeat-assets',folder/'native-repeat-assets')
    (folder/'使用说明.txt').write_text('双击“查看人物清晰度.html”。本页可离线查看，不需要启动新服务。\n'
        '原组合与两种深度权重各三个种子，共九次完整训练。支持切换种子、头肩/人物/全图、额外24帧和真实深度分散诊断。\n'
        '结论：深度0.003的全图平均改善约0.238dB，但头肩平均约-0.015dB；头肩中心深度分散增大，人物清晰度尚未通过验收。\n'
        '本包是结果快照，不含新的三维PLY。已有12模型查看器仍位于原文件夹、8840端口，旧包完整保留。\n'
        '后续深度分散约束仍在本地实验，本包不包含其尚未完成的效果。\n',encoding='utf-8')
    package=DEST/'MiniGS2_RepeatAudit_20261019.zip';assert not package.exists()
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in folder.rglob('*'):
            if path.is_file():archive.write(path,Path(folder.name)/path.relative_to(folder))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None;count=len(archive.namelist())
    receipt=dict(status='packaged_not_transferred',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        package=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        files=count,crc='passed',report_only=True,model_count_in_metrics=9,new_ply_files=0,
        local_stage=str(folder),old_packages_preserved=True)
    (ROOT/'repeat-audit-package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
