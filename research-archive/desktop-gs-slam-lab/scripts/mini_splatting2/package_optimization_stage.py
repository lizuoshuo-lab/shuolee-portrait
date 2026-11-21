"""Package reviewed local assets only, never source videos or model weights."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
VIEW=ROOT/'viewer'
PACKAGE=Path('C:/Users/Administrator/Downloads/05_mini_splatting2_viewer/MiniGS2_Optimization_20261018.zip')

def main():
    info=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='active',
        user_confirmed_method='Mip-Splatting',experiment_root=str(ROOT),
        work_video='HOVER_VERSA_0102.mp4',
        source_video_sha256='5d4b6c5cb38edfb17c90faa8352e0edb93370a6d230076a463d3e3ec9c70714f',
        immutable_original_ply_sha256='0751327e63353b2c1cf9db302f8f5861d72571879aaade8e791a32abfc823eb8',
        completed=['computer00 video verified via ToDesk', 'native vs gsplat rasterizer probe',
                   'four fixed-topology settings x three seeds x 2000 steps',
                   'same-camera downsampling 1,2,4,8', 'seed0 sparse SfM depth agreement',
                   'same novel distance/elevation paths', '8-view reference-conditioned Difix appearance test',
                   'filtered PLY export roundtrip and original hash checks', 'interactive viewer and visual report'],
        viewer=str(VIEW/'打开优化对照.exe'),report_url='http://127.0.0.1:8820/report.html',
        next=['conservative multi-view free-space Gaussian audit and ablation',
              'dynamic-background diagnosis and training masks',
              'region-gated diffusion and temporal consistency before geometry distillation',
              'integrate validated modules into full Mini-Splatting2 training',
              'paired real distance and elevation coverage validation'],
        unresolved=['large pitch appearance and geometry', 'floaters elimination',
                    'physical distance generalization', 'dynamic human reconstruction'],
        evidence=['stage-summary.json','capture-audit.json','delivery-verification.json',
                  'geometry-evaluation/sfm-depth-agreement.json','difix-appearance-test-r1/manifest.json'],
        scope='Post-training fixed topology; appearance development holdouts share original SfM. No final independent test. No cloud upload.',
        remote_delivery=dict(status='not_yet_transferred',destination='computer00 Desktop/工作/复现miniGS2',archive=str(PACKAGE)))
    (LAB/'manifests/mini-splatting2-optimization.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    (VIEW/'使用说明.txt').write_text('解压整个压缩包后，双击“打开优化对照.exe”。\n在查看器左侧切换原模型和四种优化模型，观察位置保持不变。\n点击“原图 / 实验效果与结论”查看照片、缩小渲染、距离俯仰视频和扩散对照。\n当前是第一阶段局部微调结果：Mip改善缩小渲染，深度约束改善部分稀疏几何一致性；大幅俯仰与漂浮物仍未完成验收。\n所有模型均保留，原始结果不覆盖。Difix图片是二维修复结果，未写入三维模型。\n',encoding='utf-8')
    if PACKAGE.exists():raise FileExistsError(PACKAGE)
    with zipfile.ZipFile(PACKAGE,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in VIEW.rglob('*'):
            if p.is_file() and p.name not in ['viewer-host.log']:
                z.write(p,Path('MiniGS2_Optimization')/p.relative_to(VIEW))
    receipt=dict(path=str(PACKAGE),bytes=PACKAGE.stat().st_size,sha256=hashlib.sha256(PACKAGE.read_bytes()).hexdigest())
    with zipfile.ZipFile(PACKAGE) as z:
        receipt['crc_check']=z.testzip();receipt['files']=len(z.namelist())
    (ROOT/'package-receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':main()
