"""Export source-resolution frames and measured overlays for the local work log."""
from pathlib import Path
import hashlib
import html
import json
import shutil
import subprocess

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/mnt/d/Projects/Shuolee-analysis-20261028')
RUN = Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
OUT = Path('/mnt/d/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图')
rows = json.loads((RUN/'data/training-inputs.json').read_text())['records']
selected = [0,18,37,56,75,94]
fit = dict(np.load(RUN/'fit/fitted.npz'))
obs = dict(np.load(RUN/'data/observations.npz'))
weights = dict(np.load(RUN/'fit/observation-weights.npz'))
receipt = json.loads((BASE/'gray-reproduction.json').read_text())
video = Path(receipt['video'])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert np.array_equal(fit['frame_ids'], [r['frame'] for r in rows])
assert np.array_equal(obs['frame_ids'], fit['frame_ids'])
assert not OUT.exists(), 'Preserve existing work-log delivery.'
OUT.mkdir(parents=True)
temp = OUT/'source-frames'
temp.mkdir()
select = '+'.join(f'eq(n\\,{rows[i]["source_frame_index"]})' for i in selected)
subprocess.run(['ffmpeg','-nostdin','-hide_banner','-v','error','-n','-i',str(video),
                '-vf',f'select={select}','-pix_fmt','rgb24','-vsync','0',str(temp/'%02d.png')],check=True)
frames = sorted(temp.glob('*.png'))
assert len(frames) == 6
records = []
sections = []
for source,i in zip(frames,selected):
    row = rows[i]
    folder = OUT/row['frame']
    folder.mkdir()
    original = folder/'01_完整原图.png'
    shutil.move(str(source),str(original))
    im = np.asarray(Image.open(original).convert('RGB'))
    H,W = im.shape[:2]
    factor = np.array([W/row['wh'][0],H/row['wh'][1]])
    keep = weights['face'][i] > 0
    targets = obs['keypoints'][i,40:91][keep]
    projected = fit['projected_face51'][i,keep]
    error = np.linalg.norm(targets-projected,axis=-1)
    native_targets = (targets+.5)*factor-.5
    native_projected = (projected+.5)*factor-.5
    for name,points,color,marker in [('02_检测点完整图.png',native_targets,'#00ef98','o'),
                                    ('03_SMPL-X投影完整图.png',native_projected,'#ff7d24','+')]:
        fig = plt.figure(figsize=(W/160,H/160),dpi=160)
        ax = fig.add_axes([0,0,1,1])
        ax.imshow(im,interpolation='none')
        if marker=='o':
            ax.scatter(points[:,0],points[:,1],s=14,facecolors='none',edgecolors=color,linewidths=.8)
        else:
            ax.scatter(points[:,0],points[:,1],s=20,marker=marker,c=color,linewidths=.9)
        ax.set_xlim(-.5,W-.5);ax.set_ylim(H-.5,-.5);ax.set_axis_off()
        fig.savefig(folder/name,dpi=160,pad_inches=0)
        plt.close(fig)
        assert Image.open(folder/name).size==(W,H)
    low = np.maximum(0,np.floor(np.concatenate([native_targets,native_projected]).min(0)-20*factor))
    high = np.minimum([W,H],np.ceil(np.concatenate([native_targets,native_projected]).max(0)+20*factor))
    fig,axes = plt.subplots(1,3,figsize=(12,5.2),squeeze=False)
    for ax,title,points in zip(axes[0],['Original video frame','Detector observations','SMPL-X projection'],
                               [None,native_targets,native_projected]):
        ax.imshow(im,interpolation='none')
        if title=='Detector observations':
            ax.scatter(points[:,0],points[:,1],s=25,facecolors='none',edgecolors='#00ef98',linewidths=1.2)
        if title=='SMPL-X projection':
            ax.scatter(points[:,0],points[:,1],s=33,marker='+',c='#ff7d24',linewidths=1.2)
        ax.set_xlim(low[0],high[0]);ax.set_ylim(high[1],low[1]);ax.set_aspect('equal')
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=12)
    fig.suptitle(f'{row["frame"]} | median residual {np.median(error):.2f} px at {row["wh"][0]} x {row["wh"][1]}',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(folder/'04_脸部对照.png',dpi=180);plt.close(fig)
    record = dict(frame=row['frame'],source_frame_index=row['source_frame_index'],time_seconds=row['time_seconds'],
                  native_wh=[W,H],analysis_wh=row['wh'],face_median_analysis_px=float(np.median(error)),
                  native_mapping='native_uv = (decoded_uv + 0.5) * native_wh / decoded_wh - 0.5',
                  files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),wh=list(Image.open(p).size)) for p in sorted(folder.glob('*.png'))])
    records.append(record)
    items=''.join(f'<a href="{row["frame"]}/{p.name}">{html.escape(p.stem)}</a>' for p in sorted(folder.glob('*.png')))
    sections.append(f'<section><h2>{row["frame"]} · {row["time_seconds"]:.2f} 秒</h2><p>脸部残差中位数 {np.median(error):.2f} 像素（899 × 1600 分析域）</p><img src="{row["frame"]}/04_脸部对照.png" alt="原图、检测点和 SMPL-X 投影"><nav>{items}</nav></section>')
temp.rmdir()
manifest = dict(created='2026-10-29',source_video=str(video),source_video_sha256=receipt['video_sha256'],
                run=str(RUN),server_result=False,frames=records,source_files_modified=False,
                source_pixels='FFmpeg RGB24 decoded source-resolution frames, no enhancement or alignment correction',
                observations_are_not_independent_physical_truth=True,quality_accepted=False)
(OUT/'清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>灰衣 SMPL-X 对齐原图</title>
<style>body{margin:32px auto;max-width:1150px;padding:0 20px;font:17px/1.6 system-ui;color:#18222b;background:#f7f9fa}h1{font-size:28px}section{background:white;border:1px solid #dae0e5;border-radius:10px;padding:22px;margin:24px 0}h2{font-size:21px;margin:0}img{width:100%;height:auto}nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#17628c}p{margin:12px 0}</style>
<h1>灰衣 SMPL-X 对齐原图</h1><p>本地 video3，从原视频提取六帧。每帧单独保存完整原图、检测点完整图、SMPL-X 投影完整图和一张脸部对照图。完整图片保持视频原始解码尺寸，点击链接可单独打开。</p>
<p>绿色圆圈：参与拟合的二维检测点。橙色十字：SMPL-X 三维模型经原相机投影的位置。图上残差使用训练预处理的 899 × 1600 像素域；检测点不等于独立测量真值。未做图像配准或美化。</p>'''+''.join(sections)+'</html>'
(OUT/'查看原图.html').write_text(page,encoding='utf-8')
print(json.dumps(dict(output=str(OUT),frames=len(records),images=sum(len(r['files']) for r in records),native_wh=records[0]['native_wh'],source_unchanged=True),ensure_ascii=False))
