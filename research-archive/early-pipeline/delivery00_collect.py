from pathlib import Path
import shutil,json,hashlib,re,datetime,subprocess
ROOT=Path('D:/00'); LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab'); TOP=LAB/'outputs/mini-splatting2-mip-depth-001'; BASE=TOP/'shuolee'; PAIR=BASE/'current/avatar-repository-pair-114'; SRC=TOP/'viewer'; VIEW=ROOT/'工作/查看器'; HUMAN=ROOT/'工作/人体重建'; LOG=ROOT/'工作日志/2026-10-25'; BUILD=Path(__file__).with_name('delivery00-build')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,s):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def save(p,d):write(p,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
records=[]
def cp(src,dst,kind='source'):
    s=str(src).replace('\\','/');src=Path(s[5]+':'+s[6:] if re.match(r'^/mnt/[a-z]/',s) else s);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists() or dst.stat().st_size!=src.stat().st_size or sha(dst)!=sha(src):shutil.copy2(src,dst)
    records.append(dict(path=dst.relative_to(ROOT).as_posix(),bytes=dst.stat().st_size,sha256=sha(dst),source=str(src),kind=kind))
for p in [VIEW,HUMAN,LOG/'图片',LOG/'点云',LOG/'证据',BUILD]:p.mkdir(parents=True,exist_ok=True)
ids=['whole-body-avatar114-ga','whole-body-avatar114-smplx','whole-body-static-ncc-control','whole-body-native-align','original','full-scene-localization-074']
for name in ['index.html','viewer.js','viewer.css','point-cloud-loader.js','ViewerHost.cs','高斯与点云查看器.exe','cameras.json','latest-result.json']:
    cp(SRC/name,VIEW/name,'viewer')
for f in (SRC/'vendor').rglob('*'):
    if f.is_file():cp(f,VIEW/f.relative_to(SRC),'viewer-dependency')
for id in ids:
    meta=read(SRC/f'model-{id}.json');rel=meta['model'];cp(SRC/rel,VIEW/rel,'gaussian' if id!='full-scene-localization-074' else 'sfm')
    meta['report']='progress.html#'+id;save(VIEW/f'model-{id}.json',meta)
js=(VIEW/'viewer.js').read_text('utf-8');js=re.sub(r'const BUILTINS = \[.*?\];','const BUILTINS = '+json.dumps(ids)+';',js,count=1)
write(VIEW/'viewer.js',js)
html=(VIEW/'index.html').read_text('utf-8').replace('body-smplx.html#boundary-analysis','progress.html#diagnosis').replace('report.html','progress.html')
write(VIEW/'index.html',html)
decision=read(PAIR/'decision.json')
for arm,short in [('gaussianavatar','114a_GaussianAvatar'),('smplx','114b_SMPLX本地适配')]:
    root=PAIR/arm;run=root/'gaussian-run'
    for f in sorted(run.glob('render-*.png')):cp(f,LOG/'图片'/f'{short}_18000步_实拍相机_{f.stem.removeprefix("render-")}_原生渲染.png','unaltered-render')
    export=run/'viewer-export';cp(export/'point_cloud.ply',LOG/'点云'/f'{short}_18000步_Y向上_未通过重影验收.ply','compatible-gaussian')
    for sub in ['source-inputs.json','viewer-ui-review.json','gaussian-run/config.json','gaussian-run/training-cameras.json','gaussian-run/native-vs-ply.json','gaussian-run/viewer-export/delivery.json','evaluation/result.json']:
        if (root/sub).exists():cp(root/sub,LOG/'证据'/short/Path(sub),'evidence')
rows=read(PAIR/'smplx/data/training-inputs.json')['records']
for i in [0,33,69,99,135,174]:
    r=rows[i];cp(r['image'],LOG/'图片'/f'原图_相机{r["source_frame_index"]:06d}_整图去畸变.png','unaltered-input')
    cp(r['mask'],LOG/'图片'/f'遮罩_相机{r["source_frame_index"]:06d}_原始mask.png','unaltered-mask')
for rel in ['decision.json','protocol.json','source-audit.json','completion-receipt.json']:
    cp(PAIR/rel,LOG/'证据/114双路线'/rel,'evidence')
D=BASE/'current/portrait-causal-diagnosis-117'
for rel in ['protocol.json','queue-state.json','official-versa-research.json','D0/result.json','D1-conditioning/result.json','D1-intrinsics-only/fixed074-result.json','D1-intrinsics-only/shared_K_only-result.json','D1-intrinsics-only/interruption.json']:
    if (D/rel).exists():cp(D/rel,LOG/'证据/117相机诊断'/rel,'evidence')
write(LOG/'今日记录.md', '''# 2026-10-25 · 人像重影与工程整理

114 两条路线完成18,000步后已审阅：GaussianAvatar 较平滑，SMPL-X适配纹理较清楚；鼻嘴、镜框与身体边界重影仍在，两组均未通过质量验收。两组都是175真实整图相机、20万初始点、独立拟合和从头训练。这里是今日可复查的结果归档，不表示今天又训练了一轮。

图片文件名说明来源、路线、步数和相机。原图、遮罩、原生渲染分别保存，不在图片上写字、不做二维扭曲。两组对照应选择同名相机。诊断没有新训练，不生成冒充117结果的点云。

115 已确认解码、去畸变、训练像素与CUDA投影公式一致；这不等于物理标定准确。116 三个开源候选没有产生通过验收的相机。117 复用2684条背景轨迹，固定外参只检验共享内参：两项求解已完成，但独立留出评价受运行中断影响尚未完成，不能据拟合残差变化宣称根因解决。原视频含 HOVER 元数据，尚无官方字段语义或原始tdsp，现用074相机来自COLMAP估计。

## 加载点云

打开 `../../工作/查看器/高斯与点云查看器.exe`，点击“打开文件”，选择本目录“点云”中的PLY。已转换为世界坐标、+Y朝上，位置、完整协方差和SH一起转换，不需要再加载人体模板或蒙皮。选择“高斯渲染”观察外观；把高斯球尺寸调小观察分布；“点云”显示中心。1×为原始渲染尺寸，缩小不表示质量被修复。

兼容核验记录在证据中。本次只核验本机，未声称已在电脑00异地打开。原有资料保留，当前仅整理必要对照，不批量删除历史文件。
''')
cards=[]
for id in ids:
    meta=read(VIEW/f'model-{id}.json');cards.append(f'<article id="{id}"><h2>{meta.get("label",id)}</h2><p>{meta.get("status","")}</p><a href="./?model={id}">打开模型</a> · <a href="{meta["model"]}" download>下载 PLY</a></article>')
imgs=[]
for f in (LOG/'图片').glob('*渲染.png'):
    cp(f,VIEW/'evidence/2026-10-25'/f.name,'viewer-review');imgs.append(f'<figure><img loading="lazy" src="evidence/2026-10-25/{f.name}"><figcaption>{f.stem}</figcaption></figure>')
progress='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>实验进展 · Shuolee</title><style>body{background:#101618;color:#e7ecec;font:18px/1.65 "Microsoft YaHei",sans-serif;max-width:1200px;margin:40px auto;padding:0 26px}a{color:#a5e7c8}article{border-top:1px solid #435450;padding:20px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}figure{margin:0}img{width:100%;display:block}figcaption{font-size:13px}h1{font-size:36px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>实验进展 · 2026-10-25</h1><p><a href="./?model=latest">打开最新可加载模型</a> · 模型格式通过与重影质量通过分开记录。</p><article id="diagnosis"><h2>当前结论：重影根因尚未确定</h2><p>114两条人像路线完成，但鼻嘴、眼镜和身体边界仍未对齐。115像素和CUDA投影一致；116没有合格相机。117继续将相机标定、跨视角材料匹配、人物微动分开诊断。固定外参的共享K求解已完成，独立留出评测中断，尚未作通过结论。本页没有新训练或新117点云。</p><p>新查看器提供0.05–2倍高斯显示尺寸，中心位置不变。今日日志保留未叠字原生图片和+Y点云；原视频的官方相机字段仍待核实。</p></article>'''+''.join(cards)+'<h2>同相机原生渲染</h2><div class="grid">'+''.join(imgs)+'</div></html>'
write(VIEW/'progress.html',progress);write(VIEW/'report.html',progress);write(SRC/'progress.html',progress)
for f in (VIEW/'evidence/2026-10-25').glob('*'):shutil.copy2(f,(SRC/'evidence/2026-10-25'/f.name)) if (SRC/'evidence/2026-10-25').exists() else None
(SRC/'evidence/2026-10-25').mkdir(parents=True,exist_ok=True)
for f in (VIEW/'evidence/2026-10-25').glob('*'):shutil.copy2(f,SRC/'evidence/2026-10-25'/f.name)
write(VIEW/'使用说明.md','''# 高斯与点云查看器

Windows双击“高斯与点云查看器.exe”；保持整个目录完整。它只监听本机127.0.0.1，端口8870–8879自动选择。“最新实验与当天资料”打开本轮结论和原生图片。

- 高斯球尺寸0.05–2×：只缩放每个高斯的标准差，中心、旋转、颜色和原文件均不变；恢复1×回到原始尺寸。点模式的像素大小独立控制。
- 打开文件支持PLY/SPLAT/SPZ/KSPLAT/SOG和普通PLY/PCD/XYZ/PTS。整套文件夹可导入多个迭代结果并锁定观察位置。
- 实拍相机仅对预设模型提供；直接导入文件先适应整体。不要把坐标轴改变当成人体拟合修复。
- 如需以后同步，运行“同步实验资料.ps1”（本机维护用，需要原研究目录）；它不触发训练。
- 异地只需下载本目录及模型子目录，不需要Python、CUDA或人体模板。浏览器需支持WebGL2。
- 当前候选均未通过重影验收。不得把能打开或能放大缩小写成重建成功。
''')
save(ROOT/'整理状态.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),status='in_progress',projects=['minisplatting2','人体重建','查看器'],viewer_scale='implemented_ui_review_pending',daily_log='2026-10-25',human_git='preparing_private_repository',wps_sync='not_configured',quality_accepted=False))
save(ROOT/'文件清单.json',dict(created=datetime.datetime.now().astimezone().isoformat(),files=records))
print(json.dumps(dict(files=len(records),bytes=sum(x['bytes'] for x in records),root=str(ROOT)),ensure_ascii=False))
