"""Full-image comparison and standard world/Y-up PLY + calibrated camera exports."""
from pathlib import Path
from types import SimpleNamespace
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import argparse,copy,importlib.util,json,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
BASE=Path('/opt/gs-slam-lab/msv2-demo-fullframe-20261104')
ex=load('fullframe',BASE/'experiment.py')
ROOT=ex.RECORD
LABELS={1:'原始完整图',2:'完整图插值2倍',4:'完整图SwinIR超分4倍'}

def export(scale):
    ex.choose(scale);torch=ex.runtime()
    from scene import GaussianModel
    run=ex.RUN;execution=ex.read(run/'execution.json');assert execution['state']=='training_completed'
    final=run/'model/point_cloud/iteration_18000/point_cloud.ply';assert ex.read(final.parent/'ply-parity.json')['passed']
    out=ROOT/ex.NAMES[scale];out.mkdir(parents=True,exist_ok=False)
    h=load('fullscene_export_helper',Path(__file__).with_name('review_msv2_upsampled_demo.py'))
    h.ex=ex;h.RUN=run;h.OUT=out;h.PLY_STEM=f'demo_fullframe_{ex.NAMES[scale]}_msv2_18000';h.BUNDLE=f'demo_完整图_{scale}倍_MSV2_18000步_点云与相机.zip'
    world=out/(h.PLY_STEM+'_world.ply');shutil.copy2(final,world);assert ex.sha(world)==ex.sha(final)
    for f in ['config.json','execution.json','preflight.json','input-contract.json']:
        shutil.copy2(run/f,out/f)
    shutil.copy2(BASE/'plan.json',out/'plan.json');shutil.copy2(final.parent/'ply-parity.json',out/'native-ply-parity.json')
    rows=ex.read(run/'training-inputs.json')['records'];ex.save(out/'cameras_world.json',h.camera_records(rows))
    ex.save(out/'camera-input-mapping.json',dict(records=rows,pixel_convention='pixel_corners',world='Original scene SfM coordinates',mask_used=False,full_frame=True))
    model=GaussianModel(3);model.load_ply(str(final));model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    upright=True
    try:h.export_upright(rows,model,pipe,torch.zeros(3,device='cuda'))
    except AssertionError:
        parity=ex.read(out/'upright-parity.json');assert parity['passed'] is False;upright=False
        with zipfile.ZipFile(out/h.BUNDLE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            z.write(world,'point_cloud.ply');z.write(out/'cameras_world.json','cameras.json')
            for f in ['plan.json','native-ply-parity.json','upright-parity.json','camera-input-mapping.json']:z.write(out/f,f)
            z.writestr('coordinate-note.txt','Training world coordinates; use paired cameras. Upright export failed the existing native render tolerance and is not included.')
    with zipfile.ZipFile(out/h.BUNDLE) as z:assert z.testzip() is None
    ex.save(out/'delivery.json',dict(bundle=h.BUNDLE,world_ply=world.name,default_ply=h.PLY_STEM+('_yup.ply' if upright else '_world.ply'),
        camera_file='cameras_yup.json' if upright else 'cameras_world.json',upright_export_verified=upright,world_export_verified=True,
        points=execution['points'],cameras=59,sha256=ex.sha(out/h.BUNDLE),bytes=(out/h.BUNDLE).stat().st_size,quality_accepted=False))
    print('FULLFRAME_EXPORTED',scale,execution['points'],upright,flush=True)

def box(r,key,pad):
    a=np.array(r[key]).reshape(2,2);d=np.maximum((a[1]-a[0])*pad,12)
    return np.r_[np.maximum(np.floor(a[0]-d),0),np.minimum(np.ceil(a[1]+d),[2704,1520])].astype(int).tolist()

def evaluate(scale):
    ex.choose(scale);torch=ex.runtime()
    from scene import GaussianModel
    from gaussian_renderer import render
    from fused_ssim import fused_ssim
    final=ex.RUN/'model/point_cloud/iteration_18000/point_cloud.ply';assert ex.read(final.parent/'ply-parity.json')['passed']
    model=GaussianModel(3);model.load_ply(str(final));model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    rows=ex.read(BASE/'native-inputs.json')['records'];regions=ex.read(ex.PARENT/'data/training-inputs.json')['records']
    results=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            c=ex.Camera(r,i);pred=render(c,model,pipe,bg)['render'];gt=c.original_image
            assert torch.isfinite(pred).all()
            error=(pred-gt).abs();mse=(pred-gt).square().mean()
            body=box(regions[i],'initial_box_xyxy',.12);face=box(regions[i],'face_box_xyxy',.3)
            item=dict(frame=r['frame'],l1=float(error.mean()),psnr=float(-10*torch.log10(mse)),ssim=float(fused_ssim(pred[None],gt[None],train=False)))
            for name,b in [('body',body),('face',face)]:
                x0,y0,x1,y1=b;item[name+'_box']=b;item[name+'_l1']=float(error[:,y0:y1,x0:x1].mean())
            out=ROOT/'views'/r['frame'];out.mkdir(parents=True,exist_ok=True)
            source=out/'source.png'
            if not source.exists():shutil.copy2(r['image'],source)
            ex.save_image(out/(ex.NAMES[scale]+'.png'),pred)
            results.append(item)
            if i%10==0 or i==58:print('FULLFRAME_EVALUATED',scale,i+1,flush=True)
            del c,pred,gt,error
    ex.save(ROOT/ex.NAMES[scale]/'metrics.json',dict(records=results,means={k:float(np.mean([r[k] for r in results])) for k in ['l1','psnr','ssim','body_l1','face_l1']},
        comparison_wh=[2704,1520],training_views_not_heldout=True,metric_mask_used=False,body_face_boxes_for_review_only=True,quality_accepted=False))

def board(paths,labels,dest,boxes=None):
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    images=[]
    for i,p in enumerate(paths):
        with Image.open(p) as im:
            im=im.convert('RGB')
            if boxes is not None:im=im.crop(tuple(boxes))
            if boxes is not None:im=im.resize((max(1,round(im.width*min(380/im.width,550/im.height))),max(1,round(im.height*min(380/im.width,550/im.height)))),Image.Resampling.LANCZOS)
            else:im.thumbnail((675,380),Image.Resampling.LANCZOS)
            images.append(im.copy())
    col=400 if boxes is not None else 690;rows=1 if boxes is not None else 2;cols=4 if boxes is not None else 2
    height=max(a.height for a in images)+42;canvas=Image.new('RGB',(cols*col,rows*height),'#eee');draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,labels)):
        x=(i%cols)*col;y=(i//cols)*height;draw.text((x+8,y+5),label,font=font,fill='#222');canvas.paste(im,(x+(col-im.width)//2,y+38))
    canvas.save(dest)

def publish():
    ROOT.mkdir(parents=True,exist_ok=True)
    methods=[1,2,4];metrics={s:ex.read(ROOT/ex.NAMES[s]/'metrics.json') for s in methods}
    rows=ex.read(ex.PARENT/'data/training-inputs.json')['records']
    labels=['原视频完整帧']+[LABELS[s] for s in methods]
    for i,r in enumerate(rows):
        folder=ROOT/'views'/r['frame'];paths=[folder/'source.png']+[folder/(ex.NAMES[s]+'.png') for s in methods]
        for name,b in [('full',None),('body',box(r,'initial_box_xyxy',.12)),('face',box(r,'face_box_xyxy',.3))]:
            board(paths,labels,folder/(name+'_compare.png'),b)
    for kind in ['full','body','face']:
        for start in range(0,59,12):
            panel=Image.new('RGB',(1600,6*350),'#eee');draw=ImageDraw.Draw(panel);font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
            for j,r in enumerate(rows[start:start+12]):
                with Image.open(ROOT/'views'/r['frame']/(kind+'_compare.png')) as im:
                    im.thumbnail((790,315),Image.Resampling.LANCZOS);x=(j%2)*800;y=(j//2)*350;draw.text((x+4,y),r['frame'],font=font,fill='black');panel.paste(im,(x,y+30))
            panel.save(ROOT/f'all_{kind}_{start//12+1:02d}.png')
    for f in ['plan.json','native-pixel-audit.json','initialization.json','loss-equivalence.json','superresolution.json']:
        shutil.copy2(BASE/f,ROOT/f)
    comparison=dict(created=ex.now(),means={ex.NAMES[s]:metrics[s]['means'] for s in methods},mask_used=False,full_image=True,quality_accepted=False)
    ex.save(ROOT/'comparison.json',comparison)
    text=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整图训练 · 原始 / 插值2倍 / 超分4倍</title>',
          '<style>body{max-width:1450px;margin:30px auto;padding:0 20px;background:#f4f4f0;color:#222;font:16px/1.65 system-ui}section{background:white;border-radius:12px;padding:22px;margin:20px 0}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#176684}</style>',
          '<h1>完整图训练 · 原始 / 插值2倍 / 超分4倍</h1><p>demo · MSV2 · 每版从头训练18,000步 · 完整背景参与训练</p>',
          '<section><h2>实际输入</h2><p>原视频2704×1520，59帧直接解码，逐像素核验无缩小、无裁切。原版2704×1520；Lanczos插值2倍5408×3040；SwinIR超分4倍10816×6080。背景与人物使用同一放大流程，内参同步缩放，外参固定。</p><p>三版均不读取mask，不做背景合成，无alpha/BCE/边界损失，也不根据前景筛选初始化或深度样本。损失为全图0.8 L1＋0.2 DSSIM。高斯模型、优化器、增密与简化时序保留；全程按实际尺寸训练。SSIM和深度回投影分段计算以节省内存，数值核验见记录。</p><p>新初始化使用整图SIFT三角化点与原场景SfM稀疏点，三版完全相同。未加载旧高斯或旧优化器。<a href="plan.json">实验配置</a> · <a href="native-pixel-audit.json">59帧像素核验</a> · <a href="loss-equivalence.json">损失与梯度核验</a></p></section>',
          '<section><h2>通用点云和相机</h2>']
    for s in methods:
        d=ex.read(ROOT/ex.NAMES[s]/'delivery.json');run=ex.read(ROOT/ex.NAMES[s]/'execution.json');p=ex.NAMES[s]+'/'
        text.append(f'<p>{LABELS[s]} · {d["points"]:,}个高斯 · 59台相机 · {run["seconds"]:.0f}秒<br><a href="{p+d["bundle"]}">PLY＋相机ZIP</a> · <a href="{p+d["default_ply"]}">标准3DGS PLY</a></p>')
    text.append('</section><section><h2>共同原图相机下的对比</h2><p>统一使用2704×1520相机渲染评测；页面预览会缩小显示，训练没有缩小。人像和脸部只用矩形区域作观察与统计，不参与训练。以下为训练视角重建误差，不是留出成绩。</p><table><tr><th>指标</th>'+''.join('<th>'+LABELS[s]+'</th>' for s in methods)+'</tr>')
    for key,label in [('l1','全图L1 ↓'),('ssim','全图SSIM ↑'),('psnr','全图PSNR ↑'),('body_l1','人像矩形L1 ↓'),('face_l1','面部矩形L1 ↓')]:
        text.append('<tr><td>'+label+'</td>'+''.join(f'<td>{metrics[s]["means"][key]:.5f}</td>' for s in methods)+'</tr>')
    text.append('</table><p>训练及格式验证完成与质量通过分别记录。<a href="comparison.json">完整统计</a></p></section>')
    findings=ex.read(ROOT/'review-findings.json') if (ROOT/'review-findings.json').exists() else {}
    if findings:text.append('<section><h2>实际画面审阅</h2>'+''.join('<p>'+escape(x)+'</p>' for x in findings['summary'])+'</section>')
    for i in [0,11,23,34,46,58]:
        n=rows[i]['frame'];text.append(f'<section><h2>{n}</h2><a href="views/{n}/full_compare.png"><img src="views/{n}/full_compare.png"></a><a href="views/{n}/body_compare.png"><img loading="lazy" src="views/{n}/body_compare.png"></a><a href="views/{n}/face_compare.png"><img loading="lazy" src="views/{n}/face_compare.png"></a></section>')
    text.append('<section><h2>全部59帧</h2>')
    for r in rows:
        n=r['frame'];text.append(f'<p>{n}：<a href="views/{n}/source.png">完整原图</a> · <a href="views/{n}/full_compare.png">整图对比</a> · <a href="views/{n}/body_compare.png">人体</a> · <a href="views/{n}/face_compare.png">面部</a></p>')
    text.append('</section></html>');html=''.join(text);(ROOT/'查看完整图对照.html').write_text(html,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
    parser=Links();parser.feed(html)
    for link in parser.links:
        url=urlsplit(link)
        if not url.scheme and url.path:assert (ROOT/unquote(url.path)).is_file(),link
    ex.save(ROOT/'archive-verification.json',dict(passed=True,links=len(parser.links)))
    print('FULLFRAME_REPORT_READY',ROOT/'查看完整图对照.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','evaluate','publish']);p.add_argument('--scale',type=int,choices=[1,2,4],default=1);a=p.parse_args()
    if a.action=='publish':publish()
    else:globals()[a.action](a.scale)
