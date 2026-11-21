"""Native multiple-background checks and transparent PNG export for MSV2."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
from html import escape
import argparse, json, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/opt/gs-slam-lab/msv2-demo-transparent-full-20261104')
BASE=Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')
RECORD=Path('/mnt/d/00/记录/2026-11-04_透明背景_MSV2/demo_18000')
SELECTED_RECORD=RECORD.parent/'demo'


def evaluator():
    source=Path(__file__).with_name('review_swinir4x_msv2_demo.py')
    code=source.read_text(encoding='utf-8').split('\ndef publish():')[0]
    code=code.replace("RUN=Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')",f'RUN=Path({str(ROOT)!r})')
    code=code.replace("PREVIOUS=Path('/opt/gs-slam-lab/msv2-demo-upsample3x-20261104')",f'PREVIOUS=Path({str(BASE)!r})')
    code=code.replace('/mnt/d/00/记录/2026-11-04_MSV2_人体3倍上采样/demo','/mnt/d/00/记录/2026-11-04_SwinIR4倍_MSV2/demo')
    code=code.replace("PREVIOUS_RECORD/'orbit'/(r['frame']+'_msv2.png')", "PREVIOUS_RECORD/'orbit'/(r['frame']+'_swinir.png')")
    code=code.replace('MSV2 插值3倍训练','MSV2 黑底监督').replace('MSV2 SwinIR4倍训练','MSV2 透明监督')
    code=code.replace('MSV2 插值3倍','MSV2 黑底监督').replace('MSV2 SwinIR4倍','MSV2 透明监督')
    m=ModuleType('transparent_review');m.__file__=str(source)
    exec(compile(code,str(source),'exec'),m.__dict__)
    m.h.PLY_STEM='demo_swinir4x_msv2_transparent_18000'
    m.h.BUNDLE='demo_SwinIR4倍_MSV2_透明训练_18000步_点云与相机.zip'
    return m


def transparent_views(m):
    ex=m.ex;torch=ex.runtime()
    _,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    from scene import GaussianModel
    from gaussian_renderer import render
    old=GaussianModel(3);old.load_ply(str(BASE/'model/point_cloud/iteration_18000/point_cloud.ply'));old.active_sh_degree=3
    new=GaussianModel(3);new.load_ply(str(ROOT/'model/point_cloud/iteration_18000/point_cloud.ply'));new.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    rows=ex.read(ROOT/'native-inputs.json')['records'];out=RECORD/'transparent-views';out.mkdir(exist_ok=False)
    checks=[];composites=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            cam=ex.Camera(r,i);mask=torch.as_tensor(cam.mask.copy(),device='cuda',dtype=torch.float32)/255
            entry=dict(frame=r['frame']);folder=out/r['frame'];folder.mkdir()
            for label,model in [('baseline',old),('transparent',new)]:
                color=render(cam,model,pipe,torch.zeros(3,device='cuda'))['render']
                alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(model.get_xyz))['render'][0:1]
                outside=mask<.01;inside=mask>.99;a=alpha[0]
                straight=color/alpha.clamp_min(1e-4)
                dark=(straight.clamp(0,1).mean(0)<.15)&(a>.5)
                entry[label]=dict(outside_alpha_mean=float(a[outside].mean()),outside_alpha_gt_01=float((a[outside]>.1).float().mean()),
                    inside_alpha_mean=float(a[inside].mean()),inside_alpha_gt_05=float((a[inside]>.5).float().mean()),
                    outside_dark_opaque_fraction=float(dark[outside].float().mean()),
                    mask_iou=float(((a>.5)&(mask>.5)).sum()/((a>.5)|(mask>.5)).sum()))
                ex.save_image(folder/(label+'_white.png'),color+(1-alpha))
                ex.save_image(folder/(label+'_gray.png'),color+.5*(1-alpha))
                if label=='transparent':
                    for name,bg in [('black',0.),('gray',.5)]:ex.save_image(folder/(name+'.png'),color+(1-alpha)*bg)
                    yy,xx=torch.meshgrid(torch.arange(cam.height,device='cuda'),torch.arange(cam.width,device='cuda'),indexing='ij')
                    checker=torch.where(((xx//32+yy//32)%2)==0,.85,.45)[None]
                    ex.save_image(folder/'checker.png',color+(1-alpha)*checker)
                    ex.save_image(folder/'alpha.png',alpha.expand(3,-1,-1))
                    straight=color/alpha.clamp_min(1e-4)
                    rgba=torch.cat([straight.clamp(0,1),alpha.clamp(0,1)],dim=0)
                    pixels=np.rint(rgba.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                    Image.fromarray(pixels,'RGBA').save(folder/'transparent.png')
                    if i in ex.SELECTED:
                        direct_white=render(cam,model,pipe,torch.ones(3,device='cuda'))['render']
                        err=(direct_white-(color+1-alpha)).abs()
                        composites.append(dict(frame=r['frame'],max_abs=float(err.max()),mean_abs=float(err.mean())))
            m.h.board([r['image'],folder/'baseline_white.png',folder/'transparent_white.png',folder/'checker.png'],
                      ['原始人物','修正前 白底','透明监督 白底','透明监督 棋盘底'],folder/'compare.png',size=(300,600),title=r['frame'])
            m.h.board([r['image'],folder/'baseline_gray.png',folder/'transparent_gray.png'],
                      ['原始人物','旧黑底监督 18000步','透明监督 18000步'],folder/'gray_compare.png',size=(300,600),title=r['frame'])
            m.h.board([r['image'],SELECTED_RECORD/r['frame']/'transparent_gray.png',folder/'transparent_gray.png'],
                      ['原始人物','选定候选 3000步','同配置完整训练 18000步'],folder/'selected_compare.png',size=(300,600),title=r['frame'])
            checks.append(entry)
            if i%10==0 or i==58:print('TRANSPARENT_VIEWS',i+1,'/59',flush=True)
    means={key:{metric:float(np.mean([r[key][metric] for r in checks])) for metric in checks[0][key]} for key in ['baseline','transparent']}
    ex.save(RECORD/'transparency-comparison.json',dict(checks=checks,means=means,rgba_pngs=59,
        rgba_png_encoding='Straight RGB plus alpha, 8-bit PNG; RGB >1 is clipped for display.',
        dark_pixel_definition='Among input-mask <0.01 pixels: rendered alpha >0.5 and mean straight RGB <0.15; pixel fraction, not a Gaussian count. Existing held-object mask uncertainty remains.',
        compositor_checks=composites,compositor_passed=all(c['max_abs']<.005 for c in composites),
        backgrounds=['black','white','gray','checker'],quality_accepted=False))
    for f in ['rgba-probe.json','frozen-baseline-diagnostic.json','configuration-parity.json']:
        shutil.copy2(ROOT/f,RECORD/f)
    for iteration in [1000,3000,8000,18000]:
        shutil.copy2(ROOT/f'gate{iteration}/gate.json',RECORD/f'gate{iteration}.json')
    print('TRANSPARENCY_MEANS',json.dumps(means),flush=True)
    import copy
    orbit=ex.read(m.OLD_CAMERAS)['novel_review_cameras']
    with torch.no_grad():
        for i,r in enumerate(orbit):
            r=copy.deepcopy(r);dummy=RECORD/'orbit'/'_canvas.png';Image.new('RGB',tuple(r['wh'])).save(dummy)
            r.update(image=str(dummy),mask=str(dummy));cam=ex.Camera(r,i)
            for label,model in [('baseline',old),('transparent',new)]:
                color=render(cam,model,pipe,torch.zeros(3,device='cuda'))['render']
                alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(model.get_xyz))['render'][0:1]
                ex.save_image(RECORD/'orbit'/(r['frame']+'_'+label+'_gray.png'),color+.5*(1-alpha))
            dummy.unlink()
            m.h.board([RECORD/'orbit'/(r['frame']+'_'+s+'_gray.png') for s in ['baseline','transparent']],
                      ['旧黑底监督 18000步','透明监督 18000步'],RECORD/'orbit'/(r['frame']+'_gray_compare.png'),size=(256,384),title=r['frame'])


def publish(m):
    ex=m.ex;f=ex.read(RECORD/'review-findings.json') if (RECORD/'review-findings.json').exists() else {}
    t=ex.read(RECORD/'transparency-comparison.json');c=ex.read(RECORD/'comparison.json');e=ex.read(RECORD/'execution.json')
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    for kind in ['body','face','background','gray','selected']:
        for begin in range(1,60,12):
            ids=list(range(begin,min(begin+12,60)));canvas=Image.new('RGB',(1230,((len(ids)+2)//3)*260),'#eee');d=ImageDraw.Draw(canvas)
            for j,i in enumerate(ids):
                n=f'frame_{i:06d}'
                if kind in ('body','face'):p=RECORD/n/(kind+'_compare.png')
                else:p=RECORD/'transparent-views'/n/('compare.png' if kind=='background' else kind+'_compare.png')
                with Image.open(p) as im:im=im.convert('RGB');im.thumbnail((400,225),Image.Resampling.LANCZOS)
                x=j%3*410;y=j//3*260;d.text((x+4,y),n,font=font,fill='black');canvas.paste(im,(x,y+28))
            canvas.save(RECORD/f'all_{kind}_{(begin-1)//12+1:02d}.png')
    for first in [0,180]:
        canvas=Image.new('RGB',(1536,888),'#eee')
        for j,a in enumerate(range(first,first+180,30)):
            with Image.open(RECORD/'orbit'/f'orbit_{a:03d}_gray_compare.png') as im:canvas.paste(im,((j%3)*512,(j//3)*444))
        canvas.save(RECORD/f'all_orbit_{first:03d}.png')
    rows=[]
    for key,label in [('outside_alpha_mean','人物外平均不透明度 ↓'),('outside_alpha_gt_01','人物外不透明度 >0.1 的比例 ↓'),('outside_dark_opaque_fraction','人物外黑色不透明像素比例 ↓'),('inside_alpha_mean','人物内平均不透明度'),('mask_iou','轮廓 IoU ↑')]:
        rows.append(f'<tr><td>{label}</td><td>{t["means"]["baseline"][key]:.5f}</td><td>{t["means"]["transparent"][key]:.5f}</td></tr>')
    for key,label in [('foreground_l1','原始尺寸人物 L1 ↓'),('foreground_ssim','原始尺寸人物 SSIM ↑'),('face_foreground_l1','原始尺寸脸部 L1 ↓')]:
        rows.append(f'<tr><td>{label}</td><td>{c["means"]["original_native"]["previous"][key]:.5f}</td><td>{c["means"]["original_native"]["swinir"][key]:.5f}</td></tr>')
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>demo · 透明监督与背景污染对照</title>',
      '<style>body{max-width:1280px;margin:30px auto;padding:0 20px;background:#f4f4f1;color:#222;font:16px/1.7 system-ui}section{padding:24px;background:white;margin:20px 0;border-radius:12px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}a{color:#176684}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.checker{background:repeating-conic-gradient(#ddd 0% 25%,white 0% 50%) 50%/24px 24px}</style>',
      '<h1>demo · 透明监督与背景污染对照</h1>',
      f'<p>SwinIR 4倍输入 · 59帧 · 1904×3840 · MSV2 18,000步时钟 · {e["points"]:,}个高斯</p>',
      '<section><h2>效果结论</h2>'+''.join('<p>'+escape(x)+'</p>' for x in f.get('summary_paragraphs',['训练与导出已完成，正在审阅完整对照。']))+'</section>',
      f'<section><h2>点云与透明图</h2><p><a href="{m.h.BUNDLE}">通用PLY＋59台配套相机</a> · <a href="{m.h.PLY_STEM}_yup.ply">独立PLY</a> · <a href="cameras_yup.json">相机JSON</a></p><p>PLY保留每个高斯的不透明度；透明PNG按渲染得到的Alpha导出，不使用人物遮罩裁掉模型伪影。</p><a href="transparent-views/frame_000001/transparent.png"><img class="checker" src="transparent-views/frame_000001/transparent.png" style="max-height:600px"></a></section>',
      '<section><h2>选定配置与检查</h2><p>按用户“手部是因为手里拿着东西，就用这版训练”的要求，沿用3,000步透明候选的同一配置从头训练。人物颜色与Alpha分开监督；颜色只在可靠人物区域计算，SSIM窗口完整排除背景；轮廓使用平衡透明度交叉熵；第2,000步深度重建仅从可靠人物区域采样。SwinIR输入、遮罩、相机、未训练初始点、种子和MSV2时序保持一致。</p><p>1%轮廓外不透明度门槛改为阶段记录，不再用于提前终止。原3,000步候选及其历史门槛结果保留。手持物按原输入保留，未修改遮罩。</p><p><a href="configuration-parity.json">与选定版本的配置核验</a> · <a href="rgba-probe.json">损失与梯度核验</a> · <a href="frozen-baseline-diagnostic.json">修正前冻结诊断</a> · <a href="plan.json">实验配置</a> · <a href="transparency-comparison.json">59视角透明度指标</a> · <a href="gate3000.json">3,000步记录</a> · <a href="gate18000.json">18,000步记录</a></p></section>',
      '<section><h2>同输入、同相机、同18,000步对照</h2><table><tr><th>指标</th><th>修正前</th><th>透明监督</th></tr>'+''.join(rows)+'</table><p>黑色不透明像素按Alpha&gt;0.5且直通RGB均值&lt;0.15统计，是像素比例，并非高斯数量。指标来自训练视角，受手持物原遮罩影响；环绕视角没有实拍真值。</p></section>',
      '<section><h2>从选定候选到完整训练</h2><p>左为原图，中为保留的3,000步候选，右为同配置重新完整训练的18,000步结果。</p><img src="transparent-views/frame_000001/selected_compare.png"></section>']
    for i in [1,12,24,28,35,47,59]:
        n=f'frame_{i:06d}';parts.append(f'<section><h2>{n}</h2><a href="transparent-views/{n}/compare.png"><img src="transparent-views/{n}/compare.png"></a><a href="{n}/face_compare.png"><img src="{n}/face_compare.png"></a><p><a href="transparent-views/{n}/transparent.png">透明PNG</a> · <a href="transparent-views/{n}/black.png">黑底</a> · <a href="transparent-views/{n}/gray.png">灰底</a> · <a href="transparent-views/{n}/transparent_white.png">白底</a></p></section>')
    parts.append('<section><h2>全部59帧</h2><div class="grid">')
    for i in range(1,60):
        n=f'frame_{i:06d}';parts.append(f'<div><a href="transparent-views/{n}/compare.png">{n}</a> · <a href="{n}/face_compare.png">脸部</a> · <a href="transparent-views/{n}/transparent.png">透明PNG</a><img loading="lazy" src="transparent-views/{n}/compare.png"></div>')
    parts.append('</div></section><section><h2>12个环绕视角</h2><div class="grid">')
    for a in range(0,360,30):parts.append(f'<a href="orbit/orbit_{a:03d}_gray_compare.png"><img loading="lazy" src="orbit/orbit_{a:03d}_gray_compare.png"></a>')
    parts.append('</div></section></html>');(RECORD/'查看对照.html').write_text(''.join(parts),encoding='utf-8')
    from html.parser import HTMLParser
    from urllib.parse import urlsplit,unquote
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('src','href'))
    parser=Links();parser.feed(''.join(parts))
    for link in parser.links:
        u=urlsplit(link)
        if not u.scheme and u.path:assert (RECORD/unquote(u.path)).exists(),link
    images=[]
    for p in RECORD.rglob('*.png'):
        with Image.open(p) as im:im.verify()
        images.append(dict(path=p.relative_to(RECORD).as_posix(),sha256=ex.sha(p)))
    ex.save(RECORD/'archive-verification.json',dict(passed=True,images=images,html_links=len(parser.links),quality_accepted=False))
    print('TRANSPARENT_REPORT_PUBLISHED',RECORD/'查看对照.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','publish']);a=p.parse_args();m=evaluator()
    if a.action=='evaluate':m.evaluate();transparent_views(m)
    else:publish(m)
