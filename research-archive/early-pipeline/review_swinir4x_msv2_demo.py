"""Compare fresh SwinIR4x/MSV2 against the frozen Lanczos3x/MSV2 result."""
from pathlib import Path
from types import SimpleNamespace
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,ast,copy,importlib.util,json,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

RUN=Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')
PREVIOUS=Path('/opt/gs-slam-lab/msv2-demo-upsample3x-20261104')
PREVIOUS_RECORD=Path('/mnt/d/00/记录/2026-11-04_MSV2_人体3倍上采样/demo')
OLD_CAMERAS=Path('/mnt/d/00/记录/人体上采样静止重建/demo_3x/cameras.json')
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ex=load_module('swinir_msv2_experiment',RUN/'experiment.py')
h=load_module('portable_msv2_review',Path(__file__).with_name('review_msv2_upsampled_demo.py'))
OUT=ex.RECORD;h.RUN=RUN;h.OUT=OUT;h.ex=ex
h.PLY_STEM='demo_swinir4x_msv2_18000';h.BUNDLE='demo_SwinIR4倍_MSV2_18000步_点云与相机.zip'


def evaluate():
    torch=ex.runtime()
    from gaussian_renderer import render
    from scene import GaussianModel
    assert ex.read(RUN/'execution.json')['state']=='training_completed'
    final=RUN/'model/point_cloud/iteration_18000/point_cloud.ply'
    assert ex.read(final.parent/'ply-parity.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False)
    source=ex.BASE/'source/src/shuolee_portrait/training.py'
    tree=ast.parse(source.read_text(encoding='utf-8'));f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='foreground_loss')
    ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec'),ns);foreground_loss=ns['foreground_loss']
    rows=ex.read(RUN/'training-inputs.json')['records'];native=ex.read(RUN/'native-inputs.json')['records']
    models={}
    for key,path in [('previous',PREVIOUS/'model/point_cloud/iteration_18000/point_cloud.ply'),('swinir',final)]:
        g=GaussianModel(3);g.load_ply(str(path));g.active_sh_degree=3;models[key]=g
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda');metrics={}
    with torch.no_grad():
        for domain,collection in [('swinir_target_4x',rows),('original_native',native)]:
            result=[]
            for i,r in enumerate(collection):
                c=ex.Camera(r,i);gt=torch.tensor(c.rgb,device='cuda',dtype=torch.float32).permute(2,0,1)/255.
                mask=torch.tensor(c.mask,device='cuda',dtype=torch.float32)[None]/255.;fg=mask[0]>.99
                x0,y0,x1,y1=h.face_box(r,False);entry=dict(frame=r['frame']);folder=OUT/r['frame'];folder.mkdir(exist_ok=True)
                for key,model in models.items():
                    pred=render(c,model,pipe,bg)['render'];_,l1,ssim=foreground_loss(pred,gt,mask)
                    face=(pred-gt).abs().mean(0)[y0:y1,x0:x1][fg[y0:y1,x0:x1]]
                    entry[key]=dict(foreground_l1=float(l1),foreground_ssim=float(ssim),face_foreground_l1=float(face.mean()))
                    if domain=='swinir_target_4x':ex.save_image(folder/(key+'_msv2_4x.png'),pred)
                    elif i in ex.SELECTED:ex.save_image(folder/(key+'_msv2_native.png'),pred)
                    del pred
                result.append(entry)
                if domain=='swinir_target_4x':
                    shutil.copy2(r['image'],folder/'input_swinir4x.png');shutil.copy2(native[i]['image'],folder/'input_native.png')
                    box=h.face_box(native[i]);box4=[v*4 for v in box]
                    with Image.open(folder/'input_native.png') as im:
                        face=im.crop(tuple(box));face.resize((face.width*4,face.height*4),Image.Resampling.LANCZOS).save(folder/'input_native_face_4x.png')
                    for name in ['input_swinir4x','previous_msv2_4x','swinir_msv2_4x']:
                        with Image.open(folder/(name+'.png')) as im:im.crop(tuple(box4)).save(folder/(name+'_face.png'))
                    names=['input_native','input_swinir4x','previous_msv2_4x','swinir_msv2_4x']
                    labels=['原始输入','SwinIR 4倍输入','MSV2 插值3倍训练','MSV2 SwinIR4倍训练']
                    h.board([folder/(n+'.png') for n in names],labels,folder/'body_compare.png',title=r['frame'])
                    h.board([folder/'input_native_face_4x.png']+[folder/(n+'_face.png') for n in names[1:]],labels,folder/'face_compare.png',size=(240,300),title=r['frame'])
                if i%10==0 or i==58:print('EVALUATION',domain,i+1,'/59',flush=True)
                del c,gt,mask
            metrics[domain]=result
        orbit=ex.read(OLD_CAMERAS)['novel_review_cameras'];(OUT/'orbit').mkdir()
        for i,r in enumerate(orbit):
            r=copy.deepcopy(r);dummy=OUT/'orbit'/'_canvas.png';Image.new('RGB',tuple(r['wh'])).save(dummy)
            r.update(image=str(dummy),mask=str(dummy));cam=ex.Camera(r,i)
            ex.save_image(OUT/'orbit'/(r['frame']+'_swinir.png'),render(cam,models['swinir'],pipe,bg)['render'])
            dummy.unlink();shutil.copy2(PREVIOUS_RECORD/'orbit'/(r['frame']+'_msv2.png'),OUT/'orbit'/(r['frame']+'_previous.png'))
            h.board([OUT/'orbit'/(r['frame']+s+'.png') for s in ['_previous','_swinir']],['MSV2 插值3倍','MSV2 SwinIR4倍'],OUT/'orbit'/(r['frame']+'_compare.png'),size=(256,384),title=r['frame'])
    means={d:{m:{k:float(np.mean([r[m][k] for r in rows_])) for k in ['foreground_l1','foreground_ssim','face_foreground_l1']} for m in models} for d,rows_ in metrics.items()}
    ex.save(OUT/'comparison.json',dict(created=ex.now(),metrics=metrics,means=means,frames=59,native_wh=[476,960],training_wh=[1904,3840],
        comparison_reference='Native original images are the common reference. 4x metrics fit a learned SwinIR target, not a captured HR truth.',
        training_views_not_heldout=True,novel_views=12,novel_views_have_no_ground_truth=True,quality_accepted=False,models_promoted=False,
        previous_model_sha256=ex.sha(PREVIOUS/'model/point_cloud/iteration_18000/point_cloud.ply'),current_model_sha256=ex.sha(final)))
    for name in ['plan.json','preflight.json','execution.json','config.json','superresolution.json']:
        shutil.copy2(RUN/name,OUT/name)
    shutil.copy2(final,OUT/(h.PLY_STEM+'_world.ply'));shutil.copy2(final.parent/'ply-parity.json',OUT/'native-ply-parity.json')
    ex.save(OUT/'cameras_world.json',h.camera_records(rows));ex.save(OUT/'camera-input-mapping.json',dict(records=rows,convention='K pixel corners; integer-center principal point is corner minus 0.5',world='Original background SfM coordinates'))
    h.export_upright(rows,models['swinir'],pipe,bg)
    del models;torch.cuda.empty_cache();h.background()
    print(json.dumps(means),flush=True)


def publish():
    assert ex.read(OUT/'upright-parity.json')['passed']
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    for kind,size in [('body',(400,225)),('face',(400,160))]:
        for begin in range(1,60,12):
            ids=list(range(begin,min(begin+12,60)));cw,ch=size[0]+10,size[1]+30
            canvas=Image.new('RGB',(3*cw,((len(ids)+2)//3)*ch),'#eeeeee');d=ImageDraw.Draw(canvas)
            for j,i in enumerate(ids):
                name=f'frame_{i:06d}';x=(j%3)*cw;y=(j//3)*ch
                with Image.open(OUT/name/(kind+'_compare.png')) as im:im=im.convert('RGB');im.thumbnail(size,Image.Resampling.LANCZOS)
                d.text((x+4,y+2),name,font=font,fill='black');canvas.paste(im,(x,y+27))
            canvas.save(OUT/f'all_{kind}_{(begin-1)//12+1:02d}.png')
    for first in [0,180]:
        canvas=Image.new('RGB',(1536,888),'#eeeeee')
        for j,a in enumerate(range(first,first+180,30)):
            with Image.open(OUT/'orbit'/f'orbit_{a:03d}_compare.png') as im:canvas.paste(im,((j%3)*512,(j//3)*444))
        canvas.save(OUT/f'all_orbit_{first:03d}.png')
    c=ex.read(OUT/'comparison.json');e=ex.read(OUT/'execution.json')
    findings=ex.read(OUT/'review-findings.json') if (OUT/'review-findings.json').exists() else {}
    findings_html=''.join('<p>'+escape(t)+'</p>' for t in findings.get('summary_paragraphs',['训练和文件回读已完成，图像审阅正在整理。']))
    table=[]
    for d,label in [('original_native','原始476×960'),('swinir_target_4x','SwinIR目标1904×3840')]:
        for k,title in [('foreground_l1','人物颜色误差 ↓'),('foreground_ssim','人物SSIM ↑'),('face_foreground_l1','脸部颜色误差 ↓')]:
            table.append(f'<tr><td>{label}</td><td>{title}</td><td>{c["means"][d]["previous"][k]:.5f}</td><td>{c["means"][d]["swinir"][k]:.5f}</td></tr>')
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>demo · SwinIR 4倍超分＋MSV2</title>',
      '<style>body{max-width:1280px;margin:30px auto;padding:0 20px;background:#f6f6f3;color:#222;font:16px/1.65 system-ui}section{background:white;padding:22px;margin:20px 0;border-radius:12px}h1{font-size:30px}h2{font-size:22px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;text-align:left;padding:8px}a{color:#176684}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}</style>',
      '<h1>demo · SwinIR 4倍超分＋MSV2</h1>',
      f'<p>59帧 · 476×960 → 1904×3840 · 宽高4倍、像素16倍 · 18,000步时钟／17,999次更新 · {e["points"]:,}个高斯</p>',
      f'<section><h2>点云与相机</h2><p><a download href="{h.BUNDLE}">通用PLY＋59台配套相机ZIP</a> · <a href="{h.PLY_STEM}_yup.ply">+Y向上PLY</a> · <a href="cameras_yup.json">配套相机JSON</a></p><p>fx、fy、cx、cy及skew同步乘4，K的最后一行保持[0,0,1]；外参保持原背景估计。+Y来自参考相机方向，不代表重力标定。</p></section>',
      '<section id="findings"><h2>效果结论</h2>'+findings_html+'</section>',
      '<section><h2>同一参考下的比较</h2><p>共同参考是原始476×960图像。超分图新增细节来自SwinIR推断，并非真实高分辨率拍摄；拟合超分图更好，不等于三维几何更准确。两版保持相同MSV2策略、相机、未训练初始点和随机种子，同时改变输入方法与倍率。</p><table><tr><th>参考图</th><th>指标</th><th>上一版：3倍插值＋MSV2</th><th>本版：SwinIR 4倍＋MSV2</th></tr>'+''.join(table)+'</table></section>',
      '<section><h2>来源与配置</h2><p>官方 SwinIR-L real_sr ×4、params_ema，FP32、256像素分块、32像素重叠。MSV2沿用fast 18k增密／深度回建／简化时序，全程使用1904×3840，无缩小预热。高斯与优化器重新初始化。</p><p><a href="https://github.com/JingyunLiang/SwinIR">SwinIR源码</a> · <a href="plan.json">配置与权重来源</a> · <a href="superresolution.json">逐帧超分记录</a> · <a href="comparison.json">逐帧评测</a> · <a href="upright-parity.json">坐标／SH／协方差核验</a></p></section>',
      '<section><h2>灰背景检查</h2><p>原图／本版灰底原生渲染／累计不透明度；零次参数更新。用于暴露黑背景掩盖的多余高斯。</p><img src="background-diagnostic/frame_000001_compare.png"><p><a href="background-diagnostic/diagnostic.json">7视角透明度记录</a></p></section>']
    for i in [1,12,24,28,35,47,59]:
        n=f'frame_{i:06d}';parts.append(f'<section><h2>{n}</h2><p>原始图／SwinIR输入／旧MSV2／本次MSV2；各列对应同一相机。</p><a href="{n}/face_compare.png"><img src="{n}/face_compare.png"></a><a href="{n}/body_compare.png"><img src="{n}/body_compare.png" loading="lazy"></a><p><a href="{n}/input_native.png">原始图</a> · <a href="{n}/input_swinir4x.png">4倍超分完整图</a> · <a href="{n}/swinir_msv2_4x.png">本次完整渲染</a></p></section>')
    parts.append('<section><h2>全部59帧</h2><div class="grid">')
    for i in range(1,60):
        n=f'frame_{i:06d}';parts.append(f'<div><a href="{n}/body_compare.png">{n} 全身</a> · <a href="{n}/face_compare.png">脸部</a><a href="{n}/body_compare.png"><img src="{n}/body_compare.png" loading="lazy"></a></div>')
    parts.append('</div></section><section><h2>相同12个环绕视角</h2><p>环绕视角没有实拍真值。</p><div class="grid">')
    for a in range(0,360,30):
        n=f'orbit_{a:03d}';parts.append(f'<a href="orbit/{n}_compare.png"><img src="orbit/{n}_compare.png" loading="lazy"></a>')
    parts.append('</div></section></html>');page=OUT/'查看对照.html';page.write_text(''.join(parts),encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src') and v)
    parser=Links();parser.feed(page.read_text(encoding='utf-8'))
    for link in parser.links:
        u=urlsplit(link)
        if not u.scheme and u.path:assert (OUT/unquote(u.path)).exists(),link
    files=[]
    for p in OUT.rglob('*.png'):
        with Image.open(p) as im:im.verify()
        files.append(dict(path=p.relative_to(OUT).as_posix(),sha256=ex.sha(p)))
    ex.save(OUT/'archive-verification.json',dict(passed=True,images=files,html_links=len(parser.links),points=e['points'],quality_accepted=False))
    print('PUBLISHED',page,len(files),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','publish']);a=p.parse_args()
    evaluate() if a.action=='evaluate' else publish()
