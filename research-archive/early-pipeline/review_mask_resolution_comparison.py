"""Review the original/2x/4x experiment with the same mask penalty."""
from pathlib import Path
from types import SimpleNamespace
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse, importlib.util, json, shutil, zipfile
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/mnt/d/00/记录/2026-11-04_MSV2_mask原始_插值2倍_4倍对比')
RUNS={
    'original':Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-native-20261104'),
    'interpolation2x':Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-interpolate2x-20261104'),
    'swinir4x':Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-20261104')}
LABELS={'original':'原始尺寸','interpolation2x':'Lanczos 插值2倍','swinir4x':'SwinIR 超分4倍'}
FOUR=ROOT.parent/'2026-11-04_MSV2_原始_2倍_4倍对比/mask_penalty4x/mask_penalty'

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def helper():
    m=load('same_mask_review',Path(__file__).with_name('review_transparent_resolution_comparison.py'))
    m.ROOT=ROOT;m.RUNS=RUNS;m.LABELS=LABELS;m.ZOOM_INPUTS=RUNS['interpolation2x']/'training-inputs.json'
    return m

def export(label):
    m=helper();h=m.helper();run=RUNS[label];ex=load('mask_resolution_export',run/'experiment.py')
    torch=ex.runtime()
    from scene import GaussianModel
    execution=ex.read(run/'execution.json');assert execution['state']=='training_completed'
    out=ROOT/label;out.mkdir(parents=True,exist_ok=False)
    h.ex=ex;h.RUN=run;h.OUT=out
    h.PLY_STEM=f'demo_mask_{label}_msv2_18000'
    h.BUNDLE=f'demo_mask_{LABELS[label].replace(" ","")}_MSV2_18000步_点云与相机.zip'
    for f in ['plan.json','config.json','execution.json','input-contract.json','preflight.json','rgba-probe.json','configuration-parity.json']:
        shutil.copy2(run/f,out/f)
    final=run/'model/point_cloud/iteration_18000/point_cloud.ply'
    assert ex.read(final.parent/'ply-parity.json')['passed']
    world=out/(h.PLY_STEM+'_world.ply')
    shutil.copy2(final,world);assert ex.sha(world)==ex.sha(final)
    shutil.copy2(final.parent/'ply-parity.json',out/'native-ply-parity.json')
    rows=ex.read(run/'training-inputs.json')['records']
    ex.save(out/'cameras_world.json',h.camera_records(rows))
    ex.save(out/'camera-input-mapping.json',dict(records=rows,convention='Pixel-corner K; integer centers cx-.5,cy-.5',world='Original background SfM coordinates'))
    model=GaussianModel(3);model.load_ply(str(final));model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    upright_ok=True
    try: h.export_upright(rows,model,pipe,torch.zeros(3,device='cuda'))
    except AssertionError:
        parity=ex.read(out/'upright-parity.json')
        if parity['passed'] is not False: raise
        upright_ok=False
        with zipfile.ZipFile(out/h.BUNDLE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            z.write(world,'point_cloud.ply');z.write(out/'cameras_world.json','cameras.json')
            for name in ['plan.json','camera-input-mapping.json','native-ply-parity.json','upright-parity.json']:
                z.write(out/name,name)
            z.writestr('coordinate-note.txt','Unchanged training world coordinates. Use the paired cameras. Upright conversion failed the existing render tolerance and is excluded.')
    default=h.PLY_STEM+('_yup.ply' if upright_ok else '_world.ply')
    ex.save(out/'delivery.json',dict(bundle=h.BUNDLE,world_ply=world.name,default_ply=default,
        upright_ply=h.PLY_STEM+'_yup.ply' if upright_ok else None,
        camera_file='cameras_yup.json' if upright_ok else 'cameras_world.json',
        sha256=ex.sha(out/h.BUNDLE),bytes=(out/h.BUNDLE).stat().st_size,
        points=execution['points'],cameras=len(rows),export_verified=True,
        world_export_verified=True,upright_export_verified=upright_ok,geometry_quality_accepted=False))
    print('MASK_RESOLUTION_EXPORTED',label,execution['points'],upright_ok,flush=True)

def evaluate():
    helper().evaluate()
    ex=load('mask_compare_metadata',RUNS['original']/'experiment.py');c=ex.read(ROOT/'comparison.json')
    c.update(mask_penalty_identical=True,boundary_width_native_pixels=8,model_architecture_unchanged=True,
        no_postprocess_pruning=True,video_pixel_lineage_verified=True,
        input_pixel_audit='原视频到训练像素核验.json',quality_accepted=False)
    ex.save(ROOT/'comparison.json',c)

def publish():
    ex=load('mask_comparison_publish',RUNS['original']/'experiment.py');c=ex.read(ROOT/'comparison.json')
    findings=ex.read(ROOT/'review-findings.json') if (ROOT/'review-findings.json').exists() else {}
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    for kind in ['body','face','white','checker','black']:
        for first in range(1,60,12):
            ids=list(range(first,min(60,first+12)));canvas=Image.new('RGB',(1600,((len(ids)+1)//2)*260),'#eee');d=ImageDraw.Draw(canvas)
            for j,i in enumerate(ids):
                frame=f'frame_{i:06d}'
                with Image.open(ROOT/'views'/frame/(kind+'_compare.png')) as im:
                    im=im.convert('RGB');im.thumbnail((790,225),Image.Resampling.LANCZOS)
                x=j%2*800;y=j//2*260;d.text((x+4,y),frame,font=font,fill='black');canvas.paste(im,(x,y+28))
            canvas.save(ROOT/f'all_{kind}_{(first-1)//12+1:02d}.png')
    for first in [0,180]:
        canvas=Image.new('RGB',(1536,1332),'#eee')
        for j,a in enumerate(range(first,first+180,30)):
            with Image.open(ROOT/'orbit'/f'orbit_{a:03d}_compare.png') as im:canvas.paste(im,((j%2)*768,(j//2)*444))
        canvas.save(ROOT/f'all_orbit_{first:03d}.png')
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同mask策略 · 原始 / 插值2倍 / 超分4倍</title>',
        '<style>body{max-width:1300px;margin:32px auto;padding:0 20px;background:#f5f5f2;color:#222;font:16px/1.7 system-ui}section{padding:24px;background:white;border-radius:12px;margin:20px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#176684}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}</style>',
        '<h1>同mask策略 · 原始 / 插值2倍 / 超分4倍</h1><p>demo · MSV2模型与训练时序不变 · 三版各自从头训练18,000步</p>',
        '<section><h2>输入没有先缩小</h2><p>原视频2704×1520，全部59帧重新使用FFmpeg直接解码，不加缩放滤镜；与训练来源PNG逐像素一致，最大误差为0。人物区域按原图整数坐标[1109,501,1585,1461]裁切为476×960，没有下采样。</p><table><tr><th>版本</th><th>实际训练尺寸</th><th>来源</th></tr><tr><td>原始</td><td>476×960</td><td>原图人物裁切，像素不变</td></tr><tr><td>插值2倍</td><td>952×1920</td><td>原始裁切直接Lanczos放大</td></tr><tr><td>超分4倍</td><td>1904×3840</td><td>原始裁切直接SwinIR超分</td></tr></table><p>训练全程使用对应全尺寸；无半尺寸预热。内参前两行同步乘以2或4，外参不变。<a href="原视频到训练像素核验.json">完整59帧像素链路核验</a></p></section>',
        '<section><h2>对比结论</h2>'+''.join('<p>'+escape(s)+'</p>' for s in findings.get('summary_paragraphs',['训练已完成，全部视角审阅进行中。']))+'</section>',
        '<section><h2>固定的训练条件</h2><p>相同59帧、相机、原始mask、200,000个未优化初始点和随机种子6060；相同MSV2优化器、增密和简化流程。RGB仅监督完整前景窗口；保留平衡mask BCE权重0.2和外侧边界BCE权重0.1。边界宽度为8个原始像素，随输入倍率一起缩放。没有训练后裁剪高斯。</p><p>2倍只使用插值。4倍沿用已完成SwinIR结果，同时改变分辨率与纹理预测，差异不能只归因于倍率。<a href="comparison-plan.json">固定条件记录</a></p></section>',
        '<section><h2>通用点云与配套相机</h2>']
    for label in RUNS:
        folder=FOUR if label=='swinir4x' else ROOT/label
        prefix='../2026-11-04_MSV2_原始_2倍_4倍对比/mask_penalty4x/mask_penalty/' if label=='swinir4x' else label+'/'
        d=ex.read(folder/'delivery.json');ply=d.get('default_ply') or d.get('upright_ply') or d['world_ply']
        coords='+Y朝向，完整坐标转换已核验' if d.get('upright_export_verified') else '原始训练世界坐标，使用包内相机'
        parts.append(f'<p>{LABELS[label]} · {d["points"]:,}点 · {coords}<br><a href="{prefix+d["bundle"]}">PLY＋59台相机ZIP</a> · <a href="{prefix+ply}">独立PLY</a></p>')
    parts.append('</section><section><h2>共同原图相机下的59帧指标</h2><table><tr><th>指标</th>'+''.join('<th>'+v+'</th>' for v in LABELS.values())+'</tr>')
    for key,label in [('face_exterior_alpha_mean','面部外侧平均Alpha ↓'),('face_inside_alpha_mean','面部内部平均Alpha'),('outside_alpha_mean','人物外侧平均Alpha ↓'),('inside_alpha_mean','人物内部平均Alpha'),('foreground_l1','人物原图L1 ↓'),('face_foreground_l1','脸部原图L1 ↓'),('foreground_ssim','人物原图SSIM ↑'),('mask_iou','轮廓IoU ↑')]:
        parts.append('<tr><td>'+label+'</td>'+''.join(f'<td>{c["means"][m][key]:.5f}</td>' for m in RUNS)+'</tr>')
    parts.append('</table><p>这是训练视角对照，不是留出评测。整人指标统一在476×960原图相机计算；脸部对比统一用952×1920相机渲染，原图仅作显示放大。环绕视角没有实拍真值。<a href="comparison.json">逐帧指标</a></p></section>')
    for i in [1,12,24,28,35,47,59]:
        n=f'frame_{i:06d}';parts.append(f'<section><h2>{n}</h2><a href="views/{n}/face_compare.png"><img src="views/{n}/face_compare.png"></a><a href="views/{n}/body_compare.png"><img src="views/{n}/body_compare.png"></a><p><a href="views/{n}/white_compare.png">白底</a> · <a href="views/{n}/checker_compare.png">棋盘底</a> · <a href="views/{n}/black_compare.png">黑底</a></p></section>')
    parts.append('<section><h2>全部59帧对照原图</h2><div class="grid">')
    for i in range(1,60):
        n=f'frame_{i:06d}';parts.append(f'<div><a href="views/{n}/source.png">{n} 原图</a> · <a href="views/{n}/face_compare.png">脸部</a> · <a href="views/{n}/body_compare.png">整人</a><img loading="lazy" src="views/{n}/face_compare.png"><p>'+ ' · '.join(f'<a href="views/{n}/{k}_transparent.png">{LABELS[k]}透明PNG</a>' for k in RUNS)+'</p></div>')
    parts.append('</div></section><section><h2>12个环绕视角</h2><div class="grid">')
    for a in range(0,360,30):parts.append(f'<a href="orbit/orbit_{a:03d}_compare.png"><img loading="lazy" src="orbit/orbit_{a:03d}_compare.png"></a>')
    parts.append('</div></section></html>');html=''.join(parts);(ROOT/'查看mask倍率对照.html').write_text(html,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
    parser=Links();parser.feed(html)
    for link in parser.links:
        u=urlsplit(link)
        if not u.scheme and u.path:assert (ROOT/unquote(u.path)).is_file(),link
    count=0
    for path in ROOT.rglob('*.png'):
        with Image.open(path) as im:im.verify()
        count+=1
    ex.save(ROOT/'archive-verification.json',dict(passed=True,PNGs=count,links=len(parser.links),pixel_audit_verified=True))
    print('MASK_RESOLUTION_REPORT_READY',ROOT/'查看mask倍率对照.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','evaluate','publish']);p.add_argument('--method',choices=['original','interpolation2x'])
    a=p.parse_args()
    if a.action=='export':export(a.method)
    else:globals()[a.action]()
