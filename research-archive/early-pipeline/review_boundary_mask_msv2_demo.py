"""Compare unchanged MSV2 with and without the added mask penalty."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import argparse,importlib.util,zipfile
from PIL import Image,ImageDraw,ImageFont

BASE=Path('/opt/gs-slam-lab/msv2-demo-transparent-full-20261104')
RUN=Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-20261104')
ROOT=Path('/mnt/d/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比/mask_penalty4x')


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def review_module():
    m=load('mask_comparison',Path(__file__).with_name('review_transparent_resolution_comparison.py'))
    m.ROOT=ROOT;m.RUNS={'before':BASE,'mask_penalty':RUN}
    m.LABELS={'before':'原有 mask 惩罚','mask_penalty':'加强轮廓 mask 惩罚'}
    return m


def evaluate():
    m=review_module();m.export('mask_penalty');m.evaluate()
    annotate_comparison()


def annotate_comparison():
    ex=load('mask_eval_record',RUN/'experiment.py');c=ex.read(ROOT/'comparison.json')
    c.update(four_x_is_learned_SR_and_two_x_is_interpolation=False,
        comparison='Identical4x inputs; only added exterior-boundary mask loss',
        model_architecture_unchanged=True,source_masks_unchanged=True,MSV2_schedule_unchanged=True,
        no_postprocess_pruning=True,only_mask_supervision_changed=True)
    ex.save(ROOT/'comparison.json',c)


def evaluate_world():
    """Explicit fallback: deliver the byte-identical world PLY after upright check failed."""
    ex=load('world_mask_export',RUN/'experiment.py');out=ROOT/'mask_penalty'
    parity=ex.read(out/'upright-parity.json')
    assert parity['passed'] is False
    assert ex.read(out/'native-ply-parity.json')['passed']
    name='demo_mask_penalty_msv2_transparent_18000_world.ply'
    assert ex.sha(out/name)==ex.sha(RUN/'model/point_cloud/iteration_18000/point_cloud.ply')
    bundle='demo_mask惩罚_MSV2_18000步_世界坐标点云与相机.zip'
    manifest=dict(bundle=bundle,world_ply=name,default_ply=name,
        cameras=59,points=ex.read(RUN/'execution.json')['points'],
        export_verified=True,world_export_verified=True,upright_export_verified=False,
        geometry_quality_accepted=False,coordinate_convention='Unchanged training-world coordinates; paired cameras included',
        upright_failure='Frame59 max render error0.0056038089 exceeds0.005; threshold unchanged; upright PLY not distributed')
    with zipfile.ZipFile(out/bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.write(out/name,'point_cloud.ply');z.write(out/'cameras_world.json','cameras.json')
        for n in ['native-ply-parity.json','configuration-parity.json','camera-input-mapping.json','plan.json','upright-parity.json']:
            z.write(out/n,n)
        import json
        z.writestr('delivery-coordinate-note.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    manifest.update(sha256=ex.sha(out/bundle),bytes=(out/bundle).stat().st_size)
    ex.save(out/'delivery.json',manifest)
    review_module().evaluate();annotate_comparison()


def stage(step):
    ex=load('mask_stage',RUN/'experiment.py')
    h=load('stage_board',Path(__file__).with_name('review_msv2_upsampled_demo.py'))
    assert (RUN/f'gate{step}/gate.json').exists()
    out=ROOT.parent/'face-noise-analysis'/f'stage_{step}';out.mkdir(parents=True,exist_ok=True)
    rows=ex.read(RUN/'native-inputs.json')['records']
    for i in sorted(set(ex.SELECTED+[27])):
        r=rows[i];folder=out/r['frame'];folder.mkdir(exist_ok=True)
        files=[r['image'],BASE/f'gate{step}'/(r['frame']+'_transparent_gray.png'),RUN/f'gate{step}'/(r['frame']+'_transparent_gray.png')]
        face=[]
        for j,f in enumerate(files):
            p=folder/f'face_{j}.png'
            with Image.open(f) as im:im.crop(h.face_box(r)).save(p)
            face.append(p)
        h.board(face,['原图','原有 mask 惩罚','加强轮廓 mask 惩罚'],folder/'face_compare.png',size=(320,360),title=f'{r["frame"]} / {step}步')
        h.board(files,['原图','原有 mask 惩罚','加强轮廓 mask 惩罚'],folder/'body_compare.png',size=(300,605),title=f'{r["frame"]} / {step}步')
    old=ex.read(BASE/f'gate{step}/gate.json')['means']['transparent']
    new=ex.read(RUN/f'gate{step}/gate.json')['means']['transparent']
    ex.save(out/'same_step_metrics.json',dict(iteration=step,before=old,mask_penalty=new,
        opacity_change=new['outside_alpha_mean']-old['outside_alpha_mean'],quality_accepted=False))
    print('MASK_SAME_STEP',step,old,new,flush=True)


def publish():
    ex=load('mask_publish',RUN/'experiment.py');c=ex.read(ROOT/'comparison.json')
    findings=ex.read(ROOT/'review-findings.json') if (ROOT/'review-findings.json').exists() else {}
    delivery=ex.read(ROOT/'mask_penalty/delivery.json')
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    for kind in ['body','face','white','checker','black']:
        for first in range(1,60,12):
            ids=list(range(first,min(60,first+12)));canvas=Image.new('RGB',(1440,((len(ids)+1)//2)*260),'#eee');d=ImageDraw.Draw(canvas)
            for j,i in enumerate(ids):
                n=f'frame_{i:06d}'
                with Image.open(ROOT/'views'/n/(kind+'_compare.png')) as im:
                    im=im.convert('RGB');im.thumbnail((710,225),Image.Resampling.LANCZOS)
                x=j%2*720;y=j//2*260;d.text((x+3,y),n,font=font,fill='black');canvas.paste(im,(x,y+28))
            canvas.save(ROOT/f'all_{kind}_{(first-1)//12+1:02d}.png')
    for first in [0,180]:
        canvas=Image.new('RGB',(1536,888),'#eee')
        for j,a in enumerate(range(first,first+180,30)):
            with Image.open(ROOT/'orbit'/f'orbit_{a:03d}_compare.png') as im:canvas.paste(im,((j%3)*512,(j//3)*444))
        canvas.save(ROOT/f'all_orbit_{first:03d}.png')
    metrics=[]
    for key,label in [('face_exterior_alpha_mean','面部区域外侧平均Alpha ↓'),('face_exterior_gt_01','面部外侧Alpha>0.1比例 ↓'),
        ('face_inside_alpha_mean','面部内部平均Alpha'),('outside_alpha_mean','整人外侧平均Alpha ↓'),
        ('inside_alpha_mean','人物内部平均Alpha'),('foreground_l1','人物L1 ↓'),('face_foreground_l1','脸部L1 ↓'),('mask_iou','轮廓IoU ↑')]:
        a=c['means']['before'][key];b=c['means']['mask_penalty'][key]
        metrics.append(f'<tr><td>{label}</td><td>{a:.5f}</td><td>{b:.5f}</td></tr>')
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>只加强mask惩罚 · 同MSV2模型对照</title>',
        '<style>body{max-width:1280px;margin:32px auto;padding:0 20px;background:#f4f4f2;color:#222;font:16px/1.7 system-ui}section{padding:24px;background:white;border-radius:12px;margin:20px 0}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#176684}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:16px}</style>',
        '<h1>只加强 mask 惩罚 · 同 MSV2 模型对照</h1><p>同59帧SwinIR4倍输入 · 同相机和遮罩 · 各自从头训练18,000步</p>',
        '<section><h2>结论</h2>'+''.join('<p>'+escape(p)+'</p>' for p in findings.get('summary_paragraphs',['训练与导出完成，正在审阅全部视角。']))+'</section>',
        '<section><h2>本次修改</h2><p>保留MSV2模型、优化器、增密与简化流程、原始遮罩、RGB监督及原有mask BCE。在原遮罩外侧8个原始像素宽的区域内，额外计算单独归一化的负样本BCE，权重0.1；不改变mask标签。</p><p>没有裁剪或删除训练后高斯，没有改变已有模型。展示图片直接来自新训练结果，未套遮罩抠图。</p><p><a href="../face-noise-analysis/mask-penalty-diagnostic.json">mask连接与梯度证据</a> · <a href="mask_penalty/configuration-parity.json">模型和配置一致性核验</a> · <a href="mask_penalty/plan.json">训练约定</a></p></section>',
        '<section><h2>同相机59帧指标</h2><table><tr><th>指标</th><th>原有mask惩罚</th><th>加强轮廓mask惩罚</th></tr>'+''.join(metrics)+'</table><p>共同原图分辨率476×960。训练视角指标，非留出评测；12个环绕视角没有实拍真值。面部外侧区域使用原始人像mask和同一脸部框定义。</p><a href="comparison.json">完整逐帧统计</a></section>',
        f'<section><h2>点云与相机</h2><p><a href="mask_penalty/{delivery["bundle"]}">通用PLY＋59台相机</a> · <a href="mask_penalty/{delivery.get("default_ply",delivery.get("upright_ply"))}">独立PLY</a></p><p>本包保持训练世界坐标，PLY与训练输出逐字节一致。+Y坐标转换在第59帧的最大渲染差异为0.005604，超过原有0.005门槛，因此未将该转换文件放入交付包。使用包内相机查看；单独PLY的默认朝向可能需要调整。<a href="mask_penalty/upright-parity.json">检查原始记录</a></p><p><a href="../查看对照.html">原始 / 插值2倍 / 超分4倍对照</a></p></section>']
    for i in [1,12,24,28,35,47,59]:
        n=f'frame_{i:06d}';parts.append(f'<section><h2>{n}</h2><a href="views/{n}/face_compare.png"><img src="views/{n}/face_compare.png"></a><a href="views/{n}/body_compare.png"><img src="views/{n}/body_compare.png"></a><p><a href="views/{n}/white_compare.png">白底</a> · <a href="views/{n}/checker_compare.png">棋盘底</a> · <a href="views/{n}/mask_penalty_transparent.png">真实透明PNG</a></p></section>')
    parts.append('<section><h2>全部59帧</h2><div class="grid">')
    for i in range(1,60):
        n=f'frame_{i:06d}';parts.append(f'<div><a href="views/{n}/face_compare.png">{n} 脸部</a> · <a href="views/{n}/body_compare.png">整人</a><img loading="lazy" src="views/{n}/face_compare.png"></div>')
    parts.append('</div></section><section><h2>12个环绕视角</h2><div class="grid">')
    for a in range(0,360,30):parts.append(f'<a href="orbit/orbit_{a:03d}_compare.png"><img loading="lazy" src="orbit/orbit_{a:03d}_compare.png"></a>')
    parts.append('</div></section></html>');html=''.join(parts);(ROOT/'查看mask对照.html').write_text(html,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
    p=Links();p.feed(html)
    for l in p.links:
        u=urlsplit(l)
        if not u.scheme and u.path:assert (ROOT/unquote(u.path)).is_file(),l
    count=0
    for path in ROOT.rglob('*.png'):
        with Image.open(path) as im:im.verify()
        count+=1
    ex.save(ROOT/'archive-verification.json',dict(passed=True,PNGs=count,links=len(p.links)))
    print('MASK_REPORT_PUBLISHED',ROOT/'查看mask对照.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','evaluate_world','stage','publish']);p.add_argument('--step',type=int,default=1000)
    args=p.parse_args()
    if args.action=='stage':stage(args.step)
    else:globals()[args.action]()
