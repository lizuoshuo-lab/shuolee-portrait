"""Publish the retained 3000-step candidate and its unchanged opacity gate evidence."""
from pathlib import Path
import importlib.util,json,shutil,copy,argparse
from types import SimpleNamespace
from PIL import Image,ImageDraw,ImageFont
import numpy as np

RUN=Path('/opt/gs-slam-lab/msv2-demo-transparent-depthcheck-20261104')
BASE=Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')
OUT=Path('/mnt/d/00/记录/2026-11-04_透明背景_MSV2/demo')
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ex=load('diagnostic_experiment',RUN/'experiment.py')
h=load('portable_export',Path(__file__).with_name('review_msv2_upsampled_demo.py'))
h.ex=ex;h.RUN=RUN;h.OUT=OUT
h.PLY_STEM='demo_swinir4x_msv2_transparent_003000_quality_failed'
h.BUNDLE='demo_SwinIR4倍_透明试验_3000步_未通过_点云与相机.zip'

def evaluate():
    assert ex.read(RUN/'execution.json')['state']=='failed'
    assert not ex.read(RUN/'gate3000/gate.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False)
    torch=ex.runtime();_,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    from scene import GaussianModel
    from gaussian_renderer import render
    rows=ex.read(RUN/'native-inputs.json')['records'];training=ex.read(RUN/'training-inputs.json')['records']
    models={}
    for label,root,step,degree in [('baseline',BASE,3000,0),('transparent',RUN,3000,0),('sent_18000',BASE,18000,3)]:
        model=GaussianModel(degree);model.load_ply(str(root/f'model/point_cloud/iteration_{step}/point_cloud.ply'));model.active_sh_degree=degree;models[label]=model
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);checks=[];parity=[];composites=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            cam=ex.Camera(r,i);folder=OUT/r['frame'];folder.mkdir()
            mask=torch.as_tensor(cam.mask.copy(),device='cuda',dtype=torch.float32)/255
            gt=torch.as_tensor(cam.rgb.copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
            shutil.copy2(r['image'],folder/'original.png');shutil.copy2(r['mask'],folder/'input_mask.png')
            entry=dict(frame=r['frame']);fb=h.face_box(r)
            for label,g in models.items():
                rgb=render(cam,g,pipe,torch.zeros(3,device='cuda'))['render']
                alpha=render(cam,g,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(g.get_xyz))['render'][0:1]
                outside=mask<.01;inside=mask>.99
                entry[label]=dict(outside_alpha_mean=float(alpha[0][outside].mean()),outside_alpha_gt_01=float((alpha[0][outside]>.1).float().mean()),
                    inside_alpha_mean=float(alpha[0][inside].mean()),foreground_rgb_l1=float((rgb-gt).abs()[:,inside].mean()))
                gray=rgb+.5*(1-alpha);ex.save_image(folder/(label+'_gray.png'),gray)
                if label=='transparent':
                    ex.save_image(folder/'black.png',rgb);ex.save_image(folder/'white.png',rgb+1-alpha)
                    ex.save_image(folder/'alpha.png',alpha.expand(3,-1,-1))
                    yy,xx=torch.meshgrid(torch.arange(cam.height,device='cuda'),torch.arange(cam.width,device='cuda'),indexing='ij')
                    checker=torch.where((xx//24+yy//24)%2==0,.85,.45)[None]
                    ex.save_image(folder/'checker.png',rgb+checker*(1-alpha))
                    rgba=torch.cat([(rgb/alpha.clamp_min(1e-4)).clamp(0,1),alpha.clamp(0,1)],dim=0)
                    Image.fromarray(np.rint(rgba.permute(1,2,0).cpu().numpy()*255).astype(np.uint8),'RGBA').save(folder/'transparent.png')
                    if i in sorted(set(ex.SELECTED+[27])):
                        expected=np.array(Image.open(RUN/'gate3000'/(r['frame']+'_transparent_gray.png')).convert('RGB'))
                        actual=np.array(Image.open(folder/'transparent_gray.png').convert('RGB'))
                        parity.append(dict(frame=r['frame'],max_abs_uint8=int(np.abs(expected.astype(int)-actual.astype(int)).max())))
                        white=render(cam,g,pipe,torch.ones(3,device='cuda'))['render']
                        composites.append(dict(frame=r['frame'],max_abs=float((white-(rgb+1-alpha)).abs().max())))
            names=['original','baseline_gray','transparent_gray']
            h.board([folder/(n+'.png') for n in names],['原始图','旧黑底训练 3000步','透明监督 3000步'],folder/'same_step.png',size=(300,600),title=r['frame'])
            h.board([folder/(n+'.png') for n in ['original','sent_18000_gray','transparent_gray']],['原始图','已发送四倍版 18000步','透明候选 3000步'],folder/'sent_comparison.png',size=(300,600),title=r['frame'])
            h.board([folder/(n+'.png') for n in ['black','white','checker','alpha']],['黑底','白底','棋盘底','渲染Alpha'],folder/'backgrounds.png',size=(240,480),title=r['frame'])
            for n in names:
                with Image.open(folder/(n+'.png')) as im:im.crop(tuple(fb)).save(folder/(n+'_face.png'))
            h.board([folder/(n+'_face.png') for n in names],['原图','旧黑底 3000步','透明监督 3000步'],folder/'face.png',size=(240,300),title=r['frame'])
            checks.append(entry)
            if i%10==0 or i==58:print('DIAGNOSTIC_VIEW',i+1,'/59',flush=True)
        orbit=ex.read('/mnt/d/00/记录/人体上采样静止重建/demo_3x/cameras.json')['novel_review_cameras'];(OUT/'orbit').mkdir()
        for j,r in enumerate(orbit):
            r=copy.deepcopy(r);dummy=OUT/'orbit'/'_canvas.png';Image.new('RGB',tuple(r['wh'])).save(dummy);r.update(image=str(dummy),mask=str(dummy));cam=ex.Camera(r,j)
            rgb=render(cam,models['transparent'],pipe,torch.zeros(3,device='cuda'))['render'];a=render(cam,models['transparent'],pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(models['transparent'].get_xyz))['render'][0:1]
            ex.save_image(OUT/'orbit'/(r['frame']+'_gray.png'),rgb+.5*(1-a));dummy.unlink()
    assert all(x['max_abs_uint8']==0 for x in parity),parity
    assert all(x['max_abs']<.005 for x in composites),composites
    means={k:{m:float(np.mean([r[k][m] for r in checks])) for m in checks[0][k]} for k in models}
    ex.save(OUT/'comparison.json',dict(created=ex.now(),frames=59,means=means,checks=checks,quality_accepted=False,
        same_step_comparison=['baseline','transparent'],sent_18000_is_different_training_budget=True,
        actual_iteration_label=3000,actual_optimizer_updates=2999,planned_18000_not_completed=True,
        points=len(models['transparent'].get_xyz),source_views_are_training_views=True,novel_views_have_no_ground_truth=True))
    ex.save(OUT/'native-ply-parity.json',dict(passed=True,checks=parity,reference='Exact uint8 native gray render saved from the live model immediately after PLY export; comparison is quantized, not a float-image assertion.'))
    ex.save(OUT/'rgba-compositor-check.json',dict(passed=True,checks=composites,transparent_pngs=59,mode='Straight RGB and alpha; values above 1 clipped for 8-bit display; no GT-mask cropping of rendered Gaussians'))
    for f in ['plan.json','execution.json','preflight.json','rgba-probe.json','frozen-baseline-diagnostic.json']:shutil.copy2(RUN/f,OUT/f)
    for step in [1000,3000]:shutil.copy2(RUN/f'gate{step}/gate.json',OUT/f'gate{step}.json')
    old_trials=[]
    for root in ['msv2-demo-transparent-20261104','msv2-demo-transparent-bce-20261104']:
        p=RUN.parent/root;old_trials.append(dict(run=root,execution=ex.read(p/'execution.json'),gate=ex.read(p/'gate1000/gate.json')))
    ex.save(OUT/'prior-trials.json',old_trials)
    shutil.copy2('/opt/gs-slam-lab/msv2-pollution-diagnostic-20261104/stages.json',OUT/'pollution-stages.json')
    shutil.copy2('/opt/gs-slam-lab/msv2-demo-transparent-20261104/input-mask-review.png',OUT/'input-mask-review.png')
    final=RUN/'model/point_cloud/iteration_3000/point_cloud.ply'
    shutil.copy2(final,OUT/(h.PLY_STEM+'_world.ply'))
    ex.save(OUT/'cameras_world.json',h.camera_records(training));ex.save(OUT/'camera-input-mapping.json',dict(records=training,convention='pixel_corners',camera_estimation='original background cameras, no person-only recovery'))
    h.export_upright(training,models['transparent'],pipe,torch.zeros(3,device='cuda'))
    print('DIAGNOSTIC_MEANS',json.dumps(means),flush=True)

def publish():
    c=ex.read(OUT/'comparison.json');font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    for i in range(1,60):
        folder=OUT/f'frame_{i:06d}';temporary=[]
        h.board([folder/(n+'.png') for n in ['original','sent_18000_gray','transparent_gray']],['原始图','已发送四倍版 18000步','透明候选 3000步'],folder/'sent_comparison.png',size=(300,600),title=folder.name)
        try:
            for name in ['original','baseline_gray','transparent_gray']:
                p=folder/(name+'_face_display.png');temporary.append(p)
                with Image.open(folder/(name+'_face.png')) as im:im.resize((im.width*4,im.height*4),Image.Resampling.LANCZOS).save(p)
            h.board(temporary,['原图 显示放大4倍','旧黑底 3000步','透明监督 3000步'],folder/'face.png',size=(240,300),title=folder.name)
        finally:
            for p in temporary:
                if p.exists():p.unlink()
    for kind in ['same_step','face','backgrounds']:
        for begin in range(1,60,12):
            ids=list(range(begin,min(begin+12,60)));canvas=Image.new('RGB',(1230,((len(ids)+2)//3)*270),'#eee');d=ImageDraw.Draw(canvas)
            for j,i in enumerate(ids):
                n=f'frame_{i:06d}'
                with Image.open(OUT/n/(kind+'.png')) as im:im=im.convert('RGB');im.thumbnail((400,235))
                x=j%3*410;y=j//3*270;d.text((x,y),n,font=font,fill='black');canvas.paste(im,(x,y+28))
            canvas.save(OUT/f'all_{kind}_{(begin-1)//12+1:02d}.png')
    canvas=Image.new('RGB',(1536,828),'#eee');d=ImageDraw.Draw(canvas)
    for j,a in enumerate(range(0,360,30)):
        with Image.open(OUT/'orbit'/f'orbit_{a:03d}_gray.png') as im:im=im.convert('RGB');im.thumbnail((384,760))
        x=j%6*256;y=j//6*414;im.thumbnail((256,384));d.text((x,y),str(a)+'°',font=font,fill='black');canvas.paste(im,(x,y+30))
    canvas.save(OUT/'all_orbit.png')
    b=c['means']['baseline'];n=c['means']['transparent'];reduction=1-n['outside_alpha_mean']/b['outside_alpha_mean']
    note=ex.read(OUT/'visual-review.json') if (OUT/'visual-review.json').exists() else {}
    summary=note.get('summary','全视角图像已经生成，正在逐页检查；效果尚未通过。')
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>demo 四倍透明候选 · 黑色污染高斯明显减少</title>',
        '<style>body{font:17px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;background:#f5f5f2;color:#222}section{background:white;margin:20px 0;padding:22px;border-radius:12px}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}a{color:#176684}.checker{background:repeating-conic-gradient(#ddd 0% 25%,white 0% 50%) 50%/24px 24px}</style>',
        '<h1>demo · 四倍透明训练对照</h1><p><strong>黑色污染高斯明显减少，保留这版3,000步候选。</strong>手部漂浮碎片与未观测区域的缺失另行记录；当前训练已停止。</p>',
        '<section><h2>已确认的问题</h2><p>已发给小润的四倍版使用黑底RGB监督，空白处仍可形成不透明黑色高斯。阶段对照确认污染在后期扩大。改为前景颜色、显式Alpha和前景深度采样后，背景RGB对颜色损失和梯度的影响为零，但这不代表所有多余几何都已消失。</p>',
        f'<p>59个训练视角，同为3,000步：人物外平均不透明度 {b["outside_alpha_mean"]:.4%} → {n["outside_alpha_mean"]:.4%}，下降 {reduction:.1%}；人物内平均不透明度 {b["inside_alpha_mean"]:.4%} → {n["inside_alpha_mean"]:.4%}。'+summary+'</p>',
        '<p>1%是本次试验预设的全部轮廓外不透明度门槛，包含浅色半透明残影，并非黑色污染高斯的数量指标。该门槛尚未达到；黑色污染高斯的明显改善单独记录。早期两次1,000步检查及本次3,000步停止的原始依据保留，本次没有继续到18,000步。<a href="prior-trials.json">此前记录</a> · <a href="gate3000.json">3,000步停止依据</a> · <a href="visual-review.json">效果审阅与用户反馈</a></p></section>',
        '<section><h2>已发送的旧版与当前诊断</h2><p>这里训练步数不同，只用来定位黑色伪影；同一步数比较在下方。</p><img src="frame_000001/sent_comparison.png"></section>',
        '<section><h2>真正透明的渲染</h2><p>PNG使用模型渲染出的Alpha，不使用输入遮罩把多余点裁掉。点击可下载。</p><a href="frame_000001/transparent.png"><img class="checker" src="frame_000001/transparent.png" style="max-height:650px"></a><img src="frame_000001/backgrounds.png"></section>',
        '<section><h2>输入遮罩的局限</h2><p>原图中可见的手持设备在不同帧遮罩里有时保留、有时排除，不能把这种矛盾监督当作准确三维轮廓；仅更换背景色不会解决它。下图为原始输入与现有遮罩对照，原始文件保持不变。</p><img src="input-mask-review.png"></section>',
        f'<section><h2>候选点云与59台相机</h2><p>{c["points"]:,}个高斯，SH0；3,000步时钟、2,999次更新。<a href="{h.BUNDLE}">通用PLY＋相机ZIP（保留候选）</a> · <a href="{h.PLY_STEM}_yup.ply">+Y向上PLY</a> · <a href="cameras_yup.json">配套相机</a>。文件名保留原始门槛状态；K首两行与图像同步放大4倍，外参不变。</p><p><a href="upright-parity.json">坐标／协方差／颜色回读</a> · <a href="rgba-compositor-check.json">透明合成核验</a> · <a href="rgba-probe.json">背景颜色与梯度隔离</a> · <a href="comparison.json">59视角指标</a> · <a href="plan.json">配置</a> · <a href="viewer-import-verification.json">高斯／点云实际导入核验</a></p></section>',
        '<section><h2>全部59个相同视角</h2>']
    for i in range(1,60):
        f=f'frame_{i:06d}';html.append(f'<p>{f} · <a href="{f}/original.png">原图</a> · <a href="{f}/transparent.png">透明PNG</a> · <a href="{f}/backgrounds.png">黑白棋盘底</a> · <a href="{f}/face.png">面部</a></p><a href="{f}/same_step.png"><img loading="lazy" src="{f}/same_step.png"></a>')
    html.append('</section><section><h2>12个环绕视角</h2><p>没有实拍真值；背面缺失没有修复。</p><img src="all_orbit.png"></section></html>')
    page=OUT/'查看对照.html';page.write_text(''.join(html),encoding='utf-8')
    from html.parser import HTMLParser
    from urllib.parse import urlsplit,unquote
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('src','href'))
    parser=Links();parser.feed(''.join(html))
    for link in parser.links:
        u=urlsplit(link)
        if not u.scheme and u.path:assert (OUT/unquote(u.path)).is_file(),link
    count=0
    for p in OUT.rglob('*.png'):
        with Image.open(p) as im:im.verify()
        count+=1
    ex.save(OUT/'archive-verification.json',dict(passed=True,images=count,html_links=len(parser.links),quality_accepted=False))
    print('TRANSPARENT_DIAGNOSTIC_PUBLISHED',page,count,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','publish']);args=p.parse_args();globals()[args.action]()
