"""Same native-pixel head/mask/current-mesh/Gaussian review and publication."""
import argparse,math,re,shutil
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

ROOT=a.REC/'demo-vhap-initialization'
GEO=ROOT/'flame-face-prestage/hat-mask-control/geometry-only-control'
OUT=ROOT/'head-gaussian-first-cycle'
URL='diagnostics/video10-global-initializer/index.html#demo-head-gaussian'

def prepare(step):
    snap=OUT/'gaussian-run'/f'step-{step:06d}';result=a.read(snap/'result.json')
    folder=snap/'review';a.lp(folder/'frames').mkdir(parents=True,exist_ok=True)
    rows=a.read(OUT/'inputs.json')['records'];sheets=[];names=[];stats=[]
    for i,r in enumerate(rows):
        name=r['frame'];d=OUT/'inputs'/name
        raw=Image.open(a.lp(d/'image.png')).convert('RGB');mask=Image.open(a.lp(d/'mask.png')).convert('RGB')
        side=raw.height;assert raw.size==(side,side)
        mesh=Image.open(a.lp(GEO/'review/frames'/f'{name}_原图_mask指导前mesh_mask指导后mesh.png')).convert('RGB').crop((2*side,0,3*side,side))
        gs=Image.open(a.lp(snap/(name+'_gaussian.png'))).convert('RGB')
        target=np.array(mask)[:,:,0]>127;alpha=np.array(Image.open(a.lp(snap/(name+'_alpha.png'))))[:,:,0]>127
        roi=np.array(Image.open(a.lp(d/'roi.png')))>127
        stats.append(dict(frame=name,alpha_iou=float((target&alpha&roi).sum()/max(((target|alpha)&roi).sum(),1)),
            missing=int((target&~alpha&roi).sum()),outside=int((~target&alpha&roi).sum()),
            RGB_L1=float(np.abs(np.array(raw).astype(float)/255-np.array(gs).astype(float)/255)[target&roi].mean()),photo_valid=r['photo_valid']))
        panels=[raw,mask,mesh,gs];im=Image.new('RGB',(4*side,side),'white')
        for j,p in enumerate(panels):assert p.size==raw.size;im.paste(p,(j*side,0))
        filename=f'{name}_原图_首个mask_当前FLAME投影_高斯{step:06d}.png'
        im.save(a.lp(folder/'frames'/filename));names.append(filename)
    for start in range(0,59,12):
        images=[Image.open(a.lp(folder/'frames'/n)) for n in names[start:start+12]];w,h=images[0].size
        page=Image.new('RGB',(2*w,math.ceil(len(images)/2)*h),'white')
        for j,im in enumerate(images):page.paste(im,((j%2)*w,(j//2)*h))
        name=f'head_{start+1:02d}-{start+len(images):02d}.png';page.save(a.lp(folder/name));sheets.append(name)
    m=dict(updated=now(),steps=step,points=result['points'],all59=True,native_scale=1,rows=stats,sheets=sheets,
        median_alpha_iou=float(np.median([x['alpha_iou'] for x in stats])),missing_total=sum(x['missing'] for x in stats),outside_total=sum(x['outside'] for x in stats),
        median_foreground_RGB_L1=float(np.median([x['RGB_L1'] for x in stats if x['photo_valid']])),training_views_only=True,quality_accepted=False)
    a.save(folder/'metrics.json',m);print({k:v for k,v in m.items() if k not in ['rows','sheets']})

def publish(step):
    snap=OUT/'gaussian-run'/f'step-{step:06d}';m=a.read(snap/'review/metrics.json');review=a.read(snap/'visual-review.json')
    assert review['actual_all59_review']
    summary=review['summary_zh'];section='<section id="demo-head-gaussian"><h2>demo：面部＋帽子高斯训练</h2><p><strong>'+summary+'</strong></p>'
    section+=f'<p>当前轮次 {step} 步，{m["points"]:,} 个高斯。使用纯形状拟合后的 FLAME 网格，一三角一高斯初始化，保留逐帧三角形绑定；59帧原生像素裁片和首个SAM2 mask。没有加载VHAP纹理，高斯自己学习前景颜色。</p>'
    section+='<p>本轮仅为面部和帽子，不是全身SMPL-X结果。相机、mesh姿态与mesh偏移保持不变。设备遮挡的7帧只监督帽子轮廓；脸下方不作为本轮头部监督。已有mask中的手持物仍可能干扰结果。</p>'
    section+=f'<p>训练视角的轮廓重合度中位数 {m["median_alpha_iou"]:.1%}，前景RGB误差 {m["median_foreground_RGB_L1"]:.4f}。这些是训练数据指标，不是未见视角或独立亚像素精度证明。</p>'
    section+='<p>每组从左至右：原图／第一个mask／当前FLAME投影／高斯渲染。所有图保持原生像素，不在图上写字；完整59帧同步到 D:/00/记录/demo。</p>'
    selected=[0,12,23,28,32,35,46,58];names=[]
    rows=a.read(OUT/'inputs.json')['records']
    for i in selected:
        name=f'{rows[i]["frame"]}_原图_首个mask_当前FLAME投影_高斯{step:06d}.png';names.append(name)
        section+=f'<figure><figcaption>{rows[i]["frame"]}</figcaption><img src="demo-head-gaussian/{name}" alt="原图、mask、当前网格与高斯对比"></figure>'
    section+='<p>纯形状网格阶段已停用RGB损失：1003次更新21.45秒，头帽轮廓IoU中位数75.3%→84.1%，帽顶改善，帽檐尚未完整。用户随后明确授权直接进行高斯训练；这不等于初始化质量已经通过。</p></section>'
    rec=a.lp('D:/00/记录/demo');rec.mkdir(exist_ok=True)
    for p in a.lp(snap/'review/frames').glob('*.png'):shutil.copy2(p,rec/('高斯_面部帽子_'+p.name))
    for p in a.lp(GEO/'review/frames').glob('*.png'):shutil.copy2(p,rec/('VHAP_纯形状_'+p.name))
    hashes=[]
    for dest in [a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer']:
        a.lp(dest/'demo-head-gaussian').mkdir(exist_ok=True)
        for name in names:shutil.copy2(a.lp(snap/'review/frames'/name),a.lp(dest/'demo-head-gaussian'/name))
        page=dest/'index.html';html=a.lp(page).read_text(encoding='utf-8')
        tag='<section id="demo-head-gaussian">'
        if tag in html:
            start=html.index(tag);end=html.index('</section>',start)+len('</section>');html=html[:start]+section+html[end:]
        else:
            start=html.index('<section id="vhap-demo-face-surface">');html=html[:start]+section+html[start:]
        html=html.replace('<h2>demo：脸部表面与帽子轮廓</h2>','<h2>历史对照：脸部外观与帽子轮廓</h2>')
        html=re.sub(r'<p class="status">.*?</p>','<p class="status">2026-10-30 · '+summary+'</p>',html,count=1)
        a.lp(page).write_text(html,encoding='utf-8');hashes.append(a.sha(page))
    assert len(set(hashes))==1
    for dest in [a.V,a.S]:
        path=dest/'latest-result.json';value=a.read(path);value['latestDiagnostic']=dict(updated=now(),round='123-demo-head-gaussian',title='面部帽子高斯',summary=summary,status='actual_reviewed',url=URL,report=URL,reportUrl=URL,qualityAccepted=False);a.save(path,value)
    a.save(OUT/'publication.json',dict(updated=now(),steps=step,report_url=URL,report_sha256=hashes[0],record_images=59,actual_UI_review=False,quality_accepted=False))
    print('HEAD_GAUSSIAN_REPORT_PUBLISHED',URL)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','publish']);p.add_argument('--step',type=int,required=True);args=p.parse_args()
    prepare(args.step) if args.mode=='prepare' else publish(args.step)
