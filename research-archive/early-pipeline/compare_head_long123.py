"""Same native pixels/meshes, short-schedule vs long-schedule review; CPU only."""
import argparse,math,re,shutil
from PIL import Image
import audit_gaussianavatars_params123 as a
import review_demo_head_long123 as review
from run_official_gaussians123 import now

root=a.REC/'demo-vhap-initialization';short=root/'head-gaussian-first-cycle';long=root/'head-gaussian-long300k'

def prepare(step):
    assert step==300000 or step%30000!=0,'Intermediate opacity-reset snapshots are not stable appearance comparisons; final300k skips reset'
    assert a.sha(short/'fitted-head-meshes.npz')==a.sha(long/'fitted-head-meshes.npz')
    assert a.sha(short/'inputs.json')==a.sha(long/'inputs.json')
    result=a.read(long/'gaussian-run'/f'step-{step:06d}'/'result.json')
    assert result['step']==step and result['native_ply_parity']['passed']
    metrics=long/'gaussian-run'/f'step-{step:06d}'/'review/metrics.json'
    if not a.lp(metrics).exists():review.prepare(step)
    dest=long/'comparison'/f'step-{step:06d}';a.lp(dest/'frames').mkdir(parents=True,exist_ok=True)
    rows=a.read(short/'inputs.json')['records'];names=[]
    for row in rows:
        frame=row['frame'];old=short/'gaussian-run/step-018000/review/frames'/f'{frame}_原图_首个mask_当前FLAME投影_高斯018000.png'
        base=Image.open(a.lp(old)).convert('RGB');side=base.height;assert base.size==(4*side,side)
        fresh=Image.open(a.lp(long/'gaussian-run'/f'step-{step:06d}'/(frame+'_gaussian.png'))).convert('RGB')
        assert fresh.size==(side,side)
        panels=[base.crop((0,0,side,side)),base.crop((2*side,0,3*side,side)),base.crop((3*side,0,4*side,side)),fresh]
        merged=Image.new('RGB',(4*side,side))
        for i,p in enumerate(panels):merged.paste(p,(i*side,0))
        name=f'{frame}_原图_当前FLAME投影_短训练018000_长训练{step:06d}.png';merged.save(a.lp(dest/'frames'/name));names.append(name)
    sheets=[]
    for start in range(0,len(names),12):
        pictures=[Image.open(a.lp(dest/'frames'/n)) for n in names[start:start+12]];w,h=pictures[0].size
        sheet=Image.new('RGB',(2*w,math.ceil(len(pictures)/2)*h),'white')
        for j,im in enumerate(pictures):sheet.paste(im,((j%2)*w,(j//2)*h))
        name=f'compare_{start+1:02d}-{start+len(pictures):02d}.png';sheet.save(a.lp(dest/name));sheets.append(name)
    old=a.read(short/'gaussian-run/step-018000/review/metrics.json');new=a.read(metrics)
    a.save(dest/'comparison.json',dict(updated=now(),step=step,points=result['points'],frames=59,images=names,sheets=sheets,
        mesh_and_observations_bit_identical=True,source_world_ply_sha256=result['world_ply_sha256'],
        baseline={k:old[k] for k in ['median_alpha_iou','missing_total','outside_total','median_foreground_RGB_L1']},
        current={k:new[k] for k in ['median_alpha_iou','missing_total','outside_total','median_foreground_RGB_L1']},
        comparison_scope='Short18k vs proportionally rescaled300k schedule, not one-variable iteration-count attribution.',
        actual_visual_review=False,quality_accepted=False,novel_views_pending=True))
    print(dest)

def publish(step):
    dest=long/'comparison'/f'step-{step:06d}';data=a.read(dest/'comparison.json');decision=a.read(dest/'visual-review.json')
    assert decision['actual_all59_review'] and decision['source_world_ply_sha256']==data['source_world_ply_sha256']
    section=f'<section id="demo-head-long"><h2>30万步长训练：{step:,}步阶段对照</h2><p><strong>{decision["summary_zh"]}</strong></p>'
    section+='<p>每组从左到右：原图／当前FLAME网格投影／短时序18,000步／长时序本阶段。59帧原生像素、相机、mask与网格完全相同；高斯均从头训练。两轮增密、透明度重置和学习率时间轴按各自总步数缩放，不能把差异只归因于迭代次数。</p>'
    section+=f'<p>训练视角轮廓IoU中位数：{data["baseline"]["median_alpha_iou"]:.1%} → {data["current"]["median_alpha_iou"]:.1%}；前景RGB误差：{data["baseline"]["median_foreground_RGB_L1"]:.4f} → {data["current"]["median_foreground_RGB_L1"]:.4f}。阶段图不能替代未知视角、全身与15分钟端到端验收。</p>'
    section+='<p>右侧点云仍是已完成导出与实际UI检查的18k版本；本栏展示真实长训练阶段图，最终点云与配套相机需等本轮完成后原生回放、格式和UI核验。</p>'
    selected=[0,12,23,28,32,35,46,58]
    for i in selected:
        name=data['images'][i];section+=f'<figure><figcaption>{name.split("_原图")[0]}</figcaption><img src="demo-head-long/{name}" alt="原图、网格、短训练与长训练同帧对照"></figure>'
    section+='</section>'
    rec=a.lp('D:/00/记录/demo')
    for f in a.lp(dest/'frames').glob('*.png'):shutil.copy2(f,rec/('高斯_面部帽子_短长时序对照_'+f.name))
    hashes=[]
    for target in [a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer']:
        a.lp(target/'demo-head-long').mkdir(exist_ok=True)
        for i in selected:shutil.copy2(a.lp(dest/'frames'/data['images'][i]),a.lp(target/'demo-head-long'/data['images'][i]))
        page=target/'index.html';text=a.lp(page).read_text(encoding='utf-8');tag='<section id="demo-head-long">'
        if tag in text:
            start=text.index(tag);end=text.index('</section>',start)+len('</section>');text=text[:start]+section+text[end:]
        else:text=text.replace('<section id="demo-head-runtime-audit">',section+'<section id="demo-head-runtime-audit">',1)
        a.lp(page).write_text(text,encoding='utf-8');hashes.append(a.sha(page))
    assert len(set(hashes))==1
    url='diagnostics/video10-global-initializer/index.html#demo-head-long'
    for target in [a.V,a.S]:
        p=target/'latest-result.json';d=a.read(p);d['latestDiagnostic']=dict(updated=now(),round='123-demo-head-long',title=f'长训练{step:,}步',
            summary=decision['summary_zh'],status='actual59reviewed_training_continues',reportUrl=url,qualityAccepted=False);a.save(p,d)
    a.save(dest/'publication.json',dict(updated=now(),report_url=url,report_sha256=hashes[0],record_images=59,actual_UI_review=False,quality_accepted=False))
    print('LONG_COMPARISON_PUBLISHED',url)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','publish']);parser.add_argument('--step',type=int,required=True)
    args=parser.parse_args();prepare(args.step) if args.mode=='prepare' else publish(args.step)
