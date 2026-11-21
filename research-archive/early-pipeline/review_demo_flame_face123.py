"""Prepare unchanged-pixel face review sheets; publish only reviewed evidence."""
import argparse, math, re, shutil
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

ROOT=a.REC/'demo-vhap-initialization'
OUT=ROOT/'flame-face-prestage'
URL='diagnostics/video10-global-initializer/index.html#vhap-demo-face3d'

def prepare():
    result=a.read(OUT/'result.json'); final=a.read(OUT/'epoch-30/result.json')
    rows=final['rows']; assert len(rows)==59 and result['steps']==(1770 if OUT.name=='pose-seeded-control' else 5720)
    folder=OUT/'review';a.lp(folder).mkdir(exist_ok=True)
    files=[]
    for suffix in ['原图_FLAME投影','原图_二维点_拟合三维点投影']:
        for start in range(0,59,12):
            names=[f'{r["frame"]}_{suffix}.png' for r in rows[start:start+12]]
            images=[Image.open(a.lp(OUT/'epoch-30'/n)).convert('RGB') for n in names]
            w,h=images[0].size; assert all(im.size==(w,h) for im in images)
            sheet=Image.new('RGB',(2*w,math.ceil(len(images)/2)*h),'white')
            for j,im in enumerate(images):sheet.paste(im,((j%2)*w,(j//2)*h))
            name=f'{suffix}_{start+1:02d}-{start+len(images):02d}.jpg'
            sheet.save(a.lp(folder/name),quality=97);files.append(dict(path=name,frames=names))
    metrics=dict(updated=now(),frames=59,steps=result['steps'],elapsed_seconds=result['elapsed_seconds'],
        fit_median_px=final['fit_median_of_frame_medians_px'],held_median_px=final['held_median_of_frame_medians_px'],
        held_worst_px=final['held_max_px'],fit_worst_px=max(r['fit_max_px'] for r in rows),
        held_frame_max_below_1px=sum(r['held_max_px']<1 for r in rows),
        worst_frames=sorted(rows,key=lambda r:r['held_max_px'],reverse=True)[:8],
        held_same_detector=True,independent_3D_ground_truth=False,quality_accepted=False,
        review_sheets=files,network_3D_template_used=False)
    a.save(folder/'metrics.json',metrics)
    print(metrics | {'review_sheets':len(files)})

def publish():
    metrics=a.read(OUT/'review/metrics.json');review=a.read(OUT/'visual-review.json');result=a.read(OUT/'result.json')
    record=a.lp('D:/00/记录/demo');record.mkdir(exist_ok=True)
    source=a.lp(OUT/'epoch-30')
    for path in source.glob('*.png'):
        shutil.copy2(path,record/('VHAP_原版FLAME面部前级_'+path.name))
    summary=review['summary_zh']; selected=['frame_000001','frame_000013','frame_000024','frame_000029','frame_000033','frame_000036','frame_000047','frame_000059']
    section='<section id="vhap-demo-face3d"><h2>demo：VHAP原版FLAME面部前级</h2><p><strong>'+summary+'</strong></p>'
    section+='<p>按用户最新要求：先拟合面部三维特征点，再交给全身VHAP。不使用3D-FAN、DECA等网络预测的三维脸；本轮沿用VHAP自带FLAME与原有关键点优化器，固定提交8980d494。59帧共享脸型，逐帧拟合姿态和表情，优选第33帧初始化。</p>'
    section+='<p>输入是原生视频的二维面部点和条件K。FLAME是参数化脸模型，拟合后的三维点仍是模型估计，并非独立测量的三维真值。没有重新估K、没有自由顶点偏移、没有启动高斯训练。</p>'
    section+=f'<p>本阶段{metrics["steps"]}次更新，含核验和导出共{metrics["elapsed_seconds"]:.1f}秒。二维拟合点的逐帧中位误差再取中位为{metrics["fit_median_px"]:.2f}px；预留语义点对应值{metrics["held_median_px"]:.2f}px，最差预留点{metrics["held_worst_px"]:.2f}px。预留点仍来自同一检测器，不能代替独立验收。</p>'
    section+='<ul>'+''.join('<li>'+x+'</li>' for x in review['findings_zh'])+'</ul>'
    section+='<p>每组图片由左到右：原图／二维检测点／拟合三维点的投影；下一张为原图／FLAME网格覆盖。相同真实帧、原像素裁剪、图内不写字。完整59帧已写入 D:/00/记录/demo。</p>'
    names=[]
    for frame in selected:
        for suffix in ['原图_二维点_拟合三维点投影','原图_FLAME投影']:
            name=f'{frame}_{suffix}.png';names.append(name)
            section+=f'<figure><figcaption>{frame} · {suffix}</figcaption><img src="demo-vhap-face3d/{name}" alt="原始图像与原版FLAME面部拟合核验"></figure>'
    section+='<p>后续只按IBUG内侧51点语义建立FLAME到SMPL-X接口；两个网格的顶点编号不能直接互用，模型尺度也不能假定是实测尺度。完整全身拟合、表面高斯与15分钟端到端目标尚未完成。</p></section>'
    hashes=[]
    for dest in [a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer']:
        imagefolder=dest/'demo-vhap-face3d';a.lp(imagefolder).mkdir(exist_ok=True)
        for name in names:shutil.copy2(a.lp(OUT/'epoch-30'/name),a.lp(imagefolder/name))
        page=dest/'index.html';html=a.lp(page).read_text(encoding='utf-8')
        if '<section id="vhap-demo-face3d">' in html:
            start=html.index('<section id="vhap-demo-face3d">');end=html.index('</section>',start)+len('</section>');html=html[:start]+section+html[end:]
        else:
            start=html.index('<section id="vhap-demo-input">');html=html[:start]+section+html[start:]
        html=re.sub(r'<p class="status">.*?</p>','<p class="status">2026-10-30 · '+summary+'</p>',html,count=1)
        a.lp(page).write_text(html,encoding='utf-8');hashes.append(a.sha(page))
    assert len(set(hashes))==1
    entry=dict(updated=now(),state=review['state'],summary=summary,steps=metrics['steps'],frames=59,
        elapsed_seconds=result['elapsed_seconds'],face_prestage_complete=True,actual_visual_review=True,
        network_3D_template_used=False,downstream_VHAP_started=False,training_running=False,fitter_running=False,
        quality_accepted=False,goal_complete=False,next=review['next'],source=str(OUT),report_url=URL)
    for dest in [a.V,a.S]:
        path=dest/'latest-result.json';data=a.read(path);data['latestDiagnostic']=dict(updated=entry['updated'],round='123-demo-vhap-face3d',title='VHAP原版面部前级',summary=summary,status=entry['state'],url=URL,report=URL,reportUrl=URL,qualityAccepted=False);a.save(path,data)
    for path in [a.R/'queue-state.json',a.REC/'queue-state.json',a.C/'local-research-status.json']:
        data=a.read(path);data.update(updated_at=entry['updated'],state=entry['state'],training_running=False,fitter_running=False,local_GPU_task_active=False,GPU_task_running=False,goal_complete=False,next=entry['next']);data['latest_demo_vhap_face3d']=entry;a.save(path,data)
    ledger=a.read(a.R/'evidence-ledger.json');ledger['demo_vhap_flame_face_prestage']=entry;a.save(a.R/'evidence-ledger.json',ledger)
    a.save(ROOT/'progress.json',entry);a.save(OUT/'publication.json',dict(**entry,report_sha256=hashes[0],actual_report_UI_verified=False))
    print(entry)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','publish']);parser.add_argument('--pose-control',action='store_true');args=parser.parse_args()
    if args.pose_control:OUT=OUT/'pose-seeded-control'
    prepare() if args.mode=='prepare' else publish()
