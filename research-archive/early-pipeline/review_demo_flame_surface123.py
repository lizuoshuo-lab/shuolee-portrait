"""Read-only, unchanged-pixel review of the completed original VHAP RGB stages."""
import argparse, math, shutil, re
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

ROOT=a.REC/'demo-vhap-initialization'
FACE=ROOT/'flame-face-prestage'
OUT=FACE/'photometric-face'
URL='diagnostics/video10-global-initializer/index.html#vhap-demo-face-surface'

def prepare():
    run=a.read(OUT/'result.json'); assert run['steps']==3270 and run['frames']==59
    baseline=a.read(OUT/'before/appearance-review-data.json')
    after=a.read(OUT/'epoch-30/appearance-review-data.json')
    before_points=a.read(FACE/'pose-seeded-control/epoch-30/result.json')
    after_points=a.read(OUT/'epoch-30/result.json')
    rows=after['rows']; assert [r['frame'] for r in rows]==[r['frame'] for r in baseline['rows']]
    folder=OUT/'review'; a.lp(folder).mkdir(exist_ok=True)
    images=folder/'frames'; a.lp(images).mkdir(exist_ok=True)
    sheets=[]
    for kind,suffix in [('纹理','原图_学习纹理模型独立渲染'),('网格','原图_FLAME投影')]:
        files=[]
        for row in rows:
            frame=row['frame']
            before_dir=OUT/'before' if kind=='纹理' else FACE/'pose-seeded-control/epoch-30'
            old=Image.open(a.lp(before_dir/f'{frame}_{suffix}.png')).convert('RGB')
            new=Image.open(a.lp(OUT/'epoch-30'/f'{frame}_{suffix}.png')).convert('RGB')
            assert old.size==new.size and new.width==2*new.height
            side=new.height
            assert np.array_equal(np.array(old)[:,:side],np.array(new)[:,:side])
            joined=Image.new('RGB',(3*side,side),'white')
            joined.paste(new.crop((0,0,side,side)),(0,0))
            joined.paste(old.crop((side,0,2*side,side)),(side,0))
            joined.paste(new.crop((side,0,2*side,side)),(2*side,0))
            name=f'{frame}_原图_表面学习前_表面学习后_{kind}.png'
            joined.save(a.lp(images/name)); files.append(name)
        for start in range(0,59,12):
            selected=files[start:start+12]
            ims=[Image.open(a.lp(images/n)) for n in selected]
            w,h=ims[0].size
            sheet=Image.new('RGB',(2*w,math.ceil(len(ims)/2)*h),'white')
            for j,im in enumerate(ims):sheet.paste(im,((j%2)*w,(j//2)*h))
            name=f'{kind}_{start+1:02d}-{start+len(ims):02d}.png'
            sheet.save(a.lp(folder/name));sheets.append(dict(file=name,frames=selected))
    metrics=dict(updated=now(),frames=59,updates=run['steps'],train_seconds=run['train_seconds'],
        elapsed_seconds=run['elapsed_seconds'],native_face_crop_pixels=109,
        raw_to_render_panel_scale=1,model_only_render=True,raw_image_background=False,
        face_hull_training_RGB_L1_before=baseline['median_face_hull_L1'],
        face_hull_training_RGB_L1_after=after['median_face_hull_L1'],
        shared_offset_norm_model_units=after['shared_offset_norm'],
        fit_median_before=before_points['fit_median_of_frame_medians_px'],
        fit_median_after=after_points['fit_median_of_frame_medians_px'],
        held_median_before=before_points['held_median_of_frame_medians_px'],
        held_median_after=after_points['held_median_of_frame_medians_px'],
        held_max_before=before_points['held_max_px'],held_max_after=after_points['held_max_px'],
        held_is_same_detector=True,independent_3D_ground_truth=False,
        image_improvement_frames=sum(r['face_hull_L1']<b['face_hull_L1'] for r,b in zip(rows,baseline['rows']) if not r['occluded']),
        photometric_observed_frames=sum(not r['occluded'] for r in rows),
        held_frames_all_below1=sum(r['held_max_px']<1 for r in after_points['rows']),
        sheets=sheets,quality_accepted=False)
    a.save(folder/'metrics.json',metrics)
    print({k:v for k,v in metrics.items() if k!='sheets'})

def publish():
    m=a.read(OUT/'review/metrics.json')
    review=dict(updated=now(),state='face_appearance_reviewed_hat_mask_in_progress',
        actual_review='All59 face RGB and mesh before/after panels in10 unchanged-pixel sheets.',
        face_appearance_learned=True,face_pose_retained=True,hat_geometry_accepted=False,
        findings_zh=['皮肤、眉毛和胡子外观已学习，侧脸朝向保留；细节仍偏糊。',
            '帽子颜色落在头部网格上，伸出头部的帽檐尚未形成几何。',
            'RGB误差下降不能代替几何验收；预留语义点误差中位数1.11→1.13px，最差仍5.67px。',
            '原库轮廓损失未启用，头皮边界梯度被屏蔽；8帧只读GPU检查确认该路径为零。'],
        quality_accepted=False,Gaussian_started=False)
    a.save(OUT/'visual-review.json',review)
    summary='面部外观已学习，帽檐几何未通过；正按原SAM2 mask联合修正共享头皮偏移。'
    html='<section id="vhap-demo-face-surface"><h2>demo：脸部表面与帽子轮廓</h2><p><strong>'+summary+'</strong></p>'
    html+='<p>沿用原版VHAP的RGB、纹理和共享顶点偏移阶段。59帧3270次更新，GPU拟合108.9秒，含导出共174.9秒。没有用网络预测三维脸。</p>'
    html+='<p>脸部原生像素区域的训练RGB误差中位数0.209→0.100；这说明外观有学习，不表示独立三维质量通过。设备遮挡鼻嘴的7帧不提供脸部颜色监督。预留语义点仍来自同一二维检测器，不能作为独立3D真值。</p>'
    html+='<p>下面每组由左到右为原图、表面学习前、表面学习后。纹理图右两列是模型在白底上的独立渲染，不用原图填补未建模的帽檐；网格图为同像素覆盖。完整59帧已放入D:/00/记录/demo。</p>'
    selected=['frame_000001','frame_000013','frame_000024','frame_000029','frame_000033','frame_000036','frame_000047','frame_000059']
    names=[]
    for frame in selected:
        for kind in ['纹理','网格']:
            name=f'{frame}_原图_表面学习前_表面学习后_{kind}.png';names.append(name)
            html+=f'<figure><figcaption>{frame} · {kind}</figcaption><img src="demo-vhap-face-surface/{name}" alt="原图与表面学习前后对比"></figure>'
    html+='<p>帽子修正使用原SAM2 mask作为目标；mesh自己的mask只作为预测。只优化共享头皮偏移，脸型、姿态、表情和相机固定。帽檐、区域外泄漏及折叠审阅未通过前，不进入高斯训练。</p></section>'
    record=a.lp('D:/00/记录/demo');record.mkdir(exist_ok=True)
    for p in a.lp(OUT/'review/frames').glob('*.png'):shutil.copy2(p,record/('VHAP_面部表面_'+p.name))
    hashes=[]
    for dest in [a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer']:
        imagefolder=dest/'demo-vhap-face-surface';a.lp(imagefolder).mkdir(exist_ok=True)
        for name in names:shutil.copy2(a.lp(OUT/'review/frames'/name),a.lp(imagefolder/name))
        page=dest/'index.html';old=a.lp(page).read_text(encoding='utf-8')
        if '<section id="vhap-demo-face-surface">' in old:
            start=old.index('<section id="vhap-demo-face-surface">');end=old.index('</section>',start)+len('</section>');old=old[:start]+html+old[end:]
        else:
            start=old.index('<section id="vhap-demo-face3d">');old=old[:start]+html+old[start:]
        old=re.sub(r'<p class="status">.*?</p>','<p class="status">2026-10-30 · '+summary+'</p>',old,count=1)
        a.lp(page).write_text(old,encoding='utf-8');hashes.append(a.sha(page))
    assert len(set(hashes))==1
    for dest in [a.V,a.S]:
        path=dest/'latest-result.json';data=a.read(path);data['latestDiagnostic']=dict(updated=now(),round='123-demo-face-surface',title='脸部表面与帽子mask',summary=summary,status=review['state'],url=URL,report=URL,reportUrl=URL,qualityAccepted=False);a.save(path,data)
    a.save(OUT/'publication.json',dict(updated=now(),summary=summary,report_url=URL,report_sha256=hashes[0],record_images=118,actual_report_UI_verified=False,quality_accepted=False))
    print('FACE_SURFACE_PUBLISHED',URL)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','publish']);args=p.parse_args()
    prepare() if args.mode=='prepare' else publish()
