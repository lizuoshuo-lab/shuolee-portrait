"""Stage a real completed 123 export; promote only after actual UI and quality review."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
import numpy as np
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';V=T/'viewer';S=Path('D:/00/工作/查看器')
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main(subject,promote,official_root=None):
    root=Path(official_root).resolve() if official_root else ROOT/subject
    result=read(root/'result.json');validation=read(root/'export/validation.json')
    src=root/'export/point_cloud.ply';digest=sha(src)
    assert validation['format_passed'] and validation['sha256']==digest and result['from_scratch']
    plan=read(REC/'protocol.json');item=next(s for s in plan['subjects'] if s['id']==subject)
    native_control=bool(official_root and (root/'control-input.json').exists())
    shared_offset=bool(official_root and read(root/'gaussian-launch/input.json')['config'].get('shared_static_offset',False))
    variant=subject+'-official-smplifyx' if official_root else subject
    if native_control:
        assert read(root/'gaussian-launch/input.json')['config']['training_height']==1520
        variant+='-native1520'
    if shared_offset:
        variant+='-shared-vertex-offset'
        offset_lr=float(read(root/'gaussian-launch/input.json')['config'].get('shared_static_offset_lr',1.e-6))
        if offset_lr!=1.e-6:
            variant+='-lr'+format(offset_lr,'.0e').replace('e-0','e-')
    adc_timing=None
    one_per_triangle=False
    initial_points=200000
    if official_root:
        run_cfg=read(root/'gaussian-launch/input.json')['config']
        initial_points=int(run_cfg['initial_points'])
        one_per_triangle=initial_points==20908
        if 'densify_from_iter' in run_cfg or 'densification_interval' in run_cfg:
            adc_timing=(int(run_cfg.get('densify_from_iter',10000)),int(run_cfg.get('densification_interval',2000)))
            variant+='-adc'+str(adc_timing[0])+'-'+str(adc_timing[1])
    if one_per_triangle:
        variant=subject+'-official-original-triangle-shared-offset'
    step=result['steps'];name=f'shuolee_123_{variant}_s{step:05d}.ply';ident=f'local-surface-123-{variant}-s{step:05d}'
    if shared_offset and offset_lr!=1.e-6:
        name=f'shuolee_123_shared-offset-lr{offset_lr:.0e}_s{step:05d}.ply'
    if adc_timing:
        name=f'shuolee_123_adc{adc_timing[0]}-{adc_timing[1]}_s{step:05d}.ply'
    if one_per_triangle:
        name=f'shuolee_123_original-triangle_s{step:05d}.ply'
    rows=read(root/'export/cameras.json')['records'];assert len(rows)==item['selected_frames']
    reference=None;reference_index=0
    if official_root:
        fit_receipt=read(root/'fit/official-fit-result.json')
        assert fit_receipt['state']=='complete' and fit_receipt['full_roster'] and fit_receipt['frames']==len(rows)
        assert fit_receipt['fit_sha256']==sha(root/'fit/fitted.npz')
        assert result['execution_kind']=='exploratory_official_smplx'
        reference=validation['reference_frame_id'];ids=[r['frame'] for r in rows]
        assert ids.count(reference)==1
        reference_index=ids.index(reference)
    cameras=[]
    for i,r in enumerate(rows):
        E=np.asarray(r['w2c']);K=np.asarray(r['K']);R=E[:3,:3].T;C=-R@E[:3,3]
        cameras.append(dict(id=i,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],rotation=R.tolist(),position=C.tolist(),
            fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),pixelConvention='pixel_corners',
            projectionModel='pinhole_export_assumption',source_frame_index=r['source_frame_index'],
            label=f'{item["label"]} 原帧{r["source_frame_index"]}',synthetic=False))
    quality_path=root/'export/quality-review.json' if official_root else REC/f'{subject}-quality-review.json'
    ui_path=root/'export/viewer-ui-review.json' if official_root else REC/f'{subject}-viewer-ui-review.json'
    quality=read(quality_path) if quality_path.exists() else {}
    ui=read(ui_path) if ui_path.exists() else {}
    contract=dict(schema='shuolee-renderer-contract-v1',cameraProjection='pinhole',gaussianEvaluation='screen-space-2d',
        trainingRenderer='local snapshot of sofa119 diff-gaussian-rasterization',previewBackend='spark',
        nativeImageEquivalence='PLY verified against training backend; Spark image equivalence not asserted',scope='preview-only')
    m=dict(label=f'{item["label"]} · 本地123自由表面 · {step}步',model=f'models/{ident}/{name}',sha256=digest,
        gaussianCount=validation['points'],initialGaussianCount=initial_points,steps=step,trained=True,fromScratch=True,
        oldCheckpointLoaded=False,representation='shared_static_world',cameraSource='capture',cameras=cameras,
        focus=validation['viewer_focus'],initialView=0,realFramesUsed=len(rows),registeredCameras=len(rows),interpolatedCameras=0,
        rendererContract=contract,coordinateSystem=validation['coordinate_space'],cameraEstimateAccepted=False,
        experimentalCameraUse=True,cameraNote='第一组条件K固定，原生特征/新外参；实际校准未验收。',
        filter=dict(preBlurAmount=0,blurAmount=.3),qualityAccepted=quality.get('quality_accepted',False),
        plyCompatibilityPassed=True,deliveryAccepted=promote,diagnosticOnly=True,formalTrainingAdmitted=False,
        report='body-smplx.html#local-surface123',status=quality.get('summary','已完成本轮导出，实际质量与查看器验收待完成。'))
    if official_root:
        m.update(label=f'{item["label"]} · 官方SMPLify-X初始化 · {step}步',
            representation='named_reference_static_snapshot',referenceFrameId=reference,
            referenceFrameIsPoseSnapshot=True,singlePlyIsNotDynamic=True,initialView=reference_index,
            localAdaptation=True,upstreamCompleteReproduction=False,
            cameraNote=f'完整K与同坐标相机随包导出；静态点云固定于{reference}。其他实拍帧姿态不同，不是同姿态静态真值。条件内参的物理标定未验收。',
            report='diagnostics/video10-global-initializer/index.html#official-gaussian-final')
        if native_control:
            m.update(label=f'{item["label"]} · 原生2704×1520对照 · {step}步',
                trainingResolution=[2704,1520],imageCacheFrames=94,
                report='diagnostics/video10-global-initializer/index.html#official-native-resolution')
        if shared_offset:
            m.update(label=f'{item["label"]} · 原版共享顶点偏移 · {step}步',
                sharedStaticOffset=True,staticOffsetLearningRate=offset_lr,
                report='diagnostics/video10-global-initializer/index.html#official-shared-vertex-offset'+('-long' if step>1000 else ''))
            if offset_lr!=1.e-6:
                m.update(label=f'{item["label"]} · 共享顶点偏移LR{offset_lr:.0e}对照 · {step}步',
                    report='diagnostics/video10-global-initializer/index.html#official-shared-offset-lr')
        if adc_timing:
            m.update(label=f'{item["label"]} · 原版增密时序{adc_timing[0]}/{adc_timing[1]}对照 · {step}步',
                densificationTiming=list(adc_timing),report='diagnostics/video10-global-initializer/index.html#official-adc-control')
        if one_per_triangle:
            m.update(label=f'{item["label"]} · 原版逐三角初始化与共享偏移 · {step}步',
                initialization='one Gaussian per triangle',
                report='diagnostics/video10-global-initializer/index.html#bound-surface-drive')
    if promote:
        assert ui.get('file_opened_through_UI') and ui.get('source_sha256')==digest
        assert ui.get('observed_count')==validation['points'] and ui.get('camera_count')==len(rows)
        assert set(ui.get('modes',[]))=={'Gaussian','points'}
        assert quality.get('actual_same_camera_review') and quality.get('continuous_novel_views_reviewed')==36
        assert quality.get('source_sha256')==digest
        for p in ui['screenshots']:assert sha(Path(p['path']))==p['sha256']
        if official_root:
            package_check=read(root/'export/package-check.json')
            assert package_check['model_sha256']==digest and package_check['cameras']==len(rows)
            assert sha(V/'models'/ident/'scene.zip')==package_check['zip_sha256']
    for folder in [V,S]:
        dest=folder/'models'/ident;dest.mkdir(parents=True,exist_ok=True)
        if not (dest/name).exists():shutil.copy2(src,dest/name)
        assert sha(dest/name)==digest
        save(dest/'camera-records.json',read(root/'export/cameras.json'))
        save(dest/'cameras.json',cameras);save(dest/'model-context.json',dict(m,label=m['label']))
        save(folder/f'model-{ident}.json',m)
        if promote:
            js=folder/'viewer.js';text=js.read_text(encoding='utf-8')
            if json.dumps(ident) not in text:
                assert text.count('const BUILTINS = [')==1
                js.write_text(text.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(ident)+', ',1),encoding='utf-8')
            old=read(folder/'latest-result.json')
            save(folder/'latest-result.json',dict(round=ident,modelId=ident,sha256=digest,points=validation['points'],
                compatibilityPassed=True,qualityAccepted=m['qualityAccepted'],updated=datetime.now(timezone.utc).isoformat(),
                viewer_url='?model=latest',download_url=f'models/{ident}/scene.zip?download=1',
                report=m['report'] if official_root else ('body-smplx.html' if folder==V else 'progress.html')+'#local-surface123',
                candidateNotBest=True,diagnosticOnly=True,status=m['status'],revision=digest,qualityReviewCompleted=True,
                latestDiagnostic=old.get('latestDiagnostic')))
    receipt=dict(updated=datetime.now(timezone.utc).isoformat(),subject=subject,modelId=ident,label=m['label'],
        ply=str(V/'models'/ident/name),source_sha256=digest,points=validation['points'],steps=step,camera_count=len(rows),
        context=str(V/'models'/ident/'model-context.json'),camera_file=str(V/'models'/ident/'cameras.json'),
        zip=str(V/'models'/ident/'scene.zip'),viewer_url=f'http://127.0.0.1:8870/?model={ident}',
        quality_accepted=m['qualityAccepted'],local_UI_verified=promote,company00_verified=False,
        reference_frame_id=reference,initial_view=reference_index,source_run=str(root))
    save(root/'export/staged-delivery.json' if official_root else REC/f'{subject}-staged-delivery.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--promote-after-ui',action='store_true')
    p.add_argument('--official-root',help='Completed full official SMPLify-X/Gaussian run; keep its reference-pose metadata and separate model identity.')
    a=p.parse_args();main(a.subject,a.promote_after_ui,a.official_root)
