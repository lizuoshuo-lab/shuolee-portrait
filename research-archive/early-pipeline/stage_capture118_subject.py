"""Deliver one actual subject export; latest promotion needs exact OpenFile proof."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/new-capture-118/all-persons-20261027'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-capture118-all-persons-20261027')
V=TOP/'viewer';SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-27/全部人物训练')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main(subject,promote):
    info=next(x for x in read(REC/'protocol.json')['subjects'] if x['id']==subject)
    root=ROOT/subject;mid='new-capture-118-'+subject
    result=read(root/'result.json');valid=read(root/'export/validation.json');src=root/'export/point_cloud.ply'
    steps=result['steps'];training=read(root/'gaussian-run/result.json')
    assert result['from_scratch'] and not result['formal_training_admitted']
    assert steps==18000 or (steps==1000 and training['state']=='stopped_after_actual_review')
    if steps!=18000:mid+='-stopped1000'
    name=f'shuolee_118_{subject}_s{steps:05d}.ply'
    assert valid['format_passed'] and sha(src)==valid['sha256']
    qualitypath=REC/f'{subject}-quality-review.json';quality=read(qualitypath) if qualitypath.exists() else {}
    quality_passed=quality.get('quality_accepted',False)
    cams=[]
    for i,r in enumerate(read(root/'export/cameras.json')['records']):
        E=r['w2c'];rot=[[E[j][k] for j in range(3)] for k in range(3)]
        pos=[-sum(rot[k][j]*E[j][3] for j in range(3)) for k in range(3)];K=r['K']
        cams.append(dict(id=i,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],position=pos,rotation=rot,
            fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2],raw_source_frame=r['source_frame_index'],
            label=f"原帧{r['source_frame_index']} · 固定K/OpenCV · {info['label']}",
            projectionModel='pinhole_export_assumption',camera_kind='fixed_factory_K_opencv_research',split='training_camera_not_independent_test'))
    assert len(cams)==result['frames']
    status=quality.get('summary',f'{steps}步研究快照导出，实际质量与查看器待审阅。')
    meta=dict(label=info['label']+(' · 新内参从头18k研究结果' if steps==18000 else ' · 1000步停止诊断'),model=f'models/{mid}/{name}',sha256=valid['sha256'],
        gaussianCount=valid['points'],initialGaussianCount=200000,steps=steps,trained=True,fromScratch=True,
        oldCheckpointLoaded=False,representation='shared_static_world',timeSnapshot=None,referenceFrameIsPoseSnapshot=False,
        singlePlyIsNotDynamic=True,localAdaptation=True,cameraSource='capture',cameras=cams,focus=valid['viewer_focus'],
        initialView=0,realFramesUsed=len(cams),registeredCameras=len(cams),interpolatedCameras=0,
        cameraEstimateAccepted=False,experimentalCameraUse=True,
        cameraNote='原始K按居中裁切与显式像素缩放用于OpenCV外参估计。FISHEYE_8未知公式未套用；物理标定及视频适用性未确认。',
        coordinateSystem=valid['coordinate_space'],filter=dict(preBlurAmount=0,blurAmount=.3),
        qualityAccepted=quality_passed,plyCompatibilityPassed=True,deliveryAccepted=promote,diagnosticOnly=True,
        formalTrainingAdmitted=False,report='body-smplx.html#all-persons118',deliveryReceipt=f'models/{mid}/delivery.json',status=status)
    receipt=dict(round=mid,subject=subject,updated=datetime.datetime.now().astimezone().isoformat(),experiment=str(root),
        steps=steps,initial_points=200000,points=valid['points'],sha256=valid['sha256'],compatibility_passed=True,
        quality_accepted=quality_passed,formal_training_admitted=False,local_standalone_open_verified=False,
        pc00_opened_this_round=False,download_path=str(V/'models'/mid/name),local_path=str(src),
        download_url=f'http://127.0.0.1:8870/models/{mid}/{name}?download=1',
        viewer_url=f'http://127.0.0.1:8870/?model={mid}',world_to_viewer=valid,quality_review=quality)
    if promote:
        ui=read(REC/f'{subject}-viewer-ui-review.json')
        assert ui['file_opened_through_UI'] and ui['observed_count']==valid['points'] and ui['source_sha256']==valid['sha256']
        assert set(ui['modes'])=={'Gaussian','points'} and all(sha(Path(s['path']))==s['sha256'] for s in ui['screenshots'])
        assert quality['actual_same_camera_review'] and quality['continuous_novel_views_reviewed']==36
        assert quality['source_sha256']==valid['sha256']
        receipt.update(local_viewer_open_file_verified=True,local_viewer_validation=ui)
    for folder in [V,SYNC]:
        dest=folder/'models'/mid;dest.mkdir(parents=True,exist_ok=True)
        if not (dest/name).exists():shutil.copy2(src,dest/name)
        assert sha(dest/name)==valid['sha256']
        save(dest/'delivery.json',receipt);save(dest/'cameras.json',read(root/'export/cameras.json'))
        m=dict(meta)
        if folder==SYNC:m['report']='progress.html#all-persons118'
        save(folder/f'model-{mid}.json',m)
        js=folder/'viewer.js';text=js.read_text(encoding='utf-8')
        if json.dumps(mid) not in text:
            assert 'const BUILTINS = [' in text
            text=text.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(mid)+', ',1);js.write_text(text,encoding='utf-8')
    finaldir=LOG/'点云';finaldir.mkdir(exist_ok=True)
    if not (finaldir/name).exists():shutil.copy2(src,finaldir/name)
    assert sha(finaldir/name)==valid['sha256']
    save(REC/f'{subject}-staged-delivery.json',receipt)
    if promote:
        for folder in [V,SYNC]:
            old=read(folder/'latest-result.json')
            latest=dict(round=mid,modelId=mid,sha256=valid['sha256'],points=valid['points'],compatibilityPassed=True,
                qualityAccepted=quality_passed,updated=receipt['updated'],viewer_url='?model=latest',
                download_url=f'models/{mid}/{name}?download=1',report=('body-smplx.html' if folder==V else 'progress.html')+'#all-persons118',
                candidateNotBest=True,diagnosticOnly=True,status=status,revision=valid['sha256'],qualityReviewCompleted=True,
                latestDiagnostic=old.get('latestDiagnostic'))
            save(folder/'latest-result.json',latest)
        ledger=read(TOP/'shuolee/gaussian-deliveries.json');ledger['rounds'][mid]=receipt;ledger['latest_round']=mid
        save(TOP/'shuolee/gaussian-deliveries.json',ledger)
        st=read(REC/f'{subject}-state.json');st.update(updated=receipt['updated'],state='delivered_reviewed' if steps==18000 else 'stopped_diagnostic_reviewed',
            training_running=False,quality_accepted=quality_passed,viewer_url=receipt['viewer_url'],summary=status)
        save(REC/f'{subject}-state.json',st)
    print(json.dumps(dict(subject=subject,model=mid,promoted_local=promote,sha256=valid['sha256'],bytes=src.stat().st_size,pc00_deployed=False),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True,choices=['grass','sofa','portrait','indoor118']);p.add_argument('--promote-after-ui',action='store_true')
    a=p.parse_args();main(a.subject,a.promote_after_ui)
