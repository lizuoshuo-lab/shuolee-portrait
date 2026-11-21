"""Stage a completed actual clothing result; promote only after actual local UI."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/clothed-surface-119'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-clothed-surface-119')
V=TOP/'viewer';SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-27/衣物表面点云119')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)

def main(subject,promote):
    root=ROOT/subject;result=read(root/'result.json');validation=read(root/'export/validation.json');source=root/'export/point_cloud.ply'
    training=read(root/'gaussian-run/result.json');step=result['steps'];name=f'shuolee_119_{subject}_s{step:05d}.ply'
    assert result['from_scratch'] and not result['formal_training_admitted']
    assert step==18000 or (step==1000 and training['state']=='stopped_after_actual_review')
    assert validation['format_passed'] and sha(source)==validation['sha256']
    model_id='new-capture-119-'+subject+'-clothed'+('' if step==18000 else '-stopped1000')
    label={'portrait':'蓝衣黄裙','sofa':'沙发坐姿'}[subject]
    quality=read(REC/f'{subject}-quality-review.json') if (REC/f'{subject}-quality-review.json').exists() else {}
    ui=read(REC/f'{subject}-viewer-ui-review.json') if (REC/f'{subject}-viewer-ui-review.json').exists() else {}
    cameras=[]
    for i,r in enumerate(read(root/'export/cameras.json')['records']):
        E=r['w2c'];rotation=[[E[j][k] for j in range(3)] for k in range(3)];K=r['K']
        cameras.append(dict(id=i,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],rotation=rotation,
            position=[-sum(rotation[k][j]*E[j][3] for j in range(3)) for k in range(3)],
            fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2],raw_source_frame=r['source_frame_index'],
            label=f'{label} 原帧{r["source_frame_index"]} · 固定K/OpenCV',
            camera_kind='fixed_factory_K_opencv_research',projectionModel='pinhole_export_assumption',split='training_not_independent_test'))
    assert len(cameras)==result['frames']
    if promote:
        assert ui['file_opened_through_UI'] and ui['source_sha256']==validation['sha256'] and ui['observed_count']==validation['points']
        assert set(ui['modes'])=={'Gaussian','points'} and all(sha(Path(x['path']))==x['sha256'] for x in ui['screenshots'])
        assert quality['actual_same_camera_review'] and quality['continuous_novel_views_reviewed']==36
        assert quality['source_sha256']==validation['sha256']
    summary=quality.get('summary','自由衣物表面高斯已导出；真实质量和查看器验收待完成。')
    model=dict(label=f'{label} · 119衣物表面自由点 · {step}步',model=f'models/{model_id}/{name}',sha256=validation['sha256'],
        gaussianCount=validation['points'],initialGaussianCount=200000,steps=step,trained=True,fromScratch=True,
        oldCheckpointLoaded=False,representation='shared_static_world',timeSnapshot=None,referenceFrameIsPoseSnapshot=False,
        singlePlyIsNotDynamic=True,localAdaptation=True,cameraSource='capture',cameras=cameras,focus=validation['viewer_focus'],initialView=0,
        realFramesUsed=len(cameras),registeredCameras=len(cameras),interpolatedCameras=0,cameraEstimateAccepted=False,experimentalCameraUse=True,
        cameraNote='本轮固定原有新内参候选和OpenCV相机；D8未知公式未套用，未解决物理标定或微动。',
        coordinateSystem=validation['coordinate_space'],filter=dict(preBlurAmount=0,blurAmount=.3),
        qualityAccepted=quality.get('quality_accepted',False),plyCompatibilityPassed=True,deliveryAccepted=promote,
        diagnosticOnly=True,formalTrainingAdmitted=False,report='body-smplx.html#clothed119',
        deliveryReceipt=f'models/{model_id}/delivery.json',status=summary)
    receipt=dict(updated=datetime.datetime.now().astimezone().isoformat(),round=model_id,subject=subject,experiment=str(root),
        steps=step,initial_points=200000,points=validation['points'],sha256=validation['sha256'],compatibility_passed=True,
        quality_accepted=quality.get('quality_accepted',False),formal_training_admitted=False,
        local_viewer_open_file_verified=promote,pc00_opened_this_round=False,world_to_viewer=validation,
        download_path=str(V/'models'/model_id/name),sync_path=str(LOG/'点云'/name),
        download_url=f'http://127.0.0.1:8870/models/{model_id}/{name}?download=1',viewer_url=f'http://127.0.0.1:8870/?model={model_id}',
        quality_review=quality,local_viewer_validation=ui if promote else None)
    for folder in [V,SYNC]:
        dest=folder/'models'/model_id;dest.mkdir(parents=True,exist_ok=True)
        if not (dest/name).exists():shutil.copy2(source,dest/name)
        assert sha(dest/name)==validation['sha256']
        save(dest/'delivery.json',receipt);save(dest/'cameras.json',read(root/'export/cameras.json'))
        m=dict(model)
        if folder==SYNC:m['report']='progress.html#clothed119'
        save(folder/f'model-{model_id}.json',m)
        js=folder/'viewer.js';text=js.read_text(encoding='utf-8')
        if json.dumps(model_id) not in text:
            assert 'const BUILTINS = [' in text
            js.write_text(text.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(model_id)+', ',1),encoding='utf-8')
    (LOG/'点云').mkdir(exist_ok=True)
    if not (LOG/'点云'/name).exists():shutil.copy2(source,LOG/'点云'/name)
    assert sha(LOG/'点云'/name)==validation['sha256']
    save(REC/f'{subject}-staged-delivery.json',receipt)
    if promote:
        for folder in [V,SYNC]:
            old=read(folder/'latest-result.json')
            save(folder/'latest-result.json',dict(round=model_id,modelId=model_id,sha256=validation['sha256'],points=validation['points'],
                compatibilityPassed=True,qualityAccepted=quality.get('quality_accepted',False),updated=receipt['updated'],viewer_url='?model=latest',
                download_url=f'models/{model_id}/{name}?download=1',report=('body-smplx.html' if folder==V else 'progress.html')+'#clothed119',
                candidateNotBest=True,diagnosticOnly=True,status=summary,revision=validation['sha256'],qualityReviewCompleted=True,
                latestDiagnostic=old.get('latestDiagnostic')))
        ledger=read(TOP/'shuolee/gaussian-deliveries.json');ledger['rounds'][model_id]=receipt;ledger['latest_round']=model_id
        save(TOP/'shuolee/gaussian-deliveries.json',ledger)
        state=read(REC/f'{subject}-state.json');state.update(stage='delivered_reviewed',training_running=False,summary=summary,updated=receipt['updated'])
        save(REC/f'{subject}-state.json',state)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True,choices=['portrait','sofa']);p.add_argument('--promote-after-ui',action='store_true')
    a=p.parse_args();main(a.subject,a.promote_after_ui)
