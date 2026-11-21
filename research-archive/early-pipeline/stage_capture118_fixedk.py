"""Stage a research PLY and actual cameras; promotion requires local OpenFile UI proof."""
from pathlib import Path
import argparse, datetime, hashlib, json, shutil
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
RECORD=TOP/'shuolee/current/new-capture-118'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
V=TOP/'viewer';SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-27/新人物118')
MID='new-capture-118-fixedk';NAME='shuolee_118_fixedK_s18000.ply'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main(promote):
 result=read(ROOT/'result.json');valid=read(ROOT/'export/validation.json');src=ROOT/'export/point_cloud.ply'
 assert result['steps']==18000 and result['from_scratch'] and not result['formal_training_admitted']
 assert valid['format_passed'] and sha(src)==valid['sha256']
 cams=[]
 for i,r in enumerate(read(ROOT/'export/cameras.json')['records']):
  E=r['w2c'];rot=[[E[j][k] for j in range(3)] for k in range(3)];pos=[-sum(rot[k][j]*E[j][3] for j in range(3)) for k in range(3)]
  K=r['K'];cams.append(dict(id=i,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],position=pos,rotation=rot,
    fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2],raw_source_frame=r['source_frame_index'],
    label=f"原帧{r['source_frame_index']} · 118固定K/OpenCV · 共享静态人物",projectionModel='pinhole_export_assumption',
    camera_kind='fixed_factory_K_opencv_research',split='training_camera_not_independent_test'))
 assert len(cams)==89
 quality=read(RECORD/'fixedk-gaussian-quality-review.json') if (RECORD/'fixedk-gaussian-quality-review.json').exists() else {}
 meta=dict(label='118 固定新内参 · 从头20万点 · 18000步研究结果',model=f'models/{MID}/{NAME}',
    sha256=valid['sha256'],gaussianCount=valid['points'],initialGaussianCount=200000,steps=18000,
    trained=True,fromScratch=True,oldCheckpointLoaded=False,representation='shared_static_world',
    timeSnapshot=None,referenceFrameIsPoseSnapshot=False,singlePlyIsNotDynamic=True,localAdaptation=True,
    cameraSource='capture',cameras=cams,focus=valid['viewer_focus'],initialView=0,realFramesUsed=89,
    registeredCameras=89,interpolatedCameras=0,cameraEstimateAccepted=False,experimentalCameraUse=True,
    cameraNote='89个整图固定K/OpenCV估计相机；居中裁切与已矫正导出是假设，FISHEYE_8未应用，物理标定未通过。',
    coordinateSystem=valid['coordinate_space'],filter=dict(preBlurAmount=0,blurAmount=.3),
    qualityAccepted=False,plyCompatibilityPassed=True,deliveryAccepted=promote,diagnosticOnly=True,
    formalTrainingAdmitted=False,report='body-smplx.html#new-capture118',
    deliveryReceipt=f'models/{MID}/delivery.json',
    status='18000步研究训练完成，导出格式通过；重影及腿脚质量以本轮实际审阅为准，尚未认定修复。')
 receipt=dict(round='118-fixedk18000',updated=datetime.datetime.now().astimezone().isoformat(),
    experiment=str(ROOT),steps=18000,initial_points=200000,points=valid['points'],sha256=valid['sha256'],
    compatibility_passed=True,quality_accepted=False,formal_training_admitted=False,
    local_standalone_open_verified=False,pc00_opened_this_round=False,
    download_path=str(V/'models'/MID/NAME),local_path=str(src),
    download_url=f'http://127.0.0.1:8870/models/{MID}/{NAME}?download=1',viewer_url=f'http://127.0.0.1:8870/?model={MID}',
    world_to_viewer=valid,quality_review=quality)
 if promote:
  ui=read(RECORD/'fixedk-viewer-ui-review.json')
  assert ui['file_opened_through_UI'] and ui['observed_count']==valid['points'] and ui['source_sha256']==valid['sha256']
  assert set(ui['modes'])=={'Gaussian','points'}
  assert all(sha(Path(s['path']))==s['sha256'] for s in ui['screenshots'])
  receipt.update(local_viewer_open_file_verified=True,local_standalone_open_verified=False,local_viewer_validation=ui)
 for folder in [V,SYNC]:
  dest=folder/'models'/MID;dest.mkdir(parents=True,exist_ok=True)
  if not (dest/NAME).exists():shutil.copy2(src,dest/NAME)
  assert sha(dest/NAME)==valid['sha256']
  save(dest/'delivery.json',receipt);save(dest/'cameras.json',read(ROOT/'export/cameras.json'))
  m=dict(meta)
  if folder==SYNC:m['report']='progress.html#new-capture118'
  save(folder/f'model-{MID}.json',m)
  js=folder/'viewer.js';text=js.read_text(encoding='utf-8')
  if json.dumps(MID) not in text:
   assert 'const BUILTINS = [' in text;text=text.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(MID)+', ',1);js.write_text(text,encoding='utf-8')
 # One moderate final PLY in the daily log, never training weights or environments.
 (LOG/'点云').mkdir(exist_ok=True)
 if not (LOG/'点云'/NAME).exists():shutil.copy2(src,LOG/'点云'/NAME)
 save(RECORD/'fixedk-staged-delivery.json',receipt)
 if promote:
  prior=read(V/'latest-result.json')
  latest=dict(round='118-fixedk18000',modelId=MID,sha256=valid['sha256'],points=valid['points'],compatibilityPassed=True,
    qualityAccepted=False,updated=receipt['updated'],viewer_url='http://127.0.0.1:8870/?model=latest',
    download_url=receipt['download_url'],report='body-smplx.html#new-capture118',candidateNotBest=True,
    diagnosticOnly=True,status=meta['status'],revision=valid['sha256'],qualityReviewCompleted=True,
    latestDiagnostic=prior.get('latestDiagnostic'))
  save(V/'latest-result.json',latest)
  sync_latest=dict(latest)
  sync_latest.update(viewer_url='?model=latest',download_url=f'models/{MID}/{NAME}?download=1',report='progress.html#new-capture118')
  sync_latest['latestDiagnostic']=read(SYNC/'latest-result.json').get('latestDiagnostic')
  save(SYNC/'latest-result.json',sync_latest)
  ledger=read(TOP/'shuolee/gaussian-deliveries.json');ledger['rounds']['118-fixedk18000']=receipt;ledger['latest_round']='118-fixedk18000';save(TOP/'shuolee/gaussian-deliveries.json',ledger)
 print(json.dumps(dict(model=MID,promoted_local=promote,sha256=valid['sha256'],bytes=src.stat().st_size,sync_model_staged=True,pc00_deployed=False),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--promote-after-ui',action='store_true');main(p.parse_args().promote_after_ui)
