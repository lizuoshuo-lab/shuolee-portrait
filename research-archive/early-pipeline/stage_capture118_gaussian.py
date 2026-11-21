"""Stage a research PLY and actual cameras; promotion requires local OpenFile UI proof."""
from pathlib import Path
import argparse, datetime, hashlib, json, shutil
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
RECORD=TOP/'shuolee/current/new-capture-118'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-new-capture-118-support')
V=TOP/'viewer';SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-26/新视频118')
MID='new-capture-118-probe';NAME='shuolee_118_s01000.ply'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main(promote):
 result=read(ROOT/'result.json');valid=read(ROOT/'export/validation.json');src=ROOT/'export/point_cloud.ply'
 assert result['steps']==1000 and result['from_scratch'] and not result['formal_training_admitted']
 assert valid['format_passed'] and sha(src)==valid['sha256']
 cams=[]
 for i,r in enumerate(read(ROOT/'export/cameras.json')['records']):
  E=r['w2c'];rot=[[E[j][k] for j in range(3)] for k in range(3)];pos=[-sum(rot[k][j]*E[j][3] for j in range(3)) for k in range(3)]
  K=r['K'];cams.append(dict(id=i,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],position=pos,rotation=rot,
    fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2],raw_source_frame=r['source_frame_index'],
    label=f"原帧{r['source_frame_index']} · 118整图SfM · 共享静态人物",projectionModel='rectified_pinhole',
    camera_kind='fresh_full_scene_sfm',split='training_camera_not_independent_test'))
 assert len(cams)==89
 quality=read(RECORD/'gaussian-quality-review.json') if (RECORD/'gaussian-quality-review.json').exists() else {}
 meta=dict(label='118 新视频 · 从头20万点 · 1000步诊断',model=f'models/{MID}/{NAME}',
    sha256=valid['sha256'],gaussianCount=valid['points'],initialGaussianCount=200000,steps=1000,
    trained=True,fromScratch=True,oldCheckpointLoaded=False,representation='shared_static_world',
    timeSnapshot=None,referenceFrameIsPoseSnapshot=False,singlePlyIsNotDynamic=True,localAdaptation=True,
    cameraSource='capture',cameras=cams,focus=[0,0,0],initialView=0,realFramesUsed=89,
    registeredCameras=89,interpolatedCameras=0,cameraEstimateAccepted=False,experimentalCameraUse=True,
    cameraNote='89个真实整图SfM相机；物理标定与静止性尚未证实。',
    coordinateSystem=valid['coordinate_space'],filter=dict(preBlurAmount=0,blurAmount=.3),
    qualityAccepted=False,plyCompatibilityPassed=True,deliveryAccepted=promote,diagnosticOnly=True,
    formalTrainingAdmitted=False,report='body-smplx.html#new-capture118',
    deliveryReceipt=f'models/{MID}/delivery.json',
    status='1000步短程训练完成，导出一致性通过；衣服外扩光晕和腿脚破碎仍明显，质量未通过。')
 receipt=dict(round='118-probe1000',updated=datetime.datetime.now().astimezone().isoformat(),
    experiment=str(ROOT),steps=1000,initial_points=200000,points=valid['points'],sha256=valid['sha256'],
    compatibility_passed=True,quality_accepted=False,formal_training_admitted=False,
    local_standalone_open_verified=False,pc00_opened_this_round=False,
    download_path=str(V/'models'/MID/NAME),local_path=str(src),
    download_url=f'http://127.0.0.1:8870/models/{MID}/{NAME}?download=1',viewer_url=f'http://127.0.0.1:8870/?model={MID}',
    world_to_viewer=valid,quality_review=quality)
 if promote:
  ui=read(RECORD/'viewer-ui-review.json')
  assert ui['file_opened_through_UI'] and ui['observed_count']==valid['points'] and ui['source_sha256']==valid['sha256']
  assert set(ui['modes'])=={'Gaussian','points'}
  assert all(sha(Path(s['path']))==s['sha256'] for s in ui['screenshots'])
  receipt.update(local_standalone_open_verified=True,local_viewer_validation=ui)
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
 save(RECORD/'staged-delivery.json',receipt)
 if promote:
  prior=read(V/'latest-result.json')
  latest=dict(round='118-probe1000',modelId=MID,sha256=valid['sha256'],points=valid['points'],compatibilityPassed=True,
    qualityAccepted=False,updated=receipt['updated'],viewer_url='http://127.0.0.1:8870/?model=latest',
    download_url=receipt['download_url'],report='body-smplx.html#new-capture118',candidateNotBest=True,
    diagnosticOnly=True,status=meta['status'],revision=valid['sha256'],qualityReviewCompleted=True,
    latestDiagnostic=prior.get('latestDiagnostic'))
  save(V/'latest-result.json',latest)
  ledger=read(TOP/'shuolee/gaussian-deliveries.json');ledger['rounds']['118-probe1000']=receipt;ledger['latest_round']='118-probe1000';save(TOP/'shuolee/gaussian-deliveries.json',ledger)
 print(json.dumps(dict(model=MID,promoted_local=promote,sha256=valid['sha256'],bytes=src.stat().st_size,sync_model_staged=True,pc00_deployed=False),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--promote-after-ui',action='store_true');main(p.parse_args().promote_after_ui)
