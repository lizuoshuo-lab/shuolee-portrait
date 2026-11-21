"""Stage a reviewed head snapshot; UI receipts require actual human observation."""
import argparse,json,shutil
import numpy as np
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

RUNS=('head-gaussian-first-cycle','head-gaussian-long300k')
FILTER=dict(preBlurAmount=.3,blurAmount=0)


def require(condition,message):
    if not condition:raise ValueError(message)


def passed(value):
    return type(value) is dict and value.get('passed') is True


def selected_snapshot(base,run,step):
    require(run in RUNS,'Unknown head run')
    require(type(step) is int and step>0,'Invalid step')
    require(run!=RUNS[1] or step==300000,'Long-run staging requires final step300000')
    prefix='demo_head_hat_long300k_' if run==RUNS[1] else 'demo_head_hat_'
    ident='local-surface-123-demo-head-hat-'+('long300k-' if run==RUNS[1] else '')+f's{step}'
    folder=base/run
    return folder,folder/'delivery'/f'step-{step:06d}',ident,prefix+f'{step:06d}_with_camera.zip'


def check_existing(existing,expected,label):
    require(existing is None or existing==expected,'Refuse different existing content: '+label)


def check_receipts(run,step,data,hashes,shot_hashes):
    """Prevent artifact mixups; these checks cannot authenticate human UI observations."""
    v,q,ctx,r,ui,s=(data[k] for k in ['validation','quality','context','package','ui','snapshot'])
    cameras,records=data['cameras'],data['records']
    n=v.get('points');ref=s.get('reference_frame');slot=s.get('reference_slot')
    require(type(n) is int and n>0,'Invalid Gaussian count')
    require(v.get('step')==ctx.get('steps')==s.get('step')==step,'Snapshot step mismatch')
    require(s.get('points')==r.get('points')==ui.get('observed_count')==n,'Gaussian count mismatch')
    require(v.get('sha256')==q.get('source_sha256')==ui.get('source_sha256')==ctx.get('model',{}).get('sha256')==hashes['ply'],'PLY identity mismatch')
    require(v.get('source_sha256')==s.get('world_ply_sha256')==hashes['world'],'Source worldPLY mismatch')
    require(v.get('format_passed') is True and passed(v.get('native_source_ply_parity')) and passed(v.get('native_rotated_render_parity')) and passed(s.get('native_ply_parity')),'Native export checks must pass')
    require(v.get('native_replay_uses_serialized_camera') is True and v.get('serialized_camera_sha256')==ui.get('serialized_camera_sha256')==hashes['camera'],'Serialized camera replay/SHA mismatch')
    require(r.get('run')==ui.get('run')==q.get('run')==run and r.get('step')==ui.get('step')==q.get('step')==step,'Receipt run/step mismatch')
    require(r.get('zip_sha256')==ui.get('package_sha256')==hashes['zip'] and hashes['package_path_matches'],'ZIP identity/path mismatch')
    require(r.get('verifiedBinding') is True and r.get('cameraCount')==1,'Package camera binding missing')
    require(type(slot) is int and 0<=slot<len(records) and len(records)==59,'Reference slot/cohort mismatch')
    require(ref==records[slot].get('frame')==v.get('reference_frame')==ctx.get('referenceFrameId')==r.get('referenceFrameId')==ui.get('reference_frame'),'Reference frame mismatch')
    require(len(cameras)==1 and cameras[0].get('img_name')==ref and cameras[0].get('pixelConvention')=='pixel_corners','Expected one paired head camera')
    require(ctx.get('referenceFrameIsPoseSnapshot') is True and ctx.get('initialView')==0,'Static reference-pose contract missing')
    require(q.get('actual_all59_review') is True and q.get('reviewed_frames')==59 and q.get('actual_same_camera_review') is True and q.get('continuous_novel_views_reviewed')==36,'Actual59-frame/reference/36-view review required')
    require(type(q.get('summary')) is str and bool(q['summary'].strip()),'Current quality summary missing')
    require(ui.get('file_opened_through_UI') is True and ui.get('actual_same_camera_review') is True,'Actual OpenFile/same-camera UI receipt required')
    require(ui.get('camera_count')==1 and ui.get('filter')==FILTER and type(ui.get('gaussianScale')) in (int,float) and ui.get('gaussianScale')==1,'UI camera/filter/scale mismatch')
    require(type(ui.get('modes')) is list and len(ui['modes'])==2 and set(ui['modes'])=={'Gaussian','points'},'Both actual UI modes required')
    shots=ui.get('screenshots',[])
    require(type(shots) is list and len(shots)==2 and all(type(x) is dict for x in shots),'Two UI screenshot records required')
    require({x.get('mode') for x in shots}=={'Gaussian','points'} and len({x.get('path') for x in shots})==2 and len({x.get('sha256') for x in shots})==2,'Distinct screenshot paths/SHA for both UI modes required')
    for shot in shots:
        require(shot.get('path') in shot_hashes and shot.get('sha256')==shot_hashes[shot['path']],'Screenshot file SHA mismatch')
    if run==RUNS[1]:
        p,g=data.get('progress',{}),data.get('run_result',{})
        require(step==300000 and p.get('state')=='completed_quality_review_pending' and 'pid' in p and p['pid'] is None and p.get('step')==step and p.get('points')==n,'Long training is not completed')
        require(g.get('state')=='completed_quality_review_pending' and g.get('steps')==step and g.get('points')==n and g.get('final_snapshot')==f'step-{step:06d}','Long final result mismatch')
    quality=q.get('quality_accepted') is True
    require(not quality or (q.get('same_camera_quality_accepted') is True and q.get('novel_quality_accepted') is True),'Quality needs both explicit acceptances')
    return n,ref,slot,quality


def main(run,step,ui_path):
    root=a.REC/'demo-vhap-initialization'
    folder,out,ident,package_name=selected_snapshot(root,run,step)
    snap=folder/'gaussian-run'/f'step-{step:06d}'
    paths=dict(validation=out/'validation.json',quality=out/'quality-review.json',context=out/'context.json',
        package=out/'package-receipt.json',ui=ui_path,cameras=out/'cameras.json',snapshot=snap/'result.json')
    data={k:a.read(p) for k,p in paths.items()};data['records']=a.read(folder/'inputs.json')['records']
    if run==RUNS[1]:
        data.update(progress=a.read(folder/'progress.json'),run_result=a.read(folder/'gaussian-run/result.json'))
    src=out/'point_cloud.ply';package=a.S/'exports'/package_name
    hashes=dict(ply=a.sha(src),world=a.sha(snap/'world.ply'),camera=a.sha(out/'cameras.json'),zip=a.sha(package),ui=a.sha(ui_path),
        package_path_matches=a.lp(data['package']['path']).resolve()==a.lp(package).resolve())
    shot_hashes={}
    for shot in data['ui'].get('screenshots',[]):
        require(type(shot) is dict and type(shot.get('path')) is str,'Invalid UI screenshot path')
        shot_hashes[shot['path']]=a.sha(shot['path'])
    n,ref,slot,quality=check_receipts(run,step,data,hashes,shot_hashes)
    require(not a.lp(out/'staged-delivery.json').exists(),'Immutable staging receipt already exists')
    with np.load(a.lp(folder/'fitted-head-meshes.npz'),allow_pickle=False) as mesh:
        require(str(mesh['frame_ids'][slot])==ref,'Mesh reference frame mismatch')
        focus=(np.median(mesh['vertices_world'][slot],axis=0)*np.array([1,-1,-1])).tolist()
        initial_points=len(mesh['faces'])
    v,q,ctx,cameras=(data[k] for k in ['validation','quality','context','cameras'])
    url='diagnostics/video10-global-initializer/index.html#'+('demo-head-long' if run==RUNS[1] else 'demo-head-gaussian')
    summary=q['summary'] if quality else '诊断候选（质量未通过）：'+q['summary']
    m=dict(label=f'demo面部＋帽子 · FLAME绑定 · {step}步',model=f'models/{ident}/point_cloud.ply',
        sha256=hashes['ply'],gaussianCount=n,initialGaussianCount=initial_points,steps=step,trained=True,fromScratch=True,
        oldCheckpointLoaded=False,representation='named_reference_static_snapshot',referenceFrameId=ref,
        referenceFrameIsPoseSnapshot=True,singlePlyIsNotDynamic=True,initialView=0,cameraSource='capture',cameras=cameras,
        focus=focus,realFramesUsed=59,registeredCameras=1,interpolatedCameras=0,rendererContract=ctx['rendererContract'],
        coordinateSystem=v['coordinate_space'],cameraEstimateAccepted=False,experimentalCameraUse=True,
        cameraNote=f'固定参考帧{ref}姿态；59帧训练包含逐帧FLAME形变，静态点云仅配这一帧实拍相机。仅面部和帽子，物理标定未验收。',
        filter=FILTER,qualityAccepted=quality,plyCompatibilityPassed=True,deliveryAccepted=True,diagnosticOnly=True,
        localAdaptation=True,upstreamCompleteReproduction=False,report=url,status=summary)
    # Validate all immutable model destinations before any publication write.
    for target in [a.V,a.S]:
        dest=target/'models'/ident
        for source,name in [(src,'point_cloud.ply'),(package,'scene.zip')]:
            p=dest/name;check_existing(a.sha(p) if a.lp(p).exists() else None,a.sha(source),str(p))
        for p,payload in [(dest/'cameras.json',cameras),(dest/'model-context.json',m),(target/f'model-{ident}.json',m)]:
            check_existing(a.read(p) if a.lp(p).exists() else None,payload,str(p))
        text=a.lp(target/'viewer.js').read_text(encoding='utf-8')
        require(json.dumps(ident) in text or text.count('const BUILTINS = [')==1,'Viewer builtin marker mismatch')
    updated=now()
    for target in [a.V,a.S]:
        dest=target/'models'/ident;a.lp(dest).mkdir(parents=True,exist_ok=True)
        for source,name in [(src,'point_cloud.ply'),(package,'scene.zip')]:
            if not a.lp(dest/name).exists():shutil.copy2(a.lp(source),a.lp(dest/name))
        for p,payload in [(dest/'cameras.json',cameras),(dest/'model-context.json',m),(target/f'model-{ident}.json',m)]:
            if not a.lp(p).exists():a.save(p,payload)
        js=target/'viewer.js';text=a.lp(js).read_text(encoding='utf-8')
        if json.dumps(ident) not in text:a.lp(js).write_text(text.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(ident)+', ',1),encoding='utf-8')
        old=a.read(target/'latest-result.json')
        a.save(target/'latest-result.json',dict(round=ident,modelId=ident,sha256=hashes['ply'],points=n,
            compatibilityPassed=True,qualityAccepted=quality,updated=updated,viewer_url='?model=latest',
            download_url=f'models/{ident}/scene.zip?download=1',report=url,candidateNotBest=not quality,diagnosticOnly=True,
            status=summary,revision=hashes['ply'],qualityReviewCompleted=True,latestDiagnostic=old.get('latestDiagnostic')))
    publication=folder/'publication.json';pub=a.read(publication) if a.lp(publication).exists() else {}
    pub.update(updated=updated,run=run,steps=step,source_sha256=hashes['ply'],actual_UI_review=True,viewer_model_id=ident,quality_accepted=quality)
    a.save(publication,pub)  # Selected run only; long publication never touches18k.
    long=root/RUNS[1];other=a.read(long/'progress.json') if a.lp(long/'progress.json').exists() else {}
    live=run!=RUNS[1] and other.get('state')=='training' and other.get('pid') is not None
    running=[dict(pid=other['pid'],script='train_demo_head_gaussians123_long.py',progress=str(long/'progress.json'))] if live else []
    status=dict(updated=updated,state='demo_head_gaussian_snapshot_staged',subject='demo',scope='Head plus hat FLAME only',
        run=run,frames=59,step=step,points=n,target_steps=300000 if run==RUNS[1] else step,source=str(folder),
        quality_accepted=quality,training_running=bool(live),training_started=True,fitter_running=False,goal_complete=False,report_url=url,summary=summary)
    for p in [a.R/'queue-state.json',a.REC/'queue-state.json',a.C/'local-research-status.json']:
        current=a.read(p);current.update(updated=updated,state=status['state'],current_stage=status['state'],phase=status['state'],
            training_running=bool(live),training_started=True,GPU_task_running=bool(live),fitting_running=False,
            running=running,latest_demo_head_gaussian=status,goal_complete=False);a.save(p,current)
    a.save(root/'progress.json',status)
    a.save(out/'staged-delivery.json',dict(updated=updated,run=run,modelId=ident,source_sha256=hashes['ply'],steps=step,points=n,
        package_sha256=hashes['zip'],serialized_camera_sha256=hashes['camera'],ui_receipt=str(ui_path),ui_receipt_sha256=hashes['ui'],
        camera_count=1,reference_frame=ref,local_UI_verified=True,quality_accepted=quality,company00_verified=False,
        viewer_url='http://127.0.0.1:8870/?model=latest'))
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',choices=RUNS,default=RUNS[0])
    parser.add_argument('--step',type=int,default=18000)
    parser.add_argument('--ui-receipt',required=True,help='Human-recorded current OpenFile Gaussian+points receipt; never synthesized by stage.')
    args=parser.parse_args();main(args.run,args.step,args.ui_receipt)
