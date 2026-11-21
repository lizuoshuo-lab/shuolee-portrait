"""Read-only diagnostic export of the retained1000/999-update officewoman snapshot.

audit: CPU and stdout only. export: one bounded WSL CUDA replay, explicit call.
package: Windows CPU, same scene-package/parser contract. copy-sync: only after
actual saved local UI proof; never updates latest, model registries or training.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001') if os.name=='nt' else Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
H=Path(__file__).resolve().parent
SUBJECT='officewoman-circle24-paired18k127'
ROOT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'/SUBJECT
SNAP=ROOT/'gaussian-run/step-001000'
OUT=ROOT/'delivery/step-001000'
SOURCE_SHA='b6fe8cf5fe68fbd129c45471f2642a6fa8036cceff4dd8bbc63e638cd4be9a29'
STEP,POINTS,UPDATES=1000,28901,999


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def audit():
    r=read(SNAP/'result.json');exit=read(ROOT/'process-exit.json');decision=read(ROOT/'review-1000-decision.json')
    failure=read(ROOT/'train-failure.json');rows=read(ROOT/'data/training-inputs.json')['records']
    assert r['step']==STEP and r['points']==POINTS and r['active_sh_degree']==0
    assert r['reference_frame']=='frame_000094' and rows[r['reference_slot']]['frame']==r['reference_frame']
    assert sha(SNAP/'world.ply')==r['world_ply_sha256']==SOURCE_SHA
    assert r['native_ply_parity']['mean_abs']<2e-6 and r['native_ply_parity']['max_abs']<.005
    assert exit['exit_code']==1 and 'Review timeout' in failure['error']
    assert decision['actual_visual_review'] and decision['decision']=='stop'
    assert decision['world_ply_sha256']==SOURCE_SHA and decision['quality_accepted'] is False
    assert decision['planned18000_completed'] is False and not decision['training_restarted']
    train=T.parent.parent/'repos/GaussianAvatars_SMPLX/train.py'
    text=train.read_text(encoding='utf-8');assert text.index('scene.save(iteration)')<text.index('gaussians.optimizer.step()')
    pinned=[ROOT/x for x in ['plan.json','progress.json','process-exit.json','train-failure.json','review-1000-decision.json','data/training-inputs.json','pixel-mapping.json']]
    pinned += [SNAP/x for x in ['result.json','world.ply','checkpoint.pt','current-meshes.npz']]
    return dict(updated=now(),subject=SUBJECT,step=STEP,points=POINTS,optimizer_updates=UPDATES,
        reference_frame=r['reference_frame'],source_hashes={str(p):sha(p) for p in pinned},
        native_source_ply_parity=r['native_ply_parity'],training_timed_out=True,planned18000_completed=False,
        training_restarted=False,optimizer_updates_added=0,diagnostic_only=True,quality_accepted=False,
        official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,company00_verified=False)


def no_training_process():
    # Read-only process checks, plus the existing exporter GPU lock. No signals.
    assert os.name!='nt','CUDA export must run in the registered WSL environment.'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,ProcessLookupError):continue
        if 'train_officewoman_circle24_sh0127.py train' in argv:
            raise RuntimeError('An officewoman training process is still alive; export will not stop/restart it.')
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    assert not q.stdout.strip(),'Existing GPU compute process: do not overlap diagnostic replay.'


def export():
    evidence=audit();no_training_process();assert not OUT.exists(),'Existing delivery: inspect; never overwrite or auto-retry.'
    import signal
    import inspect
    import difflib
    def timeout(*_):raise TimeoutError('Bounded120s diagnostic export timed out; no training resumed.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        import export_cross_subject_sh0124 as shared
        source=inspect.getsource(shared.main)
        old="    assert progress['state']=='bounded_control_completed',progress"
        new="    assert subject=='officewoman-circle24-paired18k127' and step==1000 and progress['step']==1000,progress"
        assert source.count(old)==1;adapted=source.replace(old,new)
        old="old.ROOT=root;row=rt.read(root/'data/training-inputs.json')['records'][0]"
        new="old.ROOT=root;row=rt.read(root/'data/training-inputs.json')['records'][rt.read(root/'gaussian-run/step-001000/result.json')['reference_slot']]"
        assert adapted.count(old)==1;adapted=adapted.replace(old,new)
        ast.parse(adapted);scope=dict(shared.__dict__)
        exec(compile(adapted,'<same-SH0-rigid-export-retained1000-diagnostic>','exec'),scope)
        scope['main'](SUBJECT,STEP)
        (OUT/'diagnostic-export-interface.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),adapted.splitlines(True))),encoding='utf-8')
        valid=read(OUT/'validation.json');assert valid['format_passed'] and valid['points']==POINTS
        assert valid['native_source_ply_parity']['passed'] and valid['native_rotated_render_parity']['passed']
        assert len(read(OUT/'cameras.json'))==1 and valid['reference_frame']=='frame_000094'
        valid.update(optimizer_updates=UPDATES,diagnostic_only=True,training_timed_out=True,planned18000_completed=False,
                     training_restarted=False,quality_accepted=False,viewer_ui_review='pending')
        save(OUT/'validation.json',valid)
        context=read(OUT/'context.json');context.update(label='办公室新人物 · 1000步检查点 · 999次更新 · 审核等待超时后的诊断',
            steps=STEP,optimizerUpdates=UPDATES,diagnosticOnly=True,qualityAccepted=False,
            plannedTrainingSteps=18000,plannedTrainingCompleted=False,trainingTimedOut=True,
            cameraScope='Only reference94 pose. The24 articulated source poses cannot be cameras of this one static PLY.')
        save(OUT/'context.json',context)
        assert all(sha(path)==digest for path,digest in evidence['source_hashes'].items())
        save(OUT/'diagnostic-contract.json',evidence)
        print('OFFICEWOMAN127_DIAGNOSTIC_NATIVE_EXPORT_PASSED',STEP,POINTS,UPDATES)
    except BaseException as error:
        if OUT.exists():save(OUT/'diagnostic-export-failure.json',dict(updated=now(),error=repr(error),training_restarted=False,auto_retry=False,quality_accepted=False))
        raise
    finally:signal.alarm(0)


def package():
    assert os.name=='nt','Use Windows Node for the existing local viewer package modules.'
    evidence=audit();v=read(OUT/'validation.json')
    assert v['format_passed'] and v['native_source_ply_parity']['passed'] and v['native_rotated_render_parity']['passed']
    assert v['step']==STEP and v['points']==POINTS and v['optimizer_updates']==UPDATES
    assert v['diagnostic_only'] and v['quality_accepted'] is False and v['planned18000_completed'] is False
    source=(H/'package_cross_subject_sh0124.mjs').read_text(encoding='utf-8')
    old="'football-half12-paired18k126']";assert source.count(old)==2
    adapted=source.replace(old,"'football-half12-paired18k126','officewoman-circle24-paired18k127']")
    marker='const zip=f.storedZip(files),entries=await m.unpackSceneZip(zip);';assert adapted.count(marker)==1
    note="\n诊断检查点：保存标记1000步，对应999次优化器更新、28901点。审核等待超时后进程退出1；计划18000步未完成，没有续训。只配reference94同姿态的一台完整K相机，PLY不是动态骨架。重建质量、公司00及官方SIBR/独立Spark GUI未验收。\n"
    insertion="files['README.txt']+="+json.dumps(note,ensure_ascii=False)+";\nfiles['diagnostic-contract.json']=JSON.stringify(await read('diagnostic-contract.json'),null,2);\n"
    adapted=adapted.replace(marker,insertion+marker)
    script=OUT/'package-diagnostic-interface.mjs';assert not script.exists();script.write_text(adapted,encoding='utf-8')
    node=shutil.which('node');assert node,'Windows Node required.'
    subprocess.run([node,str(script),SUBJECT,str(STEP)],check=True,timeout=60)
    result=read(OUT/'package-result.json');assert result['passed'] and result['cameraCount']==1 and result['referenceFrameId']=='frame_000094'
    assert sha(OUT/'scene.zip')==result['sha256'] and all(sha(p)==h for p,h in evidence['source_hashes'].items())
    result.update(step=STEP,optimizerUpdates=UPDATES,diagnosticOnly=True,trainingTimedOut=True,planned18000Completed=False,
                  qualityAccepted=False,localViewerUiVerified=False,officialSibrGuiVerified=False,independentSparkGuiVerified=False,company00Verified=False)
    save(OUT/'package-result.json',result)
    print('OFFICEWOMAN127_DIAGNOSTIC_PACKAGE_PARSER_PASSED_UI_PENDING',result['sha256'])


def copy_sync():
    assert os.name=='nt','Sync-folder copy runs on Windows.'
    audit();v=read(OUT/'validation.json');p=read(OUT/'package-result.json');ui=read(OUT/'viewer-ui-review.json')
    assert v['format_passed'] and v['native_rotated_render_parity']['passed'] and v['native_source_ply_parity']['passed']
    assert all(ui.get(k) is True for k in ('actual_ui_verified','preset_loaded','open_file_zip','gaussian_mode','point_mode','imported_camera_selected_in_UI'))
    assert ui['camera_count']==1 and ui['reference_frame']=='frame_000094'
    assert ui['sha256_binding_parser_verified'] and ui['derived_full_K_preserved']
    assert ui.get('quality_accepted') is False and not ui.get('company00_verified',False)
    assert ui['proofs'] and all(sha(Path(path))==ui['proof_sha256'][Path(path).name] for path in ui['proofs'])
    assert sha(OUT/'point_cloud.ply')==v['sha256'] and sha(OUT/'scene.zip')==p['sha256']
    target=Path('D:/00/工作/查看器/exports/officewoman_circle24_diagnostic_SH0_001000_with_camera.zip')
    target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists(),'Never overwrite another exported package.'
    shutil.copyfile(OUT/'scene.zip',target);assert sha(target)==p['sha256']
    save(OUT/'sync-copy-receipt.json',dict(updated=now(),path=str(target),sha256=sha(target),local_ui_verified=True,
        WPS_sync_verified=False,company00_opened=False,official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,
        latest_model_changed=False,training_restarted=False,quality_accepted=False))
    print('DIAGNOSTIC_PACKAGE_COPIED_TO_SYNC_FOLDER_NOT_00_VERIFIED',target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','export','package','copy-sync']);args=p.parse_args()
    if args.mode=='audit':print(json.dumps(audit(),ensure_ascii=False,indent=2))
    else:globals()[args.mode.replace('-','_')]()
