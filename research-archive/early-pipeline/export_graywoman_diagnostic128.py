"""Minimal128 specialization of the verified127 diagnostic export/package.

static-audit never reads the snapshot or initializes CUDA. Native export is an
explicit bounded call, after actual review; no preset/latest/sync publication.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import export_officewoman_diagnostic127 as previous

H=Path(__file__).resolve().parent
T=previous.T
SUBJECT='graywoman-half12-paired18k128'
ROOT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'/SUBJECT
SNAP=ROOT/'gaussian-run/step-001000'
OUT=ROOT/'delivery/step-001000'
STEP,POINTS,UPDATES=1000,34027,1000
SOURCE_SHA='5b6c00a693c9e3d59b4ef1cfbd8555b8e93de215f7562762486197cd2176d3b2'
PREVIOUS_SHA='e673db2a63a791f5f65c81763747580deafab7c6fde1af64ff6a8ba08aee10b7'
read,sha,save,now=previous.read,previous.sha,previous.save,previous.now


def local_path(value):
    text=str(value).replace('\\','/')
    if os.name=='nt' and text.startswith('/mnt/') and len(text)>7 and text[6]=='/':
        return Path(text[5].upper()+':/'+text[7:])
    if os.name!='nt' and len(text)>2 and text[1]==':':
        return Path('/mnt/'+text[0].lower()+'/'+text[2:].lstrip('/'))
    return Path(text)


def audit():
    """CPU evidence guard, used only after the actual1000 review exists."""
    r=read(SNAP/'result.json');progress=read(ROOT/'progress.json')
    exit_state=read(ROOT/'process-exit.json');request=read(ROOT/'review-1000-request.json')
    decision=read(ROOT/'review-1000-decision.json')
    rows=read(ROOT/'data/training-inputs.json')['records']
    assert r['step']==STEP and r['points']==POINTS and r['optimizer_updates']==UPDATES
    assert r['active_sh_degree']==0 and r['checkpoint_phase']=='after_density_adam_zero_grad'
    assert r['reference_frame']=='frame_000023' and rows[r['reference_slot']]['frame']==r['reference_frame']
    assert r['frames_rendered']==len(rows)==12
    assert sha(SNAP/'world.ply')==r['world_ply_sha256']==SOURCE_SHA
    assert r['native_ply_parity']['mean_abs']<2e-6 and r['native_ply_parity']['max_abs']<.005
    assert exit_state['exit_code']==0 and exit_state['state']=='checkpoint_ready_for_offline_review'
    assert progress['step']==1000 and progress['state']=='checkpoint_ready_for_offline_review'
    assert not (ROOT/'resume-launch.json').exists() and not (ROOT/'gaussian-run/step-018000').exists()
    assert decision['actual_visual_review'] is True and decision['decision'] in ('continue','stop')
    assert decision['request_sha256']==sha(ROOT/'review-1000-request.json')
    assert decision.get('quality_accepted',False) is False
    for field,path in [('world_ply_sha256',SNAP/'world.ply'),('snapshot_result_sha256',SNAP/'result.json'),
        ('native_checkpoint_sha256',ROOT/'gaussian-run/chkpnt1000.pth'),
        ('lifecycle_sidecar_sha256',ROOT/'gaussian-run/chkpnt1000.pth.lifecycle.pt'),
        ('lifecycle_manifest_sha256',ROOT/'gaussian-run/chkpnt1000.pth.lifecycle.json')]:
        assert decision[field]==request[field]==sha(path),field
    evidence_path=local_path(decision['review_evidence_path'])
    assert sha(evidence_path)==decision['review_evidence_sha256']
    review=read(evidence_path)
    assert review['actual_visual_review'] is True and review['step']==1000
    assert review.get('request_sha256')==sha(ROOT/'review-1000-request.json') or (
        review.get('native_checkpoint_sha256')==request['native_checkpoint_sha256'] and
        review.get('world_ply_sha256')==SOURCE_SHA)
    pinned=[ROOT/p for p in ['plan.json','progress.json','process-exit.json','review-1000-request.json',
        'review-1000-decision.json','compute-phase-train.json','data/training-inputs.json','pixel-mapping.json',
        'gaussian-run/chkpnt1000.pth','gaussian-run/chkpnt1000.pth.lifecycle.pt','gaussian-run/chkpnt1000.pth.lifecycle.json']]
    pinned += [SNAP/p for p in ['result.json','world.ply','checkpoint.pt','current-meshes.npz']]
    pinned += [evidence_path,Path(__file__),Path(previous.__file__)]
    return dict(updated=now(),subject=SUBJECT,record_id='015',step=STEP,points=POINTS,
        optimizer_updates=UPDATES,checkpoint_phase='after_density_adam_zero_grad',
        reference_frame=r['reference_frame'],source_hashes={str(p):sha(p) for p in pinned},
        native_source_ply_parity=r['native_ply_parity'],normal_post_update_offline_stop=True,
        training_timed_out=False,planned18000_completed=False,training_restarted=False,
        optimizer_updates_added=0,diagnostic_only=True,quality_accepted=False,
        known_pose_failure=True,
        known_pose_failure_note='Fresh fitted hidden hands were placed near front thighs instead of behind the body; source-side limb truth is not established. No opacity-only causal claim.',
        official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,company00_verified=False)


def no_training_process():
    assert os.name!='nt','Native replay requires the registered WSL CUDA environment.'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError,ProcessLookupError): continue
        if any(Path(arg).name=='train_graywoman_half12_sh0128.py' for arg in argv) and any(x in argv for x in ('train','resume')):
            raise RuntimeError('Current128 training/resume is alive; export will not stop or restart it.')
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True)
    assert not query.stdout.strip(),'GPU compute already active; do not overlap diagnostic replay.'


def _replace(source,old,new,count):
    assert source.count(old)==count,(old,source.count(old))
    return source.replace(old,new)


def compiled_export():
    assert sha(previous.__file__)==PREVIOUS_SHA
    source=inspect.getsource(previous.export)
    source=_replace(source,'officewoman-circle24-paired18k127',SUBJECT,1)
    source=_replace(source,"'frame_000094'","'frame_000023'",1)
    source=_replace(source,'training_timed_out=True','training_timed_out=False,known_pose_failure=True,normal_post_update_offline_stop=True',1)
    source=_replace(source,'trainingTimedOut=True','trainingTimedOut=False,knownPoseFailure=True,recordId=\'015\'',1)
    source=_replace(source,'办公室新人物 · 1000步检查点 · 999次更新 · 审核等待超时后的诊断',
                    '灰毛衣新人物 · 1000次实际更新 · 隐藏双手姿态错误的诊断',1)
    source=_replace(source,'Only reference94 pose. The24 articulated source poses cannot be cameras of this one static PLY.',
                    'Only reference23 pose. The12 articulated source poses cannot be cameras of this one static PLY.',1)
    source=_replace(source,'OFFICEWOMAN127_DIAGNOSTIC_NATIVE_EXPORT_PASSED','GRAYWOMAN128_DIAGNOSTIC_NATIVE_EXPORT_PASSED',1)
    namespace=dict(previous.__dict__,SUBJECT=SUBJECT,ROOT=ROOT,SNAP=SNAP,OUT=OUT,
        STEP=STEP,POINTS=POINTS,UPDATES=UPDATES,SOURCE_SHA=SOURCE_SHA,audit=audit,no_training_process=no_training_process)
    ast.parse(source);exec(compile(source,'<same127-native-export-new128-inputs>','exec'),namespace)
    return namespace['export'],source


def compiled_package():
    assert sha(previous.__file__)==PREVIOUS_SHA
    source=inspect.getsource(previous.package)
    source=_replace(source,'officewoman-circle24-paired18k127',SUBJECT,1)
    source=_replace(source,"'frame_000094'","'frame_000023'",1)
    source=_replace(source,'trainingTimedOut=True','trainingTimedOut=False,knownPoseFailure=True,recordId=\'015\'',1)
    source=_replace(source,'OFFICEWOMAN127_DIAGNOSTIC_PACKAGE_PARSER_PASSED_UI_PENDING',
                    'GRAYWOMAN128_DIAGNOSTIC_PACKAGE_PARSER_PASSED_UI_PENDING',1)
    tree=ast.parse(source)
    notes=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='note' for t in node.targets)]
    assert len(notes)==1
    notes[0].value=ast.Constant('\n诊断检查点：1000步为1000次实际优化器更新、34027点，原18k时钟未完成；保存完整post-update状态后正常退出，没有审核超时。只配reference23同姿态的一台完整K相机。隐藏双手拟合到大腿前等姿态问题尚未解决，不能作为透明度单变量因果证明。质量、00和官方SIBR/独立Spark GUI未验收。\n')
    ast.fix_missing_locations(tree);source=ast.unparse(tree)+'\n'
    namespace=dict(previous.__dict__,SUBJECT=SUBJECT,ROOT=ROOT,SNAP=SNAP,OUT=OUT,
        STEP=STEP,POINTS=POINTS,UPDATES=UPDATES,SOURCE_SHA=SOURCE_SHA,audit=audit)
    exec(compile(source,'<same127-scene-package-new128-inputs>','exec'),namespace)
    return namespace['package'],source


def static_package_source():
    # Evaluate only the exact package-string generation in the eventual entry;
    # there are no snapshot reads, writes, Node execution or CUDA imports here.
    _,source=compiled_package();function=ast.parse(source).body[0]
    start=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='source' for t in n.targets))
    end=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
    namespace=dict(previous.__dict__,H=H,SUBJECT=SUBJECT)
    exec(compile(ast.Module(body=function.body[start:end],type_ignores=[]),'<same-package-string-only>','exec'),namespace)
    return namespace['adapted']


def static_audit():
    _,export_source=compiled_export();_,package_source=compiled_package()
    javascript=static_package_source()
    assert SUBJECT in javascript and 'camera_input_mapping.json' in javascript
    assert 'scenePackage' in javascript and 'pairedCameras' in javascript and 'storedZip' in javascript
    assert not any(x in export_source for x in ('training_timed_out=True','trainingTimedOut=True','frame_000094'))
    node=shutil.which('node')
    if node:
        subprocess.run([node,'--input-type=module','--check'],input=javascript,text=True,check=True,timeout=30)
    result=dict(updated=now(),helper_sha256=sha(__file__),previous_export_helper_sha256=PREVIOUS_SHA,
        subject=SUBJECT,step=STEP,points_from_actual_root_checkpoint_report=POINTS,optimizer_updates=UPDATES,
        reference='frame_000023',record_id='015',native_export_AST_compiled=True,
        actual_package_generation_AST_compiled=True,package_Node_parse_checked=bool(node),
        snapshot_read=False,GPU_started=False,delivery_created=False,preset_or_latest_changed=False,
        known_pose_failure=True,planned18000_completed=False,quality_accepted=False,
        native_export_adapter_sha256=hashlib.sha256(export_source.encode()).hexdigest(),
        package_adapter_sha256=hashlib.sha256(package_source.encode()).hexdigest())
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['static-audit','audit','export','package'])
    mode=parser.parse_args().mode
    if mode=='static-audit': static_audit()
    elif mode=='audit': print(json.dumps(audit(),ensure_ascii=False,indent=2))
    elif mode=='export': compiled_export()[0]()
    else: compiled_package()[0]()
