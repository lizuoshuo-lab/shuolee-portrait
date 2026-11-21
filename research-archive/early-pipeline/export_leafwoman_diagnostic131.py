"""131 diagnostic-only specialization of the unchanged128 export/package.

static-audit compiles source/Node syntax only: no snapshot IO or CUDA.
Actual export requires the root's hash-bound1000 visual-review decision and
the stopped post-update process. No preset, latest, sync or00 publication.
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

import export_graywoman_diagnostic128 as previous

H = Path(__file__).resolve().parent
T = previous.T
SUBJECT = 'leafwoman-half12-dcbox-safe131'
ROOT = T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'/SUBJECT
FIT = ROOT.parent/'leafwoman-half12-pose129'
SNAP = ROOT/'gaussian-run/step-001000'
OUT = ROOT/'delivery/step-001000'
STEP, UPDATES, REFERENCE = 1000, 1000, 'frame_000004'
POINTS = 34105
SOURCE_SHA = '15f921862a36691fe2a75444275faa5758e1ba09a84ee417cad72cd8ea56e5ed'
TRAIN_DRIVER = H/'train_leafwoman_dcbox_safe131.py'
TRAIN_SHA = '5e8db46276792c86a7498afa58cc41848d2c5832575cba44d92433e238562608'
TRAIN_PLAN = H/'train_leafwoman_dcbox_safe131.plan.json'
TRAIN_PLAN_SHA = '5df8cba979639cab70000a8e00bd17b01abc2f1ff8a91faf33469fbc572ba75a'
DECISION_NAME = 'native-novel-1000-decision.json'
PREVIOUS_SHA = '759a312c085ef9537b654026d40fe6d43e2ff3e2e38890e4543f37592cb83421'
read, sha, save, now = previous.read, previous.sha, previous.save, previous.now
local_path = previous.local_path
assert sha(previous.__file__) == PREVIOUS_SHA


def audit():
    """CPU guard; intentionally not called by static-audit."""
    assert sha(TRAIN_DRIVER) == TRAIN_SHA and sha(TRAIN_PLAN) == TRAIN_PLAN_SHA
    result = read(SNAP/'result.json')
    progress, exit_state = read(ROOT/'progress.json'), read(ROOT/'process-exit.json')
    request, decision = read(ROOT/'review-1000-request.json'), read(ROOT/DECISION_NAME)
    rows = read(ROOT/'data/training-inputs.json')['records']
    points, source_sha = int(result['points']), sha(SNAP/'world.ply')
    assert result['step'] == STEP and points == POINTS and result['optimizer_updates'] == UPDATES
    assert result['active_sh_degree'] == 0 and result['checkpoint_phase'] == 'after_density_adam_zero_grad'
    assert result['reference_frame'] == REFERENCE == rows[result['reference_slot']]['frame']
    assert result['frames_rendered'] == len(rows) == 12
    assert source_sha == result['world_ply_sha256'] == SOURCE_SHA
    assert result['native_ply_parity']['mean_abs'] < 2e-6 and result['native_ply_parity']['max_abs'] < .005
    assert exit_state['exit_code'] == 0 and exit_state['state'] == 'checkpoint_ready_for_offline_review'
    assert progress['step'] == STEP and progress['state'] == 'checkpoint_ready_for_offline_review'
    assert not (ROOT/'resume-launch.json').exists() and not (ROOT/'gaussian-run/step-018000').exists()
    assert decision['actual_visual_review'] is True and decision['decision'] == 'diagnostic_export_only'
    assert decision['request_sha256'] == sha(ROOT/'review-1000-request.json')
    assert decision.get('quality_accepted', False) is False
    assert decision['compute_phase_sha256'] == sha(ROOT/'compute-phase-train.json')
    for field, path in [('world_ply_sha256',SNAP/'world.ply'), ('snapshot_result_sha256',SNAP/'result.json'),
        ('native_checkpoint_sha256',ROOT/'gaussian-run/chkpnt1000.pth'),
        ('lifecycle_sidecar_sha256',ROOT/'gaussian-run/chkpnt1000.pth.lifecycle.pt'),
        ('lifecycle_manifest_sha256',ROOT/'gaussian-run/chkpnt1000.pth.lifecycle.json')]:
        assert decision[field] == request[field] == sha(path), field
    evidence_path = local_path(decision['review_evidence_path'])
    assert sha(evidence_path) == decision['review_evidence_sha256']
    review = read(evidence_path)
    assert review['actual_visual_review'] is True and review['step'] == STEP
    assert review.get('request_sha256') == sha(ROOT/'review-1000-request.json') or (
        review.get('native_checkpoint_sha256') == request['native_checkpoint_sha256'] and
        review.get('world_ply_sha256') == source_sha)
    selection = read(FIT/'gs-selection/selection.json')
    subset = read(ROOT/'fit24-to-GS12-subset-receipt.json')
    fit_plan = read(FIT/'plan.json')
    assert len(fit_plan['rows']) == subset['original_fit_candidate_count'] == 24
    assert subset['selected_gaussian_count'] == len(selection['training_frames']) == 12
    assert set(selection['training_frames']) == {row['frame'] for row in rows}
    assert selection['reference_frame'] == subset['reference_frame'] == REFERENCE
    assert selection['fit_sha256'] == sha(FIT/'selected-fit.npz')
    assert selection['source_plan_sha256'] == sha(FIT/'plan.json')
    assert selection['input_source_plan_sha256'] == sha(local_path(fit_plan['source_selection']))
    assert subset['selection_sha256'] == sha(FIT/'gs-selection/selection.json')
    assert all(sha(local_path(path)) == digest for path,digest in subset['immutable_source_hashes'].items())
    pinned = [ROOT/path for path in ['plan.json','progress.json','process-exit.json','review-1000-request.json',
        DECISION_NAME,'compute-phase-train.json','data/training-inputs.json','pixel-mapping.json',
        'fit24-to-GS12-subset-receipt.json','gaussian-run/chkpnt1000.pth',
        'gaussian-run/chkpnt1000.pth.lifecycle.pt','gaussian-run/chkpnt1000.pth.lifecycle.json']]
    pinned += [SNAP/path for path in ['result.json','world.ply','checkpoint.pt','current-meshes.npz']]
    pinned += [local_path(path) for path in subset['immutable_source_hashes']]
    pinned += [evidence_path, Path(__file__), Path(previous.__file__), TRAIN_DRIVER, TRAIN_PLAN, ROOT/'dc-projection-1000-receipt.json']
    return dict(updated=now(),subject=SUBJECT,record_id='022',step=STEP,points=points,
        optimizer_updates=UPDATES,reference_frame=REFERENCE,source_world_ply_sha256=source_sha,
        source_hashes={str(path):sha(path) for path in pinned},
        native_source_ply_parity=result['native_ply_parity'],normal_post_update_offline_stop=True,
        source_fit_candidate_count=24,selected_Gaussian_input_count=12,portable_static_camera_count=1,
        camera_scope='Only reference4 pose. Fit24 and GS12 articulated inputs are not12 cameras of one static PLY.',
        training_timed_out=False,planned18000_completed=False,training_restarted=False,
        known_pose_failure=review.get('known_pose_failure'),optimizer_updates_added=0,
        diagnostic_only=True,quality_accepted=False,official_SIBR_GUI_verified=False,
        independent_Spark_GUI_verified=False,company00_verified=False)


def no_training_process():
    assert os.name != 'nt', 'Native replay uses the registered WSL CUDA environment.'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: argv = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError,ProcessLookupError): continue
        if any(Path(arg).name == 'train_leafwoman_dcbox_safe131.py' for arg in argv) and any(x in argv for x in ('train','resume')):
            raise RuntimeError('Current131 training is alive; export does not stop or restart it.')
    query = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,check=True)
    assert not query.stdout.strip(), 'GPU compute already active; do not overlap diagnostic replay.'


def _replace(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old), count)
    return source.replace(old,new)


def _namespace(original_function, identity):
    # Placeholders permit pure source compilation. CLI actual export/package
    # always supplies a completed audit, rather than guessing snapshot identity.
    value = identity or dict(points=0,source_world_ply_sha256='pending_actual_review')
    return dict(original_function.__globals__,SUBJECT=SUBJECT,ROOT=ROOT,SNAP=SNAP,OUT=OUT,
        STEP=STEP,POINTS=value['points'],UPDATES=UPDATES,SOURCE_SHA=value['source_world_ply_sha256'],
        audit=audit,no_training_process=no_training_process)


def compiled_export(identity=None):
    function, source = previous.compiled_export()
    source = _replace(source,'graywoman-half12-paired18k128',SUBJECT)
    source = _replace(source,"'frame_000023'",repr(REFERENCE))
    source = _replace(source,'known_pose_failure=True',"known_pose_failure=evidence.get('known_pose_failure')")
    source = _replace(source,"knownPoseFailure=True,recordId='015'",
        "knownPoseFailure=evidence.get('known_pose_failure'),recordId='022'")
    source = _replace(source,'灰毛衣新人物 · 1000次实际更新 · 隐藏双手姿态错误的诊断',
        '持叶DC安全边界131 · 1000次实际更新 · 未知视角诊断检查点')
    source = _replace(source,'Only reference23 pose. The12 articulated source poses cannot be cameras of this one static PLY.',
        'Only reference4 pose. Fit24 and GS12 articulated inputs cannot be12 cameras of this one static PLY.')
    source = _replace(source,'GRAYWOMAN128_DIAGNOSTIC_NATIVE_EXPORT_PASSED','LEAFWOMAN131_DIAGNOSTIC_NATIVE_EXPORT_PASSED')
    ast.parse(source)
    namespace = _namespace(function,identity)
    exec(compile(source,'<same128-native-export-new131-reference4>','exec'),namespace)
    return namespace['export'],source


def compiled_package(identity=None):
    function, source = previous.compiled_package()
    source = _replace(source,'graywoman-half12-paired18k128',SUBJECT)
    source = _replace(source,"'frame_000023'",repr(REFERENCE))
    source = _replace(source,"knownPoseFailure=True, recordId='015'",
        "knownPoseFailure=evidence.get('known_pose_failure'), recordId='022'")
    source = _replace(source,'GRAYWOMAN128_DIAGNOSTIC_PACKAGE_PARSER_PASSED_UI_PENDING',
        'LEAFWOMAN131_DIAGNOSTIC_PACKAGE_PARSER_PASSED_UI_PENDING')
    tree = ast.parse(source)
    notes = [node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='note' for target in node.targets)]
    assert len(notes) == 1
    notes[0].value = ast.Constant('\n诊断检查点：1000次实际优化器更新，原18k时钟未完成；保存完整post-update状态后正常退出。'
        '原官方拟合24候选与冻结12图未改；本候选仅采用原生数值安全DC区间，只配reference4同姿态的一台完整K相机。'
        '这份静态PLY不能把12个身体姿态当作同一几何的12台相机；不含可驱动骨架。'
        '裁掉的腿脚与遮挡部位保持未知；质量、00及官方SIBR/独立Spark GUI未验收。\n')
    ast.fix_missing_locations(tree);source = ast.unparse(tree)+'\n'
    namespace = _namespace(function,identity)
    exec(compile(source,'<same128-scene-package-new131-reference4>','exec'),namespace)
    return namespace['package'],source


def static_package_source():
    _, source = compiled_package()
    function = ast.parse(source).body[0]
    first = next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='source' for target in node.targets))
    last = next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='script' for target in node.targets))
    namespace = dict(previous.previous.__dict__,H=H,SUBJECT=SUBJECT)
    exec(compile(ast.Module(body=function.body[first:last],type_ignores=[]),'<package-string-generation-only-no-snapshot>', 'exec'),namespace)
    return namespace['adapted']


def static_audit():
    _, export_source = compiled_export();_, package_source = compiled_package()
    javascript = static_package_source()
    assert SUBJECT in javascript and 'camera_input_mapping.json' in javascript
    assert all(marker in javascript for marker in ('scenePackage','pairedCameras','storedZip'))
    assert all(marker not in export_source for marker in ('frame_000023','frame_000094','known_pose_failure=True'))
    assert REFERENCE in export_source and REFERENCE in package_source
    node = shutil.which('node')
    if node:
        subprocess.run([node,'--input-type=module','--check'],input=javascript,text=True,check=True,timeout=30)
    result = dict(updated=now(),helper_sha256=sha(__file__),previous128_export_helper_sha256=PREVIOUS_SHA,
        subject=SUBJECT,step=STEP,optimizer_updates=UPDATES,reference_frame=REFERENCE,record_id='022',
        fit_candidate_count=24,Gaussian_input_count=12,portable_static_camera_count=1,
        points_and_world_sha='Bound to actual reviewed snapshot at runtime; not read by static-audit.',
        native_export_AST_compiled=True,actual_package_generation_AST_compiled=True,
        package_Node_parse_checked=bool(node),snapshot_read=False,GPU_started=False,
        delivery_created=False,preset_or_latest_changed=False,diagnostic_only=True,
        planned18000_completed=False,quality_accepted=False,company00_verified=False,
        native_export_adapter_sha256=hashlib.sha256(export_source.encode()).hexdigest(),
        package_adapter_sha256=hashlib.sha256(package_source.encode()).hexdigest())
    save(H/'export_leafwoman_diagnostic131.static-audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['static-audit','audit','export','package'])
    mode = parser.parse_args().mode
    if mode == 'static-audit':
        static_audit()
    elif mode == 'audit':
        print(json.dumps(audit(),ensure_ascii=False,indent=2))
    elif mode == 'export':
        compiled_export(audit())[0]()
    else:
        compiled_package(audit())[0]()
