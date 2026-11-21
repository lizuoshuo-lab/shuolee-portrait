"""129 terminal18000 diagnostic delivery:17999 actual Adam updates.

Original Scene.save runs before the terminal optimizer branch and this adapter
stops there. This snapshot is not a post-update18000 resumable checkpoint.
static-audit reads code only; actual export needs a separate final review gate,
successful resume exit and an idle GPU. No preset/latest/sync publication.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import export_leafwoman_diagnostic129 as previous

H = Path(__file__).resolve().parent
T, SUBJECT, ROOT, FIT = previous.T, previous.SUBJECT, previous.ROOT, previous.FIT
STEP, UPDATES, REFERENCE = 18000, 17999, 'frame_000004'
SNAP = ROOT/'gaussian-run/step-018000'
OUT = ROOT/'delivery/step-018000'
FINAL_GATE = ROOT/'review-18000-decision.json'
PREVIOUS_SHA = 'a3514eaff75e36d5ea7c6397f26ed8a764141fbd6937e4c9b5a55b951081fc68'
read, sha, save, now = previous.read, previous.sha, previous.save, previous.now
local_path, no_training_process = previous.local_path, previous.no_training_process
assert sha(previous.__file__) == PREVIOUS_SHA


def audit():
    """Final CPU gate. Never use the1000 review as final quality approval."""
    result = read(SNAP/'result.json')
    run = read(ROOT/'gaussian-run/result.json')
    progress = read(ROOT/'progress.json')
    final_exit = read(ROOT/'final-process-exit.json')
    accounting = read(ROOT/'compute-phase-resume.json')
    decision = read(FINAL_GATE)
    rows = read(ROOT/'data/training-inputs.json')['records']
    points, world_sha = int(result['points']), sha(SNAP/'world.ply')
    assert result['step'] == STEP and points > 0 and result['active_sh_degree'] == 0
    assert result['reference_frame'] == REFERENCE == rows[result['reference_slot']]['frame']
    assert result['frames_rendered'] == len(rows) == 12 and result['world_ply_sha256'] == world_sha
    assert result['native_ply_parity']['mean_abs'] < 2e-6 and result['native_ply_parity']['max_abs'] < .005
    assert run['state'] == progress['state'] == 'bounded_control_completed'
    assert run['snapshot_step'] == progress['step'] == run['configured_schedule_steps'] == STEP
    assert run['optimizer_updates'] == UPDATES and run['points'] == points
    assert final_exit['exit_code'] == 0 and final_exit['phase'] == 'resume'
    assert final_exit['step'] == STEP and final_exit['optimizer_updates'] == UPDATES
    assert accounting['mode'] == 'resume' and accounting['successful'] is True
    assert accounting['state'] == 'bounded_control_completed'
    assert 0 < accounting['prior_compute_seconds'] < accounting['cumulative_compute_seconds'] < 2400
    assert read(ROOT/'gaussian-run/resume-state-verification.json')['actual_first_resumed_iteration'] == 1001
    assert decision['actual_visual_review'] is True and decision['decision'] == 'export_diagnostic'
    assert decision['step'] == STEP and decision['optimizer_updates'] == UPDATES
    assert decision.get('quality_accepted', False) is False
    for field, path in [('world_ply_sha256',SNAP/'world.ply'),('snapshot_result_sha256',SNAP/'result.json'),
        ('final_process_exit_sha256',ROOT/'final-process-exit.json')]:
        assert decision[field] == sha(path), field
    evidence_path = local_path(decision['review_evidence_path'])
    assert sha(evidence_path) == decision['review_evidence_sha256']
    review = read(evidence_path)
    assert review['actual_visual_review'] is True and review['step'] == STEP
    assert review['world_ply_sha256'] == world_sha
    assert review['snapshot_result_sha256'] == sha(SNAP/'result.json')
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
    pinned = [ROOT/path for path in ['plan.json','progress.json','resume-launch.json','final-process-exit.json',
        'compute-phase-resume.json','data/training-inputs.json','pixel-mapping.json','fit24-to-GS12-subset-receipt.json',
        'gaussian-run/result.json','gaussian-run/resume-state-verification.json','adapter.py',
        'gaussian-run/upstream-training-function.py']]
    pinned += [SNAP/path for path in ['result.json','world.ply','checkpoint.pt','current-meshes.npz','sampling.json','background-receipt.json']]
    pinned += [local_path(path) for path in subset['immutable_source_hashes']]
    pinned += [FINAL_GATE,evidence_path,Path(__file__),Path(previous.__file__)]
    return dict(updated=now(),subject=SUBJECT,record_id='022',step=STEP,points=points,
        optimizer_updates=UPDATES,reference_frame=REFERENCE,source_world_ply_sha256=world_sha,
        source_hashes={str(path):sha(path) for path in pinned},
        native_source_ply_parity=result['native_ply_parity'],source_fit_candidate_count=24,
        selected_Gaussian_input_count=12,portable_static_camera_count=1,
        snapshot_phase='Scene.save before terminal optimizer branch;17999 completed updates',
        terminal_snapshot_pre_update=True,post_update18000_complete_checkpoint=False,
        normal_post_update_offline_stop=False,training_timed_out=False,planned18000_completed=True,
        same_experiment_resume_completed=True,training_restarted=False,optimizer_updates_added=0,
        known_pose_failure=review.get('known_pose_failure'),diagnostic_only=True,quality_accepted=False,
        final_actual_review_gate=True,official_SIBR_GUI_verified=False,independent_Spark_GUI_verified=False,
        company00_verified=False)


def _replace(source, old, new, count=1):
    assert source.count(old) == count, (old,source.count(old),count)
    return source.replace(old,new)


def _namespace(function, identity):
    value = identity or dict(points=0,source_world_ply_sha256='pending_final_review')
    return dict(function.__globals__,ROOT=ROOT,SNAP=SNAP,OUT=OUT,STEP=STEP,UPDATES=UPDATES,
        POINTS=value['points'],SOURCE_SHA=value['source_world_ply_sha256'],audit=audit,
        no_training_process=no_training_process)


def compiled_export(identity=None):
    function, source = previous.compiled_export(identity)
    source = _replace(source,"and step==1000 and progress['step']==1000", "and step==18000 and progress['step']==18000")
    source = _replace(source,'gaussian-run/step-001000/result.json','gaussian-run/step-018000/result.json')
    source = _replace(source,'normal_post_update_offline_stop=True','normal_post_update_offline_stop=False,terminal_snapshot_pre_update=True')
    source = _replace(source,'planned18000_completed=False','planned18000_completed=True')
    source = _replace(source,'plannedTrainingCompleted=False','plannedTrainingCompleted=True')
    source = _replace(source,'持叶新拍摄 · 1000次实际更新 · fit24到GS12的诊断检查点',
        '持叶新拍摄 · 18000步终点 · 17999次实际更新 · fit24到GS12')
    source = _replace(source,'LEAFWOMAN129_DIAGNOSTIC_NATIVE_EXPORT_PASSED','LEAFWOMAN129_FINAL_NATIVE_EXPORT_PASSED')
    ast.parse(source)
    namespace = _namespace(function,identity)
    exec(compile(source,'<unchanged-native129-terminal18000-export>','exec'),namespace)
    return namespace['export'],source


def compiled_package(identity=None):
    function, source = previous.compiled_package(identity)
    source = _replace(source,"v['planned18000_completed'] is False","v['planned18000_completed'] is True")
    source = _replace(source,'planned18000Completed=False','planned18000Completed=True')
    source = _replace(source,'LEAFWOMAN129_DIAGNOSTIC_PACKAGE_PARSER_PASSED_UI_PENDING',
        'LEAFWOMAN129_FINAL_PACKAGE_PARSER_PASSED_UI_PENDING')
    tree = ast.parse(source)
    notes = [node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='note' for target in node.targets)]
    assert len(notes) == 1
    notes[0].value = ast.Constant('\n18000步终点对应17999次实际Adam更新。原Scene.save先于终轮优化器分支；'
        '此终点快照不是post-update18000完整可恢复检查点。1000实际更新完整检查点离线审阅后，同实验续至原18k终点，resume进程exit0。'
        '原官方拟合24候选，冻结筛选12图训练高斯；只配reference4同姿态的一台完整K训练裁片相机，不能把12个身体姿态当作同一静态PLY的12台相机。'
        '本包不含可驱动骨架；裁掉腿脚及遮挡部位保持未知。最终导出门控与重建质量验收分开；质量、00及官方SIBR/独立Spark GUI未验收。\n')
    ast.fix_missing_locations(tree);source = ast.unparse(tree)+'\n'
    namespace = _namespace(function,identity)
    exec(compile(source,'<unchanged129-scene-package-terminal18000>','exec'),namespace)
    return namespace['package'],source


def static_package_source():
    function, source = compiled_package()
    body = ast.parse(source).body[0].body
    first = next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='source' for target in node.targets))
    last = next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='script' for target in node.targets))
    namespace = dict(function.__globals__)
    exec(compile(ast.Module(body=body[first:last],type_ignores=[]),'<terminal-package-string-only-no-snapshot>','exec'),namespace)
    return namespace['adapted']


def static_audit():
    _, export_source = compiled_export();_, package_source = compiled_package()
    javascript = static_package_source()
    assert 'step-001000' not in export_source and 'normal_post_update_offline_stop=True' not in export_source
    assert 'step-018000' in export_source and 'planned18000_completed=True' in export_source
    assert REFERENCE in export_source and REFERENCE in package_source
    assert all(marker in javascript for marker in ('scenePackage','pairedCameras','storedZip','camera_input_mapping.json'))
    node = shutil.which('node')
    if node:
        subprocess.run([node,'--input-type=module','--check'],input=javascript,text=True,check=True,timeout=30)
    result = dict(updated=now(),helper_sha256=sha(__file__),frozen1000_export_helper_sha256=PREVIOUS_SHA,
        subject=SUBJECT,step=STEP,optimizer_updates=UPDATES,reference_frame=REFERENCE,record_id='022',
        snapshot_phase='pre-update original terminal Scene.save',post_update18000_checkpoint_claim=False,
        final_gate_path=str(FINAL_GATE),separate_final_actual_review_required=True,
        native_export_AST_compiled=True,package_Node_parse_checked=bool(node),
        source_fit_candidate_count=24,Gaussian_input_count=12,portable_static_camera_count=1,
        snapshot_or_final_gate_read=False,GPU_started=False,delivery_created=False,
        preset_or_latest_changed=False,quality_accepted=False,company00_verified=False,
        native_export_adapter_sha256=hashlib.sha256(export_source.encode()).hexdigest(),
        package_adapter_sha256=hashlib.sha256(package_source.encode()).hexdigest())
    save(H/'export_leafwoman_final129.static-audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['static-audit','audit','export','package'])
    mode = parser.parse_args().mode
    if mode == 'static-audit': static_audit()
    elif mode == 'audit': print(json.dumps(audit(),ensure_ascii=False,indent=2))
    elif mode == 'export': compiled_export(audit())[0]()
    else: compiled_package(audit())[0]()
