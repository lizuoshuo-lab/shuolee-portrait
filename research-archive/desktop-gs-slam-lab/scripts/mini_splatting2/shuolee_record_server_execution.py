"""Reconcile observed remote deployment and completed local experiments."""
import json
from datetime import datetime, timezone
from pathlib import Path

HOME = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
now = datetime.now(timezone.utc).isoformat()
status_path = HOME / 'work-status.json'
status = json.loads(status_path.read_text(encoding='utf-8'))
assessment_path = HOME / 'current/company-problem-audit/retention-pilot/assessment.json'
assessment = json.loads(assessment_path.read_text(encoding='utf-8'))
server = {
    'observed_at': now,
    'route': 'local ToDesk -> 00 WinSCP Company-Training -> Linux',
    'workspace': '/home/zhouhaoyi/shuolee',
    'source_and_input_archive_sha256_verified': True,
    'source_video_sha256_verified': True,
    'isolated_environment': '/home/zhouhaoyi/shuolee/envs/shuolee-minigs-001',
    'parent_torch': '1.10.0+cu113',
    'nvcc': '11.4',
    'cuda_minor_policy': 'same major explicitly allowed; synthetic kernels and gradients passed',
    'extensions_compiled': 4,
    'kernel_smoke': 'passed',
    'kernel_run': '/home/zhouhaoyi/shuolee/runs/shuolee-smoke-20261019-191819',
    'image_decoder': 'opencv-python-headless 4.9.0.80 installed in isolated environment',
    'image_rebuild': {'status': 'started_not_yet_verified', 'pid': 1916, 'expected_images': 59},
    'real_data_smoke': 'not_started',
    'controlled_training': 'not_started',
    'quality_accepted': False,
    'verification_source': 'visible WinSCP Linux console output; remote logs not yet downloaded',
}
(HOME / 'current/company-server/execution-status.json').write_text(json.dumps(server, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
status.update(updated=now, current_phase='Local retention gate rejected; server kernels passed and images rebuilding; head-observation diagnosis continuing', server_training_started=False, local_training_running=False)
status['parallel_work_after_access'] = {
    'local': 'Completed paired 4200 retention gate; candidate not promoted. Continue independent head-motion observability diagnosis.',
    'company_server': 'Four isolated CUDA extensions compiled and synthetic gradient/optimizer tests passed. Rebuilding 59 input images before bounded real-data smoke.',
}
status['latest_server_execution'] = 'current/company-server/execution-status.json'
status['diagnostics']['retention_pilot'] = 'current/company-problem-audit/retention-pilot/assessment.json'
status['diagnostics']['first_actual_full_resolution_iteration'] = 3010
status['next_bounded_experiment'] = 'Same common 3009 state, identical points and budget; refresh culling only at first actual full-resolution view; stop at 3020 then 4200 after integrity checks.'
status['latest_retention_decision'] = {
    'status': assessment['status'], 'accepted_as_best': False,
    'iteration': 4200, 'gaussians_per_arm': 459873,
    'head_psnr_delta_db': assessment['evidence']['development_head_psnr_mean_delta_db'],
    'continuous_video_human_acceptance': False,
    'continue_to_18000': False,
    'evidence': 'current/company-problem-audit/retention-pilot/assessment.json',
}
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
ledger_path = HOME / 'results-ledger.json'
ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
entry = dict(id='shuolee-roi-retention-4200-20261019', created=now,
    method='Fixed-budget training-only head contribution swaps at first simplification',
    status='not_promoted', iterations=4200, original_schedule=18000,
    gaussians_per_arm=459873, swapped_points=22993,
    development_head_psnr_delta_db=-0.0042853,
    result='Immediate simplification damage reduced, but the gain did not persist through the 4200 gate; no stable head clarity improvement in real development views or fixed-path keyframes.',
    integrity='Same common decision, initial parameters, camera sequence and RNG streams verified.',
    limitations='Single shared decision and source scene; repeatedly used development cameras; novel paths have no photographic GT; full continuous human review not complete.',
    decision='Stop this simple 5 percent retention candidate; do not tune the exchange percentage on this development set.',
    model_deleted=False, evidence='current/company-problem-audit/retention-pilot/assessment.json')
ledger['decisions'] = [x for x in ledger['decisions'] if x.get('id') != entry['id']] + [entry]
ledger['updated'] = now
ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
plan_path = HOME / 'long-term-plan.md'
plan = plan_path.read_text(encoding='utf-8')
plan = plan.replace('原视频已从00直接传到服务器，源码和冻结输入正在同步。服务器训练仍未开始，必须先通过环境、输入与 CUDA 梯度检查。', '原视频已从00直接传到服务器，源码和冻结输入压缩包均已同步并通过SHA-256核对。四个CUDA扩展已在独立环境编译完成，合成渲染、反向传播与优化器检查通过。正在从同一视频重建59张图并逐张比对像素；真实训练尚未开始，须先完成图像一致性与短程真实数据验证。')
old = '当前本地开展等高斯预算的第一次简化保留对照，从同一2,999步状态恢复，原18,000步调度保留、先运行到4,200步。'
new = '等高斯预算的第一次简化保留对照已完成：从同一2,999步状态恢复，原18,000步调度保留，两臂均运行到4,200步。'
assert old in plan or new in plan
plan = plan.replace(old, new)
anchor = '服务器准备独立检查分辨率切换后的可见性掩码刷新，先核实实际训练视图栈的尺度切换时机；不把节点审计的全分辨率探针当作真实训练输入时机。'
replacement = '4,200步验收中，两臂均为459,873个高斯。头部优先保留使简化瞬间的三个训练视图PSNR提高约0.81／1.58／1.76 dB，但后续差异缩小为约0.01／0.18／−0.02 dB；8个开发视图头部均值变化约−0.004 dB。真实裁片和俯仰、距离路径关键帧没有稳定清晰度收益。该简单5%保留策略不晋级、不继续18,000步，也不利用同一开发集调换比例。完整连续视频的人类验收仍未完成。\n\n实际训练相机栈从第3,010步才首次使用全分辨率；第3,000—3,009步仍消费旧的半分辨率栈，诊断探针分辨率不代表训练输入。服务器下一项对照只在首次真实全分辨率训练前刷新可见性掩码，保留集合、总点数、权重因子、损失与Mip不变。服务器先运行一次公共前缀，完整保存3,009步状态再分叉两臂，避免独立原子累加带来的初始差异；先到3,020步查完整性，再决定到4,200步，不自动长训。'
assert anchor in plan or replacement in plan
plan = plan.replace(anchor, replacement)
plan_path.write_text(plan, encoding='utf-8')
print(json.dumps({'updated': now, 'files': ['work-status.json', 'results-ledger.json', 'long-term-plan.md', 'current/company-server/execution-status.json'], 'server_real_training': False, 'new_best_model': False}))
