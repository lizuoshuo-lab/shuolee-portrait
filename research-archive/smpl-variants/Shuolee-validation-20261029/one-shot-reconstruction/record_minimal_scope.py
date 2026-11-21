from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

project = Path('D:/Projects/Shuolee')
evidence = Path('D:/Projects/Shuolee-validation-20261029/one-shot-reconstruction')
status_path = Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
status = json.loads(status_path.read_text(encoding='utf-8-sig'))
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
entry = json.loads((evidence / 'real-entry-validation.json').read_text(encoding='utf-8'))
record = dict(updated=datetime.now(timezone.utc).isoformat(), state='minimal_official_interface_pending',
    project=str(project), branch='smplx/reconstruction', current_base_commit=commit,
    new_changes_committed=False, new_branch_published=False,
    scope='Necessary 2D observation/confidence, full K/coordinates, SMPL-X output to existing free-clothing Gaussian interfaces only.',
    decision_policy='Solver, loss, stage, precision, initialization or shared-identity strategy changes require advance explanation of benefit, negative impact, evidence and rollback plus direct user confirmation.',
    experimental_solver_preserved=True, experimental_solver_reachable_from_default_cli=False,
    old_static_fallback=False, original_results_preserved=True,
    latest_verified_user_messages=['01a1883b-22b2-7562-ab45-f25f26ff9ea5',
        '01a1883b-7fb5-7211-b958-93077987df1a', '01a1883e-a3c7-7af0-a70d-99b62c3c1664'],
    actual_local_execution=dict(videos=['video3.mp4', 'video4.mp4'],
        orientations=['portrait', 'landscape'], source_probe_and_first_frame_decode=True,
        first_error='CAMERA_PIXEL_DOMAIN_UNRESOLVED',
        camera_config='Explicit diagnostic record of unresolved export mapping from available FISHEYE_8 calibration; no fabricated rectified K.',
        full_preprocessing=False, body_fit=False, gaussian_training=False,
        evidence=str(evidence / 'real-entry-validation.json')),
    default_for_resolved_camera='OFFICIAL_SMPLIFYX_CONNECTION_PENDING',
    local_default_routing_checks=dict(tests=8, passed=True, GPU=False),
    source_integration_checks='Fresh original prior16; sequence inputs/streaming46; bundle15; native components37 CPU checks. These do not validate real training.',
    full_regression_run_after_scope_change=False, ci_run_for_new_changes=False,
    remaining=['Official SMPLify-X minimal input/output integration and approved shared-identity semantics',
        'Automatic noisy face observations distinct from independent evaluation truth',
        'Camera output pixel mapping/common scale provenance',
        'Automatic external-occlusion validity and pinned native backend deployment'],
    server_verified=False, end_to_end_real_training_verified=False, quality_accepted=False)
record.update(
    priority='Complete one real official SMPLify-X to GaussianAvatars_SMPLX to standard PLY/cameras/viewer round first; acceleration follows.',
    timing=dict(scope='source video input through exported model', target_seconds=900,
                abort_first_round_at_target=False, actual_total_seconds=None),
    shared_identity='One updatable shared beta for the whole sequence, initialized from frame32. If updated from cross-view evidence, refit poses consistently; no permanent one-frame lock or independent final per-frame betas.',
    official_entry_readonly_audit=dict(
        smplifyx='Official loss/monitor reusable; local archived package lacks unconditional imported mesh_viewer/lbfgs_ls. Old VPoser API and default camera/initialization resets require thin compatibility hooks. Explicit matching stage configuration is required.',
        gaussian_avatars='Base local e44c1a6 lacks README-listed train_smplx.py/model; sibling files are untracked local adapters, not a complete public author entry. Research owner confirms this was known. Latest direction reuses the existing mesh-bound Gaussian training loop with named-frame mesh selection; trainable local offsets can leave the body surface. No new native training loop is selected.',
        full_round_started_by_company00=False),
    body_estimation='SMPLify-X replaces GVHMR. Default official route must not execute fresh or cached GVHMR. VPoser2 remains the official pose prior.',
    minimal_connection='Mask and 2D/full-K inputs -> official SMPLify-X -> existing local GaussianAvatars SMPL-X training adapter with named-frame mesh selection -> selected-pose standard world PLY and camera export. Await actual official output from research owner; no new solver or training loop.',
    remaining=['Complete pinned official fit_single_frame package and thin full-K/VPoser/shared-beta input-output hooks without GVHMR',
               'Bind actual official completion output to existing mesh-bound training adapter and named-frame geometry',
               'Complete real training and bake the reference frame to standard PLY plus full camera export and viewer'])
record['latest_verified_user_messages'] += ['01a18843-4500-71c3-9fc6-cc98ceab23ec', '01a18848-4aed-7710-86ef-05f6bd7d873a']
record['latest_verified_user_messages'] += ['01a1884e-4efd-7983-aaa5-05b1f7a5e722', '01a18850-9b35-7a81-b1a1-5b2f0eeaa269']
status['one_shot_reconstruction'] = record
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(evidence / 'minimal-scope-status.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
page = Path('D:/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图/修复进展.html')
html = page.read_text(encoding='utf-8')
section = '''<section id="minimal-interface-current"><h2>当前工程状态（2026-10-29）</h2><p>原图仍按帧单独保存。按最新要求，工程只推进二维观测、完整相机输入和 SMPL-X 输出到现有高斯链的必要接口；此前全序列求解代码和结果保留，默认入口已禁止自动启动该实验求解器。本轮本地横、竖屏视频都已完成格式检查和首帧解码；现有鱼眼标定缺少导出视频处理映射，未将其冒充已确认的去畸变相机。官方 SMPLify-X 最小接口仍待接通，尚无新版本完整训练和质量通过结论。本轮未提交或发布这些实验改动，也未启动新的 GPU 训练。</p></section>'''
section = '''<section id="minimal-interface-current"><h2>当前工程状态（2026-10-30）</h2><p>六帧原图与对应完整叠加图已分帧整理。当前主流程收敛为：视频与 mask、二维关键点和相机 → 官方 SMPLify-X → 现有 SMPL-X 高斯训练适配 → 指定姿态的通用 PLY 和相机。GVHMR 不再作为主流程初始化。共享体型由参考帧初始化，允许根据跨视角结果更新；保留官方求解阶段及已有高斯训练方法，只接必要输入输出。</p><p>首轮先完成可查看结果并记录实际总耗时，15 分钟目标在完整流程后优化。研究端正在运行官方拟合，00 等待真实输出接入；本地分支尚未发布，旧部署检查仍有 GVHMR 依赖需同步清除。不能将组件检查、输入解码或拟合启动视为完整训练通过。历史实验和原始结果均保留。</p></section>'''
import re
if 'id="minimal-interface-current"' in html:
    html = re.sub(r'<section id="minimal-interface-current">.*?</section>', section, html, flags=re.S)
else:
    html = html.replace('</html>', section + '</html>')
page.write_text(html, encoding='utf-8')
# This helper was prepared but never executed; its cancelled test sweep is not retained as a deliverable.
unused = evidence / 'run_cpu_checks.py'
if unused.is_file():
    unused.unlink()
print(json.dumps(dict(status_updated=True, branch=record['branch'], commit=commit,
                     new_branch_published=False, experimental_default_disabled=True)))
