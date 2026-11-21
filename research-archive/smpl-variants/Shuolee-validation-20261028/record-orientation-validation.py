"""Record completed, independently checked real-video validation; no training edits."""
from pathlib import Path
from datetime import datetime, timezone
import json

repo = Path('/mnt/d/Projects/Shuolee')
art = Path('/mnt/d/Projects/Shuolee-validation-20261028')
coord = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-coordination')
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

receipt = read(art / 'orientation-validation.json')
independent = read(art / 'orientation-independent-exports.json')
assert receipt['state'] == independent['state'] == 'passed'
assert receipt['source_git'] == independent['source_git']
assert receipt['local_validation_service_stopped'] and receipt['restart_did_not_retrain']
old = read(repo / 'provenance/deployment-validation.json')
results = independent['results']
for name, row in results.items():
    row['source_sha256'] = receipt['jobs'][name]['source_sha256']
    row['intrinsics_mode'] = receipt['jobs'][name]['intrinsics']['mode']
    assert row['steps'] == 18000 and row['state'] == 'passed'

proof = dict(
    schema='smpl-x-deployment-validation-v2', date='2026-10-28', application='smpl-x',
    branch='codex/smpl-x', scope=old['scope'], source_git=receipt['source_git'],
    local={**old['local'], 'cpu_regression_tests': 58},
    video_categories_confirmed_by_user=['landscape', 'portrait'],
    video_input_checks=old['video_input_checks'],
    fresh_video_runs=results,
    camera_profile=dict(mode='rectified_center_crop', reference_wh=[4000, 3000],
        calibration_sha256='7cded3ee3a21d6c6c0e496a61295db726fa964140b59db6eb5082788f7fe1f50',
        source='Existing MONO_MAIN_MFR first matrix and original resize/center-crop procedure',
        old_poses_loaded=False, focal_estimation_required=False, images_warped=False,
        exact_device_integer_ROI_verified=False, physical_calibration_accepted=False),
    folder_service=dict(state='passed', entrypoint='deploy.sh',
        deployment_source='clean sparse checkout from GitHub',
        partial_upload_ignored=receipt['partial_upload_ignored'],
        restart_did_not_retrain=receipt['restart_did_not_retrain'],
        completed_jobs_after_restart=2, persistent_serial_queue=True,
        viewer_files_in_deployment=receipt['viewer_deployed'],
        local_test_service_stopped_after_validation=receipt['local_validation_service_stopped'],
        content_deduplication=True,
        content_deduplication_evidence='Previous actual folder-service validation and current queue regression tests'),
    naming_change=old['naming_change'],
    ci=dict(tests_passed=58, os='Ubuntu 24.04', source_git=receipt['source_git'],
        url='https://github.com/lizuoshuo-lab/shuolee-portrait/actions/runs/40601192477', conclusion='success'),
    local_evidence=dict(directory='D:/Projects/Shuolee-validation-20261028',
        deployment_log='orientation-deploy.log', folder_integration='orientation-validation.json',
        independent_export='orientation-independent-exports.json', gpu_serialization='gpu-coordination.json'),
    previous_attempts=dict(landscape_focal_estimation=old['fresh_video'],
        portrait_focal_estimation=old['portrait_full_pipeline']),
    current_acceptance=dict(deployment_and_naming='passed',
        landscape_reference_full_training='passed_18000_steps',
        portrait_reference_full_training='passed_18000_steps',
        two_video_categories_confirmed_by_user=True,
        all_inputs_reconstruct_successfully='not_claimed; two full runs and eight decode checks'),
    company00_deployed=False, server113_deployed=False, quality_accepted=False,
    fresh_dependency_install_verified=False,
)
save(repo / 'provenance/deployment-validation.json', proof)

readme = repo / 'README.md'
text = readme.read_text(encoding='utf-8')
previous = '本地验证现状：横屏样本已完成18000步及通用导出；竖屏样本在焦距求解阶段失败，尚未通过完整重建验证。'
assert previous in text
text = text.replace(previous, '本地已用00数据集的横屏、竖屏各一段，从视频文件夹自动完成18000步训练与通用导出；分别为89帧和112帧。独立PLY/COLMAP读取检查通过，服务重启不会重复训练。')
readme.write_text(text, encoding='utf-8')

workflow = repo / 'docs/workflow.md'
text = workflow.read_text(encoding='utf-8')
start = text.index('2026-10-28本地实测：')
text = text[:start] + '2026-10-28本地实测：00数据集的5段横屏和3段竖屏视频均通过格式检查与实际抽帧，覆盖8位和10位HEVC。横屏按1600×899、竖屏按899×1600处理。恢复原流程的已有标定映射后，横屏89帧、竖屏112帧各自从原视频完成1200步拟合、18000步训练和通用导出；独立plyfile与pycolmap读取通过，完整内参得到保留。两次运行均未复用旧人物、相机或优化器。临时上传文件被忽略，服务重启后仍只有两个完成任务。本地复用已有CUDA环境，尚未验证全新系统安装或在00/113服务器部署；流程通过不代表独立物理精度已测定。先前自动估焦导致竖屏失败的记录保留在验证文件中。\n'
workflow.write_text(text, encoding='utf-8')

video_proof = read(repo / 'configs/video119-provenance.json')
video_proof['verified']['fresh_real_video_runs'] = dict(source_git=receipt['source_git'],
    landscape_frames=results['landscape']['frames'], portrait_frames=results['portrait']['frames'],
    steps_each=18000, all_camera_frames_registered=True,
    fresh_official_detector_SAM2_RTMw=True, evidence='provenance/deployment-validation.json')
video_proof['unverified'] = [s for s in video_proof['unverified']
    if not s.startswith(('Successful full raw-video', 'Fresh GPU detector/'))]
save(repo / 'configs/video119-provenance.json', video_proof)

now = datetime.now(timezone.utc).isoformat()
status = read(coord / 'company00-status.json')
status.update(state='smplx_landscape_portrait_local_validation_passed', updated_at=now,
    next='Git deploy on the target server when needed; daily input is video upload only.',
    next_status_command='cat ~/smpl-x/queue/status.json')
status['current_local_validation'].update(state='passed', gpu_task_may_be_active=False,
    evidence='D:/Projects/Shuolee-validation-20261028/orientation-independent-exports.json',
    results=results, local_test_service_stopped=True)
status['smplx_update'] = dict(source_git=receipt['source_git'], application='smpl-x',
    video_categories=['landscape', 'portrait'], full_training_each=18000,
    evidence='D:/Projects/Shuolee/provenance/deployment-validation.json')
save(coord / 'company00-status.json', status)

delivery = read(coord / 'company00-engineering-delivery.json')
delivery.update(updated_at=now, runtime_validated_commit=receipt['source_git'],
    state='local_validation_passed', linux_ci_tests_passed=58,
    actual_frames={'landscape': 89, 'portrait': 112}, export=results,
    smplx_validation=dict(deployment='passed', landscape_full_pipeline='passed_18000_steps',
        portrait_full_pipeline='passed_18000_steps', evidence_directory='D:/Projects/Shuolee-validation-20261028'),
    evidence_directory='D:/Projects/Shuolee-validation-20261028', local_test_service_stopped=True,
    video_categories=['landscape', 'portrait'], current_camera_profile=proof['camera_profile'])
save(coord / 'company00-engineering-delivery.json', delivery)
print(json.dumps(dict(state='recorded', source_git=receipt['source_git'], results=results)))
