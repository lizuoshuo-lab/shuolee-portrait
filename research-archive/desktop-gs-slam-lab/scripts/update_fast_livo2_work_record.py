"""Record actual FAST-LIVO2 evidence, retaining all prior GS results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
def read(relative):
    return json.loads((LAB / relative).read_text(encoding='utf-8-sig'))
def digest(relative):
    return hashlib.sha256((LAB / relative).read_bytes()).hexdigest()
def write(relative, value):
    (LAB / relative).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

source_path = 'outputs/fast-livo2-source-audit-001/metrics.json'
config_path = 'outputs/fast-livo2-config-audit-002/metrics.json'
camera_path = 'outputs/fast-livo2-config-audit-001/stereo-intrinsics-calibration-001/metrics.json'
timing_path = 'outputs/fast-livo2-bright-timing-audit-002/metrics.json'
author_path = 'outputs/fast-livo2-author-eee03-position-reanalysis-001/metrics.json'
controls_path = 'outputs/fast-livo2-position-evaluator-controls-002/metrics.json'
position_audit_path = 'outputs/fast-livo2-position-independent-audit-001/metrics.json'
timing_audit_path = 'outputs/fast-livo2-bright-timing-independent-audit-001/metrics.json'
platform_path = 'outputs/fast-livo2-wsl-setup/windows-platform-status.json'
resource_path = 'outputs/fast-livo2-wsl-setup/resource-config.json'
preparation_path = 'outputs/fast-livo2-linux-bootstrap-001/final-preparation-state.json'
compatibility_path = 'outputs/fast-livo2-data-recovery-001/bright-screen-compatibility.json'
retry_path = 'outputs/fast-livo2-eee03-download-retry-001/receipt.json'
native_download_path = 'outputs/fast-livo2-linux-native-download-001/receipt.json'
post_reboot_path = 'outputs/fast-livo2-wsl-setup/post-reboot-status.json'
runtime_path = 'manifests/fast-livo2-local-runtime.json'
camera = read(camera_path)
timing = read(timing_path)
author = read(author_path)
controls = read(controls_path)
assert camera['status'].startswith('completed_actual_')
assert timing['required_sensor_topics_observed']
assert author['provenance'] == 'author_saved' and author['input_estimate_point_frame'] == 'prism'
assert author['orientation_GT_available'] is False and author['full_SE3_RPE_computed'] is False
assert all(controls['gates'].values())
archive = LAB / 'data/fast-livo2/ntu-viral-eee03.zip'
expected_size = 1995367404
download_bytes = archive.stat().st_size if archive.exists() else 0
receipt = LAB / 'outputs/fast-livo2-data-discovery-001/eee03-download-receipt.json'
evidence = [source_path, config_path, camera_path, timing_path, author_path,
            controls_path, position_audit_path, timing_audit_path,
            platform_path, resource_path, preparation_path, compatibility_path, retry_path]
platform = read(platform_path)
compatibility = read(compatibility_path)
assert compatibility['unchanged'] and compatibility['compressed_images_inspected'] == 668
post_reboot = read(post_reboot_path) if (LAB / post_reboot_path).exists() else {}
runtime = read(runtime_path) if (LAB / runtime_path).exists() else {}
native_download = read(native_download_path) if (LAB / native_download_path).exists() else {}
ubuntu_ready = post_reboot.get('ubuntu_install_and_boot_verified', False)
environment_ready = runtime.get('compiled_environment_accepted', False)
local_runs = runtime.get('accepted_local_replay_runs', 0)
attempted_runs = runtime.get('actual_local_replay_runs', 0)
for optional_path in (post_reboot_path, runtime_path, native_download_path):
    if (LAB / optional_path).exists():
        evidence.append(optional_path)
if local_runs:
    fusion_status = 'actual_local_replay_completed_no_ground_truth_accuracy'
    fusion_blocker = 'Bright Screen has no known GT; EEE03 full sensor input and further calibration/sync validation remain pending'
elif attempted_runs:
    fusion_status = 'actual_local_replay_outputs_observed_shutdown_failure'
    fusion_blocker = 'Actual numerical outputs were observed, but child-process shutdown acceptance failed; investigate the saved crash before accepting complete replay'
elif environment_ready:
    fusion_status = 'compiled_environment_accepted_actual_replay_pending'
    fusion_blocker = 'Actual LIO/LIVO replay and output acceptance are pending'
elif ubuntu_ready:
    fusion_status = 'Ubuntu_WSL2_boot_verified_ROS_compilation_not_yet_accepted'
    fusion_blocker = 'ROS installation and C++ environment acceptance have not completed'
else:
    fusion_status = 'not_compiled_or_replayed_locally'
    fusion_blocker = 'Windows platform changes require manual restart before Ubuntu/ROS installation'
priorities = read('manifests/project-priorities.json') if (LAB / 'manifests/project-priorities.json').exists() else {'order': ['FAST-LIVO2', 'SplaTAM']}
priority = priorities['order'].index('FAST-LIVO2') + 1
state = {
    'priority': priority, 'project': 'FAST-LIVO2', 'observed_utc': datetime.now(timezone.utc).isoformat(),
    'official_commit': '0d2c0346107b75b59934975adec9a6eeeb913c64',
    'fusion': {'status': fusion_status, 'source_files_verified': 75,
               'local_estimator_runs': attempted_runs, 'accepted_local_replay_runs': local_runs,
               'source_local_patch': runtime.get('source_local_patch'), 'blocker': fusion_blocker},
    'environment_setup': {'windows_platform': platform, 'resources': read(resource_path),
                          'linux_build_preparation': read(preparation_path),
                          'post_reboot_observation': post_reboot,
                          'local_runtime_evidence': runtime,
                          'ubuntu_installed_and_boot_verified': ubuntu_ready,
                          'ROS_installed_and_compiled': environment_ready,
                          'resume_entry': 'Desktop/Continue-FAST-LIVO2.cmd'},
    'first_fusion_input': {'dataset': 'Bright_Screen_Wall', 'status': 'complete_read_and_all_JPEG_decode_checked',
                          'compressed_images_decoded': 668, 'image_size': [1280, 1024],
                          'official_launch_already_republishes_compressed_to_raw': True,
                          'ground_truth_available': False},
    'calibration': {'camera_intrinsics_and_stereo': 'completed_actual_public_board_fit_and_heldout_validation',
                    'LiDAR_camera_extrinsics_reestimated': False, 'camera_IMU_reestimated': False,
                    'clock_offset_reestimated': False, 'upstream_config_replaced': False},
    'synchronization': {'complete_bag_read': True, 'sensor_messages': 14850,
                        'IMU_header_reversals': 2, 'physical_hardware_sync_verified': False,
                        'actual_runtime_buffer_replay_verified': False, 'raw_acceleration_units': 'native_not_independently_calibrated'},
    'drift': {'local_FAST_LIVO2_accuracy_measured': False,
              'position_evaluator_controls_passed': len(controls['gates']),
              'author_saved_trajectory_reanalysis_completed': True,
              'author_saved_position_APE_RMSE_m': author['aligned_position_APE_RMSE_m'],
              'no_orientation_truth_or_full_SE3_RPE': True},
    'EEE03_sensor_download': {'observed_bytes': download_bytes, 'expected_bytes': expected_size,
        'bytes_fraction': download_bytes / expected_size,
        'status': 'complete_size_pending_integrity_readback' if download_bytes == expected_size else 'incomplete_download_no_active_attempt',
        'last_attempt_receipt': native_download_path if native_download else retry_path,
        'native_attempt': {key: native_download.get(key) for key in
            ['status', 'bytes_written', 'expected_bytes', 'size_matches', 'official_md5_matches',
             'zip_crc_all_entries_passed', 'error', 'finished_utc', 'exit_code']},
        'earlier_partial_preserved': 'data/fast-livo2/ntu-viral-eee03.interrupted-317325312-20261009-120000.zip',
        'complete_download_receipt_exists': receipt.exists(),
        'expected_official_MD5': '9bf10b5455ee6d2962324ef557f86433',
        'scope': 'Size observation only; a partial ZIP is not a usable complete bag.'},
    'evidence': {path: {'sha256': digest(path)} for path in evidence},
    'website_scope': 'Local knowledge and small evidence files only; no cloud publication or raw dataset upload.'
}
write('manifests/fast-livo2-work.json', state)

progress = read('manifests/progress.json')
progress['updated'] = datetime.now().astimezone().date().isoformat()
if priority == 1:
    progress['current_stage'] = 'FAST_LIVO2_' + fusion_status
    progress['next_steps'] = ([] if environment_ready else ['继续 D 盘 Ubuntu 20.04/ROS Noetic 安装编译并验收实际可执行程序']) + ([] if local_runs else [
    '先用完整 Bright Screen 数据做同段 LIO/LIVO 运行验证，逐次归档输出；该数据无已知 GT',
    ]) + [
    '恢复完整 EEE03 输入并校验后，再进行真实位置精度评测',
    '独立 GT 位置评测、时差控制、LiDAR/IMU 标定证据补齐']
new = [
    {'id': 'fast-livo2-official-source-001', 'title': 'FAST-LIVO2 固定官方源码与融合流程',
     'status': fusion_status, 'metrics': {'source_files_verified': 75, 'shared_state_dimensions': 19,
       'local_estimator_runs': attempted_runs, 'accepted_local_replay_runs': local_runs},
     'artifacts': [source_path, 'outputs/fast-livo2-source-audit-001/report.md', 'knowledge/20_fast_livo2_fusion.md', 'manifests/fast-livo2-work.json',
                   platform_path, resource_path, preparation_path, compatibility_path]
                   + ([post_reboot_path] if post_reboot else []) + ([runtime_path] if runtime else []),
     'scope': '固定官方提交全部文件匹配；环境与实际回放状态分别取自已保存的运行证据，无真值的 Bright Screen 不提供本机定位精度。'},
    {'id': 'fast-livo2-actual-stereo-calibration-001', 'title': '真实 NTU 棋盘左右相机与双目标定',
     'status': camera['status'], 'metrics': {'stereo_pairs': 135, 'left_refit_heldout_RMSE_px': .187594744875,
       'left_official_heldout_RMSE_px': .188581012211, 'right_refit_heldout_RMSE_px': .180026,
       'right_official_heldout_RMSE_px': .171907, 'independent_readback_gates': 21},
     'artifacts': [camera_path, config_path, 'knowledge/21_fast_livo2_calibration.md'],
     'scope': '真实图像求解和预先留出，右侧变差保留；未覆盖官方参数，未完成 LiDAR/IMU/时钟重新标定。'},
    {'id': 'fast-livo2-real-bright-timing-002', 'title': '完整 LiDAR/IMU/相机时间与原始字节核验',
     'status': timing['status'], 'metrics': {'lidar_messages': 667, 'camera_messages': 668, 'imu_messages': 13515,
       'IMU_header_reversals': 2, 'raw_CSV_cells_independently_checked': 89100},
     'artifacts': [timing_path, timing_audit_path, 'knowledge/22_fast_livo2_synchronization.md'],
     'scope': '只读真实 bag；单位标签修正保留原记录。未求新时差、未证明硬同步或实际运行缓存行为。'},
    {'id': 'fast-livo2-position-evaluator-author-saved-001', 'title': '位置评测控制与作者保存 EEE03 轨迹重分析',
     'status': 'author_saved_reanalysis_independently_checked_not_local_estimator_run',
     'metrics': {'controls_passed': len(controls['gates']), 'author_estimate_rows': author['estimate_rows'],
       'matched_rows': author['matched_rows'], 'matched_fraction': author['matched_fraction'],
       'author_saved_position_APE_RMSE_m': author['aligned_position_APE_RMSE_m'], 'local_estimator_runs': 0},
     'artifacts': [controls_path, author_path, position_audit_path, 'knowledge/23_fast_livo2_drift.md'],
     'scope': '固定作者保存轨迹来源和 GT header 时钟，尺度1；不是本机精度。GT 无姿态，不生成旋转或完整 SE3 RPE。'}
]
for side in ('left', 'right'):
    heldout = camera['per_camera'][side]['heldout_other_27_corners']
    new[1]['metrics'][f'{side}_refit_heldout_RMSE_px'] = heldout['reestimated']['euclidean_pixel_rmse']
    new[1]['metrics'][f'{side}_official_heldout_RMSE_px'] = heldout['published']['euclidean_pixel_rmse']
updated_ids = {item['id'] for item in new}
existing = [item for item in progress['results'] if item['id'] not in updated_ids]
progress['results'] = new + existing
write('manifests/progress.json', progress)
print(json.dumps({'priority': priority, 'results': len(progress['results']), 'download_bytes': download_bytes,
                  'local_estimator_runs': attempted_runs, 'accepted_local_replay_runs': local_runs}, ensure_ascii=False))
