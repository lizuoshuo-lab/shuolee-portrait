from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

lab = Path(__file__).resolve().parents[2]
path = lab / 'manifests/mini-splatting2-computer00.json'
record = json.loads(path.read_text(encoding='utf-8-sig'))
record.update(
    priority='Current Mini-Splatting2 reproduction request; prior actor23 fixed-budget runs completed and preserved.',
    transport='ToDesk remote desktop, remote terminal and bidirectional file transfer connected successfully.',
    remote_desktop_seen=True,
    desktop_materials_read=True,
    remote_shell_connected=True,
    remote_environment_modified=False,
    connection_observation='Connected computer00 (ZEROZER-77P4RDS). Original SSH endpoint is reachable from computer00, but not from the local machine. Server SSH currently awaits password entry; automated ToDesk keyboard events did not reach the remote console. No authenticated Linux session yet.',
    last_recorded_utc=datetime.now(timezone.utc).isoformat(),
)
record['remote_computer'] = {
    'hostname': 'ZEROZER-77P4RDS',
    'desktop': r'C:\Users\zerozero\Desktop',
    'os': 'Windows 11 IoT Enterprise LTSC, 10.0.26100',
    'cpu': 'Intel Core Ultra 5 225U, 12 cores / 14 logical processors',
    'physical_memory_bytes': 16017846272,
    'gpu': 'Intel Graphics; no NVIDIA GPU reported by Win32_VideoController; nvidia-smi not installed',
    'cuda_training_available': False,
    'disk_free_bytes': {'C:': 144208080896, 'D:': 808998182912},
    'existing_cli_tools_found': ['wsl.exe', 'ssh.exe'],
    'wsl_distribution_status': 'not established; query did not promptly complete',
    'original_ssh_endpoint_from_computer00': 'TCP reachable',
    'inspection_artifacts_created': ['Desktop/00_mini_splatting2_audit', 'Desktop/01_mini_splatting2_environment', 'Desktop/02_mini_splatting2_server'],
    'installed_software': [],
    'remote_training_started': False,
}
source = lab / 'inputs/computer00-local-pipeline'
record['materials'] = {
    'remote_source': r'C:\Users\zerozero\Desktop\local_pipeline',
    'local_archive': str(source),
    'files': [{'path': str(p.relative_to(source)), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source.rglob('*')) if p.is_file()],
    'named_documents': 'No files literally named 高斯渲染流程 or 渲染资料 on observed Windows Desktop; broader location still unresolved.',
    'scripts_already_use_mini_splatting2': True,
    'source_code_findings': [
        'test_gs.py invokes ../mini-splatting2-main/msv2/train.py with config/fast, indoor, resolution 1, without --eval.',
        'colmap2nerf.py matching arguments omit the -- prefix for SiftMatching.guided_matching and SiftMatching.use_gpu.',
        'colmap2nerf.py mapper command ends with --Mapper.ba_refine_principal_point 1 1, leaving an extra positional argument.',
        'run_ffmpeg and run_colmap delete existing image/database/sparse outputs; archived source is not executed.',
        'test_preprocess_material.py catches errors and prints complete without returning a failing exit status.',
        'read_extrinsics_binary intentionally returns only registered-image count; this is not a dict/int comparison bug.',
    ],
}
record['input_video'] = {
    'name': 'HOVER_VERSA_0102.mp4', 'duration_s': 7.366083,
    'width': 1520, 'height': 2704, 'frame_count': 177,
    'observed_content': 'Handheld orbit around a mostly standing person in an office; background people move. Seven sampled overview frames inspected.',
    'scope': 'Static baseline and input-pipeline check; not synchronized multiview dynamic reconstruction or evidence of full-body moving-person clarity.',
    'local_sfm': str(lab / 'outputs/mini-splatting2-computer00-001/hover-sfm-001/state.json'),
}
record['todesk_auto_lock_request'].update(completed=True,
    action_result='Changed only PrivateScreenLockScreen and autoLockScreen from 1 to 0 in the local ToDesk configuration; exact two-byte change verified. User subsequently confirmed the computer no longer auto-locks. ToDesk was not restarted.')
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
main_path = lab / 'manifests/dynamic-human-work.json'
main = json.loads(main_path.read_text(encoding='utf-8-sig'))
main['current_user_priority'] = 'Use ToDesk to control computer00, inspect its Gaussian rendering materials, reproduce Mini-Splatting2. Connected; archived desktop scripts/video; checking reachable Linux server and independently preparing input camera poses.'
main['mini_splatting2_computer00_manifest'] = str(path)
main['last_recorded_utc'] = datetime.now(timezone.utc).isoformat()
main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(path)
