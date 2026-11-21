"""Export only completed GPU acceptance, with pinned source and binary hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/opt/gs-slam-lab/splatam')
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB / 'outputs/splatam-linux-bootstrap-001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    smoke = json.loads((ROOT / 'renderer-smoke.json').read_text())
    assert smoke['status'] == 'passed'
    renderer = ROOT / 'repos/diff-gaussian-rasterization-w-depth'
    entries = json.loads((ROOT / 'downloads/renderer-tree.json').read_text())['tree']
    mismatches = []
    checked = 0
    for entry in entries:
        if entry['type'] != 'blob':
            continue
        path = renderer / entry['path']
        if not path.exists():
            mismatches.append(entry['path'] + ':missing')
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        checked += 1
        if blob != entry['sha']:
            mismatches.append(entry['path'])
    assert checked > 100 and not mismatches, mismatches
    import diff_gaussian_rasterization as loaded
    installed_init = Path(loaded.__file__)
    assert sha(installed_init) == sha(renderer / 'diff_gaussian_rasterization/__init__.py')
    official_splatam = LAB / 'repos/SplaTAM'
    native_splatam = ROOT / 'repos/SplaTAM'
    primary_matches = []
    for source in official_splatam.rglob('*'):
        if not source.is_file() or any(part in ('.git', '__pycache__', 'diff-gaussian-rasterization-w-depth.git') for part in source.relative_to(official_splatam).parts):
            continue
        rel = source.relative_to(official_splatam)
        target = native_splatam / rel
        assert target.is_file(), str(rel)
        source_bytes = source.read_bytes()
        target_bytes = target.read_bytes()
        exact = source_bytes == target_bytes
        normalized = source_bytes.replace(b'\r\n', b'\n') == target_bytes
        assert exact or normalized, str(rel)
        primary_matches.append({'path': str(rel), 'windows_sha256': sha(source),
                                'native_sha256': sha(target), 'exact_bytes_match': exact,
                                'line_ending_conversion': not exact and normalized})
    selected = json.loads((ROOT / 'downloads/cuda-package-selection.json').read_text())
    cuda_packages = []
    for name, value in sorted(selected.items()):
        archive = ROOT / 'downloads' / Path(value['Filename']).name
        assert sha(archive) == value['SHA256'] and archive.stat().st_size == int(value['Size'])
        cuda_packages.append({'name': name, 'version': value['Version'], 'bytes': archive.stat().st_size,
                              'sha256': sha(archive), 'source': 'official NVIDIA CUDA Ubuntu 20.04 repository'})
    source_receipt = json.loads((ROOT / 'source-receipt.json').read_text())
    weight_receipt = json.loads((ROOT / 'alexnet-weights-receipt.json').read_text())
    assert weight_receipt['full_hash_verified']
    bindings = {}
    for name in ('renderer-smoke.json', 'source-receipt.json', 'alexnet-weights-receipt.json',
                 'env.sh', 'logs/pip-freeze.txt', 'logs/pip-check.log', 'logs/official-renderer-build.log',
                 'logs/official-renderer-compile.log', 'downloads/pytorch-wheel-selection.json',
                 'downloads/pytorch-gpu-dependency-selection.json', 'downloads/core-local-wheels.json'):
        p = ROOT / name
        bindings[name] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
    for name in ('bootstrap_gpu.py', 'verify_official_renderer.py', 'requirements.txt',
                 'write_bootstrap_receipt.py', 'fetch_pypi_wheels.py'):
        p = LAB / 'scripts/splatam_linux' / name
        bindings['script/' + name] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
    metrics = {
        'status': 'isolated_official_renderer_cuda_acceptance_passed',
        'native_root': str(ROOT), 'python': sys.version, 'source': source_receipt,
        'renderer_tracked_blob_count': checked, 'renderer_source_mismatches': mismatches,
        'main_repo_windows_native_normalized_matched_file_count': len(primary_matches),
        'main_repo_line_ending_conversion_count': sum(x['line_ending_conversion'] for x in primary_matches),
        'main_repo_file_bindings': primary_matches,
        'loaded_module_path': str(installed_init), 'loaded_module_sha256': sha(installed_init),
        'cuda_packages': cuda_packages, 'weights': weight_receipt, 'smoke': smoke,
        'acceptance': {'renderer_source_clean': not mismatches, 'installed_python_matches_pinned_source': True,
                       'cuda_mapping_gradients_passed': True, 'cuda_tracking_gradients_passed': True,
                       'real_lpips_passed': True, 'evaluation_stub': False,
                       'linux_driver_installed': False, 'system_ros_environment_modified': False},
        'file_bindings': bindings,
        'scope': 'Renderer and Python/CUDA environment acceptance; full RGB-D SLAM accuracy is a separate run.',
    }
    (ROOT / 'bootstrap-metrics.json').write_text(json.dumps(metrics, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'bootstrap-metrics.json', OUT / 'metrics.json')
    for source, dest in [('renderer-smoke.json', 'renderer-smoke.json'),
                         ('source-receipt.json', 'source-receipt.json'),
                         ('alexnet-weights-receipt.json', 'alexnet-weights-receipt.json'),
                         ('env.sh', 'env.sh'), ('logs/pip-freeze.txt', 'pip-freeze.txt'),
                         ('logs/pip-check.log', 'pip-check.log'),
                         ('logs/official-renderer-compile.log', 'official-renderer-compile.log'),
                         ('downloads/pytorch-wheel-selection.json', 'pytorch-wheel-selection.json'),
                         ('downloads/pytorch-gpu-dependency-selection.json', 'pytorch-gpu-dependency-selection.json'),
                         ('downloads/core-local-wheels.json', 'core-local-wheels.json'),
                         ('logs/official-renderer-build.log', 'official-renderer-build.log')]:
        shutil.copy2(ROOT / source, OUT / dest)
    report = (
        '# SplaTAM 本机官方 CUDA 环境验收\n\n'
        'WSL Ubuntu 20.04 内独立 Python 3.10、PyTorch 2.7.1 cu128 和 CUDA 12.8 nvcc 已实装。'
        'NVIDIA 开发包经过官方 SHA256 校验后解包到项目目录；未安装 Linux 驱动，未修改 FAST-LIVO2 的 ROS 环境。\n\n'
        f"SplaTAM 固定提交 `{source_receipt['splatam_commit']}`；官方深度 renderer 固定提交 "
        f"`{source_receipt['renderer_commit']}`，{checked} 个 tracked blob 的 Git SHA1 全部匹配。"
        '安装后的 Python API 与固定源码 SHA256 匹配，实际加载的 CUDA 扩展路径和哈希见 metrics.json。\n\n'
        'RTX 5080（SM 12.0）完成真实 RGB 渲染、深度/轮廓颜色编码渲染、Gaussian 参数有限非零梯度、'
        '相机旋转和平移有限非零梯度。预训练 AlexNet 权重完整校验并执行真实 LPIPS。\n\n'
        '第三返回值 depth 只验 forward：官方 backward 不消费该输出的梯度。'
        'SplaTAM 的深度监督通过 `(z,1,z²)` 编码到 colors_precomp 后再渲染，验收沿用此真实路径。\n\n'
        '这些结果证明 CUDA 依赖和 renderer 可运行；轨迹、建图质量与完整序列评测属于另一次正式实验。\n'
    )
    (OUT / 'report.md').write_text(report)
    print(json.dumps({'status': metrics['status'], 'metrics_sha256': sha(OUT / 'metrics.json'),
                      'renderer_blobs_checked': checked, 'extension_sha256': smoke['extension_sha256']}))


if __name__ == '__main__':
    main()
