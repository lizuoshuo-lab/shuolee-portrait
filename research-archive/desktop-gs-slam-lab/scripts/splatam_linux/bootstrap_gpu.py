"""Isolated WSL RTX 5080 bootstrap for the pinned upstream SplaTAM renderer.

Run using Ubuntu's python3. System packages and the Linux GPU driver are not
installed. Native CUDA development debs are verified and unpacked below ROOT.
Stages are explicit so a failed download never becomes a successful receipt.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path('/opt/gs-slam-lab/splatam')
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
SPLATAM_COMMIT = 'da6bbcd24c248dc884ac7f49d62e91b841b26ccc'
RENDERER_COMMIT = 'cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110'
CUDA_REPO = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/'
CUDA_PACKAGES = ('cuda-nvcc-12-8', 'cuda-cudart-dev-12-8', 'cuda-cccl-12-8',
                 'cuda-crt-12-8', 'cuda-nvvm-12-8', 'cuda-driver-dev-12-8', 'cuda-cudart-12-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest, expected_size=None, expected_sha=None):
    if dest.exists() and expected_sha and sha(dest) == expected_sha:
        return
    partial = dest.with_name(dest.name + '.partial')
    with urllib.request.urlopen(url, timeout=180) as source, partial.open('wb') as out:
        shutil.copyfileobj(source, out, length=1024 * 1024)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError('Truncated download: ' + str(dest))
    if expected_sha and sha(partial) != expected_sha:
        raise RuntimeError('SHA256 mismatch: ' + str(dest))
    partial.replace(dest)


def run(argv, env=None, log_name=None, cwd=None):
    print('RUN', ' '.join(str(x) for x in argv), flush=True)
    if log_name:
        with (ROOT / 'logs' / log_name).open('w') as out:
            result = subprocess.run([str(x) for x in argv], env=env, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode:
            print((ROOT / 'logs' / log_name).read_text(errors='replace')[-8000:], flush=True)
            raise RuntimeError('Command failed: ' + log_name)
    else:
        subprocess.run([str(x) for x in argv], env=env, cwd=cwd, check=True)


def cuda_stage():
    metadata = ROOT / 'downloads/Packages.gz'
    if not metadata.exists():
        download(CUDA_REPO + 'Packages.gz', metadata)
    selected = {}
    for block in gzip.decompress(metadata.read_bytes()).decode().split('\n\n'):
        fields = {}
        for line in block.splitlines():
            if ': ' in line and not line.startswith(' '):
                name, value = line.split(': ', 1)
                fields[name] = value
        if fields.get('Package') in CUDA_PACKAGES:
            name = fields['Package']
            if name not in selected or fields['Version'] > selected[name]['Version']:
                selected[name] = fields
    if set(selected) != set(CUDA_PACKAGES):
        raise RuntimeError('Missing CUDA development package')
    (ROOT / 'downloads/cuda-package-selection.json').write_text(json.dumps(selected, indent=2))

    def fetch(item):
        name, fields = item
        archive = ROOT / 'downloads' / Path(fields['Filename']).name
        download(CUDA_REPO + fields['Filename'].lstrip('./'), archive,
                 int(fields['Size']), fields['SHA256'])
        return name, archive

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        archives = list(pool.map(fetch, selected.items()))
    target = ROOT / 'cuda-extracted'
    target.mkdir(exist_ok=True)
    for name, archive in archives:
        run(['dpkg-deb', '-x', archive, target])
    run([target / 'usr/local/cuda-12.8/bin/nvcc', '--version'])


def python_stage():
    uv = ROOT / 'tools/uv'
    if not uv.exists():
        archive = ROOT / 'downloads/uv-0.9.2.tar.gz'
        download('https://github.com/astral-sh/uv/releases/download/0.9.2/uv-x86_64-unknown-linux-gnu.tar.gz', archive)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile() and Path(member.name).name in ('uv', 'uvx'):
                    source = tar.extractfile(member)
                    dest = ROOT / 'tools' / Path(member.name).name
                    with dest.open('wb') as out:
                        shutil.copyfileobj(source, out)
                    dest.chmod(0o755)
    uv_env = os.environ.copy()
    uv_env['UV_HTTP_TIMEOUT'] = '180'
    run([uv, 'python', 'install', '3.10', '--install-dir', ROOT / 'python'], env=uv_env, log_name='managed-python.log')
    candidates = list((ROOT / 'python').glob('cpython-3.10.*-linux-x86_64-gnu/bin/python3.10'))
    if not candidates:
        raise RuntimeError('Managed Python 3.10 unavailable')
    if not (ROOT / '.venv/bin/python').exists():
        run([uv, 'venv', '--python', sorted(candidates)[-1], ROOT / '.venv'])
    run([uv, 'pip', 'install', '--python', ROOT / '.venv/bin/python', 'pip==25.2', 'setuptools==80.9.0', 'wheel==0.45.1', 'ninja==1.13.0'], env=uv_env, log_name='base-python-packages.log')
    # Official R2 links return HTTP 403 from this WSL network. The original
    # official PyTorch host serves the identical wheel and index-provided hash.
    wheels = ROOT / 'downloads/pytorch-wheel-selection.json'
    if not wheels.exists():
        import re
        selection = {}
        for name, version in [('torch', '2.7.1'), ('torchvision', '0.22.1')]:
            with urllib.request.urlopen('https://download.pytorch.org/whl/cu128/' + name + '/', timeout=180) as response:
                page = response.read().decode()
            entry = next(url for url in re.findall(r'href="([^"]+)"', page)
                         if name + '-' + version + '%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl' in url)
            selection[name] = {'official_index_url': entry, 'download_url': entry.replace('download-r2.pytorch.org', 'download.pytorch.org')}
        wheels.write_text(json.dumps(selection, indent=2))
    selection = json.loads(wheels.read_text())
    gpu_selection_path = ROOT / 'downloads/pytorch-gpu-dependency-selection.json'
    if not gpu_selection_path.exists():
        import email
        import re
        with urllib.request.urlopen(selection['torch']['download_url'].split('#')[0] + '.metadata', timeout=180) as response:
            metadata = email.message_from_string(response.read().decode())
        wanted = [re.match(r'(nvidia-[a-z0-9-]+)==([0-9.]+)', line).groups()
                  for line in metadata.get_all('Requires-Dist') if line.startswith('nvidia-')]
        wanted.append(('triton', '3.3.1'))

        def select_gpu(item):
            name, version = item
            with urllib.request.urlopen('https://download.pytorch.org/whl/cu128/' + name + '/', timeout=180) as response:
                page = response.read().decode()
            link = next(url for url in re.findall(r'href="([^"]+)"', page)
                        if name.replace('-', '_') + '-' + version + '-' in url
                        and 'x86_64.whl' in url and ('cp310-' in url or 'py3-none-' in url))
            return name, {'version': version, 'official_index_url': link,
                          'download_url': link.replace('download-r2.pytorch.org', 'download.pytorch.org')}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            gpu_selection = dict(pool.map(select_gpu, wanted))
        gpu_selection_path.write_text(json.dumps(gpu_selection, indent=2))
    gpu_selection = json.loads(gpu_selection_path.read_text())
    gpu_local_path = ROOT / 'downloads/pytorch-gpu-local-wheels.json'
    if not gpu_local_path.exists():
        import urllib.parse
        wheel_dir = ROOT / 'downloads/wheels'
        wheel_dir.mkdir(exist_ok=True)

        def fetch_gpu(item):
            name, fields = item
            url, fragment = fields['download_url'].split('#', 1)
            expected = fragment.split('=', 1)[1]
            dest = wheel_dir / urllib.parse.unquote(url.rsplit('/', 1)[1])
            download(url, dest, expected_sha=expected)
            print('GPU wheel SHA256 passed:', name, flush=True)
            return name, str(dest)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            gpu_local = dict(pool.map(fetch_gpu, gpu_selection.items()))
        gpu_local_path.write_text(json.dumps(gpu_local, indent=2))
    gpu_local = json.loads(gpu_local_path.read_text())
    run([uv, 'pip', 'install', '--python', ROOT / '.venv/bin/python',
         selection['torch']['download_url'], selection['torchvision']['download_url']]
        + list(gpu_local.values()),
        env=uv_env, log_name='pytorch-cu128.log')
    deps_stage()


def deps_stage():
    core = ROOT / 'downloads/core-local-wheels.json'
    if not core.exists():
        run([ROOT / '.venv/bin/python', LAB / 'scripts/splatam_linux/fetch_pypi_wheels.py'],
            log_name='verified-cpu-wheel-download.log')
    wheels = json.loads(core.read_text())
    env = os.environ.copy()
    env['UV_HTTP_TIMEOUT'] = '180'
    run([ROOT / 'tools/uv', 'pip', 'install', '--python', ROOT / '.venv/bin/python',
         '-r', LAB / 'scripts/splatam_linux/requirements.txt'] + [item['path'] for item in wheels.values()],
        env=env, log_name='splatam-python-packages.log')


def source_stage():
    source = LAB / 'repos/SplaTAM'
    dest = ROOT / 'repos/SplaTAM'
    if not dest.exists():
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns('.git', '__pycache__', 'diff-gaussian-rasterization-w-depth.git'))
    renderer = ROOT / 'repos/diff-gaussian-rasterization-w-depth'
    archive = ROOT / 'downloads/renderer-cb65e4b.tar.gz'
    if not renderer.exists():
        download('https://codeload.github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/tar.gz/' + RENDERER_COMMIT, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(ROOT / 'repos')
        (ROOT / 'repos' / ('diff-gaussian-rasterization-w-depth-' + RENDERER_COMMIT)).rename(renderer)
    # GitHub's tree object proves the gitlink commit; it is not guessed from a release.
    metadata = ROOT / 'downloads/renderer-tree.json'
    if not metadata.exists():
        download('https://api.github.com/repos/JonathonLuiten/diff-gaussian-rasterization-w-depth/git/trees/' + RENDERER_COMMIT + '?recursive=1', metadata)
    entries = json.loads(metadata.read_text())['tree']
    glm = next(entry for entry in entries if entry['path'] == 'third_party/glm')
    glm_archive = ROOT / 'downloads' / ('glm-' + glm['sha'] + '.tar.gz')
    glm_target = renderer / 'third_party/glm'
    if not (glm_target / 'glm/glm.hpp').exists() and glm['type'] == 'commit':
        download('https://codeload.github.com/g-truc/glm/tar.gz/' + glm['sha'], glm_archive)
        temp = ROOT / 'downloads/glm-extracted'
        temp.mkdir(exist_ok=True)
        with tarfile.open(glm_archive) as tar:
            tar.extractall(temp)
        extracted = temp / ('glm-' + glm['sha'])
        shutil.copytree(extracted, glm_target, dirs_exist_ok=True)
    receipt = {'splatam_commit': SPLATAM_COMMIT, 'renderer_commit': RENDERER_COMMIT,
               'glm_tree_or_gitlink': glm['sha'], 'glm_object_type': glm['type'],
               'renderer_archive_sha256': sha(archive),
               'glm_archive_sha256': sha(glm_archive) if glm_archive.exists() else None}
    (ROOT / 'source-receipt.json').write_text(json.dumps(receipt, indent=2))


def environment():
    env = os.environ.copy()
    cuda = ROOT / 'cuda-extracted/usr/local/cuda-12.8'
    env.update(CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='4',
               TORCH_HOME=str(ROOT / 'torch-cache'), WANDB_MODE='disabled', PYTHONUNBUFFERED='1',
               PATH=str(cuda / 'bin') + ':' + str(ROOT / '.venv/bin') + ':' + env.get('PATH', ''),
               LD_LIBRARY_PATH='/usr/lib/wsl/lib:' + str(cuda / 'lib64') + ':' + env.get('LD_LIBRARY_PATH', ''))
    return env


def build_stage():
    env = environment()
    run([ROOT / '.venv/bin/python', '-m', 'pip', 'install', '--no-build-isolation', ROOT / 'repos/diff-gaussian-rasterization-w-depth'], env=env, log_name='official-renderer-build.log')


def compile_stage():
    # Compiling source artifacts can overlap pure-Python dependency downloads;
    # installation into the venv is done separately once those finish.
    run([ROOT / '.venv/bin/python', 'setup.py', 'build_ext', '--inplace'],
        env=environment(), log_name='official-renderer-compile.log',
        cwd=ROOT / 'repos/diff-gaussian-rasterization-w-depth')


def weights_stage():
    dest = ROOT / 'torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth'
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
    expected = '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
    download(url, dest, expected_size=244408911, expected_sha=expected)
    receipt = {'url': url, 'path': str(dest), 'bytes': dest.stat().st_size,
               'sha256': sha(dest), 'hash_prefix_verified': True, 'full_hash_verified': True}
    (ROOT / 'alexnet-weights-receipt.json').write_text(json.dumps(receipt, indent=2))


def verify_stage():
    env = environment()
    run([ROOT / '.venv/bin/python', '-m', 'pip', 'check'], env=env, log_name='pip-check.log')
    run([ROOT / '.venv/bin/python', LAB / 'scripts/splatam_linux/verify_official_renderer.py'],
        env=env, log_name='renderer-smoke.log')
    run([ROOT / '.venv/bin/python', '-m', 'pip', 'freeze'], env=env, log_name='pip-freeze.txt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['cuda', 'python', 'deps', 'source', 'compile', 'build', 'weights', 'verify'])
    args = parser.parse_args()
    for name in ('downloads', 'tools', 'logs', 'repos'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    (ROOT / 'env.sh').write_text(
        'export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8\n'
        'export PATH=/opt/gs-slam-lab/splatam/.venv/bin:$CUDA_HOME/bin:$PATH\n'
        'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\n'
        'export TORCH_CUDA_ARCH_LIST=12.0\nexport MAX_JOBS=4\n'
        'export TORCH_HOME=/opt/gs-slam-lab/splatam/torch-cache\n'
        'export WANDB_MODE=disabled\nexport PYTHONUNBUFFERED=1\n')
    started = time.time()
    globals()[args.stage + '_stage']()
    print(json.dumps({'stage': args.stage, 'status': 'passed', 'elapsed_s': time.time()-started}), flush=True)


if __name__ == '__main__':
    main()
