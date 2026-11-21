"""Download pinned Linux CPU wheels with official PyPI size and SHA256 checks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path('/opt/gs-slam-lab/splatam')
REQ = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/splatam_linux/requirements.txt')
TAGS = set(sys_tags())


def fetch(item):
    name, version = item
    with urllib.request.urlopen('https://pypi.org/pypi/' + name + '/' + version + '/json', timeout=180) as response:
        metadata = json.load(response)
    wheel = next(value for value in metadata['urls'] if value['packagetype'] == 'bdist_wheel'
                 and TAGS.intersection(parse_wheel_filename(value['filename'])[3]))
    dest = ROOT / 'downloads/wheels' / wheel['filename']
    started = time.time()
    if not dest.exists():
        partial = dest.with_suffix('.whl.partial')
        digest = hashlib.sha256()
        with urllib.request.urlopen(wheel['url'], timeout=180) as source, partial.open('wb') as out:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                digest.update(block)
                out.write(block)
        assert digest.hexdigest() == wheel['digests']['sha256'], name
        assert partial.stat().st_size == wheel['size'], name
        partial.replace(dest)
    else:
        with dest.open('rb') as source:
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == wheel['digests']['sha256'], name
    print(name, dest.stat().st_size, 'SHA256 passed', round(time.time() - started, 2), flush=True)
    return name, {'path': str(dest), 'url': wheel['url'], 'sha256': wheel['digests']['sha256'],
                  'size': wheel['size'], 'version': version}


def main():
    (ROOT / 'downloads/wheels').mkdir(parents=True, exist_ok=True)
    items = [line.split('==') for line in REQ.read_text().splitlines() if line and not line.startswith('#')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        selected = dict(pool.map(fetch, items))
    (ROOT / 'downloads/core-local-wheels.json').write_text(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()
