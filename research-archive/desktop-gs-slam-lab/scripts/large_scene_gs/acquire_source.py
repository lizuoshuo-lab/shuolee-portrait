"""Download a pinned GitHub source archive without changing global Git settings."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    if len(a.commit) != 40 or any(c not in '0123456789abcdef' for c in a.commit):
        raise ValueError('Exact hexadecimal commit required')
    if len(a.repo.split('/')) != 2 or not a.name.replace('-', '').isalnum():
        raise ValueError('Invalid repository or local name')
    destination = ROOT / 'repos' / a.name
    archive = ROOT / 'outputs' / 'large-scene-gs-001' / 'source-archives' / f'{a.name}-{a.commit}.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://codeload.github.com/{a.repo}/zip/{a.commit}'
    if not archive.exists():
        part = archive.with_suffix('.zip.partial')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response, part.open('wb') as f:
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > 100 * 1024 * 1024:
                            raise ValueError('Source archive exceeds 100 MiB')
                        f.write(block)
                part.replace(archive)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        total = sum(x.file_size for x in z.infolist())
        if total > 500 * 1024 * 1024:
            raise ValueError('Expanded source exceeds 500 MiB')
        for entry in z.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or len(path.parts) < 1:
                raise ValueError('Unsafe archive path')
            relative = Path(*path.parts[1:])
            target = destination / relative
            if destination.resolve() not in target.resolve().parents and target.resolve() != destination.resolve():
                raise ValueError('Source escaped repository directory')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = z.read(entry)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Existing modified source preserved: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    receipt = {'repo': a.repo, 'commit': a.commit, 'archive_url': url,
               'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'source_path': str(destination), 'expanded_bytes': total,
               'license_file': str(destination / 'LICENSE')}
    (archive.parent / f'{a.name}.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
