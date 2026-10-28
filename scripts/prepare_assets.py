"""Prepare pinned server models without changing an existing model directory."""
import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.request
from urllib.parse import urlsplit
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / 'configs/server-assets.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def matches(path, row):
    if not path.is_file() or path.stat().st_size != row['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == row['sha256']


def candidates(root, environment, explicit=None):
    found = [Path(explicit).expanduser()] if explicit else []
    for profile in (root / 'queue/deployment.json',
                    Path.home() / 'shuolee/queue/deployment.json',
                    environment / 'install/deployment119.json'):
        if profile.is_file():
            try:
                found.append(Path(read(profile)['assets']).expanduser())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f'Ignoring unreadable model profile {profile}: {exc}', file=sys.stderr)
    found.extend((root / 'models', root / 'assets',
                  Path.home() / 'shuolee/models/113', Path('/data/shuolee-models')))
    return list(dict.fromkeys(p.resolve() for p in found))


def download(url, destination, row):
    """Resume interrupted transfers; publish only a matching complete file."""
    if matches(destination, row):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    proxies = urllib.request.getproxies()
    # urllib does not map ALL_PROXY automatically, unlike common download CLIs.
    if urlsplit(proxies.get('all', '')).scheme in ('http', 'https'):
        proxies.setdefault('http', proxies['all'])
        proxies.setdefault('https', proxies['all'])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    last_error = None
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= row['bytes']:
                if matches(partial, row):
                    partial.replace(destination)
                    return
                partial.unlink()
                offset = 0
            headers = {'User-Agent': 'smpl-x-model-setup/1', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with opener.open(request, timeout=45) as response:
                append = response.status == 206 and offset > 0
                if response.status == 206:
                    expected = f'bytes {offset}-'
                    if not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Unexpected download Content-Range')
                if not append:
                    offset = 0
                print(f'Downloading {destination.name} ({offset}/{row["bytes"]} bytes)',
                      file=sys.stderr, flush=True)
                updated = time.monotonic()
                with partial.open('ab' if append else 'wb') as stream:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        offset += len(block)
                        if offset > row['bytes']:
                            raise ValueError('Downloaded file exceeds pinned size')
                        stream.write(block)
                        if time.monotonic() - updated >= 10:
                            print(f'  {destination.name}: {offset}/{row["bytes"]} bytes',
                                  file=sys.stderr, flush=True)
                            updated = time.monotonic()
            if not matches(partial, row):
                if partial.stat().st_size >= row['bytes']:
                    partial.unlink()
                raise ValueError('Downloaded size or SHA-256 does not match the model manifest')
            partial.replace(destination)
            return
        except (OSError, ValueError, http.client.HTTPException) as exc:
            last_error = exc
            print(f'Model download attempt {attempt + 1}/3 failed: {exc}', file=sys.stderr)
    raise RuntimeError(f'Cannot download {destination.name} from {url}: {last_error}. '
                       'Rerun deploy.sh to continue; verified files are reused.')


def copy_verified(source, destination, row):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    shutil.copyfile(source, partial)
    if not matches(partial, row):
        partial.unlink()
        raise RuntimeError(f'Model changed while copying: {source}')
    partial.replace(destination)


@contextmanager
def model_lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prepare(root, sources, manifest=None):
    manifest = read(MANIFEST) if manifest is None else manifest
    rows = manifest['files']
    # Check every entry, not just the licensed template used by the old fitter.
    for source in sources:
        if all((source / row['path']).is_file() for row in rows):
            if all(matches(source / row['path'], row) for row in rows):
                print(f'Models verified: {source} ({len(rows)} files)', file=sys.stderr)
                return source

    destination = root / 'models'
    with model_lock(destination):
        # Locate and validate licensed files before downloading any public model.
        available = {}
        for row in rows:
            for source in dict.fromkeys([destination, *sources]):
                path = source / row.get('reuse', row['path'])
                if matches(path, row):
                    available[row['path']] = path
                    break
        missing_licensed = [row['path'] for row in rows
                            if not row.get('download') and not row.get('reuse')
                            and row['path'] not in available]
        if missing_licensed:
            raise RuntimeError('Licensed server models are missing or have incorrect hashes: '
                               f'{missing_licensed}. Set SMPLX_ASSETS to the existing authorized '
                               'model directory. See docs/models.md. No training was started.')
        # Preserve old server assets: assemble into the new runtime directory.
        for row in rows:
            target = destination / row['path']
            if matches(target, row):
                continue
            source = available.get(row['path'])
            if source is not None and source != target:
                copy_verified(source, target, row)
        for row in rows:
            target = destination / row['path']
            if matches(target, row):
                continue
            if row.get('reuse'):
                source = destination / row['reuse']
                if not matches(source, row):
                    raise RuntimeError(f'Cannot reuse model: {source}')
                copy_verified(source, target, row)
                continue
            spec = row['download']
            if 'archive' not in spec:
                download(spec['url'], target, row)
                continue
            archive_row = manifest['archives'][spec['archive']]
            archive = destination / '.downloads' / spec['archive']
            download(archive_row['url'], archive, archive_row)
            # Extract only explicitly named members, never arbitrary ZIP paths.
            with zipfile.ZipFile(archive) as package:
                for member_row in rows:
                    member_spec = member_row.get('download', {})
                    if member_spec.get('archive') != spec['archive']:
                        continue
                    output = destination / member_row['path']
                    if matches(output, member_row):
                        continue
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.extract-') as tmp:
                        staged = Path(tmp) / 'model'
                        if member_spec.get('normalize'):
                            pipeline = json.loads(package.read(member_spec['member']))
                            transforms = pipeline['pipeline']['tasks'][0]['transforms']
                            normalization = next(t for t in transforms if t['type'] == 'Normalize')
                            text = json.dumps(normalization, indent=2) + '\n'
                            if member_spec.get('newline') == 'crlf':
                                text = text.replace('\n', '\r\n')
                            staged.write_bytes(text.encode('utf-8'))
                        else:
                            with package.open(member_spec['member']) as stream, staged.open('wb') as out:
                                shutil.copyfileobj(stream, out)
                        if not matches(staged, member_row):
                            raise RuntimeError(f'Extracted model does not match manifest: {output}')
                        staged.replace(output)
            archive.unlink()
        bad = [row['path'] for row in rows if not matches(destination / row['path'], row)]
        if bad:
            raise RuntimeError(f'Model preparation incomplete: {bad}')
        print(f'Models verified: {destination} ({len(rows)} files)', file=sys.stderr)
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--assets')
    args = parser.parse_args()
    try:
        result = prepare(args.root.resolve(), candidates(args.root, args.environment, args.assets))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
    print(result)


if __name__ == '__main__':
    main()
