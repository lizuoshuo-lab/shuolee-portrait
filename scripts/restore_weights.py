#!/usr/bin/env python3
"""Restore lossless WPS weight parts; Python standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'Unsafe manifest path: {relative}')
    return path


def restore(bundle, destination, verify_only=False):
    bundle, destination = bundle.resolve(), destination.resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        for part in item['parts']:
            source = within(bundle, part['path'])
            if not source.is_file() or source.stat().st_size != part['bytes'] or digest(source) != part['sha256']:
                raise RuntimeError(f'Missing or damaged part: {source}. Download this part again.')
        if verify_only:
            h = hashlib.sha256()
            total = 0
            for part in item['parts']:
                with within(bundle, part['path']).open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        h.update(chunk); total += len(chunk)
            if total != item['bytes'] or h.hexdigest() != item['sha256']:
                raise RuntimeError(f'Concatenated checksum failed: {item["target"]}')
            print(f'PASS {item["target"]}: all parts and original SHA-256 match', flush=True)
            continue
        target = within(destination, item['target'])
        if target.exists():
            if target.stat().st_size == item['bytes'] and digest(target) == item['sha256']:
                print(f'Already verified: {target}', flush=True)
                continue
            raise FileExistsError(f'Refusing to overwrite a different file: {target}')
        if os.name == 'nt' and destination.is_relative_to(PROJECT):
            raise RuntimeError('Do not restore >2 GB weights inside this WPS project on Windows. Use --destination D:/GS-SLAM-Data/shuolee-portrait-runtime-assets.')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.assembling')
        if temp.exists():
            raise FileExistsError(f'Previous incomplete assembly exists; inspect it before retrying: {temp}')
        h = hashlib.sha256()
        total = 0
        with temp.open('xb') as out:
            for part in item['parts']:
                with within(bundle, part['path']).open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        out.write(chunk); h.update(chunk); total += len(chunk)
        if total != item['bytes'] or h.hexdigest() != item['sha256']:
            raise RuntimeError(f'Assembly checksum failed; retained for diagnosis: {temp}')
        temp.replace(target)
        print(f'Restored and verified: {target}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', type=Path, default=PROJECT / 'weight-parts')
    parser.add_argument('--destination', type=Path, default=PROJECT / 'assets')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--copy-small-assets', action='store_true', help='Also copy non-split assets to an external runtime directory.')
    args = parser.parse_args()
    restore(args.parts, args.destination, args.verify_only)
    if args.copy_small_assets and not args.verify_only:
        for source in (PROJECT / 'assets').rglob('*'):
            if not source.is_file():
                continue
            target = within(args.destination, source.relative_to(PROJECT / 'assets'))
            if source.resolve() == target.resolve():
                continue
            if target.exists():
                if digest(source) != digest(target):
                    raise FileExistsError(f'Different asset already exists: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


if __name__ == '__main__':
    main()
