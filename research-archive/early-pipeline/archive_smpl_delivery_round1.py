"""Verify round-one receipts and archive same-frame image comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive_head_appearance_records as common


def run_root(subject):
    name = 'head-appearance' if subject == '016' else 'delivery-round1'
    return common.LAB / f'shuolee-smpl-{name}-{subject}'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_scene(run, result):
    scene = run / result['scene']
    common.require(common.sha(scene) == result['scene_sha256'], 'Static ZIP differs from export receipt.')
    mesh = scene.parent / 'reference-mesh.obj'
    camera = run / result['camera']
    ply = run / result['ply']
    manifest = read(scene.parent / 'scene.json')
    members = ((manifest['model'], ply), (manifest['cameras'], camera), (manifest['referenceMesh'], mesh))
    with ZipFile(scene) as package:
        names = package.namelist()
        common.require(package.testzip() is None, 'Corrupted static ZIP member.')
        for item, path in members:
            names_for_file = [name for name in names if Path(name).name == item['file']]
            common.require(len(names_for_file) == 1 and common.sha(path) == item['sha256']
                           and hashlib.sha256(package.read(names_for_file[0])).hexdigest() == item['sha256'],
                           f'Packaged asset differs: {path}')
    common.require(manifest['referenceFrameIsPoseSnapshot'] is True
                   and manifest['referenceMesh']['referenceFrameId'] == manifest['referenceFrameId']
                   and manifest['referenceFrameId'] == result['reference_frame_id'] + '.png'
                   and manifest['cameras']['count'] == 1,
                   'Static camera and mesh refer to different poses.')
    return mesh


def bundle(run, snapshot, result, output):
    """Keep the portable scene and exact local replay inputs together for handoff."""
    output.mkdir(parents=True, exist_ok=True)
    src = run / result['scene']
    dst = output / 'scene.zip'
    if not dst.exists():
        with src.open('rb') as source, dst.open('xb') as target:
            shutil.copyfileobj(source, target)
    common.require(common.sha(dst) == result['scene_sha256'], 'Delivery scene ZIP differs.')
    files = [run / name for name in ('fit/fitted.npz', 'data/training-inputs.json',
                                    'fit/selection.json', 'gaussian-run/config.json',
                                    'gaussian-run/training-cameras.json')]
    files += [snapshot / name for name in ('bound-local.ply', 'bound-replay.json', 'world.ply',
                                           'reference-mesh.npz', 'native-ply-parity.json')]
    bindings = output / 'bound-replay.zip'
    if not bindings.exists():
        with ZipFile(bindings, 'x') as package:
            for path in files:
                package.write(path, path.relative_to(run).as_posix())
    with ZipFile(bindings) as package:
        common.require(len(package.namelist()) == len(files) and package.testzip() is None,
                       'Delivery replay package has invalid members.')
        for path in files:
            name = path.relative_to(run).as_posix()
            common.require(hashlib.sha256(package.read(name)).hexdigest() == common.sha(path),
                           f'Delivery replay input differs: {name}')
    return dict(scene=str(dst), scene_sha256=common.sha(dst), bound_replay=str(bindings),
                bound_replay_sha256=common.sha(bindings))


def collect(subject, publish, delivery):
    root = run_root(subject)
    run = root / 'run'
    comparison_dir = common.LAB / f'shuolee-smpl-delivery-round1-{subject}' / 'view-comparison'
    terminal = read(root / 'wrapper-terminal.json')
    result = read(run / 'result.json')
    training = read(run / 'gaussian-run/config.json')
    runtime = read(run / 'gaussian-run/result.json')
    require = common.require
    sha = common.sha
    result_hash = sha(run / 'result.json')
    require(terminal['candidate_commit'] == common.COMMIT and terminal['result_sha256'] == result_hash,
            'Round-one production revision or terminal differs.')
    require(training['native_resolution'] and training['initial_points'] == 20908
            and training['optimizer_initial_state_entries'] == 0
            and training['opacity_reset_steps'] == [1800, 3600, 5400, 7200]
            and training['sampling_policy'] == 'seeded_equal_occupied_source_bin_cycles'
            and runtime['steps'] == 10000 and runtime['schedule_steps'] == 18000,
            'Round-one training profile differs.')
    mesh = verify_scene(run, result)
    review_path = comparison_dir / 'result.json'
    review = read(review_path)
    require(review.get('subject', subject) == subject, 'Comparison receipt belongs to another person.')
    records = common.RECORDS / subject
    require(records.resolve(strict=True) == records, 'Use the existing record folder.')
    prefix = '分析_' + subject + '_交付01_762d6d0_'
    with TemporaryDirectory(prefix='shuolee-round1-record-') as scratch:
        files, pins = [], {}
        for row in review['panels']:
            filename = Path(row['file'])
            require(filename.name == row['file'] and filename.suffix == '.png', 'Invalid panel path.')
            path = comparison_dir / filename
            digest = sha(path)
            require(digest == row['sha256'], f'Comparison source changed: {path}')
            pins[path] = digest
            frame = row['frame']
            require(re.fullmatch(r'frame_\d{6}', frame), 'Invalid comparison frame.')
            target = records / (prefix + filename.stem + '.jpg')
            tmp = Path(scratch) / target.name
            with Image.open(path) as image:
                image.convert('RGB').save(tmp, quality=95, subsampling=0)
            if target.exists():
                require(sha(tmp) == sha(target), f'Preserve different existing record: {target}')
            else:
                files.append((tmp, target))
        if subject == '006':
            common.archive(subject, result_hash if publish else None, root=root,
                           expected_count=24, tag='交付01_762d6d0')
        else:
            common.archive(subject)
        if publish and files:
            common.publish_images(files, pins)
    snapshot = run / 'gaussian-run' / result['final_snapshot']
    packages = bundle(run, snapshot, result, delivery / subject) if delivery else None
    return dict(subject=subject, commit=common.COMMIT, result=str(run / 'result.json'),
        result_sha256=result_hash, scene=str(run / result['scene']), ply=str(run / result['ply']),
        cameras=str(run / result['camera']), mesh=str(mesh),
        bound_ply=str(snapshot / 'bound-local.ply'), bound_manifest=str(snapshot / 'bound-replay.json'),
        posed_replay=str(run / result['bound_replay']), native_review=str(review_path),
        native_review_sha256=sha(review_path), records=str(records), frames=result['frames'],
        points=result['points'], reference=result['reference_frame_id'], coverage=terminal['coverage'], packages=packages,
        published=publish, quality_accepted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', choices=('016', '006'))
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--delivery', type=Path)
    options = parser.parse_args()
    receipt = collect(options.subject, options.publish, options.delivery)
    if options.delivery:
        path = options.delivery / options.subject / 'receipt.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    print(json.dumps(receipt, ensure_ascii=False))
