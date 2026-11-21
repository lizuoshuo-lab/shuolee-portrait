"""Validate both extracted round-five bound packages without a CUDA load."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-main-20261111')
ASSET = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
SNAPSHOT = Path('gaussian-run/step-010000')
INPUTS = ('fit/fitted.npz', 'data/training-inputs.json', 'fit/selection.json',
          'gaussian-run/config.json', 'gaussian-run/training-cameras.json')
SNAPSHOT_FILES = ('bound-local.ply', 'bound-replay.json', 'world.ply',
                  'reference-mesh.npz', 'native-ply-parity.json')
MEMBERS = set(INPUTS) | {(SNAPSHOT / name).as_posix() for name in SNAPSHOT_FILES}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(path, digest):
    from shuolee_portrait import bound_replay as bound

    with path.open('rb') as source:
        with ZipFile(source) as archive, TemporaryDirectory(prefix='shuolee-bound-zip-cpu-') as scratch:
            names = archive.namelist()
            require(len(names) == len(MEMBERS) and set(names) == MEMBERS, f'Unexpected replay ZIP members: {path}')
            require(sum(info.file_size for info in archive.infolist()) <= 512 * 1024**2, 'Unexpected replay ZIP size.')
            root = Path(scratch)
            for name in names:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as stream, target.open('xb') as output:
                    shutil.copyfileobj(stream, output)
            source.seek(0)
            zip_sha256 = hashlib.file_digest(source, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else bound.rt.sha(path)
            require(digest is None or zip_sha256 == digest, f'Replay ZIP SHA differs: {path}')
            snapshot = root / SNAPSHOT
            manifest_path = snapshot / bound.MANIFEST
            manifest = bound.rt.read(manifest_path)
            ply = snapshot / bound.LOCAL_PLY
            require(manifest['schema'] == bound.SCHEMA and manifest['bound_ply'] == bound.LOCAL_PLY
                    and manifest['sh_degree'] == 0 and manifest['optimizer_updates'] == 0
                    and bound.rt.sha(ply) == manifest['bound_ply_sha256'], f'Invalid bound replay manifest: {path}')
            fit, _, facts = bound._inputs(root, snapshot, ASSET, manifest)
            binding, counts = bound.inspect_bound_ply(ply, len(fit['faces']))
            require(len(binding) == manifest['points'], f'Extracted point count differs: {path}')
            selection = bound.rt.read(root / 'fit/selection.json')
            cameras = bound.rt.read(root / 'gaussian-run/training-cameras.json')
            config = bound.rt.read(root / 'gaussian-run/config.json')
            ids = [frame['frame'] for frame in facts['frames']]
            require(set(ids) == set(selection['selected_frames']) and set(ids).isdisjoint(selection['retained_holdouts'])
                    and ids == [row['frame'] for row in cameras['records']]
                    and cameras['source_sha256'] == facts['inputs_sha256']['training_rows']
                    and config['gaussian_profile'] == 'smpl_sh0' and config['iterations'] == 10000,
                    f'Extracted schedule, camera or protected frame roster differs: {path}')
            return dict(zip=str(path), zip_sha256=zip_sha256, snapshot=SNAPSHOT.name,
                        bound_manifest_sha256=bound.rt.sha(manifest_path), points=len(binding),
                        minimum_points_per_triangle=int(counts.min()), pose_slots=len(ids),
                        fit_sha256=facts['inputs_sha256']['fitted'],
                        asset_sha256=facts['asset_sha256'], source_files_sha256=facts['source_files_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for subject in ('016', '006'):
        parser.add_argument(f'--zip-{subject}', type=Path, required=True)
        parser.add_argument(f'--sha-{subject}')
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(PROJECT / 'src'))
    result = dict(schema='shuolee-smpl-round5-extracted-bound-cpu-v1', state='cpu_extracted_readback_passed',
                  gpu_calls=0, quality_accepted=False, subjects={subject: validate(args[f'zip_{subject}'], args[f'sha_{subject}'])
                    for subject in ('016', '006')})
    torch = sys.modules.get('torch')
    require(torch is None or not torch.cuda.is_initialized(), 'Unexpected CUDA initialization.')
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args['output']:
        with args['output'].open('x', encoding='utf-8') as output:
            output.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
