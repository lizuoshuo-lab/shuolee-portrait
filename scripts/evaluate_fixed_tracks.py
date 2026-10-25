"""Read-only fixed-camera point diagnostics from an explicitly supplied track bank.

This tool does not extract tracks or enable training. It refuses to overwrite a
report and never modifies a source run. A bank must identify its exact camera
manifest, image bytes, point-fit views, held views and observation IDs.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.fixed_tracks import evaluate_fixed_tracks


def evaluate(run, tracks, report):
    run, tracks, report = (Path(p).resolve() for p in (run, tracks, report))
    if report.exists():
        raise FileExistsError(f'Report already exists: {report}')
    if report.is_relative_to(run):
        raise ValueError('Put the diagnostic report outside the immutable source run.')
    bank_hash = rt.sha(tracks)
    bank = rt.read(tracks)
    relative = Path(bank.get('camera_manifest_relative_path', 'data/training-inputs.json'))
    manifest = (run / relative).resolve()
    if relative.is_absolute() or not manifest.is_relative_to(run):
        raise ValueError('Camera manifest must stay inside the supplied source run.')
    manifest_hash = rt.sha(manifest)
    if bank['camera_manifest_sha256'] != manifest_hash:
        raise ValueError('Track bank and current camera manifest do not match.')
    rows = rt.read(manifest)['records']
    for row in rows:
        if rt.sha(row['image']) != row['source_sha256']:
            raise ValueError(f'Image bytes changed: {row["frame"]}')
    result = evaluate_fixed_tracks(rows, bank)
    if rt.sha(manifest) != manifest_hash or rt.sha(tracks) != bank_hash:
        raise ValueError('Camera manifest or track bank changed during evaluation.')
    result.update(camera_manifest_sha256=manifest_hash, track_bank_sha256=bank_hash,
                  image_bytes_verified=True, source_run_modified=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.run, args.tracks, args.report)
    print(json.dumps({k: v for k, v in result.items() if k != 'tracks'}, indent=2))
