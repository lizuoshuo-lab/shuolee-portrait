"""Extract current-video texture candidates without changing the source run."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait.material_tracks import extract_material_tracks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('The standalone diagnostic output must be outside the source run.')
    result = extract_material_tracks(args.run, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('features', 'pairs')}, indent=2))
