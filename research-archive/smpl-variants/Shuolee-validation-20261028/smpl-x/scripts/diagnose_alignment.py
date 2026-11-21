"""Read-only source-run region diagnostics of a native pre-SfM holdout."""
import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.alignment_diagnostics import diagnose

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('Standalone report must be outside the immutable source run.')
    result=diagnose(args.run,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('regions','source_hashes')},indent=2))
    raise SystemExit(0 if result['measured_tracks'] else 3)
