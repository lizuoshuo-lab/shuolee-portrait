"""Read-only frame/pose preflight; does not grant training admission."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args()
    try:
        bundle=load_frame_pose_bundle(args.manifest)
        bundle.assert_unchanged()
        print(json.dumps(bundle.report(),ensure_ascii=False,indent=2))
    except (ValueError,KeyError,OSError) as error:
        print(json.dumps(dict(contract_passed=False,error=str(error),training_admitted=False),ensure_ascii=False),file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    sys.exit(main())
