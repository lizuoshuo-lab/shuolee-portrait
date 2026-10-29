"""Bootstrap only the pinned official source, before installing Python packages."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from shuolee_portrait.official_deployment import prepare_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    print(prepare_source(args.root, args.source_root)['source_root'])


if __name__ == '__main__': main()
