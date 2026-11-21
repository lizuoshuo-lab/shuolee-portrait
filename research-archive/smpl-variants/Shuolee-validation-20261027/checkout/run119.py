"""Portable Git checkout entry for the 119 video reconstruction workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from shuolee_portrait.route119 import main

if __name__ == '__main__':
    main()
