"""Fresh video-only camera/observation preparation; never starts training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.video119 import main

if __name__ == '__main__':
    main()
