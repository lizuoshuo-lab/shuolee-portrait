"""Background video-folder worker; deployed by deploy119.sh."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from shuolee_portrait.queue119 import main

if __name__ == '__main__':
    main()
