"""Server entry point: fresh raw-video reconstruction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from shuolee_portrait.cli import main
if __name__=='__main__':main()
