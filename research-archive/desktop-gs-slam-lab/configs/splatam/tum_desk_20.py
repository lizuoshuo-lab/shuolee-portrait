from pathlib import Path
import runpy

config = runpy.run_path(str(Path(__file__).with_name("_common.py")))["make_config"](20)
