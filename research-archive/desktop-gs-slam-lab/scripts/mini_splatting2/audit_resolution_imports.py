"""Record actual renderer modules without running GPU training."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
runpy.run_path(str(Path(__file__).with_name('evaluate_resolution_stage.py')))
records={}
for name in ['diff_gaussian_rasterization','diff_gaussian_rasterization_ms']:
    module=__import__(name);library=module._C.__file__
    records[name]=dict(python=module.__file__,cuda=library,sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())
label=sys.argv[1] if len(sys.argv)>1 else 'initial';assert label in ['initial','corrected']
out=ROOT/('native-resolution-import-audit-'+label+'.json');assert not out.exists()
out.write_text(json.dumps(dict(modules=records,path=sys.path[:8]),indent=2));print(json.dumps(records))
