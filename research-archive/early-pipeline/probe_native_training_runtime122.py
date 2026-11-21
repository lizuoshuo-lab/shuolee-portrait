"""Use the same existing gsplat source checkout as completed R1 suites."""
from pathlib import Path
import sys, json, hashlib
import probe_native_training122 as probe

probe.O=probe.O/'existing-backend-path'
plan=json.loads((probe.O/'plan.json').read_text(encoding='utf-8-sig'))
repo=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
for name,digest in plan['runtime_source_hashes'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(repo))
probe.run()
