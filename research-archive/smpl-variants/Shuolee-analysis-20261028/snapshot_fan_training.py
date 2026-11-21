"""Record source identity for the running, isolated FAN training control."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
root=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/gray-dynamic-face-fan')
source=Path('/mnt/d/Projects/Shuolee-analysis-20261028/train_head_motion.py')
snapshot=root/'code/train_head_motion.py'
snapshot.parent.mkdir(exist_ok=True)
assert not snapshot.exists()
shutil.copyfile(source,snapshot)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_base_commit='9225e1da6a18fe6ae8aac63ec48bafd65a3794a1',
    training_script_sha256=sha(snapshot),code_snapshot=str(snapshot),
    snapshot_timing='Captured during the run; training script unchanged since launch',
    production_source_modified=False)
(root/'source-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
