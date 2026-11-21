"""Preserve001 wrong fixture constant name; compare actual immutable PILOT_PIN in002."""
from pathlib import Path
import hashlib,json
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
source=ROOT/'scripts/large_scene_gs/review_small_city_supervision_revision002_CPU_001.py'
failure=ROOT/'outputs/large-scene-gs-001/recovery-audit/small-city-thin-supervision-revision002-independent-review-CPU-001/failure.json'
assert not failure.exists()
failure.write_text(json.dumps({'status':'CPU_peer_fixture_failed_preserved','actual_exit_code':1,
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'error':'Fixture referenced nonexistent PILOT_SHA; immutable supervisor actually uses PILOT_PIN source lock',
    'production_source_or_model_fault_claimed':False,'main_or_GPU_or_release_executed':False},indent=2)+'\n',encoding='utf-8')
s=source.read_text(encoding='utf-8').replace('small-city-thin-supervision-revision002-independent-review-CPU-001','small-city-thin-supervision-revision002-independent-review-CPU-002').replace("'PILOT_SHA'","'PILOT_PIN'")
target=ROOT/'scripts/large_scene_gs/review_small_city_supervision_revision002_CPU_002.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
