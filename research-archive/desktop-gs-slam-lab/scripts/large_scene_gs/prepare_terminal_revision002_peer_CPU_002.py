"""Preserve001 fixture initialization omission and bind actual launch metadata in002."""
from pathlib import Path
import json,hashlib
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
source=ROOT/'scripts/large_scene_gs/review_post006_terminal_revision002_CPU_001.py'
failure=ROOT/'outputs/large-scene-gs-001/recovery-audit/post006-terminal-revision002-independent-review-CPU-001/failure.json'
assert not failure.exists()
failure.write_text(json.dumps({'status':'CPU_fixture_failed_preserved','actual_exit_code':1,
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'error':"NameError: global launch not initialized in independent fixture",
    'production_main_initializes_launch':True,'production_parser_or_reconciliation_defect_claimed':False,
    'release_or_GPU_executed':False},indent=2)+'\n',encoding='utf-8')
s=source.read_text(encoding='utf-8').replace('post006-terminal-revision002-independent-review-CPU-001','post006-terminal-revision002-independent-review-CPU-002')
s=s.replace("r.recorded_stage_parent_PIDs={7940}","r.launch=read(BASE/'official-hierarchy-stage-supervision-launch-006/city006-post00-i64-stage-001/launch-receipt.json')\nr.recorded_stage_parent_PIDs={7940}")
target=ROOT/'scripts/large_scene_gs/review_post006_terminal_revision002_CPU_002.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
