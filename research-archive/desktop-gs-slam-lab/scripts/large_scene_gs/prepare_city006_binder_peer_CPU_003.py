"""Preserve002 detected concurrent source revision; review restored b275 in distinct003."""
from pathlib import Path
import json,hashlib
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
src=ROOT/'scripts/large_scene_gs/review_city006_anchor_boundary_perceptual_CPU_002.py'
fail=ROOT/'outputs/large-scene-gs-001/recovery-audit/city006-anchor-boundary-perceptual-independent-review-CPU-002/failure.json'
assert not fail.exists();fail.write_text(json.dumps({'status':'CPU_peer_source_binding_rejected_preserved','actual_exit_code':1,
    'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'error':'b275 anchor source pin changed at initial check during owner pose-byte-gate revision',
    'actual_briefly_observed_anchor_source_SHA':'0148a71ae3ccf9678a6824e06f83cf1cc73c909db3001695ed09a509736ed1a2',
    'owner_restored_original_b275_bytes_then_created_separate_posebytes_source':True,
    'numerical_or_main_or_GPU_controls_started':False,'reviewer_modified_owner_source':False},indent=2)+'\n',encoding='utf-8')
s=src.read_text(encoding='utf-8').replace('city006-anchor-boundary-perceptual-independent-review-CPU-002','city006-anchor-boundary-perceptual-independent-review-CPU-003')
target=ROOT/'scripts/large_scene_gs/review_city006_anchor_boundary_perceptual_CPU_003.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
