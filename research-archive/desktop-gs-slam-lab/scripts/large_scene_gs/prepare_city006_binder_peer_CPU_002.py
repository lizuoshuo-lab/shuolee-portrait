"""Preserve001 early hash-literal typo; continue unchanged CPU-only review in002."""
from pathlib import Path
import json,hashlib
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
src=ROOT/'scripts/large_scene_gs/review_city006_anchor_boundary_perceptual_CPU_001.py'
fail=ROOT/'outputs/large-scene-gs-001/recovery-audit/city006-anchor-boundary-perceptual-independent-review-CPU-001/failure.json'
assert not fail.exists();fail.write_text(json.dumps({'status':'CPU_peer_fixture_failed_preserved','actual_exit_code':1,
    'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'error':'Old CPU SHA literal transcribed with extra e; failed before numerical/main controls.',
    'reviewed_source_or_algorithm_fault_established':False,'GPU_or_CUDA_or_role_mock_started':False},indent=2)+'\n',encoding='utf-8')
s=src.read_text(encoding='utf-8').replace('city006-anchor-boundary-perceptual-independent-review-CPU-001','city006-anchor-boundary-perceptual-independent-review-CPU-002')
s=s.replace('1ec28a295b93869c7da414f830ee2d70f8c1c4a02ef1f763de36ac8cfeeb6f9fc','1ec28a295b93869c7da414f830e2d70f8c1c4a02ef1f763de36ac8cfeeb6f9fc')
target=ROOT/'scripts/large_scene_gs/review_city006_anchor_boundary_perceptual_CPU_002.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
