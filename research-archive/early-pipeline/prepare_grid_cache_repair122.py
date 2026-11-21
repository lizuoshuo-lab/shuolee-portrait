"""Prepare exact array-access repair after confirmed terminated first attempt."""
from pathlib import Path
import json,hashlib
H=Path(__file__).parent
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
O=I/'clip03-grid-coverage'
failure=json.loads((O/'failure.json').read_text(encoding='utf-8'))
assert "d['descriptor'][m]" in failure['traceback'] and 'decompress' in failure['traceback']
src=H/'compare_white_coverage122.py';s=src.read_text(encoding='utf-8')
s=s.replace("O=I/'clip03-grid-coverage';P=I/'clip03-grid-coverage-plan.json'", "O=I/'clip03-grid-coverage/array-cache-repair';P=I/'clip03-grid-coverage/array-cache-repair-plan.json'")
old="d=np.load(W/f'features_{f:06d}.npz');kp=sift.detect(gray,None)"
new="archive=np.load(W/f'features_{f:06d}.npz');d={name:archive[name] for name in archive.files};archive.close();kp=sift.detect(gray,None)"
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("CPU_seconds=180", "CPU_seconds=95,original_total_budget_seconds=180,prior_active_seconds=84.687")
s=s.replace("time.monotonic()-started<180", "time.monotonic()-started<95")
s=s.replace("files += [W/f'features_{f:06d}.npz'", "files += [I/'clip03-grid-coverage-plan.json',I/'clip03-grid-coverage/failure.json',Path(__file__).parent/'compare_white_coverage122.py']\n    files += [W/f'features_{f:06d}.npz'")
s=s.replace("source_sha256=sha(Path(__file__)),controls=controls(),", "source_sha256=sha(Path(__file__)),controls=controls(),implementation_repair='First attempt was interrupted at repeatedNPZdecompression in equalitycheck;3.364GBread,84.687active seconds. Cacheeacharrayonce. Samefixedselection/matching/familypolicy;no firstattemptmatching result existed. Remaining95seconds staysbelow original180total. Originalscript/plan/failurepreserved.',")
dst=H/'compare_white_coverage122_repaired.py';assert not dst.exists();dst.write_text(s,encoding='utf-8')
audit=dict(reason='NpzFile __getitem__ decompresses full descriptorarray for each sharedpoint comparison',
 traceback_evidence=str(O/'failure.json'),process_observation=dict(pid=387,rchar=3364717794,status='SIGINT terminated; execsession12623 exited1; ps-p387 missing'),
 first_execution_started='2026-10-27T19:17:10.112012578+00:00',interrupted=failure['updated'],prior_active_seconds=84.687,
 repair='Read all NPZ arrays once into dictionary; values/identity checks unchanged',remaining_execution_budget_seconds=95,original_total_budget_seconds=180,
 original_script_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),repaired_script_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
 original_artifacts_preserved=True,K_or_match_parameters_changed=False)
(O/'implementation-repair.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print('CACHE_REPAIR_PREPARED_WITH_REMAINING95S_BUDGET')
