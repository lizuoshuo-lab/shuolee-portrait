from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

project = Path('D:/00/工作/人体重建')
base = Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab')
source = base/'shuolee-portrait-clean-001/source'
failed = base/'shuolee-portrait-server-check-007'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

assert read(failed/'status.json')['state']=='failed'
assert 'ModuleNotFoundError: No module named \'wis3d\'' in (failed/'logs/prior.log').read_text()
assert not (base/'shuolee-portrait-server-check-008').exists()
now = datetime.now(timezone.utc).isoformat()
save(project/'provenance/server-interface-check-007.json', {
    'updated': now, 'state': 'failed_preserved', 'stage': 'prior',
    'completed': read(failed/'status.json')['completed'],
    'cause': 'Missing wis3d dependency imported by upstream GVHMR configuration and inference modules.',
    'secondary_preflight_issue': 'SAM2 and GVHMR initialize different Hydra roots; isolated inference configuration now restores the caller context.',
    'camera_or_quality_failure': False, 'gaussian_training_started': False,
    'repair': 'Pin wis3d; compose the actual prior graph in doctor; rerun from raw video in a new directory.',
    'resume_failed_run': False, 'stable_release': False})
files = list((source/'src/shuolee_portrait').glob('*.py'))
files += [source/p for p in ['run.py','configs/smoke.json','configs/server-assets.json','upstream.lock.json','requirements.txt','constraints-linux-cu128.txt']]
manifest = {p.relative_to(source).as_posix(): sha(p) for p in files}
save(project/'provenance/server-interface-check-008-plan.json', {
    'created': now, 'purpose': 'Fresh raw-video integration check in isolated environment after dependency repair.',
    'source_hashes': manifest, 'environment': '/opt/gs-slam-lab/shuolee-portrait-clean-001/env',
    'output': '/opt/gs-slam-lab/shuolee-portrait-server-check-008',
    'config': 'configs/smoke.json', 'stop_after': 'fit-gate',
    'raw_video': 'Existing HOVER_VERSA_0102 reference video; not a new person or capture.',
    'from_scratch': True, 'prior_subject_or_gaussians_loaded': False,
    'gaussian_training_allowed': False, 'camera_causal_claim_allowed': False})
save(base/'shuolee-portrait-clean-001/env/install/check008-source.json', manifest)
print('Failure evidence and immutable source hashes recorded; new output does not exist.')
