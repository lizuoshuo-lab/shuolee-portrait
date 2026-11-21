"""Remove only this task's disposable second installation environment after verified acceptance."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil

base=Path('/opt/gs-slam-lab/shuolee-portrait-clean-002')
target=base/'env'
evidence=base/'install-evidence'
summary=json.loads((evidence/'summary.json').read_text())
assert summary['state']=='private_git_clone_and_uninterrupted_new_venv_install_passed'
assert not summary['reconstruction_quality_accepted']
assert target.resolve()==Path('/opt/gs-slam-lab/shuolee-portrait-clean-002/env')
assert not target.is_symlink() and target.parent.resolve()==base
assert (target/'pyvenv.cfg').is_file()
assert (target/'lib/python3.10/site-packages').is_dir()
# Only this disposable environment is removed. The first, actually exercised
# video environment, all inputs/results, Git checkout and download cache remain.
keeper=Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env')
assert (keeper/'bin/python').exists()
assert (keeper/'install/environment-check-complete.json').is_file()
for name in ['setup.log','installed.txt','environment-check.json','doctor.json','projection-export.json','package-versions.json']:
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(target/'install'/name)==digest(evidence/name)
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try: command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if str(target) in command:raise RuntimeError('Environment still used by PID '+proc.name)
size=0;count=0
for root,dirs,files in os.walk(target,followlinks=False):
    for name in files:
        path=Path(root)/name
        if not path.is_symlink():size+=path.stat().st_size;count+=1
record=dict(created=datetime.now(timezone.utc).isoformat(),target=str(target),
    resolved_target=str(target.resolve()),state='prepared',files=count,logical_bytes=size,
    keeper=str(keeper),evidence=str(evidence),training_outputs_or_assets_removed=False,
    source_checkout_removed=False,download_cache_removed=False)
receipt=base/'temporary-environment-cleanup.json'
receipt.write_text(json.dumps(record,indent=2)+'\n')
shutil.rmtree(target)
assert not target.exists() and (keeper/'bin/python').exists()
record.update(state='complete',completed=datetime.now(timezone.utc).isoformat())
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
