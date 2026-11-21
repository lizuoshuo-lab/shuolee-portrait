"""Independently verify final continuation checkpoint and input membership."""
from pathlib import Path
import hashlib,json,torch,shutil
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
PARENT=LAB/'outputs/dynamic-human-4k4d-official-035'
OUT=PARENT/'continuation-10000'
torch.set_num_threads(4)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
completion=json.loads((OUT/'training-completion.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
assert completion['completed_updates_this_process']==9000 and completion['history_updates']==10000
assert digest(PARENT/'model/latest.pt')==protocol['parent_sha256']
assert digest(OUT/'model/latest.pt')==completion['checkpoint_sha256']
c=torch.load(OUT/'model/latest.pt',map_location='cpu',weights_only=False)
assert c['epoch']==179
steps=[int(v['step']) for v in c['optimizer']['state'].values() if 'step' in v]
assert steps and max(steps)==10000
assert all(bool(torch.isfinite(v).all()) for v in c['model'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
assert {k:list(v.shape) for k,v in c['model'].items() if k.startswith('sampler.pcds.')}=={'sampler.pcds.0':[262144,3]}
checks=[]
for line in (OUT/'training.log').read_text(errors='replace').splitlines():
    if line.startswith('AUDITED_STEP '):checks.append(json.loads(line[len('AUDITED_STEP '):]))
assert [x['completed_updates_this_process'] for x in checks]==[1]+list(range(50,9001,50))
assert all(x['history_updates']==1000+x['completed_updates_this_process'] for x in checks)
assert all(x['target'] not in [0,6,12] and not set(x['sources'])&{0,6,12} for x in checks)
config=json.loads((OUT/'resolved-configuration.json').read_text())
assert config['val_dataloader_cfg']['dataset_cfg']['src_view_sample']==[1,2,3,4,5,7,8,9,10,11,13,14,15,16,17]
load=json.loads((OUT/'parent-load-audit.json').read_text())
assert load['exact_model_parameters'] and load['optimizer_steps']==[1000]
report=dict(status='continuation10000_checkpoint_and_logged_membership_verified',new_updates=9000,parent_updates=1000,history_updates=10000,
    checkpoint_sha256=completion['checkpoint_sha256'],parent_sha256=protocol['parent_sha256'],epoch=179,
    optimizer_min_step=min(steps),optimizer_max_step=max(steps),optimizer_entries=len(steps),recorded_checks=len(checks),
    quality_verdict='pending native visual comparison',goal_complete=False)
(OUT/'independent-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
names=['prepare_4k4d_continuation.py','train_4k4d_continuation.py','run_4k4d_continuation.sh','audit_4k4d_continuation_state.py','run_4k4d_continuation_eval.sh','evaluate_4k4d_pilot.py','audit_4k4d_native_images.py','compare_4k4d_continuation.py','queue_4k4d_continuation_evaluation.py']
entries=[]
for name in names:
    dest=snap/name
    shutil.copy2(LAB/'scripts/dynamic_human'/name,dest)
    entries.append(dict(file=name,sha256=digest(dest)))
(OUT/'source-snapshot.json').write_text(json.dumps(dict(files=entries),indent=2)+'\n')
print(json.dumps(report),flush=True)
