"""Independent checkpoint and paired-input checks for control040."""
from pathlib import Path
import hashlib,json,torch,shutil
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-4k4d-projection-control-040'
read=lambda p:json.loads(p.read_text())
parent=LAB/'outputs/dynamic-human-4k4d-mask-control-039/model/latest.pt'
parent_hash=hashlib.sha256(parent.read_bytes()).hexdigest()
reports=[];recorded=[];configs=[]
for branch in ['original_projection','calibrated_projection']:
    p=OUT/branch;c=torch.load(p/'model/latest.pt',map_location='cpu',weights_only=False)
    done=read(p/'training-completion.json');protocol=read(p/'protocol.json');config=read(p/'resolved-configuration.json')
    assert c['epoch']==19 and done['completed_updates_this_process']==1000 and done['history_updates']==2000
    assert protocol['parent_sha256']==parent_hash
    assert read(p/'parent-load-audit.json')['exact_model_parameters']
    assert all(torch.isfinite(v).all() for v in c['model'].values() if v.is_floating_point())
    steps=[int(s['step']) for s in c['optimizer']['state'].values() if 'step'in s]
    assert min(steps)>=1000 and max(steps)==2000
    assert c['model']['sampler.pcds.0'].shape==(262144,3)
    checkpoints_hash=hashlib.sha256((p/'model/latest.pt').read_bytes()).hexdigest()
    assert checkpoints_hash==done['checkpoint_sha256']
    records=[json.loads(l.removeprefix('AUDITED_STEP ')) for l in (p/'training.log').read_text(errors='replace').splitlines() if l.startswith('AUDITED_STEP ')]
    assert [r['completed_updates_this_process'] for r in records]==[1]+list(range(50,1001,50))
    assert all(r['target'] not in [0,6,12] and not set(r['sources'])&{0,6,12} for r in records)
    recorded.append([(r['completed_updates_this_process'],r['target'],r['sources']) for r in records])
    configs.append(config)
    report=dict(status='exact_parent_and_2000history_checkpoint_verified',branch=branch,checkpoint_sha256=checkpoints_hash,parent_sha256=parent_hash,
        optimizer_state_entries=len(steps),optimizer_min_step=min(steps),optimizer_max_step=max(steps),recorded_membership_checks=len(records),
        all_floating_tensors_finite=True,point_count=262144,goal_complete=False)
    (p/'independent-state-audit.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
    del c
assert recorded[0]==recorded[1], 'Paired target/source sample sequence differs'
for key in ['model_cfg','dataloader_cfg','val_dataloader_cfg']:
    assert configs[0][key]==configs[1][key],key
assert configs[0]['projection_offset_file'] is None or configs[0]['projection_offset_file']==''
assert configs[1]['projection_offset_file'].endswith('dynamic-human-enerf-static-calibration-037/calibration.json')
report=dict(status='paired_state_and_recorded_sampling_verified',branches=reports,matched_model_dataset_configs=True,matched_recorded_sampling=True,
    qualification='Runtime hook enforces allowed membership every forward;21recorded optimizer snapshots compared. Empirical calibration uses separate static photos from all18cameras.',goal_complete=False)
(OUT/'independent-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
names=['prepare_4k4d_projection_control.py','train_4k4d_projection_control.py','run_4k4d_projection_control.sh','fourkd_projection_offsets.py','evaluate_4k4d_pilot.py','audit_4k4d_projection_state.py','audit_4k4d_native_images.py','compare_4k4d_projection.py']
for n in names:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
(OUT/'source-snapshot.json').write_text(json.dumps(dict(files=[dict(file=n,sha256=hashlib.sha256((snap/n).read_bytes()).hexdigest()) for n in names]),indent=2)+'\n')
print(json.dumps(report))
