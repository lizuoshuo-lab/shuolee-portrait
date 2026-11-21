from pathlib import Path
import json,torch,subprocess,hashlib,shutil,os
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=Path(os.environ.get('FOURKD_RUN_DIR',str(LAB/'outputs/dynamic-human-4k4d-official-035')))
cp_path=OUT/'model/latest.pt';c=torch.load(cp_path,map_location='cpu',weights_only=False)
completion=json.loads((OUT/'training-completion.json').read_text())
assert c['epoch']==19 and completion['completed_updates_this_process']==1000
assert completion['max_allocated_bytes']<16*1024**3
assert all(bool(torch.isfinite(v).all()) for v in c['model'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
opt_steps=[int(v['step']) for v in c['optimizer']['state'].values() if 'step' in v]
assert len(opt_steps)>0 and max(opt_steps)==1000
pcds={k:list(v.shape) for k,v in c['model'].items() if k.startswith('sampler.pcds.')}
assert pcds=={'sampler.pcds.0':[262144,3]}
log=(OUT/'training.log').read_text(errors='replace').splitlines();audits=[]
for line in log:
    if line.startswith('AUDITED_STEP '):audits.append(json.loads(line[len('AUDITED_STEP '):]))
assert [x['completed_updates_this_process'] for x in audits]==[1]+list(range(10,1001,10))
assert all(x['target'] not in [0,6,12] and not set(x['sources'])&{0,6,12} for x in audits)
configs=json.loads((OUT/'resolved-configuration.json').read_text())
assert configs['val_dataloader_cfg']['dataset_cfg']['src_view_sample']==[1,2,3,4,5,7,8,9,10,11,13,14,15,16,17]
hashv=hashlib.sha256(cp_path.read_bytes()).hexdigest()
assert hashv in [r['sha256'] for r in completion['checkpoints'] if r['file']=='latest.pt']
report=dict(status='official1000update_state_and_recorded_input_membership_verified',epoch=19,optimizer_state_entries=len(opt_steps),optimizer_min_step=min(opt_steps),optimizer_max_step=max(opt_steps),
            point_parameters=pcds,all_floating_model_tensors_finite=True,checkpoint_sha256=hashv,recorded_update_checks=len(audits),
            scope='All training forwards enforced allowed camera membership by runtime hook;101recorded optimizer-step snapshots additionally checked. This is not a visual quality verdict.',goal_complete=False)
(OUT/'independent-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'requirements.freeze.txt').write_text(subprocess.check_output(['/opt/gs-slam-lab/4k4d/.venv/bin/python','-m','pip','freeze'],text=True))
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True);files=[]
for name in ['setup_4k4d_runtime.py','build_4k4d_tcnn.sh','probe_4k4d_runtime.py','prepare_4k4d_pilot.py','probe_4k4d_data.py','run_4k4d_data_probe.sh','train_4k4d_pilot.py','run_4k4d_pilot.sh','evaluate_4k4d_pilot.py','run_4k4d_pilot_eval.sh','audit_4k4d_native_images.py','audit_4k4d_pilot_state.py']:
    p=snap/name;shutil.copy2(LAB/'scripts/dynamic_human'/name,p)
    files.append(dict(file=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'source-snapshot.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print(json.dumps(report),flush=True)
