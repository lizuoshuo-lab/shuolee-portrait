from pathlib import Path
import json,yaml,hashlib
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
PARENT=LAB/'outputs/dynamic-human-4k4d-mask-control-039'
OUT=LAB/'outputs/dynamic-human-4k4d-projection-control-040';OUT.mkdir(exist_ok=True)
cp=PARENT/'model/latest.pt';assert (PARENT/'independent-state-audit.json').exists()
cal=LAB/'outputs/dynamic-human-enerf-static-calibration-037/calibration.json'
protocol=dict(status='fixed_before_paired_training',parent_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),parent_updates=1000,new_updates_each=1000,history_updates_each=2000,
    branches=['original_projection','calibrated_projection'],intervention='Frozen037percamera image-plane offsets applied after decoding/cropping to target/source projectionaliases only. Parentmodel,Adam,sourcepixels,masks,sourcecameras,losses,budget and freshseed matched.',
    initialization='Exact039checkpoint; no hullreconstruction for thispair. Both samplings restart fixedseed; not RNGcontinuation claim.',
    learning_rate='Both rebase officialExponentialLR at parentcurrent rates, gamma0.1over1000newupdates,min_lr5e-5.',
    calibration_data='All18camera staticbackground photos at700fit,710/720checked; extraaccess to appearance-heldoutcamera calibration photos explicitlyallowed for thisdiagnostic. No750heldoutpersonRGBusedtofit.',
    calibration_sha256=hashlib.sha256(cal.read_bytes()).hexdigest(),calibration_file=str(cal),
    evaluation='Sixnativefullframeswithsame targets/sources; calibratedbranch retainsprojectionoffsetat inference; targetRGB/matte removedbeforeforward. Previouslyinspecteddevelopmentviews, notfinaltest.',
    interpretation='Empirical image-plane registration control; not proof of physicalcalibration or synchronization.',goal_complete=False)
assert not (OUT/'protocol.json').exists();(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for branch in protocol['branches']:
    dest=OUT/branch;dest.mkdir(exist_ok=True)
    cfg=yaml.safe_load((PARENT/'pilot.yaml').read_text());cfg['exp_name']='enerf_mask039_'+branch
    cfg['model_cfg']['sampler_cfg']['skip_loading_points']=True
    cfg['parent_checkpoint']=str(cp);cfg['projection_offset_file']=str(cal) if branch=='calibrated_projection' else ''
    cfg['runner_cfg'].update(epochs=20,ep_iter=50,eval_ep=20,save_ep=20,save_latest_ep=1,trained_model=str(dest/'model'))
    cfg['runner_cfg']['visualizer_cfg']['result_dir']=str(dest/'official-render');cfg['runner_cfg']['recorder_cfg']['record_dir']=str(dest/'tensorboard')
    (dest/'pilot.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (dest/'protocol.json').write_text(json.dumps(dict(protocol,branch=branch),indent=2)+'\n')
print('040pairedprotocolready')
