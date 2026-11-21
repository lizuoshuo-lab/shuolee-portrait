from pathlib import Path
import json,yaml,hashlib
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');SCRIPTS=LAB/'scripts/dynamic_human'
PARENT=LAB/'outputs/dynamic-human-4k4d-projection-control-040/original_projection'
OUT=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044';OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists()
parent=PARENT/'model/latest.pt';sha=hashlib.sha256(parent.read_bytes()).hexdigest()
protocol=dict(status='fixed_before_matched_training',parent_sha256=sha,parent_updates=2000,new_updates_each=1000,history_updates_each=3000,
    branches=['original_blending','source_support_blending'],intervention='Only training/inference IBR weights renormalized over source-mask support>=0.5; no-support points keep original weights. No RGB unpremultiplication, source hardmax, camera correction or alpha-formula change.',
    fixed='Exact shared model and Adam state; same seeded source/target schedule, native pixels and masks, model parameterization, training budget and original projection.',
    learning_rate='Both rebase officialExponentialLR at parent rates, gamma0.1over1000updates,min_lr5e-5.',
    evaluation='Same6development views, targetRGB/matte removedbeforeforward. Samecasealreadyinspected, notfinaltest.',goal_complete=False)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for branch in protocol['branches']:
    dest=OUT/branch;dest.mkdir(exist_ok=True);cfg=yaml.safe_load((PARENT/'pilot.yaml').read_text())
    cfg.update(exp_name='enerf_'+branch+'_044',parent_checkpoint=str(parent),projection_offset_file='',source_foreground_gate=branch=='source_support_blending')
    cfg['runner_cfg'].update(trained_model=str(dest/'model'))
    cfg['runner_cfg']['visualizer_cfg']['result_dir']=str(dest/'official-render');cfg['runner_cfg']['recorder_cfg']['record_dir']=str(dest/'tensorboard')
    (dest/'pilot.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(dest/'protocol.json').write_text(json.dumps(dict(protocol,branch=branch),indent=2)+'\n')
text=(SCRIPTS/'train_4k4d_projection_control.py').read_text()
text=text.replace('parent_updates=1000','parent_updates=2000').replace('batch.meta.iter+=1000;batch.iter+=1000;batch.meta.frac=batch.meta.iter/2000;batch.frac=batch.iter/2000','batch.meta.iter+=2000;batch.iter+=2000;batch.meta.frac=batch.meta.iter/3000;batch.frac=batch.iter/3000')
text=text.replace("==1000+state['completed_updates_this_process']","==2000+state['completed_updates_this_process']")
text=text.replace("history_updates=1000+state['completed_updates_this_process']","history_updates=2000+state['completed_updates_this_process']").replace('2000history_updates_complete_review_pending','3000history_updates_complete_review_pending')
anchor='    del cp;runner.load_model=lambda:0'
assert anchor in text;text=text.replace(anchor,anchor+"\n    if cfg.source_foreground_gate:\n        from fourkd_source_support import enable_source_foreground_support\n        enable_source_foreground_support(runner.model.sampler)")
target=SCRIPTS/'train_4k4d_source_gate.py';assert not target.exists();target.write_text(text)
(OUT/'training-script-derivation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256((SCRIPTS/'train_4k4d_projection_control.py').read_bytes()).hexdigest(),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n')
print('044 matched training prepared')
