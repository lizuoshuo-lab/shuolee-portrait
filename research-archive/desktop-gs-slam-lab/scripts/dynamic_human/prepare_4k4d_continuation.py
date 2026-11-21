from pathlib import Path
import json,yaml,hashlib,torch
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
PARENT=LAB/'outputs/dynamic-human-4k4d-official-035';OUT=PARENT/'continuation-10000'
OUT.mkdir(exist_ok=True)
cp=PARENT/'model/latest.pt';parent=torch.load(cp,map_location='cpu',weights_only=False);assert parent['epoch']==19
cfg=yaml.safe_load((PARENT/'pilot.yaml').read_text())
cfg['exp_name']='enerf_actor5_6_train15_frame750_continuation10000'
cfg['model_cfg']['sampler_cfg']['skip_loading_points']=True
cfg['runner_cfg'].update(epochs=180,ep_iter=50,eval_ep=180,save_ep=30,save_latest_ep=1,trained_model=str(OUT/'model'))
cfg['runner_cfg']['visualizer_cfg']['result_dir']=str(OUT/'official-render')
cfg['runner_cfg']['recorder_cfg']['record_dir']=str(OUT/'tensorboard')
cfg['parent_checkpoint']=str(cp)
text=yaml.safe_dump(cfg,sort_keys=False);target=OUT/'pilot.yaml'
if target.exists():assert target.read_text()==text
else:target.write_text(text)
protocol=dict(status='fixed_before_continuation',parent_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),parent_updates=1000,new_updates=9000,history_updates=10000,
    reason='Single-frame1000update pilot still has blurred hands/objects and black dotted boundaries in supervised and heldout views; determine longer-budget fitting before motion expansion.',
    retained='Exact parent model and Adam moments; all source views, native pixel density, losses, point count and calibration retained.',
    learning_rate='Rebase exponential schedule at parent current learning rates, gamma0.1 over9000newupdates, official min_lr5e-5. This is a continuation schedule, not a fresh10000run.',
    iteration='Model metadata uses1000+newupdate index, preserving original CNN warmup cutoff. Data sampling restarts seeded rather than claiming RNG-state continuation.',
    data='Same15training and3appearance-heldout cameras, actor5_6frame750. Previouslyinspected development views; not an untouched final benchmark.',
    selection='Fixed10000history final; no image sharpening/postprocessing or early selection by heldout score.',goal_complete=False)
pp=OUT/'protocol.json'
if not pp.exists():pp.write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps(dict(out=str(OUT),new_updates=9000,history_updates=10000)),flush=True)
