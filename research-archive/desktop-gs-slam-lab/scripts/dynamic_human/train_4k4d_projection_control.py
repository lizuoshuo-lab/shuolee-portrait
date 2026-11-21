from pathlib import Path
import torch,json,time,hashlib,traceback
from easyvolcap.engine import cfg
from easyvolcap.scripts.main import train
from easyvolcap.runners.schedulers import ExponentialLR
from fourkd_projection_offsets import apply_projection_offsets
OUT=Path(cfg.runner_cfg.trained_model).parent;protocol=json.loads((OUT/'protocol.json').read_text())
torch.set_num_threads(8);start=time.time();assert not (OUT/'model/latest.pt').exists()
save=lambda n,x:(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
state=dict(status='constructing',completed_updates_this_process=0,parent_updates=1000,branch=protocol['branch'])
save('training-live.json',state)
offsets=None
if cfg.projection_offset_file:
    path=Path(cfg.projection_offset_file);assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['calibration_sha256']
    offsets={r['camera']:r['shift_xy_px'] for r in json.loads(path.read_text())['rows']}
try:
    save('resolved-configuration.json',cfg._cfg_dict.to_dict())
    assert cfg.model_cfg.let_user_handle_input and cfg.model_cfg.camera_cfg.type=='NoopCamera'
    runner=train(cfg,dry_run=True);cfg.runner=runner
    cp=torch.load(cfg.parent_checkpoint,map_location='cpu',weights_only=False)
    assert hashlib.sha256(Path(cfg.parent_checkpoint).read_bytes()).hexdigest()==protocol['parent_sha256']
    runner.model.load_state_dict(cp['model'],strict=True);runner.optimizer.load_state_dict(cp['optimizer'])
    assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in runner.model.state_dict().items())
    for g in runner.optimizer.param_groups:g['initial_lr']=g['lr']
    runner.scheduler=ExponentialLR(runner.optimizer,decay_iter=1000,gamma=.1,min_lr=5e-5)
    save('parent-load-audit.json',dict(exact_model_parameters=True,optimizer_steps=sorted(set(int(s['step']) for s in runner.optimizer.state.values() if 'step'in s)),learning_rates=[dict(name=g.get('name'),lr=g['lr']) for g in runner.optimizer.param_groups]))
    del cp;runner.load_model=lambda:0
    def before(model,inputs):
        batch=inputs[0];ds=runner.dataloader.dataset if model.training else runner.val_dataloader.dataset
        target=int(ds.view_inds[int(batch.meta.view_index[0])]);src=[int(ds.view_inds[int(x)]) for x in batch.meta.src_inds[0]]
        assert not set(src)&{0,6,12}
        if model.training:
            assert target not in [0,6,12]
            batch.meta.iter+=1000;batch.iter+=1000;batch.meta.frac=batch.meta.iter/2000;batch.frac=batch.iter/2000
            assert int(batch.meta.iter)==1000+state['completed_updates_this_process']
            state.update(target=target,sources=src)
        if offsets is not None:apply_projection_offsets(batch,target,src,offsets)
    def after(model,inputs,output):
        if model.training:
            assert torch.isfinite(output.loss).all();state['loss']=float(output.loss.detach().mean())
    def step(optimizer,args,kwargs):
        state['completed_updates_this_process']+=1
        if state['completed_updates_this_process']%50==0 or state['completed_updates_this_process']==1:
            state.update(status='training',history_updates=1000+state['completed_updates_this_process'],elapsed_seconds=time.time()-start,max_allocated_bytes=torch.cuda.max_memory_allocated())
            save('training-live.json',state);print('AUDITED_STEP',json.dumps(state),flush=True)
    runner.model.register_forward_pre_hook(before);runner.model.register_forward_hook(after);runner.optimizer.register_step_post_hook(step)
    runner.train();assert state['completed_updates_this_process']==1000
    assert all(bool(torch.isfinite(p).all()) for p in runner.model.parameters())
    save('training-completion.json',dict(state,status='2000history_updates_complete_review_pending',epoch=19,checkpoint_sha256=hashlib.sha256((OUT/'model/latest.pt').read_bytes()).hexdigest(),goal_complete=False))
except Exception:
    state.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.time()-start);save('training-failure.json',state);save('training-live.json',state);raise
