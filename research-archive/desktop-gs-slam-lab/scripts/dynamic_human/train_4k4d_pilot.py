"""Run official training with finite-loss and actual source-membership receipts."""
from pathlib import Path
import torch,json,time,hashlib,traceback,subprocess,os
from easyvolcap.engine import cfg
from easyvolcap.scripts.main import train
OUT=Path(os.environ.get('FOURKD_RUN_DIR','/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-4k4d-official-035'))
torch.set_num_threads(8)
assert not (OUT/'training-completion.json').exists()
start=time.time();state={'status':'constructing_official_model','completed_updates_this_process':0}
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
save('training-live.json',state)
try:
    import tinycudann as tcnn
    encoder=tcnn.Encoding(2,dict(otype='HashGrid',n_levels=2,n_features_per_level=8,log2_hashmap_size=12,base_resolution=16,per_level_scale=2),dtype=torch.float32).cuda()
    x=torch.rand(128,2,device='cuda',requires_grad=True); y=encoder(x); y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    save('tcnn-gpu-probe.json',dict(status='forward_backward_pass',shape=list(y.shape),module=tcnn.__file__,source_commit=subprocess.check_output(['git','-C','/opt/gs-slam-lab/4k4d/tiny-cuda-nn','rev-parse','HEAD'],text=True).strip()))
    del encoder,x,y;torch.cuda.empty_cache()
    save('resolved-configuration.json',cfg._cfg_dict.to_dict())
    runner=train(cfg,dry_run=True);cfg.runner=runner
    ds=runner.dataloader.dataset; val=runner.val_dataloader.dataset
    assert [int(x) for x in ds.view_inds]==[1,2,3,4,5,7,8,9,10,11,13,14,15,16,17]
    assert set(int(val.view_inds[int(x)]) for x in val.src_view_inds).isdisjoint({0,6,12})
    state.update(status='training',initial_points=[len(p) for p in runner.model.sampler.pcds],train_cameras=[int(x) for x in ds.view_inds],appearance_heldout=[0,6,12])
    def prehook(model,inputs):
        if not model.training:return
        batch=inputs[0]
        target=int(ds.view_inds[int(batch.meta.view_index[0])])
        sources=[int(ds.view_inds[int(x)]) for x in batch.meta.src_inds[0]]
        assert target not in [0,6,12] and not set(sources)&{0,6,12}
        state.update(target=target,sources=sources)
    def posthook(model,inputs,output):
        if not model.training:return
        assert torch.isfinite(output.loss).all(), 'Nonfinite official objective'
        state.update(loss=float(output.loss.detach().mean()))
    runner.model.register_forward_pre_hook(prehook)
    runner.model.register_forward_hook(posthook)
    def after_step(optimizer,args,kwargs):
        state['completed_updates_this_process']+=1
        if state['completed_updates_this_process']%10==0 or state['completed_updates_this_process']==1:
            state.update(elapsed_seconds=time.time()-start,max_allocated_bytes=torch.cuda.max_memory_allocated())
            save('training-live.json',state)
            print('AUDITED_STEP',json.dumps(state),flush=True)
    runner.optimizer.register_step_post_hook(after_step)
    runner.train()
    assert state['completed_updates_this_process']==1000
    finite=all(bool(torch.isfinite(p).all()) for p in runner.model.parameters())
    assert finite
    checkpoints=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (OUT/'model').glob('*') if p.is_file()]
    save('training-completion.json',dict(state,status='1000official_updates_complete_quality_review_pending',finite=finite,checkpoints=checkpoints,goal_complete=False))
except Exception:
    state.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.time()-start)
    save('training-failure.json',state);save('training-live.json',state)
    raise
