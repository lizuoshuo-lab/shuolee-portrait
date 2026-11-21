"""One original VHAP global-stage trial initialized by all59 SMPLify-X fits.

Compared with the completed surface-only control, restore the repository's
original optimizable parameter groups and corresponding regularizers.
"""
import os,time,types,json,dataclasses,fcntl,subprocess,traceback
import numpy as np
import fit_smplify_vhap_surface123 as bridge
from vhap_demo_adapter123 import ROOT,REPO,COMMIT,make_tracker
from run_official_gaussians123 import read,save,sha,now

OUT=ROOT/'smplify-vhap-joint'
PLAN=ROOT/'smplify-vhap-joint-plan.json'
STAGE='rgb_global_tracking'


def main():
    import torch
    from torch.utils.data import DataLoader
    plan=read(PLAN)
    assert plan['execution_authorized'] and plan['source_sha256']==sha(__file__)
    assert not OUT.exists(),'Do not repeat or overwrite this trial.'
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert subprocess.check_output(['git','-C',str(REPO),'-c','core.autocrlf=true','-c','core.filemode=false','status','--porcelain'],text=True)==''
    for path,digest in {**plan['input_hashes'],**plan['source_hashes']}.items():assert sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();bridge.OUT=OUT;start=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(6060);torch.backends.cuda.matmul.allow_tf32=False
    save(OUT/'progress.json',dict(updated=now(),state='checking_smplify_vhap_joint_bridge',pid=os.getpid(),steps=0,quality_accepted=False))
    try:
        t,rows=make_tracker(OUT)
        t.cfg.render.background_train='white'
        expected=('cam','pose','shape','joints','expr','texture','lights','static_offset','dynamic_offset')
        assert t.cfg.pipeline.rgb_global_tracking.optimizable_params==expected
        assert t.cfg.pipeline.rgb_global_tracking.num_epochs==30
        assert t.calibrated and t.scaled and not t.cfg.model.use_dynamic_offset
        receipts=bridge.initialize(t,rows)
        save(OUT/'bridge-check.json',dict(state='passed',frames=receipts,optimizer_updates=0,quality_accepted=False))
        save(OUT/'config.json',json.loads(json.dumps(dataclasses.asdict(t.cfg),default=str)))
        params=t.get_train_parameters(STAGE)
        trainable={id(p) for group in params.values() for p in group}
        for key in bridge.PARAMS:getattr(t,key).requires_grad_(id(getattr(t,key)) in trainable)
        t.vposer.eval();t.vposer.requires_grad_(False)
        assert all(id(getattr(t,k)) in trainable for k in ('betas',)+bridge.PER_FRAME+('static_offset',))
        assert 'cam' not in params and 'dynamic_offset' not in params
        save(OUT/'trainable-groups.json',dict(groups=list(params),framewise=list(bridge.PER_FRAME),
            shared=['betas','static_offset','tex_extra','lights'],camera_K_fixed=True,
            dynamic_offset_disabled=True,regularizers_activated_by_upstream_opt_dict=True))
        bridge.checkpoint(t,rows,0,{})
        def evaluate(self,make_visualization=True,epoch=0,evaluate=True):
            bridge.checkpoint(self,rows,epoch,{})
            if time.monotonic()-start>1800:raise RuntimeError('Joint trial30-minute budget reached.')
        def log_scalars(self,logs,timestep,**kw):
            elapsed=time.monotonic()-start
            if elapsed>1800:raise RuntimeError('Joint trial30-minute budget reached.')
            item=dict(updated=now(),state='vhap_joint_fitting',pid=os.getpid(),step=self.global_step+1,
                stage=kw.get('stage'),
                frame=rows[int(timestep)]['frame'],seconds=elapsed,frames_total=59,
                smplifyx_fitting_used=True,quality_accepted=False,
                losses={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in logs.items()})
            save(OUT/'progress.json',item);print(json.dumps(item),flush=True)
        t.evaluate=types.MethodType(evaluate,t);t.log_scalars=types.MethodType(log_scalars,t)
        init_updates=0
        if plan.get('restore_original_offset_initialization'):
            frame=plan['initialization_frame']
            anchor=next(i for i,r in enumerate(rows) if r['frame']==frame)
            assert t.cfg.pipeline.rgb_init_offset.num_steps==500
            t.optimize_stage('rgb_init_offset',sample=t.dataset.one(anchor),lr_scale=1.0)
            init_updates=t.global_step
            assert init_updates==500
            initial_offset=t.static_offset.detach().cpu().numpy()
            np.savez_compressed(OUT/'offset-initialization-params.npz',
                **{key:getattr(t,key).detach().cpu().numpy() for key in bridge.PARAMS})
            offset_mm=np.linalg.norm(initial_offset[0],axis=-1)*1000
            save(OUT/'offset-initialization.json',dict(updated=now(),frame=frame,steps=init_updates,
                stage='rgb_init_offset',original_stage=True,lr_scale=1.0,
                offset_mm=dict(median=float(np.median(offset_mm)),p90=float(np.quantile(offset_mm,.9)),
                    maximum=float(offset_mm.max())),quality_accepted=False))
        loader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(6060))
        t.optimize_stage(STAGE,dataloader=loader,lr_scale=.1)
        assert t.global_step==1770+init_updates
        for path,digest in {**plan['input_hashes'],**plan['source_hashes']}.items():assert sha(path)==digest,path
        result=dict(updated=now(),state='complete_pending_visual_review',steps=t.global_step,frames=59,
            elapsed_seconds=time.monotonic()-start,smplifyx_fitting_used=True,quality_accepted=False,
            gaussian_training_started=False,original_optimizer_loss_scheduler=True,
            original_global_parameter_groups=True,camera_K_fixed=True,perframe_pose_frozen=False,
            offset_initialization_updates=init_updates,global_refinement_updates=1770,
            scope='Whole59 unified trial. Initialization, pose, shape and surface quality require visual review.')
        save(OUT/'result.json',result);save(OUT/'progress.json',dict(**result,pid=None));print(json.dumps(result),flush=True)
    except BaseException as error:
        save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,quality_accepted=False));raise


if __name__=='__main__':main()
