"""Selected-frame VHAP initialization, bidirectional tracking, joint refinement.

Uses the existing full-body VHAP stage optimizer/loss/scheduler. The only new
staging is selecting the user-requested good frame instead of frame zero, and
walking temporally outward from it before the original global stage.
"""
import copy,os,time,types,fcntl,traceback,subprocess
from pathlib import Path
import numpy as np
from vhap_demo_adapter123 import ROOT,REPO,COMMIT,DATA,make_tracker,snapshot,preflight
from run_official_gaussians123 import save,read,sha,now
from continue_vhap_demo123 import PER_FRAME,REFERENCE

OUT=ROOT/'vhap-selected-reference'


def main():
    import torch
    from torch.utils.data import DataLoader
    selected=read(ROOT/'reference-selection/selection.json')
    anchor=int(selected['slot'])
    assert not OUT.exists(),'Do not overwrite or duplicate the selected-frame run.'
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert subprocess.check_output(['git','-C',str(REPO),'-c','core.autocrlf=true','-c','core.filemode=false','status','--porcelain'],text=True)==''
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();torch.set_num_threads(4);torch.manual_seed(6060);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic()
    save(OUT/'plan.json',dict(updated=now(),state='authorized_selected_reference_and_joint_fit',
        anchor=selected,frames=59,reference_stages=REFERENCE,reference_steps=2500,
        sequential_per_frame=50,global_epochs=30,expected_total_updates=7220,
        shared=['shape','canonical vertex offset','texture','lighting'],
        framewise=['root translation/orientation','body pose','face expression','jaw/eyes','hands'],
        optimizer_and_loss_source=str(REPO/'vhap/model/tracker_smplx.py'),commit=COMMIT,
        staging='Selected anchor; walk forward and backward through adjacent real frames; original global optimizer over all59 chronological identities',
        hand_repair_deferred=True,K_changed=False,reuse_rejected_fit=False,quality_accepted=False))
    try:
        t,rows=make_tracker(OUT)
        assert rows[anchor]['source_sha256']==selected['source_sha256']
        assert sha(DATA/'data/observations.npz')==selected['observations_sha256']
        save(OUT/'input-provenance.json',dict(updated=now(),files={str(p):sha(p) for p in
             [Path(__file__),Path(__file__).with_name('vhap_demo_adapter123.py'),REPO/'vhap/model/tracker_smplx.py',REPO/'vhap/config/base.py']},
             inputs={str(DATA/p):sha(DATA/p) for p in ['data/inputs.json','data/observations.npz']},
             reference_selection=selected,initial_shared_offset='zero; no rejected checkpoint loaded'))
        preflight(t,rows,OUT)
        # Reuse the exact upstream temporal equations for the backward pass:
        # reverse only the small per-frame parameter views and queried index,
        # so upstream idx-1 means the adjacent already-fitted frame idx+1.
        # Global refinement always uses the original chronological indices.
        t.backward_pass=False
        for name in ('compute_pose_smooth_energy','compute_joint_smooth_energy','compute_expr_smooth_energy'):
            original=getattr(t,name)
            def smooth(self,idx,_original=original):
                if not self.backward_pass:return _original(idx)
                proxy=copy.copy(self)
                for key in PER_FRAME:setattr(proxy,key,getattr(self,key).flip(0))
                return _original.__func__(proxy,self.n_timesteps-1-idx)
            setattr(t,name,types.MethodType(smooth,t))
        # Independent algebra check for the reversed predecessor; no updates.
        with torch.no_grad():
            saved=t.transl.clone();t.transl[:,0]=torch.arange(t.n_timesteps,device='cuda')*.013
            t.backward_pass=True;idx=torch.tensor([20]);actual=t.compute_pose_smooth_energy(idx)
            expected=((t.transl[20]-t.transl[21])**2).mean()*t.cfg.w.smooth_trans
            expected+=((t.global_orient[20]-t.global_orient[21])**2).mean()*t.cfg.w.smooth_rot
            assert torch.allclose(actual,expected,atol=1e-8,rtol=1e-5)
            t.transl.copy_(saved);t.backward_pass=False
        save(OUT/'temporal-direction-check.json',dict(updated=now(),reversed_neighbor_matches_original_equation=True,
            frame_order_for_global='original chronological',optimizer_updates=0))
        for stage in REFERENCE:
            t.optimize_stage(stage,sample=t.dataset.one(anchor))
            snapshot(t,rows,OUT,stage,[anchor])
        completed=[];copies=[]
        def copy_frame(src,dst):
            assert abs(src-dst)==1
            with torch.no_grad():
                for name in PER_FRAME:getattr(t,name)[dst].copy_(getattr(t,name)[src])
            copies.append([src,dst])
        def fit_slot(i):
            t.optimize_stage('rgb_sequential_tracking',sample=t.dataset.one(i));completed.append(i)
            save(OUT/'sequence-progress.json',dict(updated=now(),state='bidirectional_tracking',pid=os.getpid(),
                reference_slot=anchor,completed_slots=completed,frames_completed=len(completed),step=t.global_step))
        # The anchor has no fitted predecessor yet. Seed its neighbor only for
        # original temporal regularization; that neighbor is later refitted.
        if anchor>0:copy_frame(anchor,anchor-1)
        fit_slot(anchor)
        for i in range(anchor+1,len(rows)):
            copy_frame(i-1,i);fit_slot(i)
        t.backward_pass=True
        for i in range(anchor-1,-1,-1):
            copy_frame(i+1,i);fit_slot(i)
        t.backward_pass=False
        assert sorted(completed)==list(range(59))
        def evaluate(self,make_visualization=True,epoch=0,evaluate=True):
            label='bidirectional-complete' if epoch==0 else f'global-epoch-{epoch:02d}'
            ids=list(range(59)) if epoch==30 else sorted({0,12,23,anchor,35,46,58})
            snapshot(self,rows,OUT,label,ids)
            save(OUT/'stage-completed.json',dict(updated=now(),label=label,step=self.global_step,frames=59,quality_accepted=False))
        t.evaluate=types.MethodType(evaluate,t)
        t.evaluate(epoch=0)
        loader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=4)
        t.optimize_stage(stage='rgb_global_tracking',dataloader=loader,lr_scale=.1)
        assert t.global_step==7220
        save(OUT/'result.json',dict(updated=now(),state='complete_pending_mesh_review',frames=59,
            full_sequence_complete=True,reference_slot=anchor,reference_source_frame=selected['source_frame'],
            total_updates=t.global_step,stage_counts=t.stage_counts,elapsed_seconds=time.monotonic()-start,
            neighbor_copies=copies,shared_shape_updated_by_all_frames=True,hand_repair_deferred=True,
            smplifyx_fitting_used=False,quality_accepted=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete_pending_mesh_review',pid=None,
            step=t.global_step,frames=59,reference_slot=anchor,quality_accepted=False))
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),quality_accepted=False));raise


if __name__=='__main__':main()
