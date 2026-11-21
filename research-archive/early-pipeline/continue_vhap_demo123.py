"""Continue the same demo fit with VHAP's original sequential/global stages.

Only restore the completed reference-stage checkpoint, adapt output callbacks,
and repair the upstream final-slot copy boundary. The original optimize(),
optimizers, losses, stage budgets and global scheduler are executed unchanged.
"""
from pathlib import Path
import os, types, time, fcntl, traceback, subprocess
import numpy as np
from vhap_demo_adapter123 import ROOT, REPO, COMMIT, DATA, make_tracker, snapshot
from run_official_gaussians123 import save, read, sha, now

PARAMS = ('betas','expression','global_orient','transl','pose_embedding','jaw_pose',
          'leye_pose','reye_pose','left_hand_pose','right_hand_pose','static_offset','tex_extra','lights')
PER_FRAME = ('transl','global_orient','jaw_pose','leye_pose','reye_pose','expression',
             'pose_embedding','left_hand_pose','right_hand_pose')
REFERENCE = ('lmk_init_rigid','lmk_init_all','rgb_init_texture','rgb_init_all','rgb_init_offset')


def main():
    import torch
    source = ROOT/'vhap-fit'
    out = ROOT/'vhap-sequence'
    assert read(source/'progress.json')['state'] == 'reference_mesh_review_pending'
    assert not out.exists(), 'No duplicate sequence run or overwritten output.'
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip() == COMMIT
    assert subprocess.check_output(['git','-C',str(REPO),'-c','core.autocrlf=true','-c','core.filemode=false','status','--porcelain'],text=True) == ''
    origin = read(source/'input-provenance.json')
    for p, digest in origin['files'].items():
        assert sha(p) == digest, 'Reference-stage source changed: '+p
    for p, digest in origin['inputs'].items():
        assert sha(p) == digest, 'Reference-stage input changed: '+p
    lock = open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    checkpoint = source/'rgb_init_offset/tracked-smplx-params.npz'
    plan = dict(updated=now(),state='authorized_same_fit_continuation',subject='demo',
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),frames=59,
        reference_steps_already_done=2500,sequential_steps_per_frame=50,global_epochs=30,
        global_lr_scale=0.1,global_lr_decay=0.9,expected_total_updates=7220,
        execution='unchanged GlobalTracker.optimize and optimize_stage',
        adapters=['Restore exactly the reference checkpoint; skip its five completed stages',
                  'Repair last-slot initialization boundary (upstream < n-1 excludes final frame)',
                  'Save original/mesh review images and parameter checkpoints'],
        hand_repair_deferred_by_user=True,no_smplifyx_solver=True,no_new_losses=True,
        no_camera_changes=True,no_old_subject_geometry=True,quality_accepted=False)
    save(out/'plan.json',plan)
    try:
        t, rows = make_tracker(out)
        with np.load(checkpoint,allow_pickle=False) as z, torch.no_grad():
            assert list(z['frame_ids']) == [r['frame'] for r in rows]
            for name in PARAMS:
                target = getattr(t,name)
                assert target.shape == z[name].shape
                target.copy_(torch.as_tensor(z[name],device=target.device))
                assert torch.isfinite(target).all()
        # Validate the boundary copy as an isolated identity operation; restore
        # the original reference checkpoint values before actual fitting.
        original_init = t.initialize_next_timtestep
        boundary_repairs = []
        def init_next(self, timesteps):
            original_init(timesteps)
            src = int(timesteps[-1])
            if src == self.n_timesteps-2:
                with torch.no_grad():
                    for name in PER_FRAME:
                        x = getattr(self,name)
                        x[src+1].copy_(x[src])
                boundary_repairs.append(dict(source=src,target=src+1,step=self.global_step))
        t.initialize_next_timtestep = types.MethodType(init_next,t)
        with torch.no_grad():
            saved = {k:getattr(t,k)[-1].clone() for k in PER_FRAME}
            t.initialize_next_timtestep(torch.tensor([57]))
            assert all(torch.equal(getattr(t,k)[58],getattr(t,k)[57]) for k in PER_FRAME)
            for k,v in saved.items(): getattr(t,k)[-1].copy_(v)
        boundary_repairs.clear()
        save(out/'continuation-check.json',dict(updated=now(),checkpoint_identity_passed=True,
            all_59_parameter_slots_present=True,last_slot_copy_check_passed=True,
            optimizer_updates_during_check=0,upstream_source_unchanged=True))
        seen = []
        original_stage = t.optimize_stage
        def stage(self, name=None, sample=None, dataloader=None, lr_scale=1., **kw):
            name = name if name is not None else kw['stage']
            if name in REFERENCE:
                self.global_step = 2500
                return
            result = original_stage(name,sample=sample,dataloader=dataloader,lr_scale=lr_scale)
            if name == 'rgb_sequential_tracking':
                seen.append(int(sample['timestep_index'][0]))
                save(out/'sequence-progress.json',dict(updated=now(),frames_completed=len(seen),
                    frame_slots=seen,step=self.global_step,pid=os.getpid(),quality_accepted=False))
            return result
        t.optimize_stage = types.MethodType(stage,t)
        def evaluate(self, make_visualization=True, epoch=0, evaluate=True):
            label = 'sequential-complete' if epoch == 0 else f'global-epoch-{epoch:02d}'
            assert seen == list(range(len(rows)))
            ids = list(range(len(rows))) if epoch == 30 else [0,12,23,35,46,58]
            snapshot(self,rows,out,label,ids)
            save(out/'stage-completed.json',dict(updated=now(),label=label,step=self.global_step,
                frames=len(rows),quality_accepted=False))
        t.evaluate = types.MethodType(evaluate,t)
        save(out/'progress.json',dict(updated=now(),state='vhap_full_sequence_fitting',
            step=2500,pid=os.getpid(),hand_repair_deferred=True,quality_accepted=False))
        # This is the repository's sequence solver, not a rewritten solver.
        t.optimize()
        assert seen == list(range(59)) and t.global_step == 7220
        assert len(boundary_repairs) == 1
        save(out/'result.json',dict(updated=now(),state='complete_pending_mesh_review',
            frames=59,full_sequence_complete=True,reference_steps=2500,
            sequence_and_global_updates=t.global_step-2500,total_updates=t.global_step,
            stage_counts=t.stage_counts,elapsed_seconds=time.monotonic()-started,
            last_slot_repairs=boundary_repairs,checkpoint_sha256=sha(checkpoint),
            hand_repair_deferred=True,smplifyx_fitting_used=False,quality_accepted=False))
        save(out/'progress.json',dict(updated=now(),state='complete_pending_mesh_review',
            step=t.global_step,pid=None,frames=59,quality_accepted=False))
    except BaseException as e:
        save(out/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),
            quality_accepted=False))
        raise


if __name__ == '__main__':
    main()
