"""Bounded same-candidate continuation using upstream capture/restore.

The 1000 snapshot precedes its optimizer update. Restore the 999-update state,
replay deterministic camera sampling, and let upstream redo iteration1000.
No training hyperparameters, source fit, facial boundary or losses change.
"""
from pathlib import Path
import argparse, ast, copy, difflib, json, traceback
import train_smplify_direct_gaussian123 as direct
from run_official_gaussians123 import read, save, sha, now

BASE=direct.OUT
CONT=BASE/'continuation-10000'
CHECKPOINT=BASE/'gaussian-run/step-001000/checkpoint.pt'
LIMIT=10000


def source():
    _,_,original=direct.source()
    changed=original
    def replace(old,new):
        nonlocal changed
        assert changed.count(old)==1,old
        changed=changed.replace(old,new,1)
    replace("plan = read(OUT/'plan.json')", "plan = read(CONT/'plan.json')")
    replace("run = OUT/'gaussian-run'", "run = CONT/'gaussian-run'")
    replace("save(OUT/'progress.json'", "save(CONT/'progress.json'") if changed.count("save(OUT/'progress.json'")==1 else None
    # Two status destinations exist, although bounded completion returns early.
    changed=changed.replace("save(OUT/'progress.json'", "save(CONT/'progress.json'")
    replace('        enable_shared_static_offset = enable_clothing', '''        def restore(self, model_args, training_args):
            payload=torch.load(CHECKPOINT, weights_only=False)
            with torch.no_grad(): self.static_offset.copy_(payload['static_offset'])
            super().restore(model_args, training_args)
            self.select_mesh_by_timestep(0)
            captured=self.capture()
            for i in range(1,12): assert torch.equal(captured[i],model_args[i]),i
            assert torch.equal(self.static_offset,payload['static_offset'])
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    state=self.optimizer.state[param]
                    assert int(state['step'])==999,(group['name'],state['step'])
            save(run/'restore-verified.json',dict(updated=now(),checkpoint_sha256=sha(CHECKPOINT),
                optimizer_updates_restored=999,groups=[g['name'] for g in self.optimizer.param_groups],
                gaussian_tensors_binding_density_and_offset_equal=True,upstream_restore_used=True))
        enable_shared_static_offset = enable_clothing''')
    replace('    counts = np.zeros(59, dtype=int)', '    counts = np.zeros(59, dtype=int)\n    replay_only=[False]')
    replace("            cams[i].original_image = pixels[i].float().div(255) * masks[i].float().div(255)",
        "            if not replay_only[0]:\n                cams[i].original_image = pixels[i].float().div(255) * masks[i].float().div(255)")
    replace('        return torch.utils.data.DataLoader(dataset, **kwargs)', '''        actual=torch.utils.data.DataLoader(dataset, **kwargs)
        class ReplayLoader:
            first=True
            def __iter__(self):
                it=iter(actual)
                if self.first:
                    self.first=False; replay_only[0]=True
                    for _ in range(999):
                        try: next(it)
                        except StopIteration:
                            it=iter(actual); next(it)
                    replay_only[0]=False
                    assert int(counts.sum())==999
                return it
        return ReplayLoader()''')
    replace("        if step == 1:\n", """        if step == 1000:
            expected=read(OUT/'gaussian-run/step-001000/sampling.json')['counts']
            assert counts.tolist()==expected,'Sampling replay differs from frozen first1000 observations'
            old_loss=json.loads((OUT/'gaussian-run/loss.jsonl').read_text().splitlines()[-1])
            errors={k:abs(float(v)-old_loss[k]) for k,v in losses.items()}
            assert max(errors.values())<1e-6,errors
            save(run/'replay-verified.json',dict(updated=now(),step=1000,sampling_counts_equal=True,
                loss_abs_differences=errors,completed_updates=999,pending_update_recomputed=True,
                bitwise_full_trajectory_claim=False))
        if step == 1:
""")
    replace("            assert gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0\n            assert gate['new_relative_normal_reversals']==0, 'Stop: new relative triangle normal reversal'\n            assert gate['minimum_area_ratio']>.1 and gate['max_added_offset_m']<.02", """            valid=(gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0
                and gate['new_relative_normal_reversals']==0 and gate['minimum_area_ratio']>.1
                and gate['max_added_offset_m']<.02)
            if not valid:
                scene.save(step)
                save(run/'result.json',dict(updated=now(),state='geometry_stop_preserved',
                    snapshot_step=step,optimizer_updates=step-1,configured_schedule_steps=opt.iterations,
                    points=len(model._xyz),seconds=time.monotonic()-started,snapshots=saved,
                    per_frame_counts=counts.tolist(),geometry=gate,quality_accepted=False,
                    end_to_end_15min_accepted=False,stop_reason='Unchanged all59 geometry protection failed'))
                status('geometry_stop_preserved',step,geometry=gate)
                raise ControlComplete()""")
    replace('            if step == 1000:\n                suspended_baseline()', '            if step == 10000:\n                suspended_baseline()')
    replace("stop_reason='Preregistered1000-step direct SMPLify-X comparison budget'", "stop_reason='Same direct candidate continuation budget10000'")
    replace('from_scratch=True, input_mesh_sha256=', 'from_scratch=False, input_mesh_sha256=')
    replace('full_official_reproduction=False, old_checkpoint_loaded=False',
        'full_official_reproduction=False, old_checkpoint_loaded=True, same_candidate_resume=True, resume_checkpoint_sha256=sha(CHECKPOINT)')
    replace("    try:\n        namespace['training'](dataset,opt,pipe,[],SNAPSHOTS,[],None,-1)", """    payload=torch.load(CHECKPOINT,weights_only=False)
    captured=list(payload['capture'])
    assert payload['step']==1000 and captured[0]==1
    # Only undo the SH activation that upstream repeats at iteration1000.
    # No parameters or optimizer moments had been updated at that save point.
    captured[0]=0
    restored=run/'upstream-resume-999.pth'
    torch.save((tuple(captured),999),restored)
    try:
        namespace['training'](dataset,opt,pipe,[],SNAPSHOTS,[],str(restored),-1)""")
    ns=dict(direct.source()[0].__globals__)
    ns.update(CONT=CONT,CHECKPOINT=CHECKPOINT,SNAPSHOTS=[1000,10000])
    exec(compile(changed,'<same-candidate-direct-SMPLify-X-continuation>','exec'),ns)
    return ns['train'],original,changed


def prepare():
    assert not CONT.exists(),'Preserve any previous continuation; no repeat.'
    p=copy.deepcopy(read(BASE/'plan.json'))
    for path,h in p['source_hashes'].items(): assert sha(path)==h,path
    assert read(BASE/'gaussian-run/result.json')['optimizer_updates']==999
    review=read(BASE/'quality-review.json')
    assert (BASE/'gaussian-run/step-001000/review/visual-review.json').is_file()
    # Small CPU-only checkpoint audit before reserving GPU time.
    import torch
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert cp['step']==1000 and len(cp['capture'])==14
    opt=cp['capture'][12]
    assert len(opt['param_groups'])==7 and all(int(v['step'])==999 for v in opt['state'].values())
    assert tuple(cp['static_offset'].shape)==(10475,3)
    assert torch.isfinite(cp['static_offset']).all()
    assert cp['capture'][0]==1 and not torch.any(cp['capture'][3])
    CONT.mkdir()
    _,old,new=source();ast.parse(new)
    (CONT/'continuation-adapter.py').write_text(new)
    (CONT/'continuation.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))))
    p.update(updated=now(),state='prepared',same_candidate_continuation=True,from_scratch=False,
        source_checkpoint=str(CHECKPOINT),optimizer_state_reused=True,old_gaussian_checkpoint_loaded=True,
        comparison_budget=10000,snapshots=[1000,10000],runtime_budget_seconds=1000,
        start_snapshot_step=1000,completed_updates_restored=999,
        resume_contract='Original capture/restore plus the existing shared static_offset; replay sampling and the pending1000 update. No cross-experiment transfer.',
        sampling='Deterministic original loader replay999 observations; require1000 counts and all loss components to match before advancing.',
        hypothesis='Extend the reviewed direct route without changing hyperparameters; observe detail and clothing displacement under the unchanged all59 geometry guards.',
        admission='Exploratory quality remains unaccepted. All59 body/face/hands/feet visual review at1000 is complete. Preserve failed snapshots; no automatic extra run.',
        quality_accepted=False,end_to_end_15min_accepted=False,
        full_run_bitwise_equivalence_claim=False,unmodified_options=True)
    p['source_hashes'].update({str(f):sha(f) for f in [Path(__file__).resolve(),CHECKPOINT,
        BASE/'quality-review.json',BASE/'gaussian-run/step-001000/sampling.json',BASE/'gaussian-run/loss.jsonl']})
    save(CONT/'plan.json',p)
    save(CONT/'checkpoint-audit.json',dict(updated=now(),tensor_count=14,groups=7,all_adam_steps=999,
        active_sh_degree=1,rest_coefficients_zero=True,offset_shape=[10475,3],checkpoint_sha256=sha(CHECKPOINT)))
    save(CONT/'progress.json',dict(updated=now(),state='prepared',pid=None,step=1000,quality_accepted=False))
    print('SAME_CANDIDATE_CONTINUATION_PREPARED',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','train']);a.add_argument('--pid',type=int)
    args=a.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='run':
            direct.control.OUT=CONT;direct.control.__file__=__file__;direct.control.run(args.pid)
        else:source()[0]()
    except BaseException as e:
        if CONT.exists():
            save(CONT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
            save(CONT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None,error=repr(e),quality_accepted=False))
        raise
