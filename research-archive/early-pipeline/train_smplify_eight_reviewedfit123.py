"""Fresh SH0 candidate: change only the reviewed SMPLify-X input archive.

The original upstream training function, fixed8 images, camera, Gaussian
options, shared clothing offset, and Laplacian hook are reused unchanged.
"""
import argparse,ast,copy,difflib,os,shutil,traceback
from pathlib import Path
import numpy as np
import train_smplify_eight_dc123 as pilot
from run_official_gaussians123 import read,save,sha,now,ASSETS
from replay_smplify_pose_source123 import OLD as BASE,FIT,OUT as REPLAY

OUT=BASE.parent/'body-gaussian-smplify-eight-dc-reviewedfit'
LIMIT=18000
HERE=Path(__file__).resolve().parent


def source():
    fn,_,old=pilot.source();s=old
    a='User-requested SH degree0 single-variable1000-step control; inspect before any continuation'
    assert s.count(a)==1
    s=s.replace(a,'Reviewed-SMPLify-fit input-only SH0 candidate; inspect at1000, bounded18000',1)
    s=s.replace('shared Laplacian unchanged from reference; model SH degree3->0',
                'Gaussian options and SH0 unchanged; only reviewed fitted-input source changed')
    s=s.replace('model SH degree3->0 relative to matched eight-frame Laplacian run',
                'reviewed fitted-input archive relative to matched eight-frame SH0 run')
    ns=dict(fn.__globals__);ns.update(OUT=OUT,LIMIT=LIMIT,SNAPSHOTS=[1000,10000,LIMIT])
    exec(compile(s,'<reviewed-SMPLify-input-only-SH0-candidate>','exec'),ns)
    return ns['train'],old,s


def prepare():
    assert not OUT.exists(),'Do not repeat or overwrite this candidate.'
    p=copy.deepcopy(read(BASE/'plan.json'))
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    verdict=read(REPLAY/'visual-review.json');audit=read(REPLAY/'input-audit.json')
    assert verdict['actual_visual_review'] and verdict['native_groups']==60
    assert read(REPLAY/'result.json')['gaussian_parameters_bitwise_unchanged']
    assert audit['changed_training_frames']==['frame_000058'] and audit['reference_identical']
    assert read(FIT/'closeout.json')['regional_protections_passed']
    OUT.mkdir();(OUT/'fit').mkdir();(OUT/'data').mkdir()
    # Register the variable and protections before rebuilding input tensors.
    p.update(updated=now(),state='registered_before_input_replay',baseline_run=str(BASE),
        hypothesis='The reviewed SMPLify-X input repairs known pose handoff defects; fresh Gaussians must adapt to this pose set instead of retargeting old fitted appearance.',
        only_parameter_change='Replace full59 independent SMPLify-X fitted archive with the previously reviewed face-priority guarded archive, aligned by exact frame IDs. The selected8 images, SH0, all Gaussian/loss/density/offset settings remain fixed.',
        source_fit=str(FIT/'fit/fitted.npz'),source_pose_frames_changed=audit['changed_frames'],
        selected_training_frames_changed=['frame_000058'],frame_reselection=False,
        source_fit_quality_accepted=False,comparison_budget=LIMIT,snapshots=[1000,10000,LIMIT],runtime_budget_seconds=2400,
        initial_points=20908,from_scratch=True,optimizer_state_reused=False,old_gaussian_checkpoint_loaded=False,
        face_policy='Protected face/head receive no Gaussian geometric guidance. Use the reviewed SMPLify-X2D/prior fit; no independent3D face truth claimed.',
        review_gate='At1000 inspect all59 native body/face/hands/feet against same-step prior SH0. Extend this same live candidate only if no new geometry/protected-face failure or gross visible regression. Stop at18000; no automatic new candidate.',
        known_limitation='Original600k density schedule is held fixed to isolate input source. This short candidate may remain below the final roughly200k point target. Cached fitting is not a fresh15min workflow.',
        no_automatic_new_experiment=True,quality_accepted=False,end_to_end_15min_accepted=False,model_promoted=False)
    save(OUT/'plan.json',p)
    with np.load(BASE/'fit/fitted.npz',allow_pickle=False) as z:old=dict(z)
    with np.load(FIT/'fit/fitted.npz',allow_pickle=False) as z:new=dict(z)
    ids=list(old['frame_ids'].astype(str));newids=list(new['frame_ids'].astype(str))
    assert len(ids)==len(set(ids))==len(newids)==len(set(newids))==59
    ix=[newids.index(f) for f in ids]
    new={k:(v[ix] if v.ndim and v.shape[0]==59 else v) for k,v in new.items()}
    for k in audit['immutable_fields_equal']:assert np.array_equal(new[k],old[k]),k
    assert np.array_equal(new['vertices_world'][0],old['vertices_world'][0]) and ids[0]=='frame_000033'
    new['static_offset']=np.zeros((1,10475,3),np.float32)
    np.savez_compressed(OUT/'fit/fitted.npz',**new)
    for name in ('data/training-inputs.json','selection.json','selection-rule.json'):
        shutil.copy2(BASE/name,OUT/name);assert sha(BASE/name)==sha(OUT/name)
    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    bridge=FrozenSkeletonSurface(new,new['static_offset'][0],ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    replay_error=float((bridge.base_vertices-torch.as_tensor(new['vertices_world'])).abs().max())
    assert replay_error<2e-6
    with np.load(BASE/'fit/surface-cache.npz',allow_pickle=False) as z:oldcache=dict(z)
    cache=OUT/'fit/surface-cache.npz'
    np.savez_compressed(cache,frame_ids=new['frame_ids'],faces=new['faces'],base_vertices=bridge.base_vertices.numpy(),
        linear=bridge.linear.numpy(),skeletal_joints=bridge.base_skeleton.numpy(),
        clothing_vertices=oldcache['clothing_vertices'],protected_face_vertices=oldcache['protected_face_vertices'])
    receipt=dict(updated=now(),state='reviewed_SMPLify_X_interface_verified',frames=59,vhap_used=False,
        fit_sha256=sha(OUT/'fit/fitted.npz'),training_inputs_sha256=sha(OUT/'data/training-inputs.json'),
        surface_cache_sha256=sha(cache),replay_max_abs_m=replay_error,source_fit_sha256=sha(FIT/'fit/fitted.npz'),
        explicit_frame_id_permutation=ix,reference_frame='frame_000033',initial_offset_max_abs_m=0,
        no_gaussian_or_optimizer_loaded=True,physical_quality_accepted=False,quality_accepted=False,
        prerequisites={str(REPLAY/'input-audit.json'):sha(REPLAY/'input-audit.json'),str(REPLAY/'visual-review.json'):sha(REPLAY/'visual-review.json')})
    save(OUT/'smplify-gaussian-interface-receipt.json',receipt)
    paths=[Path(__file__).resolve(),FIT/'fit/fitted.npz',FIT/'closeout.json',REPLAY/'input-audit.json',REPLAY/'visual-review.json',
           OUT/'fit/fitted.npz',cache,OUT/'data/training-inputs.json',OUT/'smplify-gaussian-interface-receipt.json',
           OUT/'selection.json',OUT/'selection-rule.json']
    p['source_hashes'].update({str(f):sha(f) for f in paths})
    p['state']='prepared';save(OUT/'plan.json',p)
    _,oldsrc,newsrc=source();ast.parse(newsrc)
    (OUT/'adapter.py').write_text(newsrc)
    (OUT/'input-handoff-only.diff').write_text(''.join(difflib.unified_diff(oldsrc.splitlines(True),newsrc.splitlines(True))))
    save(OUT/'progress.json',dict(updated=now(),state='prepared',pid=None,step=0,quality_accepted=False))
    print('REVIEWED_FIT_SH0_PREPARED',replay_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','train']);p.add_argument('--pid',type=int);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='run':
            # Reuse the existing identity-bound sole-GPU lease and finally restoration.
            control=pilot.baseline.eight.direct.control
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit() or int(proc.name) in (os.getpid(),a.pid):continue
                try:
                    command,state,_=control.process(int(proc.name))
                    if (proc/'exe').resolve().name.startswith('python') and state!='T' and (' train' in command or 'replay_smplify' in command):
                        raise RuntimeError('Another GPU candidate is live: '+command)
                except (FileNotFoundError,PermissionError,ProcessLookupError):pass
            control.OUT=OUT;control.__file__=__file__;control.run(a.pid)
        else:source()[0]()
    except BaseException as e:
        if OUT.exists():save(OUT/(a.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
