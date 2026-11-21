"""Cross-person inputs for the frozen demo SH0 bound-Gaussian adapter.

No fitter/optimizer/density changes. Historical subject fits are explicitly
reused; Gaussian parameters and optimizer always start empty.
"""
import argparse, ast, copy, difflib, fcntl, hashlib, inspect, json, os
import random, re, shutil, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from run_official_gaussians123 import PROJECT, ASSETS, read, save, sha, now
from train_demo_body_upstream123 import UP, COMMIT, module

HERE=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
DEMO=REC/'demo-vhap-initialization'
BASE=DEMO/'body-gaussian-smplify-eight-dc-reviewedfit'
ROOT=REC/'cross-subject-sh0-124'
SOURCES={'grass':dict(fit_root=REC/'smplifyx-reference-control/official-first-cycle/full94-shared-shape',
    video=Path('/mnt/d/00/工作/数据集/video5.mp4'),video_sha256='846558b3a023650b3dccd1fa55da4dc3cfde596ea87de09ab43ab494f5306e14',
    record_id='005',label='草地白衣人物')}


def generate_source(out,n,ref,*,automatic=False):
    old=(BASE/'adapter.py').read_text()
    new=re.sub(r'\b59\b',str(n),old).replace("'frame_000033'",repr(ref))
    new=new.replace('all59',f'all{n}').replace('All59',f'All{n}')
    new=new.replace('suspended_baseline()  # Parent owns the export lease and resumes the baseline in finally.',
        'suspended_baseline()  # Verify this new standalone process owns the single GPU lease.')
    new=new.replace('Reviewed-SMPLify-fit input-only SH0 candidate; inspect at1000, bounded18000',
        'Cross-subject frozen-SH0 configuration; inspect1000, bounded18000; cached subject fit')
    new=new.replace('reviewed fitted-input archive relative to matched eight-frame SH0 run',
        'different subject inputs; same SH0 training configuration')
    new=new.replace("    offset_receipt = model.enable_shared_static_offset(OUT/'fit/fitted.npz', ASSETS/'smplx/SMPLX_NEUTRAL.npz')",
        "    offset_receipt = model.enable_shared_static_offset(OUT/'fit/fitted.npz', ASSETS/'smplx/SMPLX_NEUTRAL.npz')\n    offset_receipt.update(checked_frames=len(rows),reference_frame=layout['reference_frame_id'])")
    # Demo E was identity. This subject has nonidentity E: the review archive
    # must contain camera-space vertices, without changing training geometry.
    new=new.replace("                mesh_vertices.append(model.shared_offset_vertices(layout['pose_indices'][i]).cpu().numpy())",
        "                vw=model.shared_offset_vertices(layout['pose_indices'][i]).cpu().numpy()\n                E=np.asarray(row['w2c'])\n                mesh_vertices.append(vw@E[:3,:3].T+E[:3,3])")
    new=new.replace('identity extrinsics are not a recovered scene trajectory.',
        'extrinsics and fitted poses retain the source gauge; not independently calibrated ground truth.')
    if automatic:
        marker = "    assert vars(opt)==plan['options']"
        assert new.count(marker) == 1
        new = new.replace(marker, "    vars(opt).update(plan['schedule_overrides'])\n" + marker)
        new = new.replace(
            "different subject inputs; same SH0 training configuration",
            "Automatic source roster; recorded 18k clocks and original SH0 losses",
        )
        new = new.replace(
            "Gaussian options and SH0 unchanged; only reviewed fitted-input source changed",
            "Original SH0 losses with the recorded 18k schedule",
        )
        start = new.index("            if step == LIMIT and not getattr(self,'geometry_failed',False):")
        end = new.index("    # Execute the pinned original function", start)
        new = new[:start] + new[end:]
        start = new.index("            failed=(gate['protected_face_motion_m']")
        end = new.index("        if step == 1 or step % 100 == 0:", start)
        new = new[:start] + (
            "            assert gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0\n"
        ) + new[end:]
    ast.parse(new)
    (out/'adapter.py').write_text(new)
    (out/'input-interface.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozen-demo-adapter',tofile='cross-person-input-adapter')))
    return new


def select_eight(out,n):
    """Execute the existing selection mathematics, changing roster length only."""
    import train_smplify_eight123 as old
    rule=copy.deepcopy(read(BASE/'selection-rule.json'))
    rule.update(updated=now(),preservation=f'Other{n-8} images excluded from Gaussian RGB optimization, all{n} fitted poses rendered; not independent physical truth.',
        selection_implementation='Same rank/yaw-bin rule, in camera-relative body orientation: E_rotation @ root_rotation. Demo E was identity; general nonidentity cameras require the composition.',
        coverage='Eight equal-population bins of camera-relative body yaw; same quality-rank formula and tie rule.',budget=18000)
    save(out/'selection-rule.json',rule)
    body=inspect.getsource(old.prepare)
    body=body[body.index('    import cv2\n'):body.index("    for name in ['fit/fitted.npz'")]
    body=re.sub(r'\b59\b',str(n),body)
    body=re.sub(r'\b51\b',str(n-8),body)
    body=body.replace("    rotations=fit['root_rotation']", "    rotations=fit['w2c'][:,:3,:3]@fit['root_rotation']")
    ns=dict(old.prepare.__globals__);ns.update(BASE=out,OUT=out)
    exec('def select():\n'+body,ns)
    ns['select']()
    return read(out/'selection.json')


def prepare(subject):
    out=ROOT/subject;source=SOURCES[subject]
    assert not out.exists(), 'Preserve existing experiment; never repeat silently.'
    assert sha(source['video'])==source['video_sha256'], 'Dataset content changed.'
    src=source['fit_root'];receipt=read(src/'fit/official-fit-result.json')
    assert sha(src/'fit/fitted.npz')==receipt['fit_sha256']
    assert sha(src/'data/training-inputs.json')==receipt['training_inputs_sha256']
    p=copy.deepcopy(read(BASE/'plan.json'))
    for key in ['supersedes','baseline_run','prior_baseline','selected_frames','admission']:
        p.pop(key,None)
    out.mkdir(parents=True);(out/'fit').mkdir();(out/'data').mkdir()
    with np.load(src/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    document=read(src/'data/training-inputs.json');rows=document['records']
    ids=fit['frame_ids'].astype(str).tolist();n=len(ids);ref=str(fit['reference_frame_id'])
    assert len(set(ids))==len(rows)==n and ids==[r['frame'] for r in rows]
    assert n>=8 and ref in ids
    # Existing named-pose contract requires reference pose slot0, never frame1.
    order=[ids.index(ref)]+[i for i,f in enumerate(ids) if f!=ref]
    fit={k:(v[order] if v.ndim and v.shape[0]==n else v) for k,v in fit.items()}
    rows=[rows[i] for i in order];document['records']=rows
    fit['vertices_world']=fit['vertices_world'].astype(np.float32)
    fit['static_offset']=np.zeros((1,10475,3),np.float32)
    for i,r in enumerate(rows):
        assert sha(r['image'])==r['source_sha256']==str(fit['image_sha256'][i])
        assert sha(r['mask'])==r['mask_sha256']==str(fit['mask_sha256'][i])
        assert np.array_equal(r['K_integer_centers'],fit['K_integer_centers'][i])
        assert np.array_equal(r['w2c'],fit['w2c'][i])
    np.savez_compressed(out/'fit/fitted.npz',**fit)
    document.update(vhap_used=False,quality_accepted=False,cached_subject_fit=True,
        source_fit=str(src),source_fit_sha256=receipt['fit_sha256'])
    save(out/'data/training-inputs.json',document)
    p.update(updated=now(),state='registered_before_interface_check',subject=subject,label=source['label'],
        user_authorization='跑一下其他人物这个模型',route='subject masks and official SMPLify-X -> same SH0 mesh-bound Gaussians',
        source_fit=str(src/'fit/fitted.npz'),source_video=str(source['video']),source_video_sha256=source['video_sha256'],
        reference_frame=ref,frames=n,evaluation_frames=n,training_frames=8,
        from_scratch=True,gaussian_parameters_reused=False,optimizer_state_reused=False,
        cached_subject_fitting=True,cold_start_benchmark=False,demo_person_shape_reused=False,
        only_parameter_change='Person-specific observations, shape, fitted poses and frame identities. Same deterministic selector,SH0,Gaussian/offset/Laplacian/density/loss settings.',
        comparison_budget=18000,runtime_budget_seconds=3000,snapshots=[1000,10000,18000],
        review_gate='Actual original/mesh/Gaussian inspection at1000, then at most18000 if no new implementation or severe geometry failure. No automatic retry or threshold changes.',
        admission='Cross-person exploratory test using verified cached subject fit. No claim that initialization or held-out physical geometry has passed.',
        source_fit_quality_accepted=False,quality_accepted=False,model_promoted=False,end_to_end_15min_accepted=False,
        previous_demo_failure_preserved=True,record_id=source['record_id'])
    # No history of another person is an input to the new geometry.
    p['source_hashes']={str(f):sha(f) for f in [Path(__file__),BASE/'adapter.py',BASE/'selection-rule.json',
        HERE/'train_smplify_eight123.py',HERE/'smplify_direct_surface123.py',HERE/'frozen_skeleton_surface123.py',
        HERE/'facial_geometry_boundary123.py',HERE/'body_root_sh_colors123.py',UP/'train.py',UP/'scene/gaussian_model.py',
        UP/'arguments/__init__.py',PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py',
        PROJECT/'src/shuolee_portrait/training.py',PROJECT/'src/avatar_core/gaussian_renderer/__init__.py',
        Path(p['laplacian_cache']),src/'fit/fitted.npz',src/'data/training-inputs.json',src/'fit/official-fit-result.json']}
    save(out/'plan.json',p)
    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    bridge=FrozenSkeletonSurface(fit,fit['static_offset'][0],ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    with np.load(BASE/'fit/surface-cache.npz',allow_pickle=False) as z:policy=dict(z)
    assert np.array_equal(policy['faces'],fit['faces'])
    replay=float((bridge.base_vertices-torch.as_tensor(fit['vertices_world'])).abs().max())
    assert replay<2e-6,replay
    oracle=[]
    with torch.no_grad():
        d=torch.sin(torch.arange(10475*3).reshape(10475,3).float())*.0001*torch.tensor(policy['clothing_vertices'])[:,None]
        for i in sorted(set([0,n//3,2*n//3,n-1])):
            v,j=bridge.independent_original_forward(d,i)
            ve=float((bridge.vertices(d,i)-v).abs().max());je=float((bridge.joints(i)-j).abs().max())
            assert ve<2e-6 and je<2e-6
            oracle.append(dict(frame=rows[i]['frame'],vertices_error_m=ve,joints_error_m=je))
    np.savez_compressed(out/'fit/surface-cache.npz',frame_ids=fit['frame_ids'],faces=fit['faces'],
        base_vertices=bridge.base_vertices.numpy(),linear=bridge.linear.numpy(),skeletal_joints=bridge.base_skeleton.numpy(),
        clothing_vertices=policy['clothing_vertices'],protected_face_vertices=policy['protected_face_vertices'])
    save(out/'smplify-gaussian-interface-receipt.json',dict(updated=now(),frames=n,reference_frame=ref,
        fit_sha256=sha(out/'fit/fitted.npz'),training_inputs_sha256=sha(out/'data/training-inputs.json'),
        surface_cache_sha256=sha(out/'fit/surface-cache.npz'),replay_max_abs_m=replay,oracle=oracle,
        no_demo_person_geometry=True,topology_policy_shared=True,vhap_used=False,quality_accepted=False))
    sel=select_eight(out,n);p['selected_frames']=sel['selected_frames']
    generate_source(out,n,ref)
    for name in ['adapter.py','selection.json','selection-rule.json','fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json','smplify-gaussian-interface-receipt.json']:
        p['source_hashes'][str(out/name)]=sha(out/name)
    p.update(state='prepared_input_review_pending',upstream_training_function_unchanged=True)
    save(out/'plan.json',p);save(out/'progress.json',dict(updated=now(),state=p['state'],step=0,pid=None))
    print(json.dumps(dict(subject=subject,frames=n,reference=ref,selected=sel['selected_frames'],replay_max_abs_m=replay)),flush=True)


class ControlComplete(Exception):pass


def train(subject, *, pose_selection=None, automatic=False):
    from pose_training_gate151 import verify_prepared, verify_training_launch

    out=ROOT/subject
    selection=pose_selection if pose_selection is not None else out/'pose-training-selection.json'
    if automatic:
        from prepare_reviewed_sh0 import load_automatic_selection
        selected = verify_prepared(selection,out,selection_loader=load_automatic_selection)
    else:
        selected = verify_training_launch(selection,out)
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print(f'POSE_GATE_READY {len(selected)} views',flush=True)
    save(out/'launch.json',dict(updated=now(),pid=os.getpid(),command=sys.argv,
        proc_start_identity=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],single_gpu_lock=True,
        pose_selection_sha256=sha(selection),admitted_frames=selected,automatic=automatic))
    ns=dict(globals());ns.update(OUT=out,LIMIT=18000,SNAPSHOTS=[1000,10000,18000],suspended_baseline=lambda:None)
    exec(compile((out/'adapter.py').read_text(),str(out/'adapter.py'),'exec'),ns)
    try:ns['train']()
    except ControlComplete:pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','train']);parser.add_argument('subject',choices=list(SOURCES));args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare(args.subject)
        else:train(args.subject)
    except BaseException as e:
        p=ROOT/args.subject
        if p.exists():save(p/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
