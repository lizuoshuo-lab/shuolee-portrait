"""User-authorized SMPLify-X -> bound Gaussians; bypass VHAP entirely.

Reuse pinned upstream Gaussian training, the verified root-SH interface and the
existing fixed-skeleton clothing bridge. One fresh 1000-step exploration; no
teacher from the rejected mesh pilots, no weight sweep or old optimizer.
"""
from pathlib import Path
import argparse,ast,copy,difflib,os,sys,traceback
import numpy as np
import train_demo_body_root_sh123 as control
from run_official_gaussians123 import read,save,sha,now,ASSETS

HERE=Path(__file__).resolve().parent
OUT=control.offset.base.DEMO/'body-gaussian-smplify-direct'
SOURCE=control.offset.base.DEMO/'fullbody-smplifyx/full-sequence'
AUDIT=control.BASE/'gaussian-mesh-guidance-audit'
FACE=AUDIT/'face-landmark-boundary'
LIMIT=1000


def source():
    _,_,original=control.source()
    changed=original
    def replace(before,after):
        nonlocal changed
        assert changed.count(before)==1,before
        changed=changed.replace(before,after,1)
    replace("OUT/'vhap-gaussian-interface-receipt.json'","OUT/'smplify-gaussian-interface-receipt.json'")
    replace('    from gaussian_renderer import render as original_render',
        '    from smplify_direct_surface123 import enable as enable_clothing, vertices as clothing_vertices, geometry as clothing_geometry\n    from gaussian_renderer import render as original_render')
    replace('        enable_shared_static_offset = SMPLXGaussianModel.enable_shared_static_offset','        enable_shared_static_offset = enable_clothing')
    replace('        shared_offset_vertices = SMPLXGaussianModel.shared_offset_vertices','        shared_offset_vertices = clothing_vertices')
    replace("vertex_offset='Shared canonical vertexoffset optimized at1e-6 from final VHAP values'",
        "vertex_offset='Shared clothing-only canonical offset at1e-6 from zero; direct SMPLify-X initialization, no VHAP'")
    replace('            if step == 10000:','            if step == 1000:')
    replace("stop_reason='Preregistered10k comparison budget'","stop_reason='Preregistered1000-step direct SMPLify-X comparison budget'")
    replace('            if step == 1000:\n                decision =', '            if False:\n                decision =')
    replace('only_color_direction_changed=True','only_color_direction_changed=False, direct_smplify_route=True')
    replace("        if step == 1 or step % 100 == 0:", """        if step == 1 or step % 100 == 0:
            gate=clothing_geometry(model)
            save(run/'geometry-progress.json',dict(updated=now(),step=step,**gate))
            assert gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0
            assert gate['new_relative_normal_reversals']==0, 'Stop: new relative triangle normal reversal'
            assert gate['minimum_area_ratio']>.1 and gate['max_added_offset_m']<.02
        if step == 1 or step % 100 == 0:""")
    ns=dict(control.offset.base.train.__globals__)
    ns.update(OUT=OUT,SNAPSHOTS=[LIMIT],suspended_baseline=control.suspended_baseline,
        ControlComplete=control.ControlComplete)
    exec(compile(changed,'<direct-SMPLify-X-upstream-Gaussian-adapter>','exec'),ns)
    return ns['train'],original,changed


def prepare():
    assert not OUT.exists(),'Preserve previous attempt; never silently repeat.'
    OUT.mkdir();(OUT/'fit').mkdir();(OUT/'data').mkdir()
    p=copy.deepcopy(read(control.BASE/'plan.json'))
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    assert read(SOURCE/'progress.json')['state']=='complete'
    assert read(SOURCE/'progress.json')['frames_completed']==59
    orig=SOURCE/'fit/fitted.npz';inputs=SOURCE/'data/training-inputs.json'
    with np.load(orig,allow_pickle=False) as z:fit=dict(z)
    rows=read(inputs)['records'];ids=list(fit['frame_ids'].astype(str))
    assert len(ids)==len(set(ids))==len(rows)==59
    assert ids==[r['frame'] for r in rows] and ids[0]=='frame_000033'
    source_hashes={str(f):sha(f) for f in [orig,inputs,SOURCE/'fit/official-fit-result.json',
        SOURCE/'progress.json',FACE/'vertex-boundary.npz',FACE/'face-geometry-policy.json']}
    for i,row in enumerate(rows):
        assert row['wh']==[2704,1520] and np.array_equal(fit['K_integer_centers'][i],row['K_integer_centers'])
        assert np.array_equal(fit['w2c'][i],row['w2c'])
        assert sha(row['image'])==row['source_sha256']==str(fit['image_sha256'][i])
        assert sha(row['mask'])==row['mask_sha256']==str(fit['mask_sha256'][i])
    fit['static_offset']=np.zeros((1,10475,3),np.float32)
    np.savez_compressed(OUT/'fit/fitted.npz',**fit)
    document=copy.deepcopy(read(inputs));document.update(pose_mode='direct_official_SMPLify_X_camera_relative',
        vhap_used=False,quality_accepted=False,physical_camera_calibration_accepted=False)
    save(OUT/'data/training-inputs.json',document)
    _,original,changed=source();ast.parse(changed)
    (OUT/'direct-launcher-function.py').write_text(changed)
    (OUT/'direct-interface.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True))))
    paths=[Path(__file__),HERE/'smplify_direct_surface123.py',HERE/'frozen_skeleton_surface123.py',
        HERE/'facial_geometry_boundary123.py',HERE/'body_root_sh_colors123.py',HERE/'train_demo_body_root_sh123.py',
        ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    p['source_hashes'].update({str(f):sha(f) for f in paths});p['source_hashes'].update(source_hashes)
    p.update(updated=now(),state='registered_before_interface_check',user_authorization=['跳过vhap，直接用smplify初始化训练','面部mesh不要受高斯引导'],
        route='mask -> existing original SMPLify-X full59 fit -> bound Gaussian training',vhap_used=False,
        hypothesis='Test whether direct SMPLify-X geometry is a viable initialization without inherited VHAP offsets; do not assume that receiving Gaussian gradients proves a clothing surface.',
        baseline_for_review=str(control.OUT),source_fit=str(orig),initial_points=20908,from_scratch=True,
        optimizer_state_reused=False,old_gaussian_checkpoint_loaded=False,shared_static_offset_lr=1e-6,
        static_vertex_offset='Zero initial shared canonical clothing displacement, applied through the existing audited frozen original SMPL-X LBS transform. Facial/head/neck, hands and feet offsets excluded.',
        mesh_pose='SMPLify-X pose/shape/expression and skeleton frozen for this first Gaussian route test.',
        comparison_budget=LIMIT,snapshots=[LIMIT],runtime_budget_seconds=900,configured_schedule_steps=600000,
        no_automatic_new_experiment=True,quality_accepted=False,end_to_end_15min_accepted=False,
        face_policy='Preserve all59 fitted faces exactly during Gaussian geometry training; appearance may learn. Existing SMPLify-X2D fit and model prior are retained, not represented as independent3D observations.',
        interface_changes=['Direct SMPLify-X fitted archive replaces VHAP mesh','Previously verified reference-root SH basis retained',
            'Previously verified fixed-skeleton clothing-only bridge prevents face and bone guidance'],
        attribution='Route integration test, not an isolated one-factor K/optimizer attribution. Upstream Gaussian optimizer/loss/density parameters unchanged.',
        stop=dict(nonfinite=True,new_relative_normal_reversals=0,min_area_ratio=.1,max_added_offset_m=.02,protected_face_motion_m=0),
        admission='Stop at1000 and inspect all59 same-camera native body/face/hands/feet; do not publish before quality and package checks.',
        no_gaussian_center_teacher=True,no_extra_SH_penalty=True,no_mesh_silhouette_loss=True,
        smplify_input_elapsed_seconds=read(SOURCE/'progress.json')['elapsed_seconds'])
    save(OUT/'plan.json',p);save(OUT/'progress.json',dict(updated=now(),state=p['state'],pid=None,step=0,quality_accepted=False))
    print('DIRECT_ROUTE_REGISTERED',flush=True)
    # CPU-only original SMPL-X replay. Does not contend with the current CUDA run.
    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    bridge=FrozenSkeletonSurface(fit,fit['static_offset'][0],ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    with np.load(FACE/'vertex-boundary.npz',allow_pickle=False) as z:b=dict(z)
    assert np.array_equal(b['faces'],fit['faces'])
    learn=torch.as_tensor(b['clothing_vertices']);protected=torch.as_tensor(b['protected_face_vertices'])
    assert not torch.any(learn & protected)
    error=float((bridge.base_vertices-torch.tensor(fit['vertices_world'])).abs().max());assert error<2e-6,error
    oracle=[]
    with torch.no_grad():
        d=torch.sin(torch.arange(10475*3).reshape(10475,3).float())*.0001*learn[:,None]
        for frame in ('frame_000033','frame_000015','frame_000054'):
            i=bridge.slots[frame];v,j=bridge.independent_original_forward(d,i)
            ve=float((bridge.vertices(d,i)-v).abs().max());je=float((bridge.joints(i)-j).abs().max())
            assert ve<2e-6 and je<2e-6
            assert not torch.any((bridge.vertices(d,i)-bridge.base_vertices[i])[protected])
            oracle.append(dict(frame=frame,vertices_error_m=ve,skeletal_joints_error_m=je))
    cache=OUT/'fit/surface-cache.npz'
    np.savez_compressed(cache,frame_ids=fit['frame_ids'],faces=fit['faces'],base_vertices=bridge.base_vertices.numpy(),
        linear=bridge.linear.numpy(),skeletal_joints=bridge.base_skeleton.numpy(),clothing_vertices=b['clothing_vertices'],
        protected_face_vertices=b['protected_face_vertices'])
    receipt=dict(updated=now(),state='direct_SMPLify_X_interface_verified',frames=59,vhap_used=False,
        fit_sha256=sha(OUT/'fit/fitted.npz'),training_inputs_sha256=sha(OUT/'data/training-inputs.json'),
        surface_cache_sha256=sha(cache),replay_max_abs_m=error,independent_original_forward_checks=oracle,
        source_fit_sha256=sha(orig),source_hashes=source_hashes,initial_offset_max_abs_m=0,
        no_gaussian_or_optimizer_loaded=True,physical_quality_accepted=False,quality_accepted=False)
    save(OUT/'smplify-gaussian-interface-receipt.json',receipt)
    p['source_hashes'].update({str(f):sha(f) for f in [OUT/'fit/fitted.npz',OUT/'data/training-inputs.json',cache,OUT/'smplify-gaussian-interface-receipt.json']})
    p['state']='prepared';save(OUT/'plan.json',p)
    save(OUT/'progress.json',dict(updated=now(),state='prepared',pid=None,step=0,quality_accepted=False))
    print('DIRECT_SMPLIFY_INPUT_VERIFIED',error,flush=True)


def run(pid):
    # Existing handoff verifies PID/start identity, leases the sole GPU and resumes
    # the original baseline in finally. It launches this script's train mode.
    control.OUT=OUT;control.__file__=__file__;control.run(pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','train'])
    parser.add_argument('--pid',type=int);args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='run':run(args.pid)
        else:source()[0]()
    except BaseException as e:
        if OUT.exists():
            save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
            save(OUT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None,error=repr(e),quality_accepted=False))
        raise
