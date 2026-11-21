"""Reuse the matched one-frame harness with explicit local joint parameters."""
import argparse
import difflib
import inspect
from pathlib import Path
import numpy as np
import control_sofa_body_protection124 as c
import smplify_local_joint_refine124 as local

OUT=c.a.RUN/'local-head-wrist-final-stage-control'
ALLOWED=local.FACE_HAND_NAMES+(local.PARAMETER,)


def transformed_worker():
    source=inspect.getsource(c.worker)
    replacements={
        '        trace.probe.cache_static_topology(body)\n':'        trace.probe.cache_static_topology(body)\n        vposer=local.install(body,vposer)\n',
        "for name,value in first['parameters'].items()}":"for name,value in first['parameters'].items() if name!=local.PARAMETER}",
        'and len(selected)==6':'and len(selected)==len(ALLOWED)',
        "        assert all(x<=p['prefix_parity_tolerance'] for x in prefix.values()),prefix\n":"        assert all(x<=p['prefix_parity_tolerance'] for x in prefix.values()),prefix\n        assert np.count_nonzero(first['parameters'][local.PARAMETER])==0\n",
    }
    changed=source
    for before,after in replacements.items():
        assert changed.count(before)==1,before
        changed=changed.replace(before,after)
    compile(changed,'<same-single-frame-harness-local-joints>','exec')
    return changed,''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),
        fromfile='frozen-body-control-harness',tofile='local-head-wrist-control-harness'))


def preflight():
    import torch,smplx
    torch.set_num_threads(1)
    utils,_,_,_=c.entry.load_official(c.a.OFFICIAL)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(c.a.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
        use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
        create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32)
    loader=c.entry.module(c.a.PROJECT/'src/avatar_core/shuolee_vposer.py','_local_joint_preflight124')
    loader.MODEL,loader.CODE=c.a.ASSETS/'vposer/V02_05',c.a.PROJECT/'third_party/human_body_prior'
    decoder,_,_=loader.load_vposer('cpu');decoder.eval()
    saved=c.a.load_npz(c.a.frame(c.a.RUN,16))
    body.reset_params(**{name:saved[name] for name,_ in body.named_parameters()})
    latent=torch.from_numpy(saved['pose_embedding'])
    baseline=decoder.decode(latent)['pose_body'].reshape(1,21,3)
    wrapper=local.install(body,decoder)
    zero=wrapper.decode(latent)['pose_body']
    assert torch.equal(zero,baseline)
    joints0=body(body_pose=baseline.reshape(1,63),return_verts=True).joints.detach()
    with torch.no_grad():getattr(body,local.PARAMETER).copy_(torch.arange(1,13,dtype=torch.float32).reshape(1,4,3)*.002)
    updated=wrapper.decode(latent)['pose_body']
    keep=[i for i in range(21) if i not in local.BODY_INDICES]
    assert torch.equal(updated[:,keep],baseline[:,keep])
    output=body(body_pose=updated.reshape(1,63),return_verts=True,return_full_pose=True)
    # OpenPose19..24 are selected mesh vertices, not kinematic joints. Their
    # pose-blend-shape response must be reported, not called an exact invariant.
    protected=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    invariant=float((output.joints[:,protected]-joints0[:,protected]).abs().max())
    assert invariant<1e-7,invariant
    foot_mesh_delta=float((output.joints[:,19:25]-joints0[:,19:25]).abs().max())
    face_grad=torch.autograd.grad(output.joints[:,67:].square().sum(),getattr(body,local.PARAMETER),retain_graph=True)[0]
    hand_grad=torch.autograd.grad(output.joints[:,25:67].square().sum(),getattr(body,local.PARAMETER),retain_graph=True)[0]
    assert torch.isfinite(face_grad).all() and torch.isfinite(hand_grad).all()
    assert float(face_grad[:,:2].norm())>0 and float(hand_grad[:,2:].norm())>0
    serialized=updated.detach().reshape(1,63).cpu().numpy()
    with torch.no_grad():
        direct=body(body_pose=torch.from_numpy(serialized),return_verts=True)
        raw=body(body_pose=baseline.reshape(1,63),return_verts=True)
    parity=float((direct.vertices-output.vertices.detach()).abs().max())
    negative=float((raw.vertices-output.vertices.detach()).abs().max())
    assert parity==0 and negative>1e-4,(parity,negative)
    return dict(cpu_only=True,optimizer_steps=0,zero_delta_exact_original=True,joint_names=list(local.JOINT_NAMES),
        full_joint_ids=list(local.FULL_JOINT_IDS),body_indices=list(local.BODY_INDICES),other17_body_rotations_exact=True,
        protected_body_joint_change_max_m=invariant,face_gradient_finite_nonzero=True,hand_gradient_finite_nonzero=True,
        foot_vertex_landmark_pose_blend_change_max_m=foot_mesh_delta,
        final_axis_angle_export_replay_max_m=parity,dropping_local_offsets_negative_control_max_m=negative,
        downstream_contract='Final body_pose_axis_angle must be exported and used. A plain VPoser re-decode of pose_embedding is insufficient. Existing Gaussian body_pose input is compatible, but actual candidate handoff is still required.',
        learned_prior_scope='Original VPoser latent prior remains on the frozen base. The12 added local angles have the existing same-run rotation temporal penalty, not a new learned VPoser prior.',
        quality_accepted=False)


def prepare():
    assert not OUT.exists()
    rejected=c.read(c.OUT/'visual-review.json')
    assert rejected['actual_visual_review'] and not c.read(c.OUT/'result.json')['face_hand_comparative_gate_passed']
    assert c.read(c.a.RUN/'quality-stop-decision.json')['pid'] is None
    row,_=c.base.target()
    oldplan=c.read(c.OUT/'plan.json')
    frozen=dict(oldplan['frozen'])
    for path in [Path(__file__),Path(local.__file__),c.OUT/'result.json',c.OUT/'visual-review.json',
                 c.a.ROOT.parent/'demo-vhap-initialization/smplify-vhap-wrist-decoder-audit/result.json']:
        frozen[str(path)]=c.sha(path)
    prior_failure=c.a.RUN/'local-head-wrist-interface-v1-failure.json'
    frozen[str(prior_failure)]=c.sha(prior_failure)
    source,diff=transformed_worker()
    result=preflight()
    OUT.mkdir()
    (OUT/'harness.diff').write_text(diff,encoding='utf-8')
    (OUT/'harness.py').write_text(source,encoding='utf-8')
    c.save(OUT/'interface-check.json',result)
    c.save(OUT/'plan.json',dict(updated=c.now(),frame='frame_000016',diagnostic_only=True,
        hypothesis='Full-body freezing protected the knee but made face/hands worse. Test final-stage local neck/head/wrist rotations while preserving the VPoser body/root. This changes parameter support, not the solver, observations, weights or prefix stages.',
        prior_relevant_evidence='Historical demo59 VPoser wrist Jacobians found no independent wrist direction in all59 cases. Do not repeat that audit or claim it is a new finding.',
        final_stage_trainable=list(ALLOWED),local_angle_dimensions=12,original_stage_indices=[3,4],
        initialization='Same recorded17 seed for matched diagnosis only; new local offsets start zero. No selected pose becomes a new sequence initializer.',
        prefix_parity_tolerance=2e-6,budget=dict(fitter_calls=1,frames=1,max_seconds=300),
        comparative_gate='Same frozen-body control gates: exact original stage3 prefix; protected body/root/beta/translation; no worse confidence-weighted face and hand means than saved all-parameter endpoint; native body/face/hands/feet review. No gate relaxation after execution.',
        risks=['Local rotations are not restricted to the VPoser decoder manifold; actual anatomy/temporal review is required.',
               'Neck/head position and global shape/camera errors may remain, so local rotations may still fail.',
               'Future same-run temporal seeding and Gaussian handoff must retain the actual final axis angles and local offsets. No pipeline integration or quality is claimed by the interface test.'],
        stopping='One matched diagnostic only. Stop on failed prefix, anatomy or regional protection. No full sequence, automatic retries, weight sweep, saved-pose replacement or Gaussian admission.',
        interface_check=result,frozen=frozen))
    print('LOCAL_JOINT_CONTROL_REGISTERED',result,flush=True)


def configure_harness():
    c.OUT=OUT;c.ALLOWED=ALLOWED;c.adapt_source=local.adapt_source
    return c


def worker():
    c=configure_harness()
    source,_=transformed_worker()
    assert source==(OUT/'harness.py').read_text(encoding='utf-8')
    scope=dict(c.worker.__globals__,local=local)
    exec(compile(source,'<same-single-frame-harness-local-joints>','exec'),scope)
    scope['worker']()
    result=c.read(OUT/'result.json')
    result['local_joint_adapter']=c.read(OUT/'interface-check.json')
    result['original_report'].update(five_stages=False,original_stage_indices=[3,4],
        local_final_parameter_partition=True,original_optimizer_and_loss_weights_unchanged=True,
        learned_vposer_prior_applies_to_base_latent_only=True)
    c.save(OUT/'result.json',result)


def run():
    import fcntl,subprocess,sys
    c=configure_harness();p=c.verify();assert not (OUT/'progress.json').exists()
    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (OUT/'fit.log').open('x') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=p['budget']['max_seconds'])
    print('LOCAL_JOINT_CONTROL_FINISHED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker'])
    args=parser.parse_args();dict(prepare=prepare,run=run,worker=worker)[args.mode]()
