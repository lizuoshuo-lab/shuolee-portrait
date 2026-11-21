"""One bounded full-roster SMPL-X adaptation, continuing the shared-field route.

This research runner consumes an explicitly frozen input manifest and plan.
It never re-detects points, changes K/E, or loads an optimized subject fit.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,sys,time,traceback
import numpy as np
import torch
import smplx

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
sys.path.insert(0,str(PROJECT/'src'))
import run_smplify_reference123 as reference
from shuolee_portrait.sequence_input_adapter import load_sequence_inputs
from shuolee_portrait.sequence_fitting import SharedIdentitySequence
from shuolee_portrait.sequence_optimizer import optimize_sequence

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,j):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def stats(a):
    a=np.asarray(a).reshape(-1)
    return dict(n=len(a),median=float(np.median(a)),max=float(a.max()),p90=float(np.quantile(a,.9))) if len(a) else dict(n=0)

def export_frames(model,observations,bundle,outpath,*,deadline=None):
    """No all-frame SMPL-X graph is constructed during output either."""
    vertices_c=[];vertices_w=[];joints=[];pixels=[];metrics=[]
    fit=observations.fitting_confidence().cpu().numpy()
    targets=bundle.targets
    with torch.no_grad():
        for i,frame in enumerate(model.frame_ids):
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Registered frame export budget exhausted.')
            out=model.forward_state(frame_indices=[i])
            uv=model.project(out,frame_indices=[i])[0].cpu().numpy()
            vc=out.vertices[0].cpu().numpy();vw=model.world_vertices(out,frame_indices=[i])[0].cpu().numpy()
            if not all(np.isfinite(a).all() for a in [uv,vc,vw]):raise RuntimeError('Nonfinite output at '+frame)
            if float(out.joints[...,2].min())<=.05:raise RuntimeError('Invalid positive depth at '+frame)
            distance=np.linalg.norm(uv-targets[i],axis=-1)
            metrics.append(dict(frame=frame,source_frame=int(bundle.source_frames[i]),
                body_fit_errors_px=stats(distance[:25][fit[i,:25]>0]),
                face_fit_errors_px=stats(distance[67:][observations.face_active('FIT')[i].cpu().numpy()]),
                face_held_errors_px=stats(distance[67:][observations.face_active('HELD')[i].cpu().numpy()]),
                face_review_complete=bool(np.all(bundle.face_visibility[i]!='unknown')),physical_face_truth=False))
            vertices_c.append(vc.astype(np.float32));vertices_w.append(vw.astype(np.float32))
            joints.append(out.joints[0].cpu().numpy());pixels.append(uv)
    rows=read(next(p for p,_ in bundle.bound_files if p.name=='inputs.json'))['records']
    np.savez_compressed(outpath,**{k:v.detach().cpu().numpy() for k,v in model.state.items()},
        vertices_camera=np.stack(vertices_c),vertices_world=np.stack(vertices_w),joints_camera=np.stack(joints),
        projected=np.stack(pixels),targets=bundle.targets,confidence=bundle.confidence,fitting_confidence=fit,
        faces=model.body.faces,frame_ids=np.asarray(model.frame_ids),source_frames=bundle.source_frames,
        image_sha256=np.asarray([r['source_sha256'] for r in rows]),mask_sha256=np.asarray([r['mask_sha256'] for r in rows]),
        K_integer_centers=model.K.cpu().numpy(),w2c=model.w2c.cpu().numpy(),source_w2c=bundle.source_w2c,
        reference_frame_id=np.asarray(bundle.reference_frame_id),shared_betas=model.state['betas'].detach().cpu().numpy(),
        camera_length_contract_json=np.asarray(json.dumps(model.camera_length_contract,sort_keys=True)))
    return metrics

def run(plan_path):
    plan_path=Path(plan_path);plan_hash=sha(plan_path);plan=read(plan_path);dest=Path(plan['output'])
    assert plan['mode']=='one_full_sequence_research_adaptation'
    assert plan['max_real_trials']==1 and plan['evaluation_mode']=='streamed'
    assert dest.is_dir() and not (dest/'progress.json').exists(),'Never restart or resume this fit.'
    for path,h in plan['source_sha256'].items():assert sha(path)==h,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    save(dest/'progress.json',dict(state='running',stage='load_inputs',pid=os.getpid(),started=now(),gaussian_steps=0))
    try:
        bundle=load_sequence_inputs(plan['inputs'],decode_images=True)
        assert len(bundle.frame_ids)==plan['frames']==94
        save(dest/'input-readback.json',bundle.report())
        parts=reference.setup();utils,_,_,mapping,vposer,_=parts
        body=smplx.SMPLX(str(reference.A/'smplx/SMPLX_NEUTRAL.npz'),num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,batch_size=1,
            joint_mapper=utils.JointMapper(mapping)).to(device='cuda',dtype=torch.float64)
        vposer=vposer.double()
        with torch.no_grad():
            latent=vposer.encode(torch.tensor(bundle.body_pose.copy(),device='cuda',dtype=torch.float64)).mean.detach().clone()
        n=len(bundle.frame_ids)
        initial=dict(global_orient=bundle.global_orient.copy(),transl=bundle.transl.copy(),latent=latent,
            jaw_pose=np.zeros((n,3)),expression=np.zeros((n,10)))
        model=SharedIdentitySequence(body,vposer,bundle.frame_ids,bundle.K_integer_centers.copy(),bundle.source_w2c.copy(),initial,
            camera_length_contract=dict(bundle.camera_length_contract),dtype=torch.float64)
        observations=bundle.make_observations(device='cuda',dtype=torch.float64)
        assert model.state['betas'].shape==(1,10) and not model.state['betas'].any()
        before=export_frames(model,observations,bundle,dest/'initial.npz',deadline=started+plan['budget']['seconds'])
        K0=model.K.clone();E0=model.w2c.clone()
        def progress(info,theta):
            save(dest/'progress.json',dict(state='running',pid=os.getpid(),updated=now(),**info,gaussian_steps=0))
            print(json.dumps(info,ensure_ascii=False),flush=True)
        remaining=plan['budget']['seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Setup consumed registered wall budget.')
        result,theta=optimize_sequence(model,observations,bundle.wh[:,1].tolist(),seconds=remaining,
            max_evaluations=plan['budget']['max_evaluations'],progress=progress,
            root_policy='provided_initialization',initialization_source=bundle.report()['initialization_evidence'],evaluation_mode='streamed')
        assert torch.equal(K0,model.K) and torch.equal(E0,model.w2c)
        save(dest/'progress.json',dict(state='running',stage='final_export',pid=os.getpid(),updated=now(),gaussian_steps=0))
        after=export_frames(model,observations,bundle,dest/'fit.npz',deadline=time.monotonic()+plan['budget']['export_seconds'])
        bundle.assert_unchanged()
        for path,h in plan['source_sha256'].items():assert sha(path)==h,path
        assert sha(plan_path)==plan_hash,'Registered plan changed during execution.'
        torch.save(dict(model_state=model.state_dict(),frame_ids=model.frame_ids,reference_frame_id=bundle.reference_frame_id,
            inputs_sha256=sha(plan['inputs']),plan_sha256=plan_hash,optimizer_result=result),dest/'fit-state.pt')
        save(dest/'result.json',dict(updated=now(),state=result['state'],optimizer=result,frames=n,before=before,after=after,
            shared_identity=True,all_named_frames_exported=True,reference_frame_id=bundle.reference_frame_id,
            fresh_optimizer=True,old_optimized_fit_loaded=False,cached_prior_reused_explicitly=True,
            all_K_E_unchanged=True,camera_length_contract=model.camera_length_contract,
            output_fit_sha256=sha(dest/'fit.npz'),plan_sha256=plan_hash,seconds=time.monotonic()-started,
            budget_scope='Input/setup/initial export and optimization share fitting seconds; final frame export has its separately registered seconds, hard wrapper bounds whole run.',
            native_visual_review_pending=True,physical_quality_accepted=False,quality_accepted=False,
            gaussian_steps=0,goal_complete=False))
        save(dest/'progress.json',dict(state=result['state'],pid=None,finished=now(),gaussian_steps=0))
        print(json.dumps(dict(state=result['state'],frames=n,seconds=time.monotonic()-started)),flush=True)
    except BaseException:
        save(dest/'failure.json',dict(updated=now(),traceback=traceback.format_exc()))
        save(dest/'progress.json',dict(state='failed',pid=None,updated=now(),gaussian_steps=0))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();run(args.plan)
