"""One read-only replay of saved shared-identity state; no optimization."""
from pathlib import Path
import argparse, copy, fcntl, json, os, time, traceback
import numpy as np
import torch
import fit_shared_sequence123 as f
from shuolee_portrait.sequence_fitting import SharedIdentitySequence, sequence_residuals
r=f.r
OUT=f.OUT/'final-state-diagnosis'

def prepare():
    assert not OUT.exists(), 'This registered read-only diagnosis is once-only.'
    parent=r.read(f.OUT/'next-plan.json')
    assert parent['budget']['optimizer_steps']==0
    sources=[Path(__file__),Path(f.__file__),Path(r.__file__),
             f.PROJECT/'src/shuolee_portrait/sequence_fitting.py',
             f.PROJECT/'src/shuolee_portrait/sequence_optimizer.py',
             f.OUT/'fit.npz',f.OUT/'result.json',f.OUT/'observation-review.json',f.OUT/'next-plan.json']
    OUT.mkdir()
    r.save(OUT/'plan.json',dict(registered=r.now(),parent_plan=str(f.OUT/'next-plan.json'),
      source_hashes={str(p):r.sha(p) for p in sources},
      purpose='Exact saved-state loss/gradient decomposition and eligibility audit, not a new fit.',
      budget=dict(seconds=120,optimizer_steps=0,new_detector_passes=0,gaussian_steps=0),
      checks=['saved geometry/projection parity','data plus prior gradient equals total',
              'all face residuals classified by frozen observation eligibility',
              'per-variable and per-frame gradient blocks at unchanged final state',
              'final-state local column-normalized gradient, not a convergence certificate',
              'exact stage stopping reasons, input hash preservation']))

def run():
    plan=r.read(OUT/'plan.json');assert not (OUT/'progress.json').exists()
    for p,h in plan['source_hashes'].items():assert r.sha(p)==h,p
    start=time.monotonic()
    def bound():
        if time.monotonic()-start>120:raise TimeoutError('Read-only120second budget ended.')
    r.save(OUT/'progress.json',dict(state='running',pid=os.getpid(),started=r.now()))
    saved=dict(np.load(f.OUT/'fit.npz'));fitresult=r.read(f.OUT/'result.json')
    _,rows,obs,pred=r.inputs();parts=r.setup()
    ref=r.Ref(parts,rows[0]['K_integer_centers'],r.parameters(pred,rows[0]['official_prediction_slot']))
    initial={k:saved[k] for k in ['global_orient','transl','latent','jaw_pose','expression']}
    model=SharedIdentitySequence(ref.body,copy.deepcopy(parts[4]),saved['frame_ids'],saved['K_integer_centers'],saved['w2c'],initial)
    with torch.no_grad():model.state['betas'].copy_(torch.as_tensor(saved['betas'],device='cuda'))
    observations=f.make_observations(model,rows,obs)
    assert np.array_equal(observations.targets.cpu().numpy(),saved['targets'])
    assert np.array_equal(observations.fitting_confidence().cpu().numpy(),saved['fitting_confidence'])
    theta=model.flatten().requires_grad_(True)
    residual,out=sequence_residuals(model,theta,observations,4,[rows[s]['wh'][1] for s in f.SLOTS])
    uv=model.project(out);n=len(model.frame_ids);ndata=n*118*2
    parity=dict(vertices_max=float(np.max(np.abs(out.vertices.detach().cpu().numpy()-saved['vertices_camera']))),
                projection_max_px=float(np.max(np.abs(uv.detach().cpu().numpy()-saved['projected']))))
    assert parity['vertices_max']<1e-10 and parity['projection_max_px']<1e-8,parity
    lengths=[('data',ndata),('latent',n*32),('bending',n*4),
             ('left_hand_constant',out.left_hand_pose.numel()),('right_hand_constant',out.right_hand_pose.numel()),
             ('expression',n*10),('jaw',n*3),('shared_shape',10)]
    terms={};offset=0
    for key,size in lengths:terms[key]=residual[offset:offset+size].square().sum();offset+=size
    assert offset==residual.numel()
    loss=residual.square().sum();data=terms['data'];prior=loss-data
    gradients={key:torch.autograd.grad(value,theta,retain_graph=True)[0] for key,value in [('total',loss),('data',data),('prior',prior)]}
    add_error=float((gradients['total']-gradients['data']-gradients['prior']).abs().max())
    assert add_error<1e-8
    recorded=fitresult['optimizer']['stages'][-1]['best_loss']
    assert abs(float(loss)-recorded)<1e-8*max(1,abs(recorded))
    # Full residual sensitivity only at the saved state. No step or parameter update.
    J=[]
    for a in range(0,residual.numel(),8):
        bound();m=min(8,residual.numel()-a)
        basis=torch.zeros((m,residual.numel()),device='cuda',dtype=theta.dtype)
        basis[torch.arange(m,device='cuda'),a+torch.arange(m,device='cuda')]=1
        J.append(torch.autograd.grad(residual,theta,grad_outputs=basis,is_grads_batched=True,retain_graph=True)[0].detach())
    J=torch.cat(J);scale=torch.rsqrt((2*J.square().sum(0)).clamp_min(1))
    gradient_replay_error=float((2*J.T@residual.detach()-gradients['total']).abs().max())
    assert gradient_replay_error<1e-7
    def metrics(a,b):
        g={key:value[a:b].detach() for key,value in gradients.items()}
        denominator=float(g['data'].norm()*g['prior'].norm())
        return dict(**{key:dict(norm=float(v.norm()),abs_max=float(v.abs().max()),
                               final_column_scaled_abs_max=float((scale[a:b]*v).abs().max())) for key,v in g.items()},
                    data_prior_cosine=float(torch.dot(g['data'],g['prior']))/denominator if denominator>0 else None,
                    residual_fraction=float(g['total'].norm())/max(1e-30,float(g['data'].norm()+g['prior'].norm())))
    blocks={}
    for key,(a,b,shape) in model.layout().items():
        blocks[key]=dict(shape=shape,**metrics(a,b))
        if shape[0]==n:
            width=(b-a)//n
            blocks[key]['frames']=[dict(frame=model.frame_ids[i],**metrics(a+i*width,a+(i+1)*width)) for i in range(n)]
    records=[];summaries=[]
    dist=torch.linalg.vector_norm(uv-observations.targets,dim=-1).detach().cpu().numpy()
    effective=observations.fitting_confidence().cpu().numpy()
    reviewed=r.read(f.OUT/'observation-review.json')['records'];by={(v['slot'],v['ibug68_index']):v for v in reviewed}
    for i,slot in enumerate(f.SLOTS):
        groups={'active_fit':[],'reviewed_held_semantic':[],'unknown_or_unverified':[]}
        for k in range(51):
            p=by[slot,k+17];eligible=p['visibility']=='observed' and p['semantic_identity_verified']
            group=('active_fit' if p['split']=='FIT' else 'reviewed_held_semantic') if eligible else 'unknown_or_unverified'
            value=float(dist[i,67+k]);groups[group].append(value)
            records.append(dict(frame=model.frame_ids[i],source_frame=slot*5,ibug68=k+17,group=group,
              error_px=value,delta_uv=(uv.detach().cpu().numpy()[i,67+k]-saved['targets'][i,67+k]).tolist(),
              fitting_confidence=float(effective[i,67+k]),physical_identity_verified=False))
        summaries.append(dict(source_frame=slot*5,groups={k:r.stats(v) for k,v in groups.items()},
                              body_active=r.stats(dist[i,:25][effective[i,:25]>0]),
                              body_excluded=r.stats(dist[i,:25][effective[i,:25]==0])))
    root_policy=[]
    for i,slot in enumerate(f.SLOTS):
        root_policy.append(dict(source_frame=slot*5,root_joint_indices=[9,12,2,5],
          raw_confidence=saved['confidence'][i,[9,12,2,5]].tolist(),
          joint_stage_confidence=effective[i,[9,12,2,5]].tolist(),
          old_root_uses_unit_weight_for_all_four=True))
    bound()
    for p,h in plan['source_hashes'].items():assert r.sha(p)==h,p
    np.savez_compressed(OUT/'gradients.npz',theta=theta.detach().cpu().numpy(),
      **{k:v.detach().cpu().numpy() for k,v in gradients.items()},final_local_scale=scale.cpu().numpy())
    result=dict(updated=r.now(),state='complete',seconds=time.monotonic()-start,
      saved_state_replay=parity,objective=float(loss),objective_terms={k:float(v) for k,v in terms.items()},
      gradient_additivity_max_error=add_error,jacobian_gradient_max_error=gradient_replay_error,
      parameter_blocks=blocks,face_error_groups=summaries,face_points=records,
      stages=fitresult['optimizer']['stages'],
      stage3_reached_call30_but_stopped_by='relative_objective',stage4_stopped_by='outer_budget',
      root_eligibility_audit=root_policy,
      final_state_stationary_under_1e9_scaled_gradient=float((scale*gradients['total']).abs().max())<=1e-9,
      local_scale_warning='Recomputed at the saved final state for diagnosis; differs from frozen stage-entry optimizer scale.',
      anatomical_holdouts_are_not_material_truth=True,detector_passes=0,optimizer_steps=0,gaussian_steps=0,
      original_fit_unchanged=True,unique_cause_identified=False,quality_accepted=False,
      limits=['No endpoint state was logged for old LBFGS stopping checks.',
              'Data/prior gradient conflict is descriptive; cannot identify camera or detector error uniquely.',
              'No independent physical face measurements or profile semantic holdout.'])
    r.save(OUT/'result.json',result)
    r.save(OUT/'progress.json',dict(state='complete',pid=None,finished=r.now(),seconds=result['seconds']))
    print(json.dumps(dict(state='complete',seconds=result['seconds'],objective=result['objective'],
      objective_terms=result['objective_terms'],blocks={k:v['total']['final_column_scaled_abs_max'] for k,v in blocks.items()},
      groups=summaries)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception:
            r.save(OUT/'failure.json',dict(updated=r.now(),traceback=traceback.format_exc()))
            r.save(OUT/'progress.json',dict(state='failed',pid=None));raise
