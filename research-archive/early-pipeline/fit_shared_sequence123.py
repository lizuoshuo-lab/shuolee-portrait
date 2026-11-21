"""Once-only three-frame shared-identity adaptation on frozen raw targets."""
from pathlib import Path
import argparse,copy,fcntl,json,os,sys,time,traceback
import numpy as np
import torch
import run_smplify_reference123 as r

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.sequence_fitting import SharedIdentitySequence,SequenceObservations
from shuolee_portrait.sequence_optimizer import optimize_sequence

OUT=r.O/'framewise-canonical-route/shared-three-frame'
SLOTS=[0,31,62]

def prepare():
    assert not OUT.exists(),'One shared-identity trial, no parameter sweep.'
    control=r.read(OUT.parent/'shared-sequence-control/result.json');assert control['state']=='passed'
    _,rows,obs,_=r.inputs()
    OUT.mkdir()
    # Actual raw/detector crop pairs were inspected before this inventory.
    # This is anatomical visibility, never independently measured material truth.
    visible={0:[30],31:[27,28,29,30,31,32,33,34,35,36,39,48,50,51,52,54,56,57,58],
             62:[30,31,32,33,34,35,48,50,51,52,54,56,57,58]}
    descriptions={0:'Profile. Hair and self-occlusion hide eyes/brows and far-side anatomy. Only nose-tip identity is clearly supported; no other face point promoted.',
      31:'Frontal. Nose and lip curves/corners visible; two near-eye corners recognizable. Other eye/brow points remain unknown due blur/hair.',
      62:'Three-quarter. Nose-base and lip regions visible; eyelids closed/blurred and far brow/hair ambiguous, kept unknown.'}
    records=[]
    for slot in SLOTS:
        for ibug in range(17,68):
            records.append(dict(slot=slot,frame=str(rows[slot]['frame']),source_frame_index=rows[slot]['source_frame_index'],
              ibug68_index=ibug,uv_native=obs['keypoints'][slot,23+ibug].tolist(),
              visibility='observed' if ibug in visible[slot] else 'unknown',
              semantic_identity_verified=ibug in visible[slot],physical_identity_verified=False,
              split='HELD' if (ibug-17)%5==0 else 'FIT',
              family_id=f'anatomical_ibug68:{ibug}',family_kind='semantic, not verified physical material',
              visibility_source='assistant_raw_image_review',localization_uncertainty_px=None,
              source_image=str(r.O/f'input_slot{slot:03d}_face_raw_observations.png'),note=descriptions[slot]))
    review=dict(updated=r.now(),records=records,source_coordinates_changed=False,
                reviewed_before_fitting=True,confidence_not_used_to_assign_visibility=True,
                no_residual_based_selection=True,physical_truth_or_subpixel_localization_established=False,
                hand_policy='prior_only_all_frames',
                body_policy='Preserve generic relative detector confidence as heuristic; shoulders/hips under cloth are not directly measured skeletal joints.')
    r.save(OUT/'observation-review.json',review)
    sources=[Path(__file__),Path(r.__file__),PROJECT/'src/shuolee_portrait/sequence_fitting.py',PROJECT/'src/shuolee_portrait/sequence_optimizer.py',
             r.R/'data/inputs.json',r.R/'data/observations.npz',r.R/'inference/prediction-postprocessed.npz',OUT/'observation-review.json']
    sources += [r.O/f'input_slot{s:03d}_face_raw_observations.png' for s in SLOTS]
    r.save(OUT/'plan.json',dict(registered=r.now(),state='registered_not_executed',
      purpose='Three-frame anatomical fitting adaptation of the user-authorized per-frame route. Not camera validation or Gaussian admission.',
      sources={str(p):r.sha(p) for p in sources},slots=SLOTS,source_frames=[0,155,310],
      shared_identity='One zero-initialized10-beta tensor. No per-view fitted beta average or old geometry.',
      per_frame='Original same-frame GVHMR root/body pose encoded once into VPoser; fresh jaw/expression zero; neutral prior-only hands.',
      camera='Frozen diagnostic K/E; roots parameterized in each real camera. World vertices transformed with exact inverse E; no naive SMPL root translation conversion.',
      observations='All raw targets retained. Only image-reviewed face semantics enter FIT; whole nominal semantic families HELD. Physical material identity and subpixel truth remain unverified.',
      contrast_limit='Shared shape, hand-prior mode, and conservative visibility/alias handling differ from the old independent fits. Combined adaptation, not a single-factor camera experiment.',
      loss='Same five official prior schedules, fullK native projection, FP64, mean per-frame objective with shared shape prior; fixed objective-derived column scaling.',
      budget=dict(seconds=600,evaluations=6000,root_LBFGS_inner=60,stages=5,outer_per_stage=30,inner_per_outer=30,real_trials=1,gaussian_steps=0),
      stop=['invalid or behind-camera geometry','registered budget','No Gaussian admission if any raw face/body median regresses versus prior comparison, or raw review fails.'],
      physical_acceptance=False))

def make_observations(model,rows,obs):
    targets=[];confidence=[]
    for slot in SLOTS:
        gt,c=r.targets(obs,slot);targets.append(gt.double());confidence.append(c.double())
    review=r.read(OUT/'observation-review.json')['records'];by={(x['slot'],x['ibug68_index']):x for x in review}
    vis=np.array([[by[s,k]['visibility'] for k in range(17,68)] for s in SLOTS])
    split=np.array([[by[s,k]['split'] for k in range(17,68)] for s in SLOTS])
    semantic=np.array([[by[s,k]['semantic_identity_verified'] for k in range(17,68)] for s in SLOTS])
    family=np.array([[by[s,k]['family_id'] for k in range(17,68)] for s in SLOTS])
    return SequenceObservations(model.frame_ids,torch.cat(targets),torch.cat(confidence),vis,split,semantic,np.zeros((3,51),bool),family)

def report(model,observations):
    with torch.no_grad():out=model.forward_state();uv=model.project(out).cpu().numpy()
    target=observations.targets.cpu().numpy();conf=observations.confidence.cpu().numpy();views=[]
    for i,slot in enumerate(SLOTS):
        distance=np.linalg.norm(uv[i,67:]-target[i,67:],axis=-1)
        views.append(dict(slot=slot,source_frame=slot*5,all_raw_detector_errors=r.errors(uv[i:i+1],target[i:i+1],conf[i:i+1]),
          reviewed_fit_face_errors=r.stats(distance[observations.face_active('FIT')[i].cpu().numpy()]),
          reviewed_held_semantic_face_errors=r.stats(distance[observations.face_active('HELD')[i].cpu().numpy()]),
          physical_face_truth=False))
    return views,out,uv

def run():
    plan=r.read(OUT/'plan.json');assert not (OUT/'progress.json').exists()
    for path,h in plan['sources'].items():assert r.sha(path)==h,path
    r.save(OUT/'progress.json',dict(state='running',pid=os.getpid(),started=r.now(),stage='setup'))
    _,rows,obs,pred=r.inputs();parts=r.setup();params=[r.parameters(pred,rows[s]['official_prediction_slot']) for s in SLOTS]
    ref=r.Ref(parts,rows[0]['K_integer_centers'],params[0]);vp=copy.deepcopy(parts[4]).double()
    with torch.no_grad():latent=vp.encode(torch.as_tensor(np.stack([x['body_pose'] for x in params]).reshape(3,63),device='cuda',dtype=torch.float64)).mean
    initial=dict(global_orient=np.stack([x['global_orient'] for x in params]).reshape(3,3),
       transl=np.stack([x['transl'] for x in params]).reshape(3,3),latent=latent,
       jaw_pose=np.zeros((3,3)),expression=np.zeros((3,10)))
    model=SharedIdentitySequence(ref.body,vp,[rows[s]['frame'] for s in SLOTS],
          np.array([rows[s]['K_integer_centers'] for s in SLOTS]),np.array([rows[s]['w2c'] for s in SLOTS]),initial)
    observations=make_observations(model,rows,obs)
    assert not model.state['betas'].any() and not observations.physical_face_verified.any()
    K0=model.K.clone();E0=model.w2c.clone();initial_vector=model.flatten();before,initial_out,initial_uv=report(model,observations)
    np.savez_compressed(OUT/'initial.npz',theta=initial_vector.cpu().numpy(),K=model.K.cpu().numpy(),w2c=model.w2c.cpu().numpy(),
      projected=initial_uv,vertices_camera=initial_out.vertices.detach().cpu().numpy(),faces=model.body.faces,
      targets=observations.targets.cpu().numpy(),confidence=observations.confidence.cpu().numpy())
    def progress(info,theta):
        r.save(OUT/'progress.json',dict(state='running',pid=os.getpid(),updated=r.now(),**info,gaussian_steps=0))
        if 'outer' not in info or info['outer']%5==0:print(json.dumps(info),flush=True)
    fitted,theta=optimize_sequence(model,observations,[rows[s]['wh'][1] for s in SLOTS],seconds=600,max_evaluations=6000,progress=progress)
    after,out,uv=report(model,observations)
    assert torch.equal(K0,model.K) and torch.equal(E0,model.w2c)
    snapshot=model.native_snapshot(rows[31]['frame'])
    np.savez_compressed(OUT/'fit.npz',**{k:v.detach().cpu().numpy() for k,v in model.state.items()},
        vertices_camera=out.vertices.detach().cpu().numpy(),vertices_world=snapshot['vertices_world_sequence'].cpu().numpy(),
        joints_camera=out.joints.detach().cpu().numpy(),projected=uv,initial_projected=initial_uv,
        targets=observations.targets.cpu().numpy(),confidence=observations.confidence.cpu().numpy(),
        fitting_confidence=observations.fitting_confidence().cpu().numpy(),
        K_integer_centers=model.K.cpu().numpy(),w2c=model.w2c.cpu().numpy(),faces=model.body.faces,
        frame_ids=np.array(model.frame_ids),source_frames=np.array([0,155,310]),
        reference_vertices_world=snapshot['reference_vertices_world'].cpu().numpy())
    comparisons=[]
    for i,s in enumerate(SLOTS):
        previous=r.read(r.O/f'conditioned-reference/slot{s:03d}-result.json')['final']['errors_px']
        current=after[i]['all_raw_detector_errors']
        comparisons.append(dict(slot=s,previous=previous,current=current,
          face_body_median_nonregression=all(current[k]['median']<=previous[k]['median']+.01 for k in ['face51','body17'])))
    for path,h in plan['sources'].items():assert r.sha(path)==h,path
    result=dict(updated=r.now(),state=fitted['state'],optimizer=fitted,before=before,after=after,comparisons=comparisons,
       same_shared_betas_for_every_frame=model.state['betas'].detach().cpu().tolist(),frame_ids=model.frame_ids,
       all_K_E_unchanged=True,old_fit_used_only_as_comparison=True,fresh_zero_identity=True,
       diagnostic_nonregression=all(x['face_body_median_nonregression'] for x in comparisons),
       combination_effect=True,quality_accepted=False,physical_material_correspondence_verified=False,
       native_visual_review_pending=True,gaussian_steps=0,goal_complete=False,source_hashes=plan['sources'])
    r.save(OUT/'result.json',result);r.save(OUT/'progress.json',dict(state='complete' if fitted['state']=='complete' else fitted['state'],pid=None,finished=r.now(),gaussian_steps=0))
    print(json.dumps(dict(state=result['state'],seconds=fitted['seconds'],after=after,diagnostic_nonregression=result['diagnostic_nonregression'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception:
            r.save(OUT/'failure.json',dict(updated=r.now(),traceback=traceback.format_exc()));r.save(OUT/'progress.json',dict(state='failed',pid=None));raise
