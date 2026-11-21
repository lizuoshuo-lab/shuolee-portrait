"""Read-only one-component-at-a-time evaluation; no fitting or camera update."""
from audit_grass_smplify_chain123 import ROOT,O,MODEL,REGDIR,read,save,sha,now,stats
import argparse,time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def register():
    assert not (O/'mean-ablation-plan.json').exists()
    p=read(O/'plan.json')
    save(O/'mean-ablation-plan.json',dict(created=now(),inputs=p['inputs'],base='Saved per-frame prior, converted to world using fixed94cameras and shared scale; no new inference.',
        changes=['Only betas replace by median','Only21local rotations replace by SO3 means','Only root orientation replace by SO3 mean','Only world pelvis replace by median','All shared components, parity against saved initialization'],
        scope='Diagnostic perturbations, no optimized parameters or candidate model. Quantifies each replacement effect in the fixed failed setup; not physical cause identification.',
        roi_or_threshold_changes=False,cpu_seconds_budget=90,fit_steps=0,gaussian_steps=0))
def run():
    assert not (O/'mean-ablation-result.json').exists()
    plan=read(O/'mean-ablation-plan.json');assert all(sha(p)==s for p,s in plan['inputs'].items());start=time.monotonic()
    import torch,smplx
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    rows=read(ROOT/'data/inputs.json')['records'];n=len(rows);ids=np.array([r['official_prediction_slot'] for r in rows])
    pred=dict(np.load(ROOT/'inference/prediction-postprocessed.npz'));init=dict(np.load(ROOT/'fit/initialization.npz'));w=dict(np.load(ROOT/'fit/observation-weights.npz'))
    E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);K[:,:2,2]-=.5
    rot=Rotation.from_rotvec(pred['incam_global_orient'][ids]).as_matrix();p=pred['incam_joints'][ids,0]
    A=np.concatenate([E[:,:3,:3],-p[:,:,None]],-1).reshape(-1,4);rhs=-E[:,:3,3].reshape(-1)
    s=float(least_squares(lambda x:A@x-rhs,np.linalg.lstsq(A,rhs,rcond=None)[0],loss='soft_l1',f_scale=.05).x[3])
    R=E[:,:3,:3].transpose(0,2,1)@rot;t=np.einsum('nji,nj->ni',E[:,:3,:3],s*p-E[:,:3,3])
    Rmean=np.broadcast_to(Rotation.from_matrix(R).mean().as_matrix(),R.shape);tmean=np.broadcast_to(np.median(t,axis=0),t.shape)
    pose=pred['incam_body_pose'][ids];pr=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(n,21,3,3)
    posemean=np.broadcast_to(np.stack([Rotation.from_matrix(pr[:,j]).mean().as_rotvec() for j in range(21)]).reshape(1,63),pose.shape)
    beta=pred['incam_betas'][ids];betamean=np.broadcast_to(np.median(beta,axis=0),beta.shape)
    model=smplx.SMPLX(str(MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False)
    reg=(torch.load(REGDIR/'smpl_coco17_J_regressor.pt',map_location='cpu',weights_only=True)@torch.load(REGDIR/'smplx2smpl_sparse.pt',map_location='cpu',weights_only=True).to_dense()).numpy()
    def local(bp,be):
        vertices=[]
        for i in range(n):
            z=model(body_pose=torch.tensor(bp[i:i+1].copy(),dtype=torch.float32),betas=torch.tensor(be[i:i+1].copy(),dtype=torch.float32))
            vertices.append((z.vertices-z.joints[:,:1]).numpy()[0])
        return np.stack(vertices)
    v=local(pose,beta);vmshape=local(pose,betamean);vmpose=local(posemean,beta);vmall=local(posemean,betamean)
    configs=[('per_frame',v,R,t),('only_shape_median',vmshape,R,t),('only_pose_mean',vmpose,R,t),('only_root_mean',v,Rmean,t),('only_pelvis_median',v,R,tmean),('all_shared',vmall,Rmean,tmean)]
    result=[];target=w['body_target'];valid=w['body']>.25
    for name,vl,rl,tl in configs:
        world=np.einsum('nij,nvj->nvi',rl,vl)*s+tl[:,None]
        co=np.einsum('jv,nvc->njc',reg,world)
        cam=np.einsum('nij,nvj->nvi',E[:,:3,:3],co)+E[:,:3,3][:,None];q=np.einsum('nij,nvj->nvi',K,cam);uv=q[...,:2]/q[...,2:]
        err=np.linalg.norm(uv-target,axis=-1)
        result.append(dict(variant=name,reliable=stats(err[valid]),all=stats(err)))
        if name=='all_shared':parity=float(np.max(np.abs(world-init['vertices_world'])))
    save(O/'mean-ablation-result.json',dict(created=now(),seconds=time.monotonic()-start,results=result,all_shared_matches_saved_initialization_max_world=parity,
        world_pelvis_spread_in_body_units=stats(np.linalg.norm(t-tmean,axis=-1)/s),body_units_are_template_metric_not_independent_scale_truth=True,
        original_inputs_unchanged=all(sha(p)==h for p,h in plan['inputs'].items()),camera_changed=False,fit_steps=0,gaussian_steps=0,
        interpretation='Components are changed one at a time around the same prior. Effects are nonlinear, not additive. No camera, movement or correspondence cause is uniquely established.'))
    print(result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.register:register()
    elif a.run:run()
