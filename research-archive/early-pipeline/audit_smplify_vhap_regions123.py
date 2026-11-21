"""All-frame read-only shape/offset gradient and finite-difference audit."""
import os,time,json,fcntl,traceback
import numpy as np
import torch
import fit_smplify_vhap_surface123 as bridge
from vhap_demo_adapter123 import ROOT,make_tracker
from run_official_gaussians123 import read,save,sha,now

RUN=ROOT/'smplify-vhap-offset-init'
OUT=RUN/'regional-gradient-audit'
PLAN=ROOT/'smplify-vhap-regional-audit-plan.json'
REGIONS={'torso':[0,3,6,9],'arms':[13,14,16,17,18,19],
    'legs_and_feet':[1,2,4,5,7,8,10,11],'head_neck':[12,15,22,23,24],
    'hands':[20,21]+list(range(25,55))}


def summarize(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()))


def main():
    plan=read(PLAN)
    assert plan['optimizer_updates']==0 and plan['frames']==59
    assert plan['source_sha256']==sha(__file__)
    assert not OUT.exists(),'Never repeat or overwrite this audit.'
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32=False
    save(OUT/'progress.json',dict(updated=now(),state='checking',pid=os.getpid(),optimizer_updates=0))
    t,rows=make_tracker(OUT);t.cfg.render.background_train='white'
    checkpoint=RUN/'epoch-30/tracked-smplx-params.npz'
    with np.load(checkpoint,allow_pickle=False) as z,torch.no_grad():
        assert list(z['frame_ids'])==[r['frame'] for r in rows]==plan['roster']
        for name in bridge.PARAMS:getattr(t,name).copy_(torch.as_tensor(z[name],device='cuda'))
    params=t.get_train_parameters('rgb_global_tracking');trainable={id(p) for group in params.values() for p in group}
    for name in bridge.PARAMS:getattr(t,name).requires_grad_(id(getattr(t,name)) in trainable)
    t.vposer.requires_grad_(False);t.vposer.eval()
    before={k:bridge.digest_tensor(getattr(t,k)) for k in bridge.PARAMS}
    with torch.no_grad():
        cano=t.smplx.v_template+torch.einsum('vck,k->vc',t.smplx.shapedirs,t.betas)
        faces=t.smplx.faces_tensor.long();tri=cano[faces]
        fn=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1);normal=torch.zeros_like(cano)
        for c in range(3):normal.index_add_(0,faces[:,c],fn)
        normal=torch.nn.functional.normalize(normal,dim=-1)
        dominant=t.smplx.lbs_weights.argmax(-1)
        selections={k:torch.isin(dominant,torch.tensor(ids,device='cuda')) for k,ids in REGIONS.items()}
        direction=normal*(selections['torso']|selections['arms'])[:,None]
        offset0=t.static_offset.detach().clone()
        with np.load(RUN/'initial/tracked-smplx-params.npz',allow_pickle=False) as z:
            initial_beta=torch.as_tensor(z['betas'],device='cuda')
        shape_delta=torch.einsum('vck,k->vc',t.smplx.shapedirs,t.betas-initial_beta)
        shape_evidence={r:dict(vertices=int(sel.sum()),
            shape_only_mm=summarize(shape_delta[sel].norm(dim=-1).cpu().numpy()*1000),
            shared_offset_mm=summarize(offset0[0,sel].norm(dim=-1).cpu().numpy()*1000)) for r,sel in selections.items()}
    sums={};norms={};records=[];epsilon=plan['finite_difference_epsilon_m']
    for i,row in enumerate(rows):
        assert time.monotonic()-start<plan['max_seconds'],'Read-only audit budget reached'
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        t.clear_cache();sample=t.dataset.one(i)
        total,logs,*rest=t.compute_energy(sample,stage='rgb_global_tracking')
        target=sample['alpha_map'].cuda();pred=rest[-1]['pred_alpha'];area=target.sum().clamp_min(1)
        missing=t.cfg.w.photo*torch.relu(target-pred).sum()/area
        outside=t.cfg.w.photo*torch.relu(pred-target).sum()/area
        terms={'photo':logs['photo'],'landmark':logs['lmk'],'offset_L1':logs['reg_offset'],
            'offset_laplacian':logs['reg_offset_lap'],'alpha_missing':missing,'alpha_outside':outside}
        values={};frame_grad={}
        for name,loss in terms.items():
            grad,beta_grad=torch.autograd.grad(loss,(t.static_offset,t.betas),retain_graph=True,allow_unused=True)
            grad=torch.zeros_like(t.static_offset) if grad is None else grad.detach()
            assert torch.isfinite(grad).all() and torch.isfinite(loss)
            frame_grad[name]=grad[0].cpu().numpy()
            values[name]=dict(loss=float(loss.detach()),beta_gradient_l2=0 if beta_grad is None else float(beta_grad.norm()),regions={})
            for region,sel in selections.items():
                g=grad[0,sel];outward=-(g*normal[sel]).sum(-1)
                values[name]['regions'][region]=dict(offset_gradient_l2=float(g.norm()),
                    normal_descent_sum=float(outward.sum()),positive_normal_descent_sum=float(outward.clamp_min(0).sum()),
                    negative_normal_descent_sum=float(outward.clamp_max(0).sum()),
                    vertices_with_gradient=int((g.norm(dim=-1)>1e-8).sum()))
                norms.setdefault((name,region),[]).append(float(g.norm()))
            sums[name]=sums.get(name,0)+frame_grad[name]
        mask_grad=frame_grad['alpha_missing']+frame_grad['alpha_outside']
        regularizer_grad=frame_grad['offset_L1']+frame_grad['offset_laplacian']
        cosines={}
        for region,sel in selections.items():
            sel=sel.cpu().numpy();mg=mask_grad[sel].ravel();rg=regularizer_grad[sel].ravel()
            cosines[region]=float(np.dot(mg,rg)/max(float(np.linalg.norm(mg)*np.linalg.norm(rg)),1e-20))
        analytic=float((torch.tensor(mask_grad,device='cuda')*direction).sum())
        original_silhouette=float((missing+outside).detach())
        del total,logs,rest,pred,terms,grad,beta_grad,missing,outside
        # Fixed 0.1-mm signed displacement of torso/arms; restore exactly.
        perturbed=[]
        with torch.no_grad():
            for sign in (-1,1):
                t.static_offset.copy_(offset0+sign*epsilon*direction[None]);t.clear_cache()
                _,_,*output=t.compute_energy(sample,stage='rgb_global_tracking')
                pred=output[-1]['pred_alpha']
                perturbed.append(float((t.cfg.w.photo*(pred-target).abs().sum()/area).detach()))
                del output,pred
            t.static_offset.copy_(offset0)
        numerical=(perturbed[1]-perturbed[0])/(2*epsilon)
        records.append(dict(frame=row['frame'],terms=values,mask_regularizer_cosine=cosines,
            uniform_torso_arms_normal_test=dict(epsilon_m=epsilon,base_loss=original_silhouette,
                inward_loss=perturbed[0],outward_loss=perturbed[1],analytic_derivative=analytic,finite_difference_derivative=numerical)))
        save(OUT/'progress.json',dict(updated=now(),state='auditing',pid=os.getpid(),completed=i+1,total=59,
            optimizer_updates=0,seconds=time.monotonic()-start))
        del target,area,sample,values,frame_grad
    assert all(bridge.digest_tensor(getattr(t,k))==v for k,v in before.items())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    summaries={}
    for name in sums:
        summaries[name]={}
        for region,sel in selections.items():
            selected=sel.cpu().numpy();mean=sums[name][selected]/len(rows)
            summaries[name][region]=dict(
                per_frame_offset_gradient_l2=summarize(norms[name,region]),mean_gradient_l2=float(np.linalg.norm(mean)),
                cross_frame_consistency=float(np.linalg.norm(mean)/max(np.mean(norms[name,region]),1e-20)),
                summed_normal_descent_mean=float(-(mean*normal[sel].cpu().numpy()).sum()))
    test=[r['uniform_torso_arms_normal_test'] for r in records]
    result=dict(updated=now(),state='complete_read_only',frames=59,optimizer_updates=0,parameters_unchanged=True,
        source_checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-start,shape_vs_offset=shape_evidence,
        gradient_summary=summaries,uniform_normal_probe=dict(epsilon_m=epsilon,
            outward_lowers_loss=sum(r['outward_loss']<r['base_loss'] for r in test),
            inward_lowers_loss=sum(r['inward_loss']<r['base_loss'] for r in test),
            analytic_sum=sum(r['analytic_derivative'] for r in test),numerical_sum=sum(r['finite_difference_derivative'] for r in test)),
        records=records,quality_accepted=False,
        limits=['Read-only local derivative evidence is not a trained candidate or a proof of geometric accuracy.',
            'Raw beta and offset gradient magnitudes have different units and must not be compared directly.',
            'Cross-view gradient cancellation may arise from visibility, motion, correspondence or geometry; not a unique camera diagnosis.',
            'Dominant-skinning regions are fixed anatomical groups, not cloth labels or residual-selected points.'])
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state='complete_read_only',pid=None,
        completed=59,total=59,seconds=result['seconds'],optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','gradient_summary')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
