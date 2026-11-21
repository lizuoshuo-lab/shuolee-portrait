"""Read-only all59 gradient audit; no optimizer or camera update."""
import fcntl,json,time,dataclasses,traceback
import numpy as np
import torch
import fit_smplify_vhap_surface123 as bridge
from vhap_demo_adapter123 import ROOT,make_tracker
from run_official_gaussians123 import read,save,sha,now

RUN=ROOT/'smplify-vhap-joint'
OUT=RUN/'mask-gradient-audit'
STAGE='rgb_global_tracking'

def main():
    assert not OUT.exists(),'Read-only audit already has evidence; do not overwrite.'
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();torch.set_num_threads(4);torch.manual_seed(6060);start=time.monotonic()
    checkpoint=RUN/'epoch-30/tracked-smplx-params.npz';digest=sha(checkpoint)
    save(OUT/'plan.json',dict(created=now(),scope='All59 final joint-fit frames. Read-only loss and gradient decomposition.',
        checkpoint_sha256=digest,source_sha256=sha(__file__),optimizer_updates=0,max_seconds=300,
        hypothesis='The current RGB objective has no explicit silhouette term. Compare its shared-surface gradient against the normalized absolute-alpha term used by the existing head control.',
        candidate_silhouette='Existing photo weight30 times full-native-image mean abs(pred_alpha-target_alpha). No target alteration, crop or landmark filtering.',
        stop='No parameter updates. Finite losses/gradients and exact source checkpoint hash required.'))
    t,rows=make_tracker(OUT);t.cfg.render.background_train='white'
    with np.load(checkpoint,allow_pickle=False) as z,torch.no_grad():
        assert list(z['frame_ids'])==[r['frame'] for r in rows]
        for name in bridge.PARAMS:getattr(t,name).copy_(torch.as_tensor(z[name],device='cuda'))
    params=t.get_train_parameters(STAGE);trainable={id(p) for group in params.values() for p in group}
    for name in bridge.PARAMS:getattr(t,name).requires_grad_(id(getattr(t,name)) in trainable)
    t.vposer.requires_grad_(False);t.vposer.eval()
    before={k:bridge.digest_tensor(getattr(t,k)) for k in bridge.PARAMS}
    records=[]
    for i,row in enumerate(rows):
        assert time.monotonic()-start<300,'Bounded read-only audit timed out.'
        t.clear_cache();s=t.dataset.one(i)
        total,logs,*rest=t.compute_energy(s,stage=STAGE)
        target=s['alpha_map'].cuda();pred=rest[-1]['pred_alpha']
        silhouette=t.cfg.w.photo*(pred-target).abs().mean()
        regularization=logs['reg_offset']+logs['reg_offset_lap']
        terms={'existing_total':total,'photo':logs['photo'],'landmark':logs['lmk'],
            'offset_regularizers':regularization,'candidate_silhouette':silhouette}
        gradients={};values={}
        for name,loss in terms.items():
            grad=torch.autograd.grad(loss,(t.static_offset,t.pose_embedding),retain_graph=True,allow_unused=True)
            gradients[name]=[None if g is None else g.detach().clone() for g in grad]
            values[name]=dict(loss=float(loss.detach()),offset_grad_l2=float(grad[0].norm()) if grad[0] is not None else 0,
                body_latent_grad_l2=float(grad[1][i].norm()) if grad[1] is not None else 0)
            assert np.isfinite(list(values[name].values())).all(),(row['frame'],name)
        a=gradients['photo'][0].flatten();b=gradients['candidate_silhouette'][0].flatten()
        cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
        records.append(dict(frame=row['frame'],terms=values,photo_mask_offset_cosine=cosine,
            foreground_fraction=float((target>.5).float().mean()),optimizer_updates=0))
        save(OUT/'progress.json',dict(updated=now(),completed=i+1,total=59,optimizer_updates=0))
        del total,logs,rest,s,pred,target,terms,gradients,grad,a,b,silhouette,regularization
    assert sha(checkpoint)==digest
    assert all(bridge.digest_tensor(getattr(t,k))==v for k,v in before.items())
    summaries={}
    for name in records[0]['terms']:
        summaries[name]={}
        for metric in records[0]['terms'][name]:
            x=[r['terms'][name][metric] for r in records]
            summaries[name][metric]=dict(median=float(np.median(x)),maximum=float(np.max(x)))
    result=dict(updated=now(),state='complete',frames=59,optimizer_updates=0,parameters_unchanged=True,
        checkpoint_sha256=digest,seconds=time.monotonic()-start,terms=summaries,
        conflicting_photo_mask_offset_gradients=sum(r['photo_mask_offset_cosine']<0 for r in records),records=records,
        limits='Gradient availability is not evidence of successful clothing geometry or corrected elbows.')
    save(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
