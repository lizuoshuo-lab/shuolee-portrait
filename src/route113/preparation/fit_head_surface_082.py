"""Fresh shared FLAME surface under frozen full-image cameras.

The only comparison variable is shared identity shape. No camera correction,
old subject fit, per-frame pose, local vertex offset, or Gaussian is loaded.
"""
import argparse, inspect, json, time, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import shuolee_common as c
from prepare_head_surface_082 import O, A, S

def compat():
    for k,v in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
        if k not in np.__dict__:setattr(np,k,v)
    if not hasattr(inspect,'getargspec'):
        from collections import namedtuple
        Args=namedtuple('ArgSpec','args varargs keywords defaults')
        def getargspec(f):
            a=inspect.getfullargspec(f);return Args(a.args,a.varargs,a.varkw,a.defaults)
        inspect.getargspec=getargspec

def model():
    compat();torch=c.runtime()
    from flame_model.flame import FlameHead
    repo=Path(__file__).resolve().parent
    h=FlameHead(300,100,flame_model_path=str(c.PRIVATE/'models/flame2023/flame2023.pkl'),
        flame_lmk_embedding_path=str(c.PRIVATE/'models/flame/landmark_embedding_with_eyes.npy'),
        flame_template_mesh_path=str(c.PRIVATE/'models/flame/head_template_mesh.obj'),
        include_mask=False,add_teeth=False).cuda().double()
    h.requires_grad_(False)
    z=lambda k:torch.zeros((1,k),dtype=torch.float64,device='cuda')
    def native(beta):
        shape=z(300);shape[0,:len(beta)]=beta
        return h(shape,z(100),z(3),z(3),z(3),z(6),z(3),return_landmarks=True)
    with torch.no_grad():v,l=native(z(100)[0])
    basis=h.shapedirs[:,:,:100].permute(2,0,1).contiguous()
    lmk_faces=h.full_lmk_faces_idx.reshape(-1)
    bary=h.full_lmk_bary_coords.reshape(-1,3)
    lb=(basis[:,h.faces[lmk_faces]]*bary[None,:,:,None]).sum(2)
    return torch,h,native,v[0],l[0],basis,lb

def load(p):
    with np.load(p) as a:return {k:a[k] for k in a.files}

def quant(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.percentile(a,90)),mean=float(np.mean(a))) if len(a) else dict(n=0)

def project(X,K,E):
    v=np.einsum('fij,pj->fpi',E[:,:3,:3],X)+E[:,None,:3,3]
    q=np.einsum('fij,fpj->fpi',K,v)
    return q[:,:,:2]/q[:,:,2:]-.5,v[:,:,2]

def visibility(vertices,faces,points,E,scale,torch):
    """Segment/triangle visibility, no observed landmark pixels enter."""
    g=lambda x:torch.as_tensor(x,dtype=torch.float64,device='cuda')
    tri=g(vertices)[torch.as_tensor(faces,device='cuda',dtype=torch.long)]
    t0=tri[:,0];e1=tri[:,1]-t0;e2=tri[:,2]-t0
    centers=-np.einsum('fji,fj->fi',E[:,:3,:3],E[:,:3,3])
    origin=g(np.repeat(centers,len(points),axis=0));target=g(np.tile(points,(len(E),1)))
    result=[]
    with torch.no_grad():
        for start in range(0,len(origin),32):
            o=origin[start:start+32];direction=target[start:start+32]-o
            p=torch.cross(direction[:,None,:],e2[None],dim=-1)
            det=(e1[None]*p).sum(-1);valid=det.abs()>1e-12
            inv=torch.where(valid,1/det,torch.zeros_like(det));s=o[:,None,:]-t0
            u=(s*p).sum(-1)*inv;q=torch.cross(s,e1[None],dim=-1)
            v=(direction[:,None,:]*q).sum(-1)*inv;t=(e2[None]*q).sum(-1)*inv
            tol=.002*scale/torch.linalg.norm(direction,dim=-1)
            hit=valid&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(t>1e-7)&(t<(1-tol[:,None]))
            result.extend((~hit.any(-1)).cpu().tolist())
    return np.array(result).reshape(len(E),len(points))

def initialize():
    assert not (O/'initialization.npz').exists()
    a=load(O/'observations.npz');torch,h,native,v0,l0,basis,lb=model()
    # Meaningful algebra check against official native forward at nonzero shape.
    beta=torch.linspace(-.2,.2,100,dtype=torch.float64,device='cuda')
    with torch.no_grad():
        nv,nl=native(beta)
        ev=v0+torch.einsum('k,kvc->vc',beta,basis);el=l0+torch.einsum('k,kpc->pc',beta,lb)
        native_error=float(torch.max(torch.abs(nv[0]-ev)))
        landmark_error=float(torch.max(torch.abs(nl[0]-el)))
    assert native_error<1e-8 and landmark_error<1e-8
    ids=a['core_indices'];uv=a['uv'][:,ids];fit=~a['held_frames'];valid=a['finite'][:,ids]&a['inside'][:,ids]
    sigma=np.sqrt(4**2+a['flip'][:,ids]**2)
    E=a['w2c'];K=a['K'];P=K@E[:,:3];xyz=[];tri_stats=[]
    for k in range(len(ids)):
        mask=fit&valid[:,k];rows=[]
        for proj,q,s in zip(P[mask],uv[mask,k]+.5,sigma[mask,k]):
            rows.extend([(q[0]*proj[2]-proj[0])/s,(q[1]*proj[2]-proj[1])/s])
        _,_,vt=np.linalg.svd(np.array(rows),full_matrices=False);x=vt[-1,:3]/vt[-1,3]
        def residual(X):
            q=np.einsum('fij,j->fi',P[mask,:,:3],X)+P[mask,:,3]
            return ((q[:,:2]/q[:,2:]-uv[mask,k]-.5)/sigma[mask,k,None]).ravel()
        f=least_squares(residual,x,loss='soft_l1',f_scale=1,max_nfev=100)
        assert f.success
        xyz.append(f.x);tri_stats.append(dict(landmark=int(ids[k]),fit_views=int(mask.sum()),normalized_residual=quant(residual(f.x))))
    X=l0.cpu().numpy()[ids];Y=np.array(xyz)
    xc=X-X.mean(0);yc=Y-Y.mean(0);u,sv,vt=np.linalg.svd(yc.T@xc/len(X));sg=np.ones(3);sg[-1]=np.linalg.det(u@vt)
    R=u@np.diag(sg)@vt;scale=(sv*sg).sum()/np.mean(np.sum(xc**2,axis=1));t=Y.mean(0)-scale*R@X.mean(0)
    assert scale>0
    V=v0.cpu().numpy()@R.T*scale+t;L=l0.cpu().numpy()@R.T*scale+t
    pred,z=project(L[ids],K,E);assert np.min(z)>0
    vis=visibility(V,h.faces.cpu().numpy(),L[ids],E,scale,torch)&valid
    # Occlusion is fixed from fresh neutral alignment, independent of held pixels.
    assert vis.sum(axis=1).min()>=12,vis.sum(axis=1).tolist()
    np.savez_compressed(O/'initialization.npz',vertices=V,landmarks=L,faces=h.faces.cpu().numpy(),
        scale=scale,rotation=R,translation=t,shape=np.zeros(100),visibility=vis,
        initial_projection=pred,triangulated=Y,core_indices=ids)
    protocol=dict(created=c.stamp(),code_sha256=c.sha(__file__),
        objective='Compare fresh neutral rigid alignment with one shared100dim FLAME identity under fixed whole-image cameras, before any Gaussian initialization change.',
        source_cameras=str(S/'cameras.json'),camera_sha256=c.sha(S/'cameras.json'),
        observations_sha256=c.sha(O/'observations.npz'),initialization_sha256=c.sha(O/'initialization.npz'),
        native_forward_checks=dict(vertices_max_error=native_error,landmarks_max_error=landmark_error),
        official_source=str(Path(__file__).resolve().parent/'flame_model/flame.py'),official_source_sha256=c.sha(Path(__file__).resolve().parent/'flame_model/flame.py'),
        licensed_template_sha256=c.sha(c.PRIVATE/'models/flame2023/flame2023.pkl'),
        initialization='Fresh static triangulation of51core landmarks in140fitframes -> neutral FLAME proper Sim3. No held pixels enter initialization. No old personalized fit.',
        visibility='Freeze line-segment/triangle occlusion from this neutral alignment;2mm canonical endpoint tolerance scaled into scene units. No observed held pixel used. Fixed apparent jaw contour indices0..16 excluded.',
        arms=['rigid_neutral','shared_shape100'],steps=1000,shape_starts_after=250,
        loss='Equal-total-weight per frame, mean soft-L1 of2D norm divided by sqrt(4^2+flip_disagreement^2); identity prior0.05mean(beta^2). Detector scores are not probabilities and are not thresholded.',
        variables='One global proper rotation, translation, scale; second arm adds one shared100dim identity beta=3tanh(raw). Expression/jaw/neck/eyes zero; no frame-dependent geometry.',
        fixed='175full-image camera matrices and intrinsics, images, observed coordinates, visibility, fit/held split, optimizer budget',
        optimizer=dict(name='Adam',rotation_lr=.003,translation_lr_in_initial_scale=.001,log_scale_lr=.002,shape_raw_lr=.01),
        gates='Held nose/mouth median>=10percent improvement over rigid control; upper-corep90<=+2nativepx; no nonfinite/negative depth;>=95percent initially visible held queries still mesh-visible; max abs identity<2.95; source photograph and neck/collar visual review before use. Training readiness is not implied by numeric gates.',
        stop=['1000updates per arm','nonfinite loss/gradient','initial scale ratio outside0.5..2','source hash mismatch'],
        no_camera_updates=True,no_glasses_branch=True,no_dynamic=True,no_old_person_state=True,
        no_gaussian_updates=True,quality_accepted=False,initialization_visual_review_pending=True)
    c.save(O/'fit-protocol.json',protocol)
    c.save(O/'initialization-audit.json',dict(created=c.stamp(),scale=float(scale),visible_core_per_frame=vis.sum(1).tolist(),
        initial_fit=quant(np.linalg.norm(pred-uv,axis=-1)[vis&fit[:,None]]),
        initial_held=quant(np.linalg.norm(pred-uv,axis=-1)[vis&a['held_frames'][:,None]]),
        triangulation=tri_stats,held_frames_used_for_fit=False,shape_zero=True))
    print(json.dumps(dict(initialized=True,scale=scale,visible_min=int(vis.sum(1).min()),visible_max=int(vis.sum(1).max()),native_error=native_error)),flush=True)

def fit_arm(arm):
    assert arm in ['rigid_neutral','shared_shape100']
    assert c.read(O/'initialization-review.json')['allow_bounded_fit']
    target=O/(arm+'.npz');assert not target.exists()
    a=load(O/'observations.npz');init=load(O/'initialization.npz');p=c.read(O/'fit-protocol.json')
    assert c.sha(S/'cameras.json')==p['camera_sha256'] and c.sha(O/'observations.npz')==p['observations_sha256']
    torch,h,native,v0,l0,basis,lb=model();import roma
    g=lambda x:torch.as_tensor(x,dtype=torch.float64,device='cuda')
    ids=a['core_indices'];fit=~a['held_frames'];vis=init['visibility'];chosen=vis&fit[:,None]
    uv=g(a['uv'][:,ids]);sigma=g(np.sqrt(16+a['flip'][:,ids]**2));E=g(a['w2c']);K=g(a['K'])
    mask=g(chosen);frame_count=mask.sum(1).clamp_min(1);rot=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64,device='cuda'))
    trans=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64,device='cuda'));log_scale=torch.nn.Parameter(g(np.array(0.)))
    shape=torch.nn.Parameter(torch.zeros(100,dtype=torch.float64,device='cuda'))
    params=[dict(params=[rot],lr=.003),dict(params=[trans],lr=.001),dict(params=[log_scale],lr=.002)]
    if arm=='shared_shape100':params.append(dict(params=[shape],lr=.01))
    optimizer=torch.optim.Adam(params);R0=g(init['rotation']);t0=g(init['translation']);s0=float(init['scale'])
    trace=[];start=time.monotonic()
    for step in range(1000):
        beta=3*torch.tanh(shape) if arm=='shared_shape100' and step>=250 else shape*0
        local=l0[ids]+torch.einsum('k,kpc->pc',beta,lb[:,ids])
        R=roma.rotvec_to_rotmat(rot)@R0;s=s0*torch.exp(log_scale);t=t0+s0*trans
        world=local@R.T*s+t;cam=torch.einsum('fij,pj->fpi',E[:,:3,:3],world)+E[:,None,:3,3]
        q=torch.einsum('fij,fpj->fpi',K,cam);pr=q[:,:,:2]/q[:,:,2:]-.5
        r2=((pr-uv)/sigma[:,:,None]).square().sum(-1)
        data=((2*(torch.sqrt(1+r2)-1)*mask).sum(1)/frame_count)[g(fit).bool()].mean()
        prior=.05*beta.square().mean();loss=data+prior
        assert torch.isfinite(loss) and cam[:,:,2].min()>0 and .5<float(s/s0)<2
        optimizer.zero_grad();loss.backward()
        assert all(q.grad is None or torch.isfinite(q.grad).all() for q in [rot,trans,log_scale,shape])
        optimizer.step()
        if step%100==0 or step==999:
            item=dict(step=step+1,loss=float(loss),data=float(data),prior=float(prior),seconds=time.monotonic()-start)
            trace.append(item);print(json.dumps(dict(arm=arm,**item)),flush=True)
    with torch.no_grad():
        beta=3*torch.tanh(shape) if arm=='shared_shape100' else shape*0
        nv,nl=native(beta);R=roma.rotvec_to_rotmat(rot)@R0;s=s0*torch.exp(log_scale);t=t0+s0*trans
        V=(nv[0]@R.T*s+t).cpu().numpy();L=(nl[0]@R.T*s+t).cpu().numpy()
    np.savez_compressed(target,vertices=V,landmarks=L,faces=h.faces.cpu().numpy(),shape=beta.cpu().numpy(),rotation=R.cpu().numpy(),translation=t.cpu().numpy(),scale=float(s))
    pred,z=project(L[ids],a['K'],a['w2c']);err=np.linalg.norm(pred-a['uv'][:,ids],axis=-1)
    finalvis=visibility(V,h.faces.cpu().numpy(),L[ids],a['w2c'],float(s),torch)
    held=vis&a['held_frames'][:,None];lower=np.isin(ids,np.r_[27:36,48:68]);upper=~lower
    result=dict(created=c.stamp(),arm=arm,steps=1000,seconds=time.monotonic()-start,
        fit=quant(err[chosen]),held=quant(err[held]),held_nose_mouth=quant(err[held&lower[None]]),
        held_upper=quant(err[held&upper[None]]),held_visibility_retention=float(finalvis[held].mean()),
        max_shape=float(np.abs(beta.cpu().numpy()).max()),positive_all=bool((z>0).all()),scale=float(s),
        shape_zero=bool((beta==0).all()),trace=trace,parameter_sha256=c.sha(target),source_cameras_unchanged=c.sha(S/'cameras.json')==p['camera_sha256'],
        source_code_sha256=c.sha(__file__),gaussian_updates=0,geometry_accepted=False)
    c.save(O/(arm+'-result.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','rigid_neutral','shared_shape100']);a=p.parse_args()
    if a.stage=='initialize':initialize()
    else:fit_arm(a.stage)
