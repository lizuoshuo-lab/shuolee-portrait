"""Preserve strict image parity; separately certify isolated alpha-cutoff roundoff.

No model, image, kernel, or threshold is modified. A certificate is not a quality
acceptance and never changes the recorded strict maximum-error verdict.
"""
import math
import numpy as np
from audit_indoor_export_pixel124 import buffer_fields


def covariance64(pc):
    q=pc.get_rotation.detach().cpu().numpy().astype(np.float64)
    s=pc.get_scaling.detach().cpu().numpy().astype(np.float64)
    r,x,y,z=q.T
    # Row-major form of the actual CUDA glm matrix; do not add normalization.
    R=np.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
                2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
                2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)),axis=1).reshape(-1,3,3)
    A=R*s[:,None,:]
    return A@A.transpose(0,2,1)


def capture(pc,cam,torch):
    import diff_gaussian_rasterization as dgr
    n=len(pc.get_xyz);h,w=int(cam.image_height),int(cam.image_width)
    empty=torch.Tensor([]);bg=torch.zeros(3,device='cuda')
    args=(bg,pc.get_xyz,empty,pc.get_opacity,pc.get_scaling,pc.get_rotation,1.,empty,
          cam.world_view_transform.cuda(),cam.full_proj_transform.cuda(),math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),
          h,w,pc.get_features,pc.active_sh_degree,cam.camera_center.cuda(),False,False)
    nrender,rgb,radii,geom,bins,img=dgr._C.rasterize_gaussians(*args)
    g=buffer_fields(geom,[('depth','<f4',n,1),('unused_radius_prefix','?',n,3),('unused_radii','<i4',n,1),
        ('xy','<f4',n,2),('cov3d','<f4',n,6),('conic','<f4',n,4),('rgb','<f4',n,3)])
    b=buffer_fields(bins,[('ids','<u4',nrender,1)])
    im=buffer_fields(img,[('T','<f4',h*w,1),('n','<u4',h*w,1),('ranges','<u4',h*w,2)])
    return dict(rgb=rgb,g=g,b=b,i=im,radii=radii)


def certify(pc_a,pc_b,cam_a,cam_b,torch,render,pipe,R=None,standard_scale_encoding=False):
    R=np.eye(3) if R is None else np.asarray(R,dtype=np.float64)
    eps=float(np.finfo(np.float32).eps)
    xyz_a=pc_a.get_xyz.detach().cpu().numpy().astype(float)
    xyz_b=pc_b.get_xyz.detach().cpu().numpy().astype(float)
    center_error=float(np.max(np.abs(xyz_a@R.T-xyz_b)))
    scales_a=pc_a.get_scaling.detach().cpu().numpy().astype(float)
    scales_b=pc_b.get_scaling.detach().cpu().numpy().astype(float)
    scale_relative=float(np.max(np.abs(scales_a-scales_b)/scales_a))
    cov_a=R@covariance64(pc_a)@R.T;cov_b=covariance64(pc_b)
    covariance_relative=float(np.max(np.linalg.norm(cov_a-cov_b,axis=(1,2))/np.linalg.norm(cov_a,axis=(1,2))))
    properties=dict(center_max_abs_m=center_error,scale_relative=scale_relative,covariance_relative=covariance_relative,
        color_exact=bool(torch.equal(pc_a.get_features,pc_b.get_features)),opacity_exact=bool(torch.equal(pc_a.get_opacity,pc_b.get_opacity)),
        center_allowance_m=8*eps*max(float(np.abs(xyz_a).max()),1),scale_allowance=8*eps,covariance_allowance=64*eps)
    properties['passed']=bool(center_error<=properties['center_allowance_m'] and scale_relative<=8*eps and
        covariance_relative<=64*eps and properties['color_exact'] and properties['opacity_exact'])
    if standard_scale_encoding:
        # For exp(log(s)), a constant 8*eps relative bound is invalid when
        # abs(log(s)) is large. Check the actual standard PLY encoding exactly,
        # not a newly enlarged image/scale tolerance. Covariance still checked.
        encoded=pc_a.get_scaling.log()
        exact=bool(torch.equal(encoded,pc_b._scaling))
        properties['scale_encoding_rule']='Exact float32 raw log(source physical scale); same original exp activation on readback.'
        properties['scale_encoding_exact']=exact
        properties['constant_8eps_scale_test_passed']=scale_relative<=8*eps
        properties['passed']=bool(center_error<=properties['center_allowance_m'] and exact and
            covariance_relative<=64*eps and properties['color_exact'] and properties['opacity_exact'])
    a,b=capture(pc_a,cam_a,torch),capture(pc_b,cam_b,torch)
    # Check the independent full-covariance formula against the actual kernel.
    kernel_checks=[]
    for pc,d in ((pc_a,a),(pc_b,b)):
        visible=d['radii'].detach().cpu().numpy()>0
        cv=covariance64(pc)[visible]
        expected=cv[:,(0,0,0,1,1,2),(0,1,2,1,2,2)]
        actual=d['g']['cov3d'][visible].astype(float)
        err=np.linalg.norm(expected-actual,axis=1)/np.linalg.norm(expected,axis=1)
        kernel_checks.append(float(err.max()))
    properties['kernel_covariance_relative']=kernel_checks
    properties['passed']=properties['passed'] and max(kernel_checks)<=64*eps
    bg=torch.zeros(3,device='cuda')
    assert torch.equal(a['rgb'],render(cam_a,pc_a,pipe,bg)['render'])
    assert torch.equal(b['rgb'],render(cam_b,pc_b,pipe,bg)['render'])
    delta=(a['rgb']-b['rgb']).abs();per_pixel=delta.max(0).values
    strict=dict(mean_abs=float(delta.mean()),max_abs=float(delta.max()))
    strict['passed']=bool(strict['mean_abs']<2e-6 and strict['max_abs']<.005)
    exceptions=[];h,w=per_pixel.shape
    for yy,xx in torch.nonzero(per_pixel>=.005).cpu().numpy():
        y,x=int(yy),int(xx);tile=(y//16)*((w+15)//16)+x//16;idx=y*w+x
        ids=[]
        for d in (a,b):
            lo,hi=d['i']['ranges'][tile]
            ids.append(d['b']['ids'][int(lo):int(hi),0].astype(int))
        same_candidates=np.array_equal(*ids)
        entry=dict(x=x,y=y,delta=float(per_pixel[y,x]),same_ordered_candidates=bool(same_candidates),passed=False)
        if same_candidates:
            order=ids[0];powers=[];alphas=[];uncertainties=[]
            for d in (a,b):
                xy=d['g']['xy'][order].astype(float);q=d['g']['conic'][order].astype(float)
                dx,dy=xy[:,0]-x,xy[:,1]-y
                terms=np.stack((-.5*q[:,0]*dx*dx,-.5*q[:,2]*dy*dy,-q[:,1]*dx*dy),axis=1)
                power=terms.sum(1);alpha=np.minimum(.99,q[:,3]*np.exp(np.minimum(power,0)))
                powers.append(power);alphas.append(alpha)
                # Standard gamma_n bound for <=16 FP32 arithmetic operations.
                uncertainties.append((16*eps/(1-16*eps))*np.abs(terms).sum(1)+2*eps)
            ta,tb=float(a['i']['T'][idx,0]),float(b['i']['T'][idx,0])
            filled=0 if tb==1. and ta<1. else (1 if ta==1. and tb<1. else None)
            possible=[]
            if filled is not None:
                empty=1-filled;ds=(a,b)[filled];color=ds['rgb'][:,y,x].cpu().numpy().astype(float)
                observed_alpha=1-(ta,tb)[filled];tau=1/255
                for j,pid in enumerate(order):
                    pa,pb=powers[0][j],powers[1][j]
                    bound=max(uncertainties[0][j],uncertainties[1][j])+abs(pa-pb)
                    lo=min(alphas[0][j],alphas[1][j])*math.exp(-min(bound,100))
                    hi=max(alphas[0][j],alphas[1][j])*math.exp(min(bound,100))
                    expected=ds['g']['rgb'][pid].astype(float)*observed_alpha
                    error=float(np.max(np.abs(expected-color)))
                    if max(pa,pb)<0 and lo<=tau<=hi and error<2e-6 and abs(observed_alpha-tau)<=max(hi-lo,8*eps):
                        possible.append(dict(gaussian_id=int(pid),alpha_before=alphas[0][j],alpha_after=alphas[1][j],
                            alpha_interval=[float(lo),float(hi)],observed_alpha=observed_alpha,color_explained_error=error,
                            power_bound=float(bound)))
                # Require a unique identified contributor and an exactly empty
                # opposite CUDA pixel. No exception for an unexplained residual.
                entry.update(filled_side=filled,candidates=possible,passed=len(possible)==1)
        exceptions.append(entry)
    stable=per_pixel[per_pixel<.005]
    passed=properties['passed'] and strict['mean_abs']<2e-6 and all(e['passed'] for e in exceptions)
    return dict(schema='native-export-fp32-alpha-boundary-v1',passed=bool(passed),properties=properties,
        strict_image_parity=strict,exception_count=len(exceptions),exceptions=exceptions,
        stable_pixel_max_abs=float(stable.max()) if stable.numel() else None,
        method='Same unmodified CUDA raster. Every pixel failing original0.005maximum requires a unique alpha=1/255 cutoff crossing within parameter-roundoff propagation, identical ordered candidates and exact RGB contribution explanation. Unmasked mean threshold remains2e-6.',
        scope='Read-only reference-camera numerical diagnosis. The arithmetic interval is a roundoff estimate, not a formally verified CUDA expf bound or all-view pixel-equivalence proof. Original strict maximum-image check remains separate.',
        exact_pixel_equivalence=strict['passed'],quality_accepted=False)
