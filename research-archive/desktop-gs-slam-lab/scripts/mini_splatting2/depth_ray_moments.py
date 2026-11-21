"""Read-only depth contribution moments for frozen 3D Gaussian models.

These are Gaussian-center depths, NOT ray/surfel intersection depths. The NDC
quantity shares the algebra in 2DGS Appendix A but does not reproduce 2DGS.
No Gaussian parameters, cameras, references, or training losses are changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def analytic_checks():
    # An exact mean does not determine the spread of contributing depths.
    weights=np.array([.4,.4],dtype=np.float64)
    def moments(z):
        a=weights.sum();mu=(weights*z).sum()/a
        variance=(weights*(z-mu)**2).sum()/a
        pairwise=sum(weights[i]*weights[j]*(z[i]-z[j])**2
                     for i in range(len(z)) for j in range(i))
        assert abs(pairwise-a*a*variance)<1e-12
        return dict(alpha=float(a),mean=float(mu),variance=float(variance),pairwise=float(pairwise))
    single=moments(np.array([2.,2.]));layered=moments(np.array([1.,3.]))
    assert single['mean']==layered['mean']==2 and single['variance']==0 and layered['variance']==1
    # NDC is affine in inverse depth, so its variance has a known scale factor.
    z=np.array([1.4,2.7]);near=.2;far=1000.;m=far/(far-near)*(1-near/z)
    a=weights.sum();actual=sum(weights[i]*weights[j]*(m[i]-m[j])**2 for i in range(2) for j in range(i))
    inv=1/z;var=(weights*(inv-(weights*inv).sum()/a)**2).sum()/a
    identity=a*a*(far*near/(far-near))**2*var
    assert abs(actual-identity)<1e-12
    return dict(status='passed',single_layer=single,two_layers=layered,ndc_identity_error=abs(actual-identity),
        meaning='Mathematical non-identifiability only; not evidence that this actual scene is multi-layered')


def render_moments(model,c,mode='mip',divisor=1):
    import torch
    from gsplat import rasterization
    width,height=round(c['width']/divisor),round(c['height']/divisor)
    sx,sy=width/c['width'],height/c['height']
    k=torch.tensor([[c['fx']*sx,0,c.get('cx',c['width']/2)*sx],
                    [0,c['fy']*sy,c.get('cy',c['height']/2)*sy],[0,0,1]],
                    device='cuda',dtype=torch.float32)[None]
    view=c['viewmat'][0];z=(model['means']@view[:3,:3].T+view[:3,3])[:,2]
    positive=z>.2
    assert positive.any()
    scale=z[positive].detach().median()
    q=z/scale;r=scale/z.clamp_min(.2)
    values=torch.stack([q,q.square(),r,r.square(),torch.ones_like(z)],-1)
    rendered,alpha,_=rasterization(model['means'],model['quats'],model['scales'].exp(),
        model['opacities'].sigmoid(),values,c['viewmat'],k,width,height,sh_degree=None,
        packed=False,near_plane=.2,rasterize_mode='antialiased' if mode=='mip' else 'classic',
        eps2d=.1 if mode=='mip' else .3)
    a=alpha[0,...,0].double();s=scale.double()
    e=rendered[0].double()/a.clamp_min(1e-8)[...,None]
    vq=e[...,1]-e[...,0].square();vr=e[...,3]-e[...,2].square()
    valid=(a>.95)&(e[...,0]>0)
    assert torch.isfinite(e).all()
    channel_error=float((rendered[0,...,4].double()-a).abs().max())
    assert channel_error<2e-5,'Constant feature must equal accumulated alpha'
    # Float32 raster accumulation can produce a tiny negative difference.
    bad_q=valid&(vq < -1e-4*e[...,1].abs().clamp_min(1e-8))
    bad_r=valid&(vr < -1e-4*e[...,3].abs().clamp_min(1e-8))
    assert not bad_q.any() and not bad_r.any(),'Moment cancellation exceeds declared tolerance'
    mean=e[...,0]*s;variance=vq.clamp_min(0)*s.square()
    invmean=e[...,2]/s;invvariance=vr.clamp_min(0)/s.square()
    near,far=.2,1000.
    ndc=a.square()*(near*far/(far-near))**2*invvariance
    output=dict(mean_z=mean,variance_z=variance,relative_std=variance.sqrt()/mean.clamp_min(.2),
        mean_inverse_z=invmean,relative_inverse_std=invvariance.sqrt()/invmean.clamp_min(1e-8),
        ndc_pairwise_proxy=ndc,alpha=a)
    diagnostics=dict(alpha_channel_max_error=channel_error,
        negative_q_variance_fraction=float((vq[valid]<0).float().mean()) if valid.any() else 0.,
        negative_r_variance_fraction=float((vr[valid]<0).float().mean()) if valid.any() else 0.,
        normalization_scale=float(s),significantly_negative_pixels=0)
    return output,diagnostics


def gpu_checks():
    import torch
    from local_refinement import render
    torch.set_num_threads(4)
    camera=dict(width=64,height=64,fx=48.,fy=48.,viewmat=torch.eye(4,device='cuda')[None])
    def model(zvalues):
        n=len(zvalues)
        return dict(means=torch.tensor([[0.,0.,z] for z in zvalues],device='cuda'),
            quats=torch.tensor([[1.,0.,0.,0.]]*n,device='cuda'),
            scales=torch.tensor([[.18,.18,.18]]*n,device='cuda').log(),
            opacities=torch.full((n,),torch.logit(torch.tensor(.8)).item(),device='cuda'),
            sh0=torch.zeros(n,1,3,device='cuda'),shN=torch.zeros(n,15,3,device='cuda'))
    records={}
    with torch.no_grad():
        for name,depths in [('single',[2.]),('same_depth',[2.,2.]),('two_layers',[1.5,2.5])]:
            m=model(depths);out,diag=render_moments(m,camera,mode='classic')
            ref,alpha,_=render(m,camera,mode='classic',depth=True)
            good=alpha[...,0]>.1
            err=float((out['mean_z'][good]-ref[...,3][good]).abs().max())
            assert err<2e-5
            pixel=(32,32);a=float(out['alpha'][pixel]);mu=float(out['mean_z'][pixel]);var=float(out['variance_z'][pixel])
            if name!='two_layers':assert var<2e-5 and abs(mu-2)<2e-5
            else:
                # Render each point separately to obtain its own footprint alpha.
                one=[]
                for index in range(2):
                    sliced={key:value[index:index+1] for key,value in m.items()}
                    oi,_=render_moments(sliced,camera,mode='classic');one.append(float(oi['alpha'][pixel]))
                w=np.array([one[0],(1-one[0])*one[1]]);z=np.array(depths)
                expected_mean=float((w*z).sum()/w.sum());expected_var=float((w*(z-expected_mean)**2).sum()/w.sum())
                assert abs(mu-expected_mean)<2e-5 and abs(var-expected_var)<2e-5
                assert var>.05
            records[name]=dict(alpha=a,mean=mu,variance=var,expected_depth_max_error=err,diagnostics=diag)
    return dict(status='passed',cases=records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    target=Path(a.output);assert not target.exists(),'Preserve earlier validation evidence'
    result=dict(cpu=analytic_checks())
    if a.gpu:result['gpu']=gpu_checks()
    result['status']='passed';target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
