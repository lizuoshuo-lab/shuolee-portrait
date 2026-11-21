"""Analytical counterexample for a candidate auxiliary target, not GS training.

Creates a fresh, exactly specified planar Gaussian surface and its own
reference alpha image. No subject weights, fitted XYZ, cameras, or optimizer
are loaded. This tests a desired property, not merely implementation parity.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np

O = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/material-support-gradient-110')
OUT = O / 'target-bias-control.json'
PROTOCOL = O / 'target-bias-protocol.json'
assert not OUT.exists() and not PROTOCOL.exists()

def save(p, v):
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    t.replace(p)

protocol = dict(
    created=datetime.now().astimezone().isoformat(),
    purpose='Does the center-only finite-support target preserve a valid finite-footprint planar representation?',
    representation='Two fresh coplanar Gaussians, finite anisotropic covariance; fixed frontal pinhole camera.',
    constants=dict(depth=2., fx=1000., d=.02, scales=[.03,.03,.001], sigma=.01,
                   opacity=.7, queries_px=[0.,-10.,10.], image_width=121, image_height=81),
    control='Correct plane z=2, query X on that plane, self-render before perturbation is the reference.',
    perturbation='One fixed trial displacement with max center movement .002 world units (5 percent of center separation); all other attributes fixed. This is a sensitivity probe, not iterative training.',
    tests=['one valid off-center material query', 'two exact center material queries sharing contributors', 'duplicate layers around a known plane'],
    hypothesis='A full-rank center penalty may contract valid tangential support even at correct plane geometry.',
    criteria='Any nonzero in-plane center gradient on the valid plane is evidence of tangent attraction; re-rendered alpha error quantifies representation change, not proven irrecoverable training harm.',
    limitations=['Analytical pinhole/Gaussian forward, not the CUDA rasterizer.',
                 'No subject-specific cause or treatment effectiveness is inferred.',
                 'A finite trial step cannot establish convergence or irreversible degradation.',
                 'Rendering alpha is used to check support; this is not a physical optical model.'],
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    training=False, old_model_loaded=False)
save(PROTOCOL, protocol)

fx = 1000.
scales = np.array([.03, .03, .001], np.float64)
metric = np.diag(1. / (scales**2 + .01**2))
opacity = .7
pixels = np.stack(np.meshgrid(np.arange(-60,61), np.arange(-40,41)), -1).astype(np.float64)

def footprint(centers, uv):
    camera_z = centers[:,2]
    means = fx*centers[:,:2]/camera_z[:,None]
    # Same local perspective linearization used by splatting; no min-pixel floor.
    J = np.zeros((len(centers),2,3), np.float64)
    J[:,0,0] = fx/camera_z
    J[:,1,1] = fx/camera_z
    J[:,:,2] = -fx*centers[:,:2]/camera_z[:,None]**2
    cov = (J*scales[None,None,:]**2) @ J.transpose(0,2,1)
    delta = uv[...,None,:]-means
    q = np.einsum('...ji,jik,...jk->...j',delta,np.linalg.inv(cov),delta)
    return opacity*np.exp(-.5*q)

def weights(centers, uv):
    a = footprint(centers,uv)
    # Stable front-to-back ordering; equal-depth plane uses stored order.
    order = np.argsort(centers[:,2],kind='stable')
    ordered = a[...,order]
    transmission = np.concatenate([np.ones(ordered.shape[:-1]+(1,)),
                                    np.cumprod(1.-ordered,axis=-1)[...,:-1]],axis=-1)
    w = ordered*transmission
    restored=np.empty_like(w);restored[...,order]=w
    return restored

def alpha_image(centers):
    return weights(centers,pixels).sum(-1)

def auxiliary(centers, targets, w):
    delta=centers[None,:,:]-targets[:,None,:]
    q=np.einsum('kji,il,kjl->kj',delta,metric,delta)
    normalized=w/w.sum(-1,keepdims=True)
    expected=(normalized*q).sum(-1)
    value=(2.*(np.sqrt(1.+expected)-1.)).mean()
    gradient=(2.*normalized[:,:,None]*(delta@metric.T)/np.sqrt(1.+expected)[:,None,None]).mean(0)
    return float(value),gradient

def probe(name,centers,targets,uv):
    w=weights(centers,uv)
    before,gradient=auxiliary(centers,targets,w)
    old_alpha=alpha_image(centers)
    shift=-gradient*.002/max(np.linalg.norm(gradient,axis=1).max(),1e-30)
    moved=centers+shift
    fixed_after,_=auxiliary(moved,targets,w)
    rerendered_after,_=auxiliary(moved,targets,weights(moved,uv))
    alpha_error=np.abs(alpha_image(moved)-old_alpha)
    return dict(name=name, centers_before=centers.tolist(), targets=targets.tolist(),
                frozen_weights=w.tolist(), center_gradient=gradient.tolist(),
                max_in_plane_gradient=float(np.linalg.norm(gradient[:,:2],axis=1).max()),
                max_normal_gradient=float(np.abs(gradient[:,2]).max()),
                displacement=shift.tolist(),centers_after=moved.tolist(),
                before_loss=before,frozen_weight_after_loss=fixed_after,
                recomputed_weight_after_loss=rerendered_after,
                separation_before=float(np.linalg.norm(centers[0]-centers[1])),
                separation_after=float(np.linalg.norm(moved[0]-moved[1])),
                alpha_mae=float(alpha_error.mean()),alpha_max_error=float(alpha_error.max()),
                reference_generated_before_probe=True)

plane=np.array([[-.02,0.,2.],[.02,0.,2.]])
cases=[probe('valid_plane_single_material_point',plane,np.array([[0.,0.,2.]]),np.array([[0.,0.]])),
       probe('valid_plane_two_material_points',plane,plane.copy(),np.array([[-10.,0.],[10.,0.]])),
       probe('duplicate_layers',np.array([[0.,0.,1.98],[0.,0.,2.02]]),np.array([[0.,0.,2.]]),np.array([[0.,0.]]))]
for c in cases:
    assert np.isfinite(c['before_loss']) and np.isfinite(c['alpha_mae'])
assert cases[0]['max_in_plane_gradient']>0 and cases[0]['max_normal_gradient']==0
assert cases[1]['max_in_plane_gradient']>0 and cases[1]['max_normal_gradient']==0
assert cases[2]['max_normal_gradient']>0 and cases[2]['max_in_plane_gradient']==0
assert all(c['separation_after']<c['separation_before'] for c in cases)
result=dict(created=datetime.now().astimezone().isoformat(),cases=cases,
    conclusion='Candidate center-only penalty pulls valid tangential contributors together as well as correcting duplicate normal layers; gradient parity alone is insufficient training admission.',
    numerical_execution_passed=True,desired_representation_preservation_passed=False,
    training_allowed=False,training_ran=False,subject_artifacts_unchanged=True,
    claims='This is a controlled target-bias counterexample, not proof that this mechanism caused model098 ghosting or that all center penalties are unusable.',
    protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
save(OUT,result)
print(json.dumps(dict(result=str(OUT),desired_representation_preservation_passed=False,
    cases=[{k:c[k] for k in ['name','before_loss','recomputed_weight_after_loss','separation_before','separation_after','alpha_mae']} for c in cases]),ensure_ascii=False))
