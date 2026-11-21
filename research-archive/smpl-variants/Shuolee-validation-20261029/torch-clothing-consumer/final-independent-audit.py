"""Recheck the four repaired boundaries against independent saved fixtures."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock
from contextlib import ExitStack

import numpy as np
import torch

ROOT = Path('/mnt/d/Projects/Shuolee-torch-clothing-consumer-123')
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait.free_clothing_transport import FreeClothingTransport
from shuolee_portrait.free_clothing_torch import FreeClothingTorch, normalized_quaternion_matrix

torch.set_num_threads(1)
axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
S=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
Q=np.eye(3)+np.sin(.37)*S+(1-np.cos(.37))*S@S
V=np.array([[0.,0.,3.],[1.,0.,3.],[.3,.8,3.]])
source=FreeClothingTransport(frame_ids=('reference','posed'),reference_frame_id='reference',
    gaussian_ids=('gaussian',),vertices_world=np.stack([V,1.7*V@Q.T+[.2,.3,.4]]),
    faces=np.array([[0,1,2]]),motion_support_triangle_ids=np.array([0]),
    frame_source_sha256={'reference':'a'*64,'posed':'b'*64},translation_unit='meters')
CONTRACT=dict(translation_unit='meters',meters_per_camera_unit=1.,basis='synthetic_known',evidence='Independent synthetic fixture only.')
CAMERAS={f:np.eye(4) for f in source.frame_ids}


def prepare(dtype=torch.float64,cameras=CAMERAS):
    return FreeClothingTorch.from_cpu(source,camera_length_contract=CONTRACT,frame_source_w2c=cameras,device='cpu',dtype=dtype)


def params(dtype=torch.float64,amplitude=1.,grad=False):
    return (torch.tensor([[.2,.3,3.35]],dtype=dtype,device='cpu',requires_grad=grad),
            (torch.tensor([[1.,2.,3.,4.]],dtype=dtype,device='cpu')*amplitude).requires_grad_(grad),
            torch.tensor(np.log([[.13,.08,.04]]),dtype=dtype,device='cpu',requires_grad=grad))


def run(c,values,frame='reference',cov=True):
    return c(frame,*values,gaussian_ids=('gaussian',),return_covariance=cov)


def independent_covariance(q,logs):
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1);o=torch.zeros_like(w)
    skew=torch.stack([o,-z,y,z,o,-x,-y,x,o],-1).reshape(-1,3,3)
    R=torch.eye(3,dtype=q.dtype,device='cpu')+2*w[:,None,None]*skew+2*skew@skew
    factor=R*logs.exp()[:,None,:]
    return factor@factor.transpose(-1,-2)


fixed_flags=[]
for name in ('_rotations','_quaternions','_scales','_centroids','_residuals','_camera_centers'):
    c=prepare();table=getattr(c,name);before=table._version;table.requires_grad_(True)
    try:
        run(c,params(),cov=False);raise AssertionError('Trainable table was accepted: '+name)
    except ValueError as error:
        assert 'made trainable' in str(error)
        fixed_flags.append(dict(field=name,version_unchanged=table._version==before,rejected=True,message=str(error)))

camera_checks=[]
for n in (1,2):
    mapping=dict(CAMERAS);mapping['posed']=np.repeat(np.eye(4)[None],n,axis=0)
    try:
        prepare(cameras=mapping);raise AssertionError('Batched E accepted')
    except ValueError as error:
        assert 'exactly one 4x4' in str(error)
        camera_checks.append(dict(shape=[n,4,4],rejected=True,message=str(error)))

with torch.inference_mode():
    c=prepare();assert torch.is_inference_mode_enabled()
assert not torch.is_inference_mode_enabled()
inference_buffers={name:dict(is_inference=torch.is_inference(getattr(c,name)),version_tracked=getattr(c,name)._version==version,requires_grad=getattr(c,name).requires_grad) for name,_,version in c._buffer_guards}
assert all(not x['is_inference'] and x['version_tracked'] and not x['requires_grad'] for x in inference_buffers.values())
values=params(grad=True);out=run(c,values,frame='posed')
weight=torch.tensor([[.3,.12,-.08],[.12,.5,.04],[-.08,.04,.9]],dtype=torch.float64,device='cpu')
gradient=torch.autograd.grad(out['centers'].square().sum()+(out['covariance']*weight).sum(),values)
inference_gradient=[dict(finite=bool(torch.isfinite(g).all()),max_abs=float(g.abs().max())) for g in gradient]
assert all(g['finite'] and g['max_abs']>0 for g in inference_gradient)

amplitudes=[]
for dtype,scales in ((torch.float32,(1e-23,1e-22,1e20,1e30)),(torch.float64,(1e-162,1e-200,1e200))):
    c=prepare(dtype)
    for amplitude in scales:
        for frame in source.frame_ids:
            x,q,logs=params(dtype,amplitude,True)
            out=run(c,(x,q,logs),frame)
            oracle=(q.detach()/amplitude).requires_grad_(True)
            expected=independent_covariance(oracle,logs)
            if frame=='posed':
                rotation=torch.tensor(Q,dtype=dtype,device='cpu')
                expected=1.7**2*rotation@expected@rotation.T
            w=weight.to(dtype=dtype)
            g=torch.autograd.grad((out['covariance']*w).sum(),q,retain_graph=True)[0]
            expected_g=torch.autograd.grad((expected*w).sum(),oracle)[0]
            R=normalized_quaternion_matrix(out['quats']).double()
            covariance_error=float((out['covariance']-expected).detach().abs().max())
            gradient_error=float((g*amplitude-expected_g).abs().max())
            norm_error=float((torch.linalg.vector_norm(out['quats'].double(),dim=-1)-1).abs().max())
            tolerance=2e-6 if dtype==torch.float32 else 2e-13
            assert covariance_error < tolerance*.02 and gradient_error < tolerance*.02
            assert bool(torch.isfinite(g).all()) and norm_error < 8*torch.finfo(dtype).eps
            amplitudes.append(dict(dtype=str(dtype),amplitude=amplitude,frame=frame,quaternion_norm_error=norm_error,
                rotation_orthogonality_error=float((R.transpose(-1,-2)@R-torch.eye(3,dtype=torch.float64,device='cpu')).abs().max()),
                covariance_max_abs_error=covariance_error,amplitude_scaled_quaternion_gradient_error=gradient_error,gradient_finite=True))

rejected_inverse=[]
for dtype,amplitude in ((torch.float32,1e-40),(torch.float64,1e-320)):
    try:
        run(prepare(dtype),params(dtype,amplitude));raise AssertionError('Unrepresentable reciprocal accepted')
    except RuntimeError as error:
        assert 'finite parameter gradients' in str(error)
        rejected_inverse.append(dict(dtype=str(dtype),amplitude=amplitude,rejected=True))

c=prepare(torch.float32);values=params(torch.float32,1e-23,True)
baseline=run(c,values,cov=False)
def forbidden(*args,**kwargs):raise AssertionError('CPU/NumPy/factory/decomposition in hotpath')
with ExitStack() as stack:
    for name in ('cpu','numpy','detach','to'):stack.enter_context(mock.patch.object(torch.Tensor,name,forbidden))
    for owner,name in ((torch,'tensor'),(np,'asarray'),(torch.linalg,'eigh'),(torch.linalg,'eigvalsh'),(torch.linalg,'cholesky_ex')):
        stack.enter_context(mock.patch.object(owner,name,forbidden))
    actual=run(c,values,cov=False)
assert torch.equal(actual['quats'],baseline['quats'])
with torch.device('meta'),torch.autocast(device_type='cpu',dtype=torch.bfloat16):
    ambient=run(c,values,cov=False)
    assert torch.is_autocast_enabled('cpu') and torch.empty(0).device.type=='meta'
for key in ('centers','quats','scales','reference_view_directions'):
    assert ambient[key].dtype==torch.float32 and ambient[key].device.type=='cpu'
    assert torch.equal(ambient[key],baseline[key])

report=dict(schema='free-clothing-torch-final-independent-audit/1',outcome='passed',
    source_sha256=hashlib.sha256((ROOT/'src/shuolee_portrait/free_clothing_torch.py').read_bytes()).hexdigest(),
    test_sha256=hashlib.sha256((ROOT/'tests/test_free_clothing_torch.py').read_bytes()).hexdigest(),
    fixed_table_flag_checks=fixed_flags,single_camera_shape_checks=camera_checks,
    inference_mode_buffers=inference_buffers,inference_prepared_real_parameter_gradients=inference_gradient,
    original_tiny_quaternion_fixture_replays=amplitudes,unrepresentable_inverse_rejected=rejected_inverse,
    hotpath_no_cpu_numpy_factory_or_decomposition=True,meta_autocast_preserve_cpu_float32=True,
    original_19_test_suite_run=False,model_forwards=0,GPU_jobs=0,training_steps=0,
    physical_camera_accepted=False,training_admitted=False,CUDA_parity_validated=False)
path=Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer/final-independent-audit.json')
with path.open('x',encoding='utf-8') as stream:
    json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(dict(path=str(path),outcome=report['outcome'],source_sha256=report['source_sha256'],
    amplitude_cases=len(amplitudes),max_covariance_error=max(x['covariance_max_abs_error'] for x in amplitudes),
    max_scaled_quaternion_gradient_error=max(x['amplitude_scaled_quaternion_gradient_error'] for x in amplitudes)),indent=2))
