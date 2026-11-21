"""Compare the original fit on native person crops with its full-frame inputs."""
from pathlib import Path
import types

import numpy as np

import fit_video21_native148 as base
import official_smplifyx_entry123 as entry
from update_leafwoman129_state import read


def crop_fit(**kwargs):
    frame = Path(kwargs['result_path']).parent.name
    rows = read(base.NATIVE/'observation-review/inputs-review.json')['rows']
    x0,y0,x1,y1 = next(row['crop_xyxy'] for row in rows if row['frame']==frame)
    image = kwargs['image']
    assert image.shape == (1520,2704,3) and 0 <= x0 < x1 <= 2704 and 0 <= y0 < y1 <= 1520
    K = np.asarray(kwargs['K_integer_centers'])
    origin = np.asarray([x0,y0],np.float32)
    local_K = K.copy()
    local_K[:2,2] -= origin
    target = kwargs['keypoints'].copy()
    target[...,:2] -= origin
    rays = lambda xy,k: np.concatenate((xy,np.ones((*xy.shape[:-1],1))),axis=-1)@np.linalg.inv(k).T
    assert np.max(np.abs(rays(target[...,:2],local_K)-rays(kwargs['keypoints'][...,:2],K))) < 2e-7
    fit, receipt = entry.fit_one(**dict(kwargs,image=image[y0:y1,x0:x1],keypoints=target,K_integer_centers=local_K))
    assert np.array_equal(fit['K_integer_centers'],local_K)
    compute = K.astype(np.float32)
    q = fit['joints_camera']@compute.T
    native = q[...,:2]/q[...,2:]
    assert np.max(np.abs(native-fit['joints_projected_native']-origin)) < .001
    fit.update(K_integer_centers=K,K_compute_float32=compute,joints_projected_native=native)
    receipt.update(input_native_crop_xyxy=[x0,y0,x1,y1],ray_invariant=True,
                   original_data_weight_ratio=1520/(y1-y0),source_resampled=False)
    return fit,receipt


def shard(folder,i):
    assert read(folder/'plan.json')['crop_source_sha256']==entry.sha(base.NATIVE/'observation-review/inputs-review.json')
    adapter = types.SimpleNamespace(**dict(entry.__dict__,fit_one=crop_fit))
    worker = types.SimpleNamespace(**dict(base.original.__dict__,entry=adapter))
    scope = dict(base.__dict__,original=worker,__file__=__file__)
    types.FunctionType(base.fit_shard.__code__,scope)(folder,i)


def main():
    plan = read(base.OUT/'plan.json')
    result = read(base.OUT/'result.json')
    assert result['fitting_complete'] and result['plan_sha256']==entry.sha(base.OUT/'plan.json')
    assert plan['runner_sha256']==entry.sha(base.__file__) and plan['worker_sha256']==entry.sha(base.original.__file__)
    assert all(entry.sha(path)==digest for path,digest in plan['model_sha256'].items())
    assert plan['source_admission_sha256']==entry.sha(base.CONTRACT)
    entry_source,_ = entry.adapted_entry_source(base.original.OFFICIAL)
    assert 'data_weight = 1000 / H' in entry_source
    scope = dict(base.__dict__,fit_shard=shard)
    run = types.FunctionType(base.stage.__code__,scope)
    indices = [plan['source_frames'].index(name) for name in plan['shape_anchors']]
    frozen = dict(plan,runner_sha256=entry.sha(__file__),original_full_fit_sha256=entry.sha(base.OUT/'result.json'),
        crop_source_sha256=entry.sha(base.NATIVE/'observation-review/inputs-review.json'),
        comparison='Same fixed shared shape, full-K rays, visible RTMW observations and original stages. Native integer crop changes the upstream 1000/H data weight; no pixel resampling or model edits.')
    run('crop-weight-control',frozen,indices,dict(vector=result['fixed_shape']))


if __name__=='__main__':
    main()
