"""Matched eight-frame check of the original shoulder-to-hip depth seed."""
import inspect
import types

import numpy as np

import fit_video21_seated149 as seated
import official_smplifyx_entry123 as entry
from update_leafwoman129_state import read, sha


NAME = 'visible-shin-depth150'
OUT = seated.base.OUT/NAME


def fit_with_visible_depth(**kwargs):
    observed = kwargs['keypoints'][0]
    edges = [edge for edge in ([10,11],[13,14]) if (observed[edge,2]>0).all()]
    assert edges
    source = inspect.getsource(entry.fit_one)
    old = 'body_tri_idxs=[[5,12],[2,9]], **config)'
    assert source.count(old)==1
    scope = dict(entry.__dict__,_depth_edges=edges)
    exec(compile(source.replace(old,'body_tri_idxs=_depth_edges, **config)'),'<original fit with visible depth edges>','exec'),scope)
    output,receipt = scope['fit_one'](**kwargs)
    receipt.update(original_depth_edges=[[5,12],[2,9]],candidate_depth_edges=edges,
                   only_depth_initialization_edges_changed=True,original_camera_init_residual_ids=[9,12,2,5])
    return output,receipt


def shard(folder,i):
    plan = read(folder/'plan.json')
    assert plan['seated_fit_sha256']==sha(seated.OUT/'result.json')
    assert plan['seated_runner_sha256']==sha(seated.__file__)
    module = types.SimpleNamespace(**dict(entry.__dict__,fit_one=fit_with_visible_depth))
    scope = dict(seated.__dict__,entry=module,__file__=__file__)
    types.FunctionType(seated.shard.__code__,scope)(folder,i)


def run():
    p = read(seated.OUT/'plan.json')
    result = read(seated.OUT/'result.json')
    assert result['fitting_complete'] and result['plan_sha256']==sha(seated.OUT/'plan.json')
    assert p['runner_sha256']==sha(seated.__file__) and p['fit_entry_sha256']==sha(entry.__file__)
    assert p['source_admission_sha256']==sha(seated.base.CONTRACT)
    assert all(sha(path)==digest for path,digest in p['model_sha256'].items())
    indices = [p['source_frames'].index(frame) for frame in p['shape_anchors']]
    assert len(indices)==8 and not set(p['shape_anchors'])&set(p['retained_holdouts'])
    # Every retained shin comes from the previously frozen source visibility, not a small fitted residual.
    with np.load(seated.base.ARCHIVE,allow_pickle=False) as z:
        targets = z['observed_native_keypoints'][indices]
    assert np.logical_or((targets[:,[10,11],2]>0).all(1),(targets[:,[13,14],2]>0).all(1)).all()
    plan = dict(p,runner_sha256=sha(__file__),seated_runner_sha256=sha(seated.__file__),
        seated_fit_sha256=sha(seated.OUT/'result.json'),
        depth_hypothesis='Substitute the original guess_init shoulder-to-hip edges with the source-visible knee-to-ankle edges. Keep the camera residual, all original stages, seated prior, fixed shape, and conditional native K.',
        depth_seed_is_not_physical_depth=True,all_4_holdouts_unchanged=True)
    scope = dict(seated.base.__dict__,fit_shard=shard)
    types.FunctionType(seated.base.stage.__code__,scope)(NAME,plan,indices,dict(vector=result['fixed_shape']))


if __name__=='__main__':
    run()
