"""One side-derived seated prior around the original 021 SMPLify-X fit."""
import argparse
from pathlib import Path
import types

import numpy as np

import fit_video21_native148 as base
import official_smplifyx_entry123 as entry
import review_video21_fit148 as reviewer
from update_leafwoman129_state import atomic, read


NAME = 'seated-prior149'
OUT = base.OUT/NAME
ANCHORS = {
    'left': dict(samples=(4,12), landmarks=(13,14), joints=(1,4)),
    'right': dict(samples=(53,59), landmarks=(10,11), joints=(2,5)),
}
JOINTS = (1,2,4,5)


def side_prior(plan, result):
    contacts = {r['frame']:r for r in read(base.CONTACTS)['records']}
    center = np.zeros((4,3),np.float32)
    radius = np.zeros_like(center)
    evidence = {}
    for side,rule in ANCHORS.items():
        poses,rows = [],[]
        for row in result['frames']:
            frame = row['frame']
            number = int(frame[-6:])
            if not rule['samples'][0] <= number <= rule['samples'][1]:
                continue
            assert frame not in plan['retained_holdouts'] and entry.sha(row['output'])==row['sha256']
            with np.load(row['output'],allow_pickle=False) as z:
                assert np.array_equal(z['betas'],np.asarray(result['fixed_shape'],np.float32))
                j = z['joints_camera'][0]
                obs = z['observed_native_keypoints'][0,list(rule['landmarks'])]
                uv = z['joints_projected_native'][0,list(rule['landmarks'])]
                angles = z['body_pose_axis_angle'].reshape(21,3)[np.asarray(rule['joints'])-1]
            e = np.linalg.norm(obs[:,:2]-uv,axis=1)
            shank = float(np.linalg.norm(obs[1,:2]-obs[0,:2]))
            box = contacts[frame]['initial_box_xyxy']
            knees = reviewer.knee_flexion(j)
            eligible = (obs[:,2]>0).all() and shank>0 and e[0]<.5*shank and e[1]<=.03*(box[3]-box[1]) and all(60<=k<=120 for k in knees)
            rows.append(dict(frame=frame,knee_ankle_error_px=e.tolist(),shank_px=shank,knees_deg=knees,
                             prior_support=bool(eligible),source_fit_sha256=row['sha256']))
            if eligible:
                poses.append(angles)
        assert len(poses)>=3,(side,rows)
        slots = [JOINTS.index(j) for j in rule['joints']]
        values = np.stack(poses)
        mid = np.median(values,axis=0)
        center[slots] = mid
        radius[slots] = np.max(np.abs(values-mid),axis=0)+np.deg2rad(12.)
        evidence[side] = rows
    assert np.isfinite(center).all() and np.isfinite(radius).all() and (radius>0).all()
    return dict(joints=list(JOINTS),center_rad=center.tolist(),radius_rad=radius.tolist(),side_frames=evidence,
        source_visibility='Early left-side and late right-side profile windows show seated thigh/shank bends. Near-side estimates only; no one seed frame or heldout is used.',
        support_policy='Both 3D knees in broad seated range60..120, near observed ankle within prior3% height, near knee within half observed shank. These tests supply a prior, not a training quality gate.',
        parameterization='Componentwise side-fit median plus tanh of VPoser change from latent zero, bounded by observed max deviation plus 12 degrees. Hips/knees only; no physically measured angle claim.')


class SeatedDecoder:
    def __init__(self, original, prior):
        import torch
        self.original = original
        self.indices = torch.as_tensor(prior['joints'])-1
        self.center = torch.as_tensor(prior['center_rad'],dtype=torch.float32)
        self.radius = torch.as_tensor(prior['radius_rad'],dtype=torch.float32)
        with torch.no_grad():
            zero = self.original.decode(torch.zeros((1,32)))['pose_body'].reshape(1,21,3)
            self.zero = zero[:,self.indices]
        latent = torch.zeros((1,32),requires_grad=True)
        reference = self.original.decode(latent)['pose_body'].reshape(1,21,3)
        check = self.decode(latent)['pose_body']
        other = sorted(set(range(21))-set(self.indices.tolist()))
        assert torch.equal(check[:,other],reference[:,other])
        assert torch.equal(check[:,self.indices][0],self.center)
        derivative, = torch.autograd.grad(check[:,self.indices].square().sum(),latent)
        assert torch.isfinite(derivative).all() and derivative.abs().max()>0

    def decode(self, latent):
        pose = self.original.decode(latent)['pose_body'].reshape(latent.shape[0],21,3)
        delta = pose[:,self.indices]-self.zero
        seated = self.center+self.radius*(delta/self.radius).tanh()
        return {'pose_body':pose.index_copy(1,self.indices,seated)}


def shard(folder,i):
    prior = read(folder/'plan.json')['seated_prior']

    def fit(**kwargs):
        fit, receipt = entry.fit_one(**dict(kwargs,vposer2=SeatedDecoder(kwargs['vposer2'],prior)))
        angles = fit['body_pose_axis_angle'].reshape(21,3)[np.asarray(JOINTS)-1]
        assert np.all(np.abs(angles-prior['center_rad'])<=np.asarray(prior['radius_rad'])+1e-6)
        receipt.update(seated_prior_applied=True,original_fit_stages_unchanged=True)
        return fit,receipt

    adapter = types.SimpleNamespace(**dict(entry.__dict__,fit_one=fit))
    worker = types.SimpleNamespace(**dict(base.original.__dict__,entry=adapter))
    scope = dict(base.__dict__,original=worker,__file__=__file__)
    types.FunctionType(base.fit_shard.__code__,scope)(folder,i)


def run():
    plan = read(base.OUT/'plan.json')
    result = read(base.OUT/'result.json')
    assert not OUT.exists() and result['fitting_complete']
    assert result['plan_sha256']==entry.sha(base.OUT/'plan.json')
    assert plan['worker_sha256']==entry.sha(base.original.__file__) and plan['fit_entry_sha256']==entry.sha(entry.__file__)
    assert plan['runner_sha256']==entry.sha(base.__file__) and plan['source_admission_sha256']==entry.sha(base.CONTRACT)
    assert all(entry.sha(path)==digest for path,digest in plan['model_sha256'].items())
    prior = side_prior(plan,result)
    frozen = dict(plan,runner_sha256=entry.sha(__file__),original_full_fit_sha256=entry.sha(base.OUT/'result.json'),
        seated_prior=prior,comparison='All32 from the same observations, native raster, fixed shape and conditional K. A side-derived lower-leg prior changes only VPoser hip and knee outputs, with the original 5 stages and optimizer.')
    scope = dict(base.__dict__,fit_shard=shard)
    stage = types.FunctionType(base.stage.__code__,scope)
    stage(NAME,frozen,list(range(len(plan['source_frames']))),dict(vector=result['fixed_shape']))
    result_path = OUT/'result.json'
    atomic(result_path,dict(read(result_path),fitting_complete=True,reference_fit_sha256=entry.sha(base.OUT/'result.json'),
        fixed_shape=result['fixed_shape'],gaussian_started=False,quality_accepted=False))


def review():
    scope = dict(reviewer.__dict__,OUT=OUT)
    types.FunctionType(reviewer.review.__code__,scope)(False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','review'))
    (run if parser.parse_args().mode=='run' else review)()
