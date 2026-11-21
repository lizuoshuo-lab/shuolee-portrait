"""Run one coherent 021 roster with the validated original root-seed interface."""
import argparse
from datetime import datetime, timezone
from importlib.util import find_spec
import inspect
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from scipy.spatial.transform import Rotation

from extract_video21_shoe_depth162 import replace
import probe_video21_world_root172 as probe
import review_video21_fit148 as reviewer
from review_full_sequence123 import relocate
from update_leafwoman129_state import atomic, read, sha


BASE = probe.BASE
OUT = BASE.OUT / 'world-root-coherent173'
ARCHIVE = OUT / 'fit-observations.npz'
CONTACTS = OUT / 'contacts.json'


def initial_roots(frames, sources, cameras):
    known = {}
    for name,row in sources.items():
        assert sha(row['output']) == row['sha256']
        with np.load(row['output'],allow_pickle=False) as fit:
            known[name] = cameras[name].T @ Rotation.from_rotvec(fit['global_orient'][0]).as_matrix()
    order = list(known)
    seeds = {}
    for f in frames:
        earlier = [name for name in order if name < f]
        later = [name for name in order if name > f]
        neighbours = earlier[-2:] + later[:2]
        if len(neighbours) < 4:
            neighbours = sorted((name for name in order if name != f),
                key=lambda name:(abs(int(name[-6:])-int(f[-6:])),name))[:4]
        assert len(neighbours) == 4 and f not in neighbours
        world = np.asarray([known[name] for name in neighbours])
        relative = world[:,None].transpose(0,1,3,2) @ world[None]
        angles = np.rad2deg(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).reshape(4,4)
        medoid = int(np.argmin(angles.sum(1)))
        assert np.count_nonzero(angles[medoid] <= 30.) >= 3, f
        local = Rotation.from_matrix(cameras[f] @ world[medoid]).as_rotvec().astype(np.float32)
        seeds[f] = dict(neighbours=neighbours,medoid=neighbours[medoid],
            neighbour_angles_deg=angles[medoid].tolist(),body_fit_initial_root=local[None].tolist())
    return seeds


def prepare():
    assert not OUT.exists()
    p = read(probe.seated.OUT/'plan.json')
    old_fit = read(probe.seated.OUT/'result.json')
    assert old_fit['fitting_complete'] and old_fit['plan_sha256'] == sha(probe.seated.OUT/'plan.json')
    assert sha(BASE.ARCHIVE) == p['baseline_observation_sha256']
    trial = read(probe.OUT/'review/visual-decision.json')
    evidence = probe.AUDIT/'result.json'
    coverage = read(evidence)
    assert trial['actual_native_and_side_review'] and trial['records_sha256'][str(evidence)] == sha(evidence)
    assert coverage['new_3fps_bins'] == list(range(1,17)) and coverage['new_3fps_camera_frames'] == 54
    frames = [f for interval in p['source_view_bins'] for f in interval if f not in p['retained_holdouts']
              and coverage['per_frame'][f]['new_pass']]
    assert len(frames) == 54 and set(p['source_frames']) <= set(frames)
    assert not set(frames) & set(p['retained_holdouts'])
    camera_path = probe.CAMERA/'cameras-diagnostic.json'
    assert sha(camera_path) == coverage['source_pose_sha256']
    cameras = {r['frame']:np.asarray(r['R_w2c']) for r in read(camera_path)}
    sources = {r['frame']:r for r in old_fit['frames']}
    assert list(sources) == p['source_frames']
    init = initial_roots(frames,sources,cameras)
    trial_plan = read(probe.OUT/'plan.json')
    assert all(init[f]['body_fit_initial_root'] == r['body_fit_initial_root']
               and init[f]['neighbours'] == r['neighbours'] for f,r in trial_plan['initial_roots'].items())

    native = read(BASE.NATIVE/'result.json')
    data = relocate(native['data_root'])/'data'
    rows = read(data/'inputs.json')['records']
    obs = data/'observations.npz'
    assert sha(obs) == native['observations_sha256'] == p['source_obs_sha256']
    assert sha(data/'inputs.json') == native['observation_manifest_sha256']
    assert sha(BASE.CONTRACT) == p['source_admission_sha256']
    names = [r['frame'] for r in rows]
    assert len(names) == len(set(names)) == 59
    K = np.asarray(p['K_integer_centers'])
    slots = [names.index(f) for f in frames]
    with np.load(obs,allow_pickle=False) as z:
        assert z['frame_ids'].tolist() == names
        points, scores, boxes, flip = (z[k] for k in ('keypoints','scores','boxes','flip_disagreement'))
    for item in read(BASE.CONTRACT)['source_visibility_exclusions']:
        start,stop = item['sample_index_inclusive']
        scores[start-1:stop,item['coco133_indices']] = 0
    with np.load(BASE.ARCHIVE,allow_pickle=False) as z:
        q90, reference = z['confidence_q90'],z['observed_native_keypoints']
        assert z['frame_ids'].tolist() == list(sources) and np.array_equal(K,z['K_integer_centers'])
    # Reuse the old population normalizer so adding frames cannot change any
    # previously fitted coordinate or confidence. Assert parity on all 32.
    height = boxes[slots,3].astype(float)-boxes[slots,1].astype(float)
    confidence = np.clip(scores[slots].astype(float)/q90,0.,1.) * np.exp(
        -.25*(flip[slots].astype(float)/(.012*height[:,None]))**2)
    inputs = np.zeros((len(frames),118,3),np.float32)
    inputs[:,BASE.TARGET_IDX,:2] = points[slots][:,BASE.SOURCE_IDX]
    inputs[:,BASE.TARGET_IDX,2] = confidence[:,BASE.SOURCE_IDX]
    assert np.array_equal(inputs[[frames.index(f) for f in sources]],reference)
    contacts = []
    for i in slots:
        row = rows[i]
        assert row['wh'] == [2704,1520] and np.array_equal(row['K_integer_centers'],K)
        rgb,mask = relocate(row['image']),relocate(row['mask'])
        assert sha(rgb) == row['source_sha256'] and sha(mask) == row['mask_sha256']
        contacts.append(dict(row,paths={'body':dict(path=str(rgb),sha256=row['source_sha256'],
            native_integer_crop=True,resized=False,crop_original_xyxy=[0,0,2704,1520])}))
    paths = [Path(__file__),Path(probe.__file__),Path(probe.seated.__file__),Path(probe.root_hook.__file__),
             Path(BASE.__file__),Path(BASE.original.__file__),Path(probe.entry.__file__),
             BASE.ARCHIVE,BASE.CONTRACT,obs,data/'inputs.json',probe.seated.OUT/'plan.json',probe.seated.OUT/'result.json',
             probe.OUT/'plan.json',probe.OUT/'review/visual-decision.json',evidence,camera_path]
    OUT.mkdir()
    BASE.save_npz(ARCHIVE,frame_ids=np.asarray(frames),observed_native_keypoints=inputs,
        K_integer_centers=K,confidence_q90=q90)
    atomic(CONTACTS,dict(records=contacts))
    checks = {str(path):sha(path) for path in [*paths,ARCHIVE,CONTACTS]}
    plan = dict(p,updated=datetime.now(timezone.utc).isoformat(),source_frames=frames,
        native_indices=[r['source_frame_index'] for r in contacts],initial_roots=init,baseline_rows=sources,
        source_hashes=checks,runner_sha256=sha(__file__),baseline_observation_sha256=sha(ARCHIVE),contacts_sha256=sha(CONTACTS),
        candidate_policy='Every numerically supported original 3fps source frame; 54 of 55, all 16 frozen source time bins. Four holdouts protected; no alternate fits mixed.',
        observation_policy='Exact old 148 source visibility and q90 confidence normalization; old 32 observations bit-exact. No new visibility exclusion, loss or threshold change.',
        root_policy='172 original root hook; the two previous and two following frozen camera-aligned roots when available, otherwise nearest four. Three within 30 degrees. Target excluded.',
        candidate_poses_started=False,gaussian_started=False,quality_accepted=False)
    atomic(OUT/'plan.json',plan)
    print('COHERENT_021_PREPARED',len(frames),len(sources),flush=True)


def fit_shard(folder,shard):
    code = inspect.getsource(probe.fit_shard)
    code = replace(code,"        baseline = p['baseline_rows'][frame]\n        assert sha(baseline['output']) == baseline['sha256']",
        "        baseline = p['baseline_rows'].get(frame)\n        if baseline:\n            assert sha(baseline['output']) == baseline['sha256']")
    code = replace(code,"        with np.load(baseline['output'],allow_pickle=False) as prior:\n            assert np.array_equal(prior['camera_translation'],trace['camera_translation'])\n            assert np.array_equal(prior['betas'],arrays['betas'])\n            assert np.array_equal(prior['K_integer_centers'],arrays['K_integer_centers'])",
        "        if baseline:\n            with np.load(baseline['output'],allow_pickle=False) as prior:\n                assert np.array_equal(prior['camera_translation'],trace['camera_translation'])\n                assert np.array_equal(prior['betas'],arrays['betas'])\n                assert np.array_equal(prior['K_integer_centers'],arrays['K_integer_centers'])")
    code = replace(code,'original_camera_initialization_exact=True,','original_camera_initialization_exact=True if baseline else None,')
    native = types.SimpleNamespace(**dict(BASE.__dict__,ARCHIVE=ARCHIVE,CONTACTS=CONTACTS))
    seated = types.SimpleNamespace(**dict(probe.seated.__dict__,base=native,__file__=__file__))
    scope = dict(probe.__dict__,seated=seated)
    exec(compile(code,'<coherent-root-roster-interface>', 'exec'),scope)
    scope['fit_shard'](folder,shard)


def run():
    assert os.name=='nt' and not (OUT/'launch.json').exists()
    if find_spec('omegaconf') is None:
        raise RuntimeError('Run in the existing VPoser environment.')
    plan = read(OUT/'plan.json')
    checks = plan['source_hashes'] | plan['model_sha256']
    assert all(sha(path)==digest for path,digest in checks.items())
    shape = read(probe.seated.OUT/'result.json')['fixed_shape']
    frames = plan['source_frames']
    assert len(frames)==54 and not set(frames)&set(plan['retained_holdouts'])
    atomic(OUT/'launch.json',dict(pid=os.getpid(),command=[sys.executable,*sys.argv],frames=frames))
    scope = dict(BASE.__dict__,OUT=OUT,fit_shard=fit_shard)
    stage = types.FunctionType(BASE.stage.__code__,scope)
    try:
        rows = stage('fit',plan,list(range(len(frames))),dict(vector=shape))
        assert all(r['world_root_seed_enabled'] and r['five_stages'] for r in rows)
        assert sum(r['original_camera_initialization_exact'] is True for r in rows)==32
        assert all(sha(path)==digest for path,digest in checks.items())
        atomic(OUT/'result.json',dict(fitting_complete=True,frames=rows,fixed_shape=shape,
            seconds=read(OUT/'fit/result.json')['seconds'],plan_sha256=sha(OUT/'plan.json'),
            fit_result_sha256=sha(OUT/'fit/result.json'),gaussian_started=False,quality_accepted=False))
    except BaseException as exc:
        atomic(OUT/'failure.json',dict(error=repr(exc)))
        raise


def review():
    scope = dict(reviewer.__dict__,OUT=OUT,CONTACTS=CONTACTS)
    types.FunctionType(reviewer.review.__code__,scope)(False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run','review'))
    {'prepare':prepare,'run':run,'review':review}[parser.parse_args().mode]()
