"""Matched 021 control: fixed or optimized body-to-camera translation."""
from importlib.util import find_spec
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

import fit_video21_seated149 as seated
import official_smplifyx_entry123 as entry
from update_leafwoman129_state import atomic, read, sha


BASE = seated.base
HERE = Path(__file__).resolve().parent
OUT = BASE.OUT / 'translation-refine152'
BRANCHES = ('fixed', 'free')


def fit_shard(folder, index):
    plan = read(folder / 'plan.json')
    free_translation = folder.name == 'free'
    prior = plan['seated_prior']

    def fit(**kwargs):
        import smplify_seeded_refine124 as refine

        frame = kwargs['result_path'].parent.name
        row = plan['seeds'][frame]
        assert sha(row['output']) == row['sha256']
        with np.load(row['output'], allow_pickle=False) as archive:
            seed = dict(archive)
        assert np.array_equal(kwargs['K_integer_centers'], seed['K_integer_centers'])
        assert np.array_equal(kwargs['keypoints'], seed['observed_native_keypoints'])
        assert np.array_equal(kwargs['shared_betas'], seed['betas'])

        def fixed_source(root):
            code, diff = refine.source(root)
            old = '                body_params.append(camera.translation)'
            new = '                camera.translation.requires_grad_(False)'
            return refine.once(code, old, new), diff + '\n- ' + old.strip() + '\n+ ' + new.strip() + '\n'

        scope = dict(refine.__dict__, source=refine.source if free_translation else fixed_source)
        run = types.FunctionType(refine.fit.__code__, scope)
        inputs = dict(kwargs, vposer2=seated.SeatedDecoder(kwargs['vposer2'], prior))
        replay, replay_receipt = run(saved_seed=seed, replay_only=True, **inputs)
        replay_px = float(np.max(np.abs(replay['joints_projected_native'] - seed['joints_projected_native'])))
        replay_m = float(np.max(np.abs(replay['vertices_camera'] - seed['vertices_camera'])))
        assert replay_px < .001 and replay_m < 1e-6 and replay_receipt['anchor_calls'] == 0, (replay_px, replay_m)
        assert not kwargs['result_path'].exists()
        arrays, receipt = run(saved_seed=seed, replay_only=False, **inputs)
        translation_change = arrays['camera_translation'] - seed['camera_translation']
        assert free_translation or not translation_change.any()
        angles = arrays['body_pose_axis_angle'].reshape(21, 3)[np.asarray(seated.JOINTS) - 1]
        assert np.all(np.abs(angles - prior['center_rad']) <= np.asarray(prior['radius_rad']) + 1e-6)
        assert receipt['original_stage_indices'] == [3, 4] and receipt['anchor_calls'] > 0
        receipt.update(seed_sha256=row['sha256'], seed_replay_max_px=replay_px,
                       seed_replay_max_model_m=replay_m, translation_optimized=free_translation,
                       translation_delta_model_m=translation_change[0].tolist(),
                       initialization='Same-frame 021 seated prior fit; seed used only inside the matched diagnostic.',
                       camera_initialization='Same saved body-to-camera translation and complete conditional K in both branches.')
        return arrays, receipt

    adapter = types.SimpleNamespace(**dict(entry.__dict__, fit_one=fit))
    worker = types.SimpleNamespace(**dict(BASE.original.__dict__, entry=adapter))
    scope = dict(BASE.__dict__, original=worker, __file__=__file__)
    types.FunctionType(BASE.fit_shard.__code__, scope)(folder, index)


def run():
    assert os.name == 'nt' and not OUT.exists()
    if find_spec('omegaconf') is None:
        raise RuntimeError('Run with the existing VPoser environment that includes OmegaConf.')
    source_plan = seated.OUT / 'plan.json'
    source_result = seated.OUT / 'result.json'
    plan, result = read(source_plan), read(source_result)
    assert result['fitting_complete'] and result['plan_sha256'] == sha(source_plan)
    assert plan['baseline_observation_sha256'] == sha(BASE.ARCHIVE)
    assert plan['contacts_sha256'] == sha(BASE.CONTACTS)
    assert all(sha(path) == digest for path, digest in plan['model_sha256'].items())
    ids = plan['shape_anchors']
    assert len(ids) == 8 and not set(ids) & set(plan['retained_holdouts'])
    seeds = {row['frame']: row for row in result['frames'] if row['frame'] in ids}
    assert list(seeds) == ids and all(sha(r['output']) == r['sha256'] for r in seeds.values())
    files = [Path(__file__), source_plan, source_result, Path(seated.__file__), Path(entry.__file__),
             Path(BASE.__file__), Path(BASE.original.__file__), HERE/'smplify_seeded_refine124.py',
             HERE/'control_smplifyx_root_seed123.py', HERE/'smplify_temporal124.py']
    registered = dict(plan, runner_sha256=sha(__file__), seeds=seeds, experiment_frames=ids,
        python_executable=sys.executable, device='cpu',
        source_hashes={str(p): sha(p) for p in files}, branches=list(BRANCHES),
        single_difference='Whether the original LBFGS stages 3/4 can update body-to-camera translation.',
        fixed='Same-frame seated seed, seated hip/knee decoder, original stage3/4 weights, existing rotation anchor30, maxiters30, K, beta, pixels and 118 observations.',
        criteria='Compare 8 matched views, supported body/feet max divided by image-box height, face RMS and actual native and synthetic-side anatomy; no frame-wise branch mixing, and no automatic Gaussian launch.')
    atomic(OUT/'plan.json', registered)
    atomic(OUT/'launch.json', dict(pid=os.getpid(), command=[sys.executable, *sys.argv], stage='matched_refinement', branches=list(BRANCHES)))
    scope = dict(BASE.__dict__, fit_shard=fit_shard)
    stage = types.FunctionType(BASE.stage.__code__, scope)
    indices = [plan['source_frames'].index(f) for f in ids]
    started = time.monotonic()
    try:
        for branch in BRANCHES:
            stage(OUT.name+'/'+branch, registered, indices, dict(vector=result['fixed_shape']))
        assert all(sha(path) == digest for path, digest in registered['source_hashes'].items())
        atomic(OUT/'result.json', dict(completed=True, frames=ids, branches=list(BRANCHES),
            branch_result_sha256={b:sha(OUT/b/'result.json') for b in BRANCHES},
            plan_sha256=sha(OUT/'plan.json'), seconds=time.monotonic()-started,
            gaussians_started=False, quality_accepted=False))
    except BaseException as error:
        atomic(OUT/'failure.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    run()
