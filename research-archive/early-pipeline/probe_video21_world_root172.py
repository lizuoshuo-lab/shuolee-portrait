"""One 021 root-initialization control using the existing five-stage fitter."""
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

from audit_video21_multiview_holdout171 import CAMERA, OUT as AUDIT
import control_smplifyx_root_seed123 as root_hook
import fit_video21_seated149 as seated
import official_smplifyx_entry123 as entry
from update_leafwoman129_state import atomic, read, sha


BASE = seated.base
OUT = BASE.OUT / 'world-root-init172'


def seed_rotations(frames, roster, sources, cameras):
    result = {}
    for frame in frames:
        i = roster.index(frame)
        neighbours = roster[i-2:i] + roster[i+1:i+3]
        assert len(neighbours) == 4 and frame not in neighbours
        roots = []
        for f in neighbours:
            row = sources[f]
            assert sha(row['output']) == row['sha256']
            with np.load(row['output'], allow_pickle=False) as fit:
                local = Rotation.from_rotvec(fit['global_orient'][0]).as_matrix()
            roots.append(cameras[f].T @ local)
        roots = np.asarray(roots)
        pairs = roots[:,None].transpose(0,1,3,2) @ roots[None]
        d = np.rad2deg(Rotation.from_matrix(pairs.reshape(-1,3,3)).magnitude()).reshape(4,4)
        center = int(np.argmin(d.sum(1)))
        assert np.count_nonzero(d[center] <= 30.) >= 3
        local = Rotation.from_matrix(cameras[frame] @ roots[center]).as_rotvec().astype(np.float32)
        result[frame] = dict(neighbours=neighbours, medoid=neighbours[center],
            neighbour_angles_deg=d[center].tolist(), world_root=roots[center].tolist(),
            body_fit_initial_root=local[None].tolist())
    return result


def fit_shard(folder, shard):
    p = read(folder / 'plan.json')

    def fit(**kwargs):
        frame = kwargs['result_path'].parent.name
        start = p['initial_roots'][frame]
        root = np.asarray(start['body_fit_initial_root'],np.float32)
        baseline = p['baseline_rows'][frame]
        assert sha(baseline['output']) == baseline['sha256']

        def config(path):
            original = entry.load_official_config(path)
            original.update(_seed_pose_embedding=np.zeros((1,32),np.float32),
                            _seed_global_orient=root, _seed_root_enabled=True)
            return original

        entry._root_seed_original_source = entry.adapted_entry_source
        scope = dict(entry.__dict__, adapted_entry_source=root_hook.adapted_source,
                     load_official_config=config)
        exec(compile(inspect.getsource(entry.fit_one),'<original-body-root-initialization-only>','exec'),scope)
        arrays,receipt = scope['fit_one'](**kwargs)
        trace = entry._root_seed_trace
        assert np.array_equal(trace['body_fit_initial_root'],root)
        with np.load(baseline['output'],allow_pickle=False) as prior:
            assert np.array_equal(prior['camera_translation'],trace['camera_translation'])
            assert np.array_equal(prior['betas'],arrays['betas'])
            assert np.array_equal(prior['K_integer_centers'],arrays['K_integer_centers'])
        receipt.update(world_root_seed_enabled=True, seed_medoid=start['medoid'],
                       original_camera_initialization_exact=True,
                       initial_body_root=root.tolist(), body_latent_zero=True,
                       initialization='Five original stages from zero latent and a local camera-aligned root seed.')
        return arrays,receipt

    adapter=types.SimpleNamespace(**dict(entry.__dict__,fit_one=fit))
    scope=dict(seated.__dict__,entry=adapter)
    types.FunctionType(seated.shard.__code__,scope)(folder,shard)


def main():
    assert os.name=='nt' and not OUT.exists()
    if find_spec('omegaconf') is None:
        raise RuntimeError('Run in the existing VPoser environment.')
    review_path = AUDIT/'review/visual-decision.json'
    review=read(review_path)
    source=AUDIT/'review/root-spike-source-receipt.json'
    assert review['actual_visual_review'] and sha(source)==review['root_source_review_sha256']
    assert sha(AUDIT/'review/root-spike-source.jpg')==review['root_source_sheet_sha256']
    transport_path=AUDIT/'review/world-pose-transport.json'
    assert sha(transport_path)==read(source)['source_transport_sha256']
    p,r=read(seated.OUT/'plan.json'),read(seated.OUT/'result.json')
    assert r['fitting_complete'] and r['plan_sha256']==sha(seated.OUT/'plan.json')
    transport=read(transport_path)
    assert transport['records']['seated_prior']['fit_sha256']==sha(seated.OUT/'result.json')
    assert transport['frames']==p['source_frames']
    assert all(sha(path)==digest for path,digest in p['model_sha256'].items())
    assert sha(BASE.ARCHIVE)==p['baseline_observation_sha256'] and sha(BASE.CONTACTS)==p['contacts_sha256']
    assert sha(entry.__file__)==p['fit_entry_sha256'] and sha(seated.__file__)==p['runner_sha256']
    frames=review['root_source_frames']
    assert len(frames)==3 and not set(frames)&set(p['retained_holdouts'])
    reference=CAMERA/'cameras-diagnostic.json'
    assert sha(reference)==transport['camera_sha256']
    cameras={r['frame']:np.asarray(r['R_w2c']) for r in read(reference)}
    sources={row['frame']:row for row in r['frames']}
    init=seed_rotations(frames,p['source_frames'],sources,cameras)
    checks=[Path(__file__),Path(seated.__file__),Path(root_hook.__file__),Path(entry.__file__),
            BASE.ARCHIVE,BASE.CONTACTS,seated.OUT/'plan.json',seated.OUT/'result.json',
            review_path,source,transport_path,reference]
    frozen=dict(p, updated=datetime.now(timezone.utc).isoformat(), frames=frames,
        initial_roots=init, baseline_rows={f:sources[f] for f in frames},
        source_hashes={str(path):sha(path) for path in checks},
        single_change='Root axis angle immediately after original camera initialization. No parameter or loss changes in the five fitting stages.',
        rule='In the fixed scene gauge, choose the minimum summed geodesic distance of the two earlier and two later roots, excluding the target. Require at least three within 30 degrees. Transport that seed into the target camera. All pose parameters remain optimized by the original stages.',
        test='One source-confirmed two-sided >90-degree spike and its immediately flanking controls. Compare paired geometry and visible body/face/feet before admission.',
        old_fit_used_as_training_initialization=False, gaussian_started=False, quality_accepted=False)
    atomic(OUT/'plan.json',frozen)
    atomic(OUT/'launch.json',dict(pid=os.getpid(),command=[sys.executable,*sys.argv],frames=frames))
    scope=dict(BASE.__dict__,fit_shard=fit_shard)
    stage=types.FunctionType(BASE.stage.__code__,scope)
    indices=[p['source_frames'].index(f) for f in frames]
    start=time.monotonic()
    try:
        rows=stage(OUT.name+'/seeded',frozen,indices,dict(vector=r['fixed_shape']))
        assert all(row['world_root_seed_enabled'] and row['five_stages'] for row in rows)
        assert all(sha(path)==digest for path,digest in frozen['source_hashes'].items())
        atomic(OUT/'result.json',dict(completed=True,frames=frames,plan_sha256=sha(OUT/'plan.json'),
            fit_result_sha256=sha(OUT/'seeded/result.json'), seconds=time.monotonic()-start,
            gaussian_started=False))
    except BaseException as exc:
        atomic(OUT/'failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':
    main()
