"""Parallelize independent calls to the existing native-pixel SMPLify-X entry."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import os
from pathlib import Path
import sys
import time

import numpy as np

import official_smplifyx_entry123 as entry
from run_leafwoman_front24145 import (
    ARCHIVE, ASSET, CONTACTS, OFFICIAL, OUT, RUNTIME, SMPLX_ROOT, decode, save_npz, verify_crop,
)
from update_leafwoman129_state import atomic, read


def lock_unobserved_joints(vposer, joints):
    """Keep the VPoser latent and fix only explicitly unobserved local angles."""
    import torch

    class LockedDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vposer = vposer
            self.register_buffer('indices', torch.tensor(joints, dtype=torch.long)-1)

        def decode(self, latent):
            result = self.vposer.decode(latent)
            pose = result['pose_body']
            shaped = pose.reshape(latent.shape[0], 21, 3)
            return dict(result, pose_body=shaped.index_fill(1, self.indices, 0.).reshape_as(pose))

    decoder = LockedDecoder()
    latent = torch.zeros((1, 32), requires_grad=True)
    original = vposer.decode(latent)['pose_body'].reshape(1, 21, 3)
    constrained = decoder.decode(latent)['pose_body'].reshape(1, 21, 3)
    unlocked = sorted(set(range(21))-set(j-1 for j in joints))
    assert torch.equal(original[:, unlocked], constrained[:, unlocked])
    assert torch.count_nonzero(constrained[:, decoder.indices]) == 0
    g_hidden, = torch.autograd.grad(constrained[:, decoder.indices].sum(), latent, retain_graph=True)
    g_visible, = torch.autograd.grad(constrained[:, unlocked].square().sum(), latent)
    assert torch.count_nonzero(g_hidden) == 0 and torch.isfinite(g_visible).all() and g_visible.abs().max() > 0
    return decoder


def fit_shard(folder, shard):
    plan = read(folder/'plan.json')
    assert plan['runner_sha256'] == entry.sha(__file__)
    assert plan['baseline_observation_sha256'] == entry.sha(ARCHIVE)
    assert plan['contacts_sha256'] == entry.sha(CONTACTS)
    if 'visibility_receipt_sha256' in plan:
        assert plan['visibility_receipt_sha256'] == entry.sha(ARCHIVE.with_name('source-observation-visibility.json'))
    subset = plan['shards'][shard]
    with open(folder/f'shard-{shard}.log', 'x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            return fit_frames(folder, shard, plan, subset)


def fit_frames(folder, shard, plan, subset):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(SMPLX_ROOT))
    import smplx
    assert Path(smplx.__file__).is_relative_to(SMPLX_ROOT)
    utils, _, _, _ = entry.load_official(OFFICIAL)
    mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True, use_face_contour=False)
    loader = entry.module(RUNTIME/'src/avatar_core/shuolee_vposer.py', '_leaf145cpu_vposer')
    loader.MODEL, loader.CODE = ASSET.parent.parent/'vposer/V02_05', RUNTIME/'third_party/human_body_prior'
    vposer, _, vposer_receipt = loader.load_vposer('cpu')
    body = smplx.SMPLX(str(ASSET), batch_size=1, num_betas=10, num_expression_coeffs=10, use_pca=True,
        num_pca_comps=12, flat_hand_mean=False, use_face_contour=False, create_body_pose=False,
        create_transl=False, joint_mapper=utils.JointMapper(mapping), dtype=torch.float32).eval()
    with np.load(ARCHIVE, allow_pickle=False) as z:
        assert z['frame_ids'].tolist() == plan['source_frames']
        targets, K = z['observed_native_keypoints'], z['K_integer_centers']
    assert len(mapping) == targets.shape[1] == 118 and np.array_equal(K, plan['K_integer_centers'])
    shared = None if plan['shared_shape'] is None else np.asarray(plan['shared_shape']['vector'], np.float32)
    locked = plan.get('locked_unobserved_pose_joints', ())
    if locked:
        from smplx.joint_names import JOINT_NAMES
        expected = [JOINT_NAMES.index(f'{side}_{part}') for part in ('hip','knee','ankle','foot') for side in ('left','right')]
        assert locked == expected
        assert not targets[:, [10,11,13,14,*range(19,25)], 2].any()
        vposer = lock_unobserved_joints(vposer, locked)
    contacts = {r['frame']: r for r in read(CONTACTS)['records']}
    indices = [plan['native_indices'][i] for i in subset]
    full = (0, 0, *plan.get('native_wh', (2704, 1520)))
    progress = folder/f'shard-{shard}-progress.json'
    for n, ((index, rgb), i) in enumerate(zip(decode(indices, full, False), subset, strict=True)):
        frame = plan['source_frames'][i]
        assert index == contacts[frame]['source_frame_index']
        verify_crop(rgb, contacts[frame]['paths']['body'], full)
        dest = folder/frame
        dest.mkdir()
        atomic(progress, dict(stage='fitting', pid=os.getpid(), frame=frame, completed=n, total=len(subset)))
        started = time.monotonic()
        arrays, receipt = entry.fit_one(official_root=OFFICIAL, image=rgb.astype(np.float32)/255,
            keypoints=targets[i:i+1], K_integer_centers=K, body_model=body, vposer2=vposer,
            result_path=dest/'official-result.pkl', shared_betas=shared, freeze_betas=shared is not None,
            device='cpu', diff_path=dest/'upstream-adapter.diff')
        assert np.array_equal(arrays['K_integer_centers'], K)
        if not plan.get('automatic_delivery'):
            assert (arrays['vertices_camera'][..., 2] > .05).all()
        if shared is not None:
            assert np.array_equal(arrays['betas'], shared)
        if locked:
            assert not arrays['body_pose_axis_angle'].reshape(21, 3)[np.asarray(locked)-1].any()
        path = dest/'fit-camera-space.npz'
        save_npz(path, **arrays, observed_native_keypoints=targets[i:i+1])
        atomic(dest/'receipt.json', dict(frame=frame, source_index=index, output=str(path), sha256=entry.sha(path),
            native_crop_bitexact=True, seconds=time.monotonic()-started, **receipt))
    atomic(progress, dict(stage='complete', pid=None, completed=len(subset), total=len(subset)))
    return vposer_receipt


def prepare(mode, workers):
    old = read(OUT/'free/plan.json')
    assert old['source_fit_plan_sha256'] == entry.sha(OUT.parent/'fit-plan.json')
    assert old['fit_entry_sha256'] == entry.sha(entry.__file__)
    assert all(entry.sha(path) == digest for path, digest in old['model_sha256'].items())
    assert old['contacts_sha256'] == entry.sha(CONTACTS) and old['baseline_observation_sha256'] == entry.sha(ARCHIVE)
    assert len(old['source_frames']) == 24 and not set(old['source_frames']) & set(old['retained_holdouts'])
    shared = None
    visibility_contract = {}
    if mode == 'shared':
        free = read(OUT/'free-cpu/result.json')
        plan = read(OUT/'free-cpu/plan.json')
        assert free['successful'] and free['plan_sha256'] == entry.sha(OUT/'free-cpu/plan.json')
        assert plan['source_frames'] == old['source_frames'] and plan['model_sha256'] == old['model_sha256']
        assert [r['frame'] for r in free['frames']] == plan['source_frames']
        betas = []
        for row in free['frames']:
            assert entry.sha(row['output']) == row['sha256']
            with np.load(row['output'], allow_pickle=False) as fit:
                betas.append(fit['betas'][0])
        beta = np.median(np.stack(betas), axis=0, keepdims=True).astype(np.float32)
        assert beta.shape == (1, 10) and np.isfinite(beta).all()
        shared = dict(vector=beta.tolist(), estimator='Componentwise median across all 24 free fits. No score ranking.',
                      free_result_sha256=entry.sha(OUT/'free-cpu/result.json'), can_reestimate=True)
    if mode == 'neutral':
        baseline = OUT/'shared-cpu'
        prior = read(baseline/'plan.json')
        fit = read(baseline/'result.json')
        assert fit['successful'] and fit['plan_sha256'] == entry.sha(baseline/'plan.json')
        for key in ('source_frames','retained_holdouts','contacts_sha256','baseline_observation_sha256','K_integer_centers','model_sha256'):
            assert prior[key] == old[key], key
        assert workers == len(prior['shards'])
        shared = prior['shared_shape']
        path = ARCHIVE.with_name('source-observation-visibility.json')
        evidence = read(path)
        assert evidence['training_frames'] == prior['source_frames']
        for frame in prior['source_frames']:
            assert all(evidence['joint_group_visibility'][frame][name] is False
                       for name in ('left_knee','right_knee','left_foot','right_foot'))
        visibility_contract = dict(
            visibility_receipt_sha256=entry.sha(path),
            comparison_shared_fit_sha256=entry.sha(baseline/'result.json'),
            locked_unobserved_pose_joints=[1,2,4,5,7,8,10,11],
            lower_pose_policy='Zero local output only for the obscured hip-to-foot chains. The original hip detector coordinates still enter camera initialization.',
            lower_pose_physically_verified=False,
        )
    folder = OUT/(mode+'-cpu')
    assert not folder.exists()
    atomic(folder/'plan.json', dict(old, runner_sha256=entry.sha(__file__), device='cpu', fp32_tf32_disabled=False,
        shared_shape=shared, partition_policy='Deterministic modulo by frozen chronological roster index',
        shards=[list(range(i, 24, workers)) for i in range(workers)], torch_threads_per_worker=1,
        gpu_diagnostic_plan_sha256=entry.sha(OUT/'free/plan.json'), **visibility_contract))
    return folder


def run(mode, workers):
    assert os.name == 'nt' and workers in (1, 2, 3, 4)
    folder = prepare(mode, workers)
    plan = read(folder/'plan.json')
    started = time.monotonic()
    atomic(folder/'progress.json', dict(stage='parallel_official_fit', workers=workers, pid=os.getpid()))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit_shard, folder, i): i for i in range(workers)}
            meta = []
            for completed in as_completed(futures):
                meta.append(completed.result())
                print('SMPLIFY022_CPU_SHARD_COMPLETE', mode, futures[completed], flush=True)
        assert all(v == meta[0] for v in meta)
        rows = [read(folder/frame/'receipt.json') for frame in plan['source_frames']]
        assert all(entry.sha(r['output']) == r['sha256'] for r in rows)
        atomic(folder/'result.json', dict(plan_sha256=entry.sha(folder/'plan.json'), mode=mode, frames=rows, workers=workers,
            vposer=meta[0], seconds=time.monotonic()-started, peak_cuda_mib=0, successful=len(rows)==24,
            gaussian_started=False, quality_accepted=False))
        atomic(folder/'progress.json', dict(stage='complete', completed=len(rows), pid=None))
    except Exception as error:
        atomic(folder/'failure.json', dict(error=repr(error), completed=len(list(folder.glob('frame_*/receipt.json')))))
        atomic(folder/'progress.json', dict(stage='failed', pid=None))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('free', 'shared', 'neutral'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.mode, args.workers)
