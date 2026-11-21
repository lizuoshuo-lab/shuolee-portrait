"""Run the existing SMPLify-X single-frame entry on the frozen 022 front roster."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import official_smplifyx_entry123 as entry
from run_leafwoman_smplify_independent144 import (
    ARCHIVE, ASSET, CONTACTS, OFFICIAL, OUT as PREVIOUS, RUNTIME, SMPLX_ROOT, decode, verify_crop,
)
from run_official_sequence123 import save_npz
from update_leafwoman129_state import atomic, read


OUT = PREVIOUS.with_name('smplify-front24-145')


def run(mode):
    assert os.name == 'nt'
    native = read(OUT.parent/'fit-plan.json')
    if mode == 'shared':
        free = read(OUT/'free/result.json')
        free_plan = read(OUT/'free/plan.json')
        assert free['successful'] and free['plan_sha256'] == entry.sha(OUT/'free/plan.json')
    assert entry.sha(ARCHIVE) == native['previous_fit_sha256']
    assert entry.sha(CONTACTS) == native['source_contacts_sha256']
    source = {r['frame']: r for r in read(CONTACTS)['records']}
    frames = [r['frame'] for r in native['front_arc']]
    assert len(frames) == len(set(frames)) == 24 and not native['rear_arc']
    assert not (set(frames) & set(native['retained_holdouts']))
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        assert saved['frame_ids'].tolist() == frames
        observations = saved['observed_native_keypoints']
        K = saved['K_integer_centers']
    indices = [source[f]['source_frame_index'] for f in frames]
    assert indices == sorted(set(indices))
    model_files = (ASSET, SMPLX_ROOT/'smplx/body_models.py', SMPLX_ROOT/'smplx/lbs.py',
                   RUNTIME/'src/avatar_core/shuolee_vposer.py')
    pins = {str(p): entry.sha(p) for p in model_files}
    entry.adapted_entry_source(OFFICIAL)
    fit_dir = OUT/mode
    shared = None
    shared_inputs = None
    if mode == 'shared':
        assert free_plan['source_frames'] == frames and free_plan['model_sha256'] == pins
        assert [r['frame'] for r in free['frames']] == frames
        rows = []
        for row in free['frames']:
            path = Path(row['output'])
            assert entry.sha(path) == row['sha256']
            with np.load(path, allow_pickle=False) as fit:
                rows.append(fit['betas'][0])
        shared = np.median(np.stack(rows), axis=0, keepdims=True).astype(np.float32)
        assert shared.shape == (1, 10) and np.isfinite(shared).all()
        shared_inputs = dict(free_result_sha256=entry.sha(OUT/'free/result.json'),
                             estimator='Componentwise median across all fixed front views, without residual ranking',
                             vector=shared.tolist(), can_reestimate_for_a_new_roster=True)
    assert not fit_dir.exists()
    atomic(fit_dir/'plan.json', dict(source_frames=frames, native_indices=indices,
        source_fit_plan_sha256=entry.sha(OUT.parent/'fit-plan.json'), baseline_observation_sha256=entry.sha(ARCHIVE),
        contacts_sha256=entry.sha(CONTACTS), retained_holdouts=native['retained_holdouts'],
        runner_sha256=entry.sha(__file__), fit_entry_sha256=entry.sha(entry.__file__), model_sha256=pins,
        official_commit=entry.COMMIT, K_integer_centers=K.tolist(), camera_initialization_hips_unverified=True,
        original_five_stages=True, pose_zero_start_each_frame=True, device='cuda', fp32_tf32_disabled=True,
        shared_shape=shared_inputs, rear_frames_verified=0, quality_accepted=False, gaussian_started=False))

    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 5*2**30
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(SMPLX_ROOT))
    import smplx
    assert Path(smplx.__file__).is_relative_to(SMPLX_ROOT)
    utils, _, _, _ = entry.load_official(OFFICIAL)
    mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True, use_face_contour=False)
    assert len(mapping) == observations.shape[1] == 118
    loader = entry.module(RUNTIME/'src/avatar_core/shuolee_vposer.py', '_leaf145_vposer')
    loader.MODEL, loader.CODE = ASSET.parent.parent/'vposer/V02_05', RUNTIME/'third_party/human_body_prior'
    vposer, _, vposer_receipt = loader.load_vposer('cuda')
    body = smplx.SMPLX(str(ASSET), batch_size=1, num_betas=10, num_expression_coeffs=10, use_pca=True,
        num_pca_comps=12, flat_hand_mean=False, use_face_contour=False, create_body_pose=False,
        create_transl=False, joint_mapper=utils.JointMapper(mapping), dtype=torch.float32).cuda().eval()
    results = []
    active = None
    started = time.monotonic()
    full = (0, 0, 2704, 1520)
    try:
        for i, ((index, rgb), frame) in enumerate(zip(decode(indices, full, False), frames, strict=True)):
            assert index == source[frame]['source_frame_index']
            verify_crop(rgb, source[frame]['paths']['body'], full)
            active = frame
            dest = fit_dir/frame
            dest.mkdir()
            atomic(fit_dir/'progress.json', dict(stage='official_frame_fit', frame=frame, completed=i, total=24, pid=os.getpid()))
            begin = time.monotonic()
            arrays, receipt = entry.fit_one(official_root=OFFICIAL, image=rgb.astype(np.float32)/255,
                keypoints=observations[i:i+1], K_integer_centers=K, body_model=body, vposer2=vposer,
                result_path=dest/'official-result.pkl', shared_betas=shared, freeze_betas=shared is not None,
                device='cuda', diff_path=dest/'upstream-adapter.diff')
            assert np.array_equal(arrays['K_integer_centers'], K) and (arrays['vertices_camera'][..., 2] > .05).all()
            if shared is not None:
                assert np.array_equal(arrays['betas'], shared)
            archive = dest/'fit-camera-space.npz'
            save_npz(archive, **arrays, observed_native_keypoints=observations[i:i+1])
            row = dict(frame=frame, source_index=index, output=str(archive), sha256=entry.sha(archive),
                       native_crop_bitexact=True, seconds=time.monotonic()-begin, **receipt)
            results.append(row)
            atomic(dest/'receipt.json', row)
            print('SMPLIFY022_FRONT24', mode, i+1, frame, round(row['seconds'], 3), flush=True)
        atomic(fit_dir/'result.json', dict(plan_sha256=entry.sha(fit_dir/'plan.json'), mode=mode, frames=results,
            vposer=vposer_receipt, seconds=time.monotonic()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
            successful=len(results)==24, gaussian_started=False, quality_accepted=False))
        atomic(fit_dir/'progress.json', dict(stage='complete', completed=len(results), pid=None))
    except Exception as error:
        atomic(fit_dir/'failure.json', dict(frame=active, completed=len(results), error=repr(error),
                                           traceback=traceback.format_exc()))
        atomic(fit_dir/'progress.json', dict(stage='failed', completed=len(results), pid=None))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('free', 'shared'))
    run(parser.parse_args().mode)
