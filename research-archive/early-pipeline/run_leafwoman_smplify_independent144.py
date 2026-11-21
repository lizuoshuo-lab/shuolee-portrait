"""Independent three-frame neutral-start SMPLify-X control for 022."""
from pathlib import Path
import os
import sys
import time

import numpy as np

import official_smplifyx_entry123 as entry
from replay_leafwoman_arm138 import ARCHIVE, OFFICIAL, SMPLX_ROOT, ASSET
from review_full_sequence123 import relocate
from run_leafwoman_gvhmr143 import CONTACTS, OUT as BASE
from track_leafwoman_torso141 import decode, verify_crop
from update_leafwoman129_state import atomic, read


OUT = BASE.with_name('smplify-independent144')
RUNTIME = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')


def main():
    assert os.name == 'nt' and not OUT.exists()
    plan = read(BASE/'plan.json')
    assert entry.sha(ARCHIVE) == plan['baseline_sha256']
    assert entry.sha(CONTACTS) == plan['contacts_sha256']
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        names = saved['frame_ids'].tolist()
        observations = saved['observed_native_keypoints']
        K = saved['K_integer_centers']
        assert np.array_equal(K, plan['K_integer_centers'])
    supported = np.flatnonzero((observations[:, 2:8, 2] > 0).all(axis=1))
    slots = supported[[0, len(supported)//2, -1]]
    frames = [names[i] for i in slots]
    assert frames == read(BASE.with_name('arm-refine138')/'plan.json')['frames']
    source = {r['frame']: r for r in read(CONTACTS)['records']}
    indices = [source[f]['source_frame_index'] for f in frames]
    entry.adapted_entry_source(OFFICIAL)
    atomic(OUT/'plan.json', dict(frames=frames, source_indices=indices, observation_archive_sha256=entry.sha(ARCHIVE),
        runner_sha256=entry.sha(__file__), fit_entry_sha256=entry.sha(entry.__file__),
        same_complete_K=K.tolist(), official_commit=entry.COMMIT,
        control='Official headless single-frame five stages, original camera init and upstream schedule',
        camera_init_hips_unverified=True, prior_subject_optimized_parameters_loaded=False,
        independent_neutral_start_per_frame=True, quality_accepted=False))

    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(SMPLX_ROOT))
    import smplx
    assert Path(smplx.__file__).is_relative_to(SMPLX_ROOT)
    utils, _, _, _ = entry.load_official(OFFICIAL)
    mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True, use_face_contour=False)
    assert len(mapping) == observations.shape[1] == 118
    vposer_loader = entry.module(RUNTIME/'src/avatar_core/shuolee_vposer.py', '_leaf144_vposer')
    vposer_loader.MODEL = ASSET.parent.parent/'vposer/V02_05'
    vposer_loader.CODE = RUNTIME/'third_party/human_body_prior'
    vposer, _, vposer_receipt = vposer_loader.load_vposer('cpu')
    model = smplx.SMPLX(str(ASSET), batch_size=1, num_betas=10, num_expression_coeffs=10,
                        use_pca=True, num_pca_comps=12, flat_hand_mean=False, use_face_contour=False,
                        create_body_pose=False, create_transl=False, joint_mapper=utils.JointMapper(mapping),
                        dtype=torch.float32).eval()
    results = []
    start = time.monotonic()
    full = (0, 0, 2704, 1520)
    for (index, rgb), frame, slot in zip(decode(indices, full, False), frames, slots, strict=True):
        row = source[frame]
        assert index == row['source_frame_index']
        verify_crop(rgb, row['paths']['body'], full)
        print('SMPLIFY022_START', frame, flush=True)
        dest = OUT/frame
        dest.mkdir()
        atomic(OUT/'progress.json', dict(stage='official_fit_single_frame', frame=frame, pid=os.getpid()))
        arrays, receipt = entry.fit_one(
            official_root=OFFICIAL, image=rgb.astype(np.float32)/255, keypoints=observations[slot:slot+1],
            K_integer_centers=K, body_model=model, vposer2=vposer, result_path=dest/'result.pkl',
            shared_betas=None, freeze_betas=False, device='cpu', diff_path=dest/'upstream-adapter.diff',
        )
        assert np.array_equal(arrays['K_integer_centers'], K) and (arrays['vertices_camera'][..., 2] > .05).all()
        archive = dest/'fit-camera-space.npz'
        np.savez_compressed(archive, **arrays, observed_native_keypoints=observations[slot:slot+1])
        results.append(dict(frame=frame, source_index=index, output=str(archive), sha256=entry.sha(archive),
                            native_crop_bitexact=True, **receipt))
        print('SMPLIFY022_COMPLETE', frame, flush=True)
    atomic(OUT/'result.json', dict(frames=results, vposer=vposer_receipt, seconds=time.monotonic()-start,
        successful=True, baseline_geometry_used=False, original_hips_are_unverified=True, gaussian_started=False,
        camera_pose_world_verified=False, quality_accepted=False))
    atomic(OUT/'progress.json', dict(stage='complete', pid=None))


if __name__ == '__main__':
    main()
