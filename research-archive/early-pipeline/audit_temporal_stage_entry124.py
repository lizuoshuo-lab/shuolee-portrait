"""Read-only receipt and adjacent-rotation audit of the registered stage control."""
from pathlib import Path
import json
import numpy as np
import cv2
from run_official_gaussians123 import read, save, sha, now


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def angles(a, b):
    r = [np.stack([cv2.Rodrigues(x)[0] for x in z['body_pose_axis_angle'].reshape(-1, 3)]) for z in (a, b)]
    return np.degrees(np.arccos(np.clip((np.trace(r[0].transpose(0, 2, 1) @ r[1], axis1=1, axis2=2)-1)/2, -1, 1)))


if __name__ == '__main__':
    root = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
    run = root / 'sofa-temporal-warmstart-v2'
    plan = read(run/'plan.json')
    source_checks = {p: sha(p) == digest for p, digest in plan['sources'].items()}
    assert all(source_checks.values())
    frames = {}
    previous = {}
    checks = []
    rows = {r['frame']: r for r in read(Path(plan['source_root'])/'data/inputs.json')['records']}
    for name in plan['ordered_fit_frames']:
        folder = run/'official-fit/frames'/name
        if not (folder/'result.json').exists():
            break
        receipt = read(folder/'result.json')
        current = load(folder/'winner-camera-space.npz')
        seed = receipt['continuation']
        neighbour = seed['neighbour']
        contract = dict(frame=name, neighbour=neighbour, camera_initialization_unchanged=seed['camera_initialization_unchanged'],
                        original_stage_indices=receipt['original_stage_indices'])
        if neighbour:
            assert neighbour in frames and abs(int(name[-6:])-int(neighbour[-6:])) == 1
            contract['body_seed_matches_same_run'] = np.array_equal(np.asarray(seed['initial_body_latent'], dtype=np.float32), frames[neighbour]['pose_embedding'])
            contract['root_seed_matches_same_run'] = np.array_equal(np.asarray(seed['initial_state']['body_fit_initial_root'], dtype=np.float32), frames[neighbour]['global_orient'])
            assert contract['body_seed_matches_same_run'] and contract['root_seed_matches_same_run']
            assert receipt['original_stage_indices'] == [3, 4]
        else:
            assert name == plan['reference_frame'] and receipt['original_stage_indices'] == list(range(5))
        old_path = root/'sofa-temporal/official-fit/frames'/name/'winner-camera-space.npz'
        if old_path.exists():
            old = load(old_path)
            previous[name] = old
            contract['old_camera_beta_topology_parity'] = {k: np.array_equal(current[k], old[k]) for k in ('K_integer_centers', 'camera_translation', 'betas', 'faces')}
            assert all(contract['old_camera_beta_topology_parity'].values())
        frames[name] = current
        checks.append(contract)
    transitions = []
    names = sorted(frames)
    for a, b in zip(names[:-1], names[1:]):
        if int(b[-6:])-int(a[-6:]) != 1:
            continue
        new = angles(frames[a], frames[b])
        entry = dict(previous=a, current=b, dt_seconds=rows[b]['time_seconds']-rows[a]['time_seconds'],
                     current_angles_deg=new.tolist(), current_max_deg=float(new.max()))
        if a in previous and b in previous:
            old = angles(previous[a], previous[b])
            entry.update(previous_angles_deg=old.tolist(), previous_max_deg=float(old.max()))
        transitions.append(entry)
    result = dict(updated=now(),state='read_only_completed',frames=len(frames),source_hash_checks=source_checks,
        receipt_checks=checks,transitions=transitions,optimizer_steps=0,quality_accepted=False,
        note='A smooth pose is not necessarily correct. Compare raw/native overlays and synthetic sides separately. Progress metadata neighbour_used_as_initialization=false is inherited from the original unseeded control; the checked per-frame continuation receipts describe this run.')
    path=run/f'contract-audit-{len(frames):03d}.json'
    assert not path.exists(), 'Read the existing audit; do not overwrite evidence.'
    save(path,result)
    print(json.dumps(dict(path=str(path),frames=len(frames),source_checks=len(source_checks),contracts_passed=True,
        known_failure_transitions=[t for t in transitions if t['current'] in ('frame_000033','frame_000036')]),ensure_ascii=False))
