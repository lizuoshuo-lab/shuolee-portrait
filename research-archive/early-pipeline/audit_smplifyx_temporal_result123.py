"""Read-only objective and SO(3) audit; does not select or splice frame results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

BODY = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/fullbody-smplifyx')
OUT = BODY / 'temporal-root-sequence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def rotations(aa):
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    out = []
    for v in aa:
        theta = np.linalg.norm(v)
        if theta < 1e-12:
            out.append(np.eye(3))
            continue
        x, y, z = v / theta
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        out.append(np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross))
    return np.stack(out)


def distances(left, right):
    delta = np.asarray(left).reshape(-1, 3, 3) @ np.swapaxes(np.asarray(right).reshape(-1, 3, 3), -1, -2)
    cosine = (np.trace(delta, axis1=-2, axis2=-1) - 1) / 2
    skew = np.stack([delta[:, 2, 1]-delta[:, 1, 2], delta[:, 0, 2]-delta[:, 2, 0], delta[:, 1, 0]-delta[:, 0, 1]], -1)
    sine = np.linalg.norm(skew, axis=-1) / 2
    return np.degrees(np.arctan2(sine, cosine))


def main():
    # Independent elementary cases protect this read-only continuity calculation.
    assert np.max(distances(rotations([[0, 0, 0]]), rotations([[0, 0, 2*np.pi]]))) < 1e-10
    assert abs(distances(rotations([[0, 0, 0]]), rotations([[0, 0, np.pi/2]]))[0] - 90) < 1e-10
    rows, pairs, cached = [], [], {}
    for i in range(1, 60):
        frame = f'frame_{i:06d}'
        p = OUT/'frames'/frame/'result.json'
        if not p.exists():
            continue
        r = read(p)
        if r.get('state') != 'complete':
            continue
        old_report = read(BODY/'full-sequence/frames'/frame/'result.json')
        data = {}
        for label, root in [('baseline', BODY/'full-sequence'), ('candidate', OUT)]:
            path = root/'frames'/frame/'winner-camera-space.npz'
            data[label] = load(path)
            assert data[label]['frame_id'].item() == frame
        for key in ('faces', 'betas', 'K_integer_centers', 'w2c', 'camera_translation'):
            assert np.array_equal(data['baseline'][key], data['candidate'][key]), (frame, key)
        cached[frame] = data
        rows.append(dict(frame=frame, baseline_loss=old_report['winning_loss'], candidate_loss=r['winning_loss'],
                         delta_loss=r['winning_loss']-old_report['winning_loss'], seconds=r['elapsed_seconds'],
                         result_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        prior = f'frame_{i-1:06d}'
        if prior not in cached:
            continue
        pair = dict(previous=prior, frame=frame)
        for label, current in data.items():
            prev = cached[prior][label]
            joint = distances(rotations(current['body_pose_axis_angle']), rotations(prev['body_pose_axis_angle']))
            pair[label] = dict(root_deg=float(distances(current['root_rotation'], prev['root_rotation'])[0]),
                               local_body_joint_deg=joint.tolist(), local_joint_median_deg=float(np.median(joint)),
                               local_joint_max_deg=float(np.max(joint)), left_elbow_deg=float(joint[17]),
                               right_elbow_deg=float(joint[18]))
        pairs.append(pair)
    result = dict(updated=datetime.now(timezone.utc).isoformat(), frames=len(rows), total=59,
                  state='complete' if len(rows)==59 else 'partial', rows=rows, adjacent_pairs=pairs,
                  criterion='Same original final objective, unchanged observations and weights; no frame selection performed.',
                  purpose='Diagnose whether worse visible poses also have a worse original objective. Post-hoc association is not selector validation.',
                  continuity='SO(3) distance in degrees, not axis-angle Euclidean distance. Root is camera-relative, not a recovered world trajectory.',
                  no_frame_selection=True, candidate_fit_modified=False, Gaussian_training_started=False,
                  quality_accepted=False, goal_complete=False)
    (OUT/'review/metrics-objective.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(frames=len(rows), pairs=len(pairs), lower_objective=sum(r['delta_loss']<0 for r in rows),
                         total_fit_seconds=sum(r['seconds'] for r in rows), largest_objective_increase=max(rows,key=lambda r:r['delta_loss']))))


if __name__ == '__main__':
    main()
