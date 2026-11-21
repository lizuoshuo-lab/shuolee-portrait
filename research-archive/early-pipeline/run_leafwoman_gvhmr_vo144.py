"""Bounded upstream GVHMR VO diagnostic using the 022 conditional full K."""
import sys
import time
from pathlib import Path

import numpy as np

from official_smplifyx_entry123 import sha
from run_leafwoman_gvhmr143 import CONTACTS, REPO, WSL, atomic, read
from run_leafwoman_gvhmr_visual144 import OUT


def main():
    assert WSL and not (OUT/'vo-result.json').exists()
    plan = read(OUT/'plan.json')
    assert sha(CONTACTS) == plan['contacts_sha256']
    files = ('solver_two_view.py', 'matcher_wrapper.py', 'model/cv2_matcher.py')
    pins = {f: sha(REPO/'hmr4d/utils/preproc/relpose'/f) for f in files}
    config_path = OUT/'vo-plan.json'
    existing = read(config_path)
    assert existing['visual_plan_sha256'] == sha(OUT/'plan.json') and existing['source_pins'] == pins
    partial_path = OUT/'vo-progress.json'
    partial = read(partial_path)
    assert partial['pairs'] == len(partial['results']) and 0 < partial['pairs'] < 56
    previous_pairs = partial['results']
    progress_copy = OUT/'vo-first-session-progress.json'
    if not progress_copy.exists():
        atomic(progress_copy, partial)

    sys.path.insert(0, str(REPO))
    import cv2
    import pycolmap
    from hmr4d.utils.preproc.relpose.matcher_wrapper import Matcher
    from hmr4d.utils.preproc.relpose.solver_two_view import CameraParams, TwoPairSolver
    assert pycolmap.__version__ == '3.10.0'
    cv2.setNumThreads(2)
    cv2.setRNGSeed(144)
    start = time.monotonic()
    fx, fy = np.diag(plan['K_integer_centers'])[:2]
    cx, cy = np.asarray(plan['K_integer_centers'])[:2, 2] + .5
    w, h = 2704, 1520
    solver = TwoPairSolver(CameraParams(w, h, fx, cx, cy)).solver
    camera = pycolmap.Camera(camera_id=0, model='PINHOLE', width=w, height=h, params=[fx, fy, cx, cy])
    matcher = Matcher('sift')
    source = read(CONTACTS)['records']
    assert len(source) == 57
    measures, rotations = list(previous_pairs), [np.eye(3)]
    for j, measure in enumerate(measures):
        assert measure['pair'] == [source[j]['frame'], source[j+1]['frame']]
        if 'relative_R' in measure and len(rotations) == j+1:
            rotations.append(np.asarray(measure['relative_R'])@rotations[-1])
    previous = None
    previous_row = None

    def in_person(points, row):
        x0, y0, x1, y1 = row['initial_detector_box']
        x, y = points.T
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    for i in range(len(measures), len(source)):
        row = source[i]
        path = Path(row['source_image'])
        assert sha(path) == row['source_sha256']
        rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert rgb is not None and rgb.shape == (h, w, 3)
        if previous is not None:
            cv2.setRNGSeed(144+i)
            a, b = matcher.match_np(previous, rgb)
            xy0, xy1 = a.astype(np.float64)+.5, b.astype(np.float64)+.5
            matches = np.repeat(np.arange(len(a))[:, None], 2, axis=1)
            answer = pycolmap.estimate_calibrated_two_view_geometry(
                camera, xy0, camera, xy1, matches=matches, options=solver.options,
            )
            support = np.asarray(answer.inlier_matches)
            count = len(support)
            support_in_box = int(np.count_nonzero(
                in_person(a[support[:, 0]], previous_row) | in_person(b[support[:, 1]], row)
            )) if count else 0
            record = dict(pair=[previous_row['frame'], row['frame']], matches=len(a), inliers=count,
                          inlier_at_least_one_person_box=support_in_box, config=int(answer.config))
            pose = answer.cam2_from_cam1
            if pose is not None and count >= solver.options.min_num_inliers:
                R = pose.matrix()[:, :3]
                if np.isfinite(R).all() and np.max(abs(R.T@R-np.eye(3))) < 1e-6 and abs(np.linalg.det(R)-1) < 1e-6:
                    record['relative_R'] = R.tolist()
                    record['rotation_deg'] = float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))
                    if len(rotations) == i:
                        rotations.append(R@rotations[-1])
            measures.append(record)
            atomic(OUT/'vo-progress.json', dict(pairs=len(measures), results=measures))
            print('GVHMR022_VO', i, count, record.get('rotation_deg'), flush=True)
        previous, previous_row = rgb, row
    usable = len(rotations) == len(source)
    if usable:
        np.save(OUT/'vo-rotations.npy', np.stack(rotations), allow_pickle=False)
    atomic(OUT/'vo-result.json', dict(plan_sha256=sha(config_path), pairs=measures,
        continuation_runner_sha256=sha(__file__), first_session_pairs=read(progress_copy)['pairs'],
        valid_rotations=sum('relative_R' in r for r in measures), complete_numeric_chain=usable,
        seconds=time.monotonic()-start, physical_camera_verified=False, gaussian_started=False))


if __name__ == '__main__':
    main()
