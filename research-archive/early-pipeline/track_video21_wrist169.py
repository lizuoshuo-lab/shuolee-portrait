"""Check 021 right-wrist detector aliases and nearby native material tracks."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

from extract_video21_shoe_depth162 import FROZEN, replace
from fit_video21_native148 import ARCHIVE, CONTACTS, OUT as BASE, original
from review_full_sequence123 import bound_file
from update_leafwoman129_state import atomic, read, sha


OUT = BASE / 'right-wrist-material169'
# Rectangles in the unchanged 2704x1520 source, checked in RGB before tracking.
REGIONS = {
    'frame_000045': {
        'wrist_crease': (1257, 1028, 1275, 1042),
        'hand_dorsum': (1275, 1041, 1294, 1055),
        'finger_skin': (1275, 1063, 1286, 1078),
        'chair_armrest': (1272, 925, 1291, 937),
    },
    'frame_000053': {
        'wrist_crease': (1299, 1039, 1317, 1054),
        'hand_dorsum': (1322, 1052, 1342, 1067),
        'finger_skin': (1337, 1070, 1353, 1091),
        'chair_armrest': (1334, 926, 1354, 938),
    },
}


def audit_aliases():
    utils = original.OFFICIAL / 'smplifyx/utils.py'
    tree = ast.parse(utils.read_text(encoding='utf-8'))
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'smpl_to_openpose')
    scope = dict(np=np)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(utils), 'exec'), scope)
    mapper = scope['smpl_to_openpose'](model_type='smplx', use_hands=True, use_face=True,
        use_face_contour=False, openpose_format='coco25')
    assert mapper[4] == mapper[46] == 21 and mapper[7] == mapper[25] == 20
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        ids = archive['frame_ids'].tolist()
        targets = archive['observed_native_keypoints']
    results = {}
    for side, body, hand in (('left', 7, 25), ('right', 4, 46)):
        xy = targets[:, [body, hand], :2]
        confidence = targets[:, [body, hand], 2]
        gaps = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1)
        active = (confidence > 0).all(1)
        results[side] = dict(body=body, hand=hand, smplx_index=int(mapper[body]), active=int(active.sum()),
            median_px=float(np.median(gaps[active])), p90_px=float(np.percentile(gaps[active], 90)),
            both_below_one_px_impossible=int((active & (gaps >= 2.)).sum()),
            records=[dict(frame=f, xy=xy[i].tolist(), confidence=confidence[i].tolist(),
                gap_px=float(gaps[i]), lower_bound_max_px=float(gaps[i]/2), active=bool(active[i]))
                for i, f in enumerate(ids)])
    checked = []
    for name in ('shoe-depth-refine165/control', 'right-elbow-visibility168/control'):
        result = BASE / name / 'result.json'
        for row in read(result)['frames']:
            f = row['frame']
            if int(f[-6:]) < 50:
                continue
            assert sha(row['output']) == row['sha256']
            with np.load(row['output'], allow_pickle=False) as archive:
                assert np.array_equal(archive['observed_native_keypoints'][0], targets[ids.index(f)])
                projected = archive['joints_projected_native'][0]
                assert np.array_equal(projected[4], projected[46])
                wrist = projected[4].tolist()
            checked.append(dict(branch=name, frame=f, projection=wrist, pose_sha256=row['sha256']))
    return dict(input_sha256=sha(ARCHIVE), official_mapper_sha256=sha(utils), subject='021',
        camera_axis_unmodified=True, raw_frames_unchanged=True, records=results, shared_projection_checks=checked,
        inference='Two high-confidence labels share one projected joint. A separation d forces at least one error >= d/2. This gives no physical ownership or depth on its own.')


def main():
    assert not OUT.exists()
    previous = BASE / 'right-elbow-visibility168/review/visual-decision.json'
    assert read(previous)['actual_visual_review'] and not read(previous)['gaussian_admitted']
    source_path = FROZEN / 'source.py'
    assert sha(source_path) == read(BASE / 'fixedk-dense159-reference/review/visual-decision.json')['local_shoe_source_sha256']
    aliases = audit_aliases()
    photos = {r['frame']:r for r in read(CONTACTS)['records']}
    cv2.setNumThreads(4)
    spec, sources = [], {}
    for frame, regions in REGIONS.items():
        path = bound_file(photos[frame]['paths']['body'])
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert image.shape == (1520, 2704)
        seeds = []
        for label, (x0,y0,x1,y1) in regions.items():
            points = cv2.goodFeaturesToTrack(image[y0:y1,x0:x1], maxCorners=2, qualityLevel=.01, minDistance=7)
            if points is None:
                continue
            for i, xy in enumerate(points[:,0] + [x0,y0]):
                seeds.append((f'{label}_{i}',xy.tolist()))
        spec.append((int(frame[-6:]), [0,0], seeds))
        sources[str(path)] = sha(path)
    manifest = BASE / 'right-wrist-seeds169.json'
    assert not manifest.exists()
    atomic(manifest,dict(updated=datetime.now(timezone.utc).isoformat(), source_images=sources,
        patches=REGIONS, seeds=spec, prior_visual_sha256=sha(previous), worker_sha256=sha(__file__),
        alias_audit=aliases, retention='All candidate tracks retained. Eligibility requires original patch, split, camera and visual checks.'))
    code = source_path.read_text(encoding='utf-8')
    start,end = code.index('spec=[\n'),code.index('\nrecords={}')
    code = code[:start] + 'spec=' + repr(spec) + code[end:]
    start,end = code.index('        x0,y0,x1,y1=np.asarray(local)'),code.index('        obs={center:seed}')
    code = code[:start] + '        seed=np.asarray(local,dtype=np.float32)\n' + code[end:]
    code = replace(code, "destination=OUT/'shoe-material-local161'", "destination=OUT/'right-wrist-material169'")
    code = replace(code, 'Short-window shoe material measurements can be distinguished from detector label error using floor/chair controls.',
        'Local visible wrist and hand texture may support physical association independently of contradictory wrist aliases.')
    code = replace(code, 'Strongest original corner inside each source-reviewed material ROI; no reprojection in selection.',
        'Up to two native corners per source-checked skin or armrest ROI; coordinates fixed in the seed manifest before tracking.')
    code = replace(code, 'triangulated_from=[a[0],b[0]])',
        '''triangulated_from=[a[0],b[0]],xyz_arbitrary_scale=h[:3].tolist(),
                                  observed_uv={str(i):uv.tolist() for i,uv,_,_ in supported})''')
    sys.argv[1:] = [str(Path(__file__).resolve().parent)]
    exec(compile(code,str(source_path),'exec'),{})
    result = read(OUT / 'result.json')
    assert result['plan_sha256'] == sha(OUT / 'plan.json') and len(result['rows']) == sum(len(s[-1]) for s in spec)
    atomic(OUT / 'source-receipt.json',dict(seed_manifest_sha256=sha(manifest), result_sha256=sha(OUT / 'result.json'),
        unchanged_tracker_sha256=sha(source_path), executed_variant_sha256=__import__('hashlib').sha256(code.encode()).hexdigest(),
        actual_fit_started=False, gaussian_started=False))


if __name__ == '__main__':
    main()
