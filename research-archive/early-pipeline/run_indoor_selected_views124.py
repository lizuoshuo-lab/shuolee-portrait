"""One indoor use of the existing shared-shape fitter; no new solver."""
from pathlib import Path
from types import FunctionType
import inspect, sys, os, difflib
import numpy as np
import run_sofa_selected_views124 as base
import smplify_face_support124 as face_support
from run_official_gaussians123 import read, save, sha, now

original, shared = base.original, base.shared
SRC = original.ROOT/'indoor-input'
OUT = original.ROOT/'indoor-selected-view-pose124'
SFD = original.ROOT/'sfd-face-presence124/full167'
WRIST = original.ROOT/'selected-observation-audit124/wrist-preselection/indoor.json'
scope = shared.scope.copy()
scope.update(SRC=SRC, OUT=OUT)
exec(compile(inspect.getsource(original.data), '<same-indoor-selected-input>', 'exec'), scope)
all_data = scope['data']


def data():
    rows, target = all_data()
    review = read(OUT/'source-review.json')
    allowed = review['retained_input_frames']
    indices = [i for i, row in enumerate(rows) if row['frame'] in allowed]
    rows, target = [rows[i] for i in indices], target[indices]
    assert [r['frame'] for r in rows] == allowed
    lookup = {(r['subject'], r['frame']): r for r in read(SFD/'result.json')['records']}
    counts = []
    for row in rows:
        detection = lookup['indoor', row['frame']]
        assert detection['source_sha256'] == row['source_sha256'] == sha(row['image'])
        counts.append(len(detection['associated_indices']))
    revised = face_support.abstain_inner_face(target, np.asarray(counts, np.int64))
    assert np.array_equal(revised[..., :2], target[..., :2])
    assert np.array_equal(revised[:, :67], target[:, :67])
    return rows, revised


scope['data'] = data
runner = base.runner
# Observation-only statistics must mark an unsupported group unknown.
# This does not change the upstream fitting objective or point confidence.
old = "float(np.sqrt(np.sum(w*error[ix]**2)/w.sum()))"
assert runner.count(old) == 1
runner = runner.replace(old, "(float(np.sqrt(np.sum(w*error[ix]**2)/w.sum())) if w.sum()>0 else None)")
old = "float(error[ix][w>0].max())"
assert runner.count(old) == 1
runner = runner.replace(old, "(float(error[ix][w>0].max()) if (w>0).any() else None)")
exec(compile(runner, '<same-shared-fitter-indoor>', 'exec'), scope)


def bind(fn):
    environment = dict(fn.__globals__)
    environment.update(SRC=SRC, OUT=OUT, scope=scope, __file__=__file__)
    return FunctionType(fn.__code__, environment, fn.__name__, fn.__defaults__, fn.__closure__)


def source_review():
    environment = dict(base.__dict__, SRC=SRC, OUT=OUT, all_data=all_data)
    code = inspect.getsource(base.source_review)
    code = code.replace('/mnt/d/00/记录/006', '/mnt/d/00/记录/008')
    code = code.replace('沙发八帧124', '室内共享六帧输入124')
    exec(compile(code, '<same-source-review-indoor>', 'exec'), environment)
    environment['source_review']()
    review = read(OUT/'source-review.json')
    selection = read(SRC/'selection.json')
    guards = {r['frame']: r['screen_flags'] for r in read(WRIST)['records']}
    review.update(subject='indoor', original_selected_frames=selection['training_frames'],
        retained_input_frames=[f for f in selection['training_frames'] if not guards[f]],
        rejected_input_frames=[f for f in selection['training_frames'] if guards[f]],
        selection_before_fit=True, selection_rule='Existing frozen wrist-consistency flags only; no replacement from heldout or loss-ranked frames.',
        inner_face_rule='Existing negative-only absence adapter, indices67:118 confidence zero iff associatedSFDcount=0; all coordinates/body/hand/foot targets unchanged.',
        gaussian_admitted=False)
    save(OUT/'source-review.json', review)


def prepare():
    assert not (OUT/'plan.json').exists() and not (OUT/'progress.json').exists()
    review = read(OUT/'source-review.json')
    assert review['actual_visual_review'] and review['static_gesture_plausible']
    selection = read(SRC/'selection.json'); rows, target = data()
    ids = [r['frame'] for r in rows]
    assert ids == review['retained_input_frames'] and len(ids) == 6
    assert set(ids) < set(selection['training_frames'])
    assert not set(ids) & set(selection['review_only_frames'])
    reference = selection['reference_frame']; assert reference in ids
    for row in rows:
        assert row['wh'] == [2704, 1520] and row['K_integer_centers'] == rows[0]['K_integer_centers']
        assert np.array_equal(row['w2c'], np.eye(4))
        assert sha(row['image']) == row['source_sha256'] and sha(row['mask']) == row['mask_sha256']
    sources = [Path(__file__), Path(base.__file__), Path(original.__file__), Path(shared.__file__),
        Path(original.adapter.__file__), Path(original.entry.__file__), Path(face_support.__file__),
        original.H/'smplify_score_contract123.py', original.H/'smplify_observation_validity124.py',
        SRC/'data/inputs.json', SRC/'data/observations.npz', SRC/'selection.json',
        SRC/'reference-fit/winner-camera-space.npz', SRC/'reference-review/result.json',
        OUT/'source-review.json', WRIST, SFD/'result.json', SFD/'visual-review.json',
        original.ROOT/'face-absence-input124/preflight.json', original.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    sources += [original.OFFICIAL/f for f in original.entry.PINNED]
    import smplx
    sources += [Path(smplx.__file__).parent/'body_models.py', Path(smplx.__file__).parent/'lbs.py']
    save(OUT/'plan.json', dict(updated=now(),state='registered',subject='indoor',frames=ids,rows=rows,
        reference_for_root_seed=reference, review_only_excluded=selection['review_only_frames'],
        source_rejected_frames=review['rejected_input_frames'], source_exclusion_before_any_fit=True,
        hypothesis='Transfer the already-used joint selected-view shared-shape interface to the approximately constant hands-on-waist gesture, with frozen prefit wrist screening and the existing negative-only face51 support guard.',
        scope='Cross-person pipeline initialization. The combined input screening and joint fit cannot establish a single-cause improvement.',
        user_basis='Continue more people, choose approximately8good distributed frames, use reproducible minimal original-repo interfaces and proceed toGS rather than repeated per-frame pose repair.',
        algorithm='Same original5stages,max30,LBFGS,VPoser2,bending/L2 priors and mean-view/shared10Dbeta/shared32Dbody interface; per-view root/translation/face/fingers.',
        initialization='Own original reference105 beta seed only, trainable shared beta; original zero body latent and standard front root. No old optimized body pose or Gaussian.',
        source_camera_identity_placeholder=True, original_K_fixed=True, physical_camera_calibration=False,
        root_seed_warning='All views start front-facing because source extrinsics are identity placeholders; side/back root convergence is uncertain. No invented scene camera or detector angle.',
        inner_face_absence_adapter_opt_in=True, default_visibility_gate_promoted=False,
        original_coordinates_body_hand_foot_preserved=True, empty_support_statistics='null, never zero error',
        risks=['Static local pose can suppress real motion.', 'Positive face detections do not prove landmark visibility.',
               'Back-view false body-head points remain unchanged.', 'Cropped and furniture-hidden legs are unknown; this fitter is not a silhouette/occlusion solution.'],
        budget=dict(candidates=1,selected_views=6,GPU_seconds=600,new_gaussian_steps=0),
        stop='One fit and actual six-view plus synthetic-side review. No weight/roster/init sweep or loss-based best-frame mixture.',
        next='Only after actual review, decide whether geometry and valid image supervision permit exploratory GS; never label furniture-hidden legs empty.',
        quality_accepted=False,gaussian_admitted=False,source_hashes={str(p):sha(p) for p in sources}))
    (OUT/'executed-runner.py.txt').write_text(runner)
    (OUT/'metric-only-interface.diff').write_text(''.join(difflib.unified_diff(base.runner.splitlines(True),runner.splitlines(True))))
    _, delta = original.adapter.source(original.OFFICIAL); (OUT/'upstream-interface.diff').write_text(delta)
    _, delta = original.adapter.make_entry(len(rows), original.root_initialization(rows,reference)); (OUT/'entry-interface.diff').write_text(delta)
    print('INDOOR_SHARED_SIX_REGISTERED', ids, flush=True)


preflight = bind(base.preflight)
run = bind(base.run)
launch = bind(base.launch)

if __name__ == '__main__':
    {'source-review':source_review, 'prepare':prepare, 'preflight':preflight, 'run':run, 'launch':launch}[sys.argv[1]]()
