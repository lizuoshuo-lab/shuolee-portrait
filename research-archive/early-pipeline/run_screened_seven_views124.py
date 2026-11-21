"""One input-roster control: original shared-shape fit on reviewed seven views."""
from pathlib import Path
import inspect,sys,difflib
import numpy as np
import run_grass_selected_views124 as original
import run_shared_shape_views124 as shared
from run_official_gaussians123 import read,save,sha,now

OUT=original.ROOT/'grass-screened-seven124'
OWNERSHIP=original.ROOT/'selected-wrist-ownership124'
BASELINE=shared.OUT


def data():
    rows,target=original.data()
    valid=read(OWNERSHIP/'visual-review.json')['retained_input_frames']
    indices=[i for i,r in enumerate(rows) if r['frame'] in valid]
    assert [rows[i]['frame'] for i in indices]==valid and len(valid)==7
    return [rows[i] for i in indices],target[indices]


# Batch count and input roster only. Upstream stages/loss/weights are unchanged.
scope=shared.scope.copy();scope.update(OUT=OUT,data=data)
code=shared.run_source
for old,new in [('make_models(8)','make_models(len(p["frames"]))'),
    ('adapter.make_entry(8,','adapter.make_entry(len(p["frames"]),'),
    ('shape==(8,10475,3)','shape==(len(p["frames"]),10475,3)'),
    ("state='fitted_eight_views_review_pending'","state='fitted_selected_views_review_pending'")]:
    assert code.count(old)==1,old
    code=code.replace(old,new)
exec(compile(code,'<same-upstream-fit-screened-roster>','exec'),scope)


def prepare():
    assert not OUT.exists(),'Do not rerun this source-roster control.'
    rows,target=data();old=read(BASELINE/'plan.json');review=read(OWNERSHIP/'visual-review.json')
    assert review['rejected_input_frames']==['frame_000085']
    assert old['reference_for_root_seed'] in [r['frame'] for r in rows]
    with np.load(BASELINE/'selected-fit.npz') as z:
        ix=[z['frame_ids'].astype(str).tolist().index(r['frame']) for r in rows]
        assert np.array_equal(z['observed_native_keypoints'][ix],target)
    sources=[Path(__file__),Path(shared.__file__),Path(original.__file__),Path(original.adapter.__file__),
        Path(original.entry.__file__),BASELINE/'plan.json',BASELINE/'result.json',BASELINE/'selected-fit.npz',
        OWNERSHIP/'plan.json',OWNERSHIP/'result.json',OWNERSHIP/'visual-review.json',
        original.SRC/'data/inputs.json',original.SRC/'data/observations.npz',original.SRC/'selection.json',
        original.SRC/'reference-fit/winner-camera-space.npz',original.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    sources += [original.OFFICIAL/f for f in original.entry.PINNED]
    sources += [original.H/'smplify_score_contract123.py',original.H/'smplify_observation_validity124.py']
    for r in rows:
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered',subject='grass',frames=[r['frame'] for r in rows],rows=rows,
        baseline=str(BASELINE),reference_for_root_seed=old['reference_for_root_seed'],
        hypothesis='A visually unsupported far-hand observation in85may pull shared shape/pose. Exclude that whole unresolved input frame, then refit all retained views together from original seeds.',
        only_causal_change='Roster8to7. Same target bytes for retained frames, same original reference32beta seed, zero VPoser pose, root seed46, shared10Dbeta/localpose, per-view placement/face/fingers, K, original5stages/max30/weights/LBFGS. Mean-view normalization unchanged.',
        exclusion_reason='Source image ownership/visibility review, not fitting error. Left hand is not independently visible at its claimed2Dchest target. Neither2Dlabel nor unknown3Dhand position is replaced.',
        user_basis='About8best source-supported frames, not requiring all frames; preserve reproducible filtering and original-repo fitting.',
        original_inputs_unchanged=True,old_optimized_pose_or_shape_used=False,static_gesture_assumption=True,
        risks=['Removing the most side-on input reduces angular support.','The remaining approximately static gesture and conditional K can still be wrong.','Shapes and feet may regress.'],
        source_hashes={str(p):sha(p) for p in sources},budget=dict(candidates=1,selected_views=7,GPU_seconds=600,new_gaussian_steps=0),
        quality_accepted=False,gaussian_admitted=False))
    (OUT/'roster-interface.diff').write_text(''.join(difflib.unified_diff(shared.run_source.splitlines(True),code.splitlines(True),fromfile='eight_view_interface',tofile='selected_view_count')))
    print('SOURCE_SCREENED_SEVEN_REGISTERED',flush=True)


def preflight():
    text=inspect.getsource(shared.preflight)
    text=text.replace('for n in (1,8):','for n in (1,len(plan["frames"])):')
    text=text.replace('if n==8:','if n==len(plan["frames"]):')
    text=text.replace('original_shape_prior_not_divided_by8','original_shape_prior_not_divided_byN')
    ns=shared.__dict__.copy();ns['OUT']=OUT
    exec(compile(text,'<existing-shared-shape-preflight-selected-N>','exec'),ns)
    ns['preflight']()
    p=read(OUT/'preflight.json');rows,target=data()
    p.update(retained_count=len(rows),retained_observations_unchanged=True,removed_frame='frame_000085',
        selector_uses_source_review=True,new_fit_calls=0)
    save(OUT/'preflight.json',p)


def run():
    scope['run']()
    r=read(OUT/'result.json')
    with np.load(OUT/'selected-fit.npz') as z:beta=z['betas'].copy()
    with np.load(original.SRC/'reference-fit/winner-camera-space.npz') as z:seed=z['betas'].copy()
    r.update(beta_before=seed.tolist(),beta_after=beta.tolist(),beta_delta_l2=float(np.linalg.norm(beta-seed)),
        shared_beta_parameter_count=10,shared_shape_updated=True,freeze_betas=False,
        upstream_loss_monitor_optimizer_unchanged=False,actual_reuse='Same original shared-shape interface; only reviewed input roster8to7.',
        quality_accepted=False,gaussian_admitted=False)
    save(OUT/'result.json',r)


if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'run':run}[sys.argv[1]]()
