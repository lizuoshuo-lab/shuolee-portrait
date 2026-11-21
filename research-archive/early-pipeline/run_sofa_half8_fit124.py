"""Eight-view roster update; reuse the existing shared SMPLify-X solver."""
from pathlib import Path
from types import FunctionType
import copy, difflib, inspect, os, sys
import numpy as np
import run_sofa_selected_views124 as prior
from run_official_gaussians123 import read, save, sha, now

SRC=prior.SRC
BASE=prior.OUT
OUT=prior.original.ROOT/'sofa-half8-pose124'
AUDIT=prior.original.ROOT/'sofa-half8-input-audit124'
scope=prior.scope.copy();scope.update(SRC=SRC,OUT=OUT)
data_source=inspect.getsource(prior.original.data).replace(
    "read(SRC/'selection.json')['training_frames']", "read(OUT/'selection.json')['training_frames']")
exec(compile(data_source,'<same-sofa-eight-input-mapping>','exec'),scope)
exec(compile(prior.runner,'<unchanged-shared-sofa-eight-runner>','exec'),scope)

def bind(fn):
    env=dict(fn.__globals__);env.update(SRC=SRC,OUT=OUT,scope=scope,__file__=__file__)
    return FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)

def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    a=read(AUDIT/'result.json');ap=read(AUDIT/'plan.json')
    assert ap['candidate_priority']==['frame_000056','frame_000055','frame_000054','frame_000001','frame_000052','frame_000021']
    decisions=[]
    for r in a['records']:
        assert sha(r['body'])==r['body_sha256'] and sha(r['hands'])==r['hands_sha256']
        f=r['frame'];keep=f=='frame_000021'
        reason=('Visible wrists, near hand and body are source-consistent. Far fingers overlap and remain uncertain; not verified physical landmarks. No clear sleeve/trouser assignment as in rejected candidates.' if keep else
                'Occluded far-hand finger detections lie on visible sleeve/trousers, unsupported by the source image.')
        decisions.append(dict(frame=f,actual_visual_review=True,decision='retained_exploratory' if keep else 'rejected',reason=reason))
    save(AUDIT/'visual-review.json',dict(updated=now(),actual_visual_review=True,records=decisions,
        selected_additional_frame='frame_000021',selection_before_fit=True,physical_joint_truth=False,
        coverage_limitation='Added frontal support, not a new extreme-side angle. No back is observed; half-circle is capture scope, not validated uniform angular spacing.',
        source_targets_and_confidence_unchanged=True,quality_accepted=False))
    old=read(BASE/'source-review.json');sel=read(SRC/'selection.json')
    ids=sorted(old['retained_input_frames']+['frame_000021'])
    assert len(ids)==8 and not set(ids)&set(sel['review_only_frames']) and 'frame_000053' not in ids
    OUT.mkdir()
    save(OUT/'selection.json',dict(updated=now(),subject='sofa',capture_arc='front_half_or_less',
        user_rule=dict(half_circle_target=8,full_circle_target=16),training_frames=ids,
        reference_frame=sel['reference_frame'],review_only_frames=sel['review_only_frames'],
        preserved_seven=old['retained_input_frames'],added_frame='frame_000021',
        rule=ap['candidate_rank_rule'],input_audit=str(AUDIT/'visual-review.json'),
        no_interpolated_camera=True,angular_uniformity_not_certified=True,quality_accepted=False))
    review=copy.deepcopy(old)
    review.update(updated=now(),retained_input_frames=ids,
        rejected_input_frames=sorted(set(old['rejected_input_frames'])|{x['frame'] for x in decisions if x['decision']=='rejected'}),
        supplemental_review=str(AUDIT/'visual-review.json'),actual_visual_review=True,
        previous_seven_review_reused=True,physical_joint_truth=False)
    r=next(r for r in a['records'] if r['frame']=='frame_000021')
    review['records']=[r for r in review['records'] if r['frame'] in ids]+[dict(r,actual_visual_review=True,source_decision='retained_exploratory')]
    save(OUT/'source-review.json',review)
    rows,target=scope['data']();assert [r['frame'] for r in rows]==ids and len(rows)==8
    guards={r['frame']:r['screen_flags'] for r in read(prior.original.ROOT/'selected-observation-audit124/wrist-preselection/sofa.json')['records']}
    assert all(not guards[f] for f in ids)
    p=copy.deepcopy(read(BASE/'plan.json'))
    p.update(updated=now(),state='registered',frames=ids,rows=rows,
        source_rejected_frames=review['rejected_input_frames'],source_selection=str(OUT/'selection.json'),
        hypothesis='Transfer the registered eight-view half-circle roster to the same shared SMPLify-X and fresh SH0 pipeline.',
        user_basis='Half circle8, full circle16; reproducible selected observations, not making all frames pass.',
        scope='Roster, joint fit and current Gaussian profile are a pipeline comparison; no isolated view-count causal claim.',
        initialization='Own reference26 beta seed remains trainable. Original zero VPoser and all-front roots. No previous optimized seven-view poses, Gaussian or optimizer reused.',
        budget=dict(candidates=1,selected_views=8,GPU_seconds=600,new_gaussian_steps=0),
        stop='One eight-view fit and all-source/native/side review. No roster or parameter sweeps.',
        next='Fresh eight-view SH0 exploratory training only after actual review excludes new gross anatomical failures.',
        quality_accepted=False,gaussian_admitted=False)
    paths=[Path(x) for x in p['source_hashes']]+[Path(__file__),OUT/'selection.json',OUT/'source-review.json',AUDIT/'visual-review.json']
    for row in rows:
        assert row['wh']==[2704,1520] and row['K_integer_centers']==rows[0]['K_integer_centers'] and np.array_equal(row['w2c'],np.eye(4))
        for key,hkey in [('image','source_sha256'),('mask','mask_sha256')]:
            assert sha(row[key])==row[hkey];paths.append(Path(row[key]))
    p['source_hashes']={str(x):sha(x) for x in paths};save(OUT/'plan.json',p)
    (OUT/'executed-runner.py.txt').write_text(prior.runner)
    (OUT/'data-interface.diff').write_text(''.join(difflib.unified_diff(inspect.getsource(prior.original.data).splitlines(True),data_source.splitlines(True))))
    print('SOFA_HALF8_REGISTERED',ids,flush=True)

preflight=bind(prior.preflight)
run=bind(prior.run)
launch=bind(prior.launch)
if __name__=='__main__':{'prepare':prepare,'preflight':preflight,'run':run,'launch':launch}[sys.argv[1]]()
