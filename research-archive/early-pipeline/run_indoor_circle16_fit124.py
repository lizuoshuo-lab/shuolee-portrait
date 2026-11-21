"""Roster-only extension of the existing indoor shared-view SMPLify-X entry.

User rule: half circle approximately8, full circle approximately16.
No new sequence solver, losses, priors, optimizer, or camera interpolation.
"""
from pathlib import Path
from types import FunctionType
import copy
import difflib
import inspect
import os
import sys
import numpy as np
import run_indoor_selected_views124 as prior
from run_official_gaussians123 import read, save, sha, now

SRC = prior.SRC
OUT = prior.original.ROOT/'indoor-circle16-pose124'
BASE = prior.OUT
scope = prior.scope.copy()
scope.update(SRC=SRC, OUT=OUT)
data_source = inspect.getsource(prior.original.data).replace(
    "read(SRC/'selection.json')['training_frames']",
    "read(OUT/'selection.json')['training_frames']")
exec(compile(data_source, '<same-input-mapping-circle16>', 'exec'), scope)
mapped_data = scope['data']


def data():
    rows, targets = mapped_data()
    lookup = {(r['subject'],r['frame']):r for r in read(prior.SFD/'result.json')['records']}
    counts = []
    for row in rows:
        det = lookup['indoor',row['frame']]
        assert det['source_sha256'] == row['source_sha256'] == sha(row['image'])
        counts.append(len(det['associated_indices']))
    revised = prior.face_support.abstain_inner_face(targets,np.asarray(counts,np.int64))
    assert np.array_equal(revised[...,:2],targets[...,:2])
    assert np.array_equal(revised[:,:67],targets[:,:67])
    return rows,revised


scope['data'] = data
exec(compile(prior.runner, '<unchanged-shared-view-runner-circle16>', 'exec'), scope)


def bind(fn):
    env = dict(fn.__globals__)
    env.update(SRC=SRC,OUT=OUT,scope=scope,__file__=__file__)
    return FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)


def shortlist():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUT.exists()
    OUT.mkdir()
    src = read(SRC/'selection.json')
    rows = sorted(read(SRC/'data/inputs.json')['records'],key=lambda r:(r['time_seconds'],r['frame']))
    scores = {r['frame']:r['score'] for r in src['candidates']}
    guards = {r['frame']:r['screen_flags'] for r in read(prior.WRIST)['records']}
    old = read(BASE/'plan.json')['frames']
    chosen = set(old); held = set(src['review_only_frames']);bins=[]
    for bi,group in enumerate(np.array_split(np.arange(len(rows)),16)):
        names=[rows[int(i)]['frame'] for i in group]
        kept=sorted(chosen.intersection(names))
        available=[f for f in names if f not in held and not guards[f]]
        if not kept and available:
            best=min(available,key=lambda f:(scores[f],f));chosen.add(best);kept=[best]
        bins.append(dict(bin=bi,frames=names,selected=kept,eligible=available))
    ids=[r['frame'] for r in rows if r['frame'] in chosen]
    assert len(ids)==16 and set(old)<=set(ids) and not set(ids)&held
    assert all(not guards[f] for f in ids)
    save(OUT/'selection.json',dict(updated=now(),subject='indoor',capture_arc='full_circle',
        user_rule=dict(half_circle_target=8,full_circle_target=16,nominal_spacing_degrees=22.5),
        training_frames=ids,reference_frame=src['reference_frame'],review_only_frames=src['review_only_frames'],
        preserved_six=old,additional_frames=[f for f in ids if f not in old],bins=bins,
        rule='Preserve six previous inputs.16equal-population chronological bins; fill empty bins using the same frozen prefit score among wrist-clear non-heldout frames, tie by frame ID.',
        chronological_proxy_only=True,actual_angular_coverage_requires_fit_review=True,
        no_interpolated_training_camera=True,quality_accepted=False,gaussian_admitted=False))
    env=dict(prior.base.__dict__,SRC=SRC,OUT=OUT,all_data=data)
    code=inspect.getsource(prior.base.source_review)
    changes=[("and not OUT.exists()","and not (OUT/'source-review.json').exists()"),
             ('assert len(rows)==8','assert len(rows)==16'),('OUT.mkdir();','OUT.mkdir(exist_ok=True);'),
             ("/mnt/d/00/记录/006","/mnt/d/00/记录/008"),('沙发八帧124','室内一圈16帧124'),
             ('for start in (0,4):','for start in range(0,len(tiles),4):')]
    for a,b in changes:
        assert code.count(a)==1,a
        code=code.replace(a,b)
    exec(compile(code,'<same-raw-observation-review16>','exec'),env)
    env['source_review']()
    (OUT/'source-review-interface.diff').write_text(''.join(difflib.unified_diff(
        inspect.getsource(prior.base.source_review).splitlines(True),code.splitlines(True))))
    print('FULL_CIRCLE16_SHORTLIST_READY',ids,flush=True)


def prepare():
    assert not (OUT/'plan.json').exists()
    review=read(OUT/'source-review.json');assert review['actual_visual_review'] and review['static_gesture_plausible']
    sel=read(OUT/'selection.json');rows,targets=data()
    assert [r['frame'] for r in rows]==sel['training_frames'] and len(rows)==16
    p=copy.deepcopy(read(BASE/'plan.json'))
    p.update(updated=now(),state='registered',frames=sel['training_frames'],rows=rows,
        reference_for_root_seed=sel['reference_frame'],review_only_excluded=sel['review_only_frames'],
        source_rejected_frames=[],source_exclusion_before_any_fit=True,
        hypothesis='Check full-circle16view support instead of sparse6; existing selected-view original-repo interface unchanged.',
        user_basis='User explicitly requests half-circle8/full-circle16 after pointing out insufficient view support.',
        scope='Input roster and shared fitted initialization change together. Compare old6only as a pipeline comparison; do not claim an isolated Gaussian-view-count effect.',
        source_selection=str(OUT/'selection.json'),previous_six_fit_used_as_initialization=False,
        initialization='Same own reference105 beta seed, trainable shared beta; original zero VPoser and all-front root initialization. No fitted six-view pose or Gaussian reused.',
        algorithm='Existing shared10Dbeta/shared32DVPoser, per-view root/translation/face/hands. Same five original stages, max30, priors, weights and LBFGS.',
        budget=dict(candidates=1,selected_views=16,GPU_seconds=600,new_gaussian_steps=0),
        stop='One16view fit and actual all16source/mesh review; no frame winner mixing, no refitting threshold sweep. Review coverage before Gaussian handoff.',
        next='If no new severe failure, preserve one coherent16view initialization for a registered fresh SH0 Gaussian run; full quality remains unaccepted.',
        quality_accepted=False,gaussian_admitted=False)
    paths=[Path(x) for x in p['source_hashes']]
    paths += [Path(__file__),OUT/'selection.json',OUT/'source-review.json',Path(prior.__file__)]
    for row in rows:
        for key,hkey in [('image','source_sha256'),('mask','mask_sha256')]:
            assert sha(row[key])==row[hkey]
            paths.append(Path(row[key]))
    p['source_hashes']={str(x):sha(x) for x in paths}
    save(OUT/'plan.json',p)
    (OUT/'executed-runner.py.txt').write_text(prior.runner)
    (OUT/'data-interface.diff').write_text(''.join(difflib.unified_diff(
        inspect.getsource(prior.original.data).splitlines(True),data_source.splitlines(True))))
    print('FULL_CIRCLE16_FIT_REGISTERED',flush=True)


preflight=bind(prior.base.preflight)
run=bind(prior.base.run)
launch=bind(prior.base.launch)

if __name__=='__main__':
    {'shortlist':shortlist,'prepare':prepare,'preflight':preflight,'run':run,'launch':launch}[sys.argv[1]]()
