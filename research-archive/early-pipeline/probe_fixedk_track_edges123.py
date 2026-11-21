"""One seven-pair FIT-only assay and canonical cycle replay, no global camera fit."""
from pathlib import Path
from functools import lru_cache
import os
import time
import traceback
import json
import cv2
import numpy as np
import audit_smplify_observation_contract123 as a
from shuolee_portrait.fixedk_correspondence import fit_pair_model, symmetric_epipolar_px, cycle_supported_subtracks
from shuolee_portrait.material_tracks import mutual_matches

OUT = a.REF / 'solver-track-membership/preunion-geometry-probe'
TID = '82e918800e2525059cba07a2f837fbae3023c1fa09371cd75f34a9b12d1901b7'

def serial(x):
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, dict):
        return {k:serial(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)):
        return [serial(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x

def main():
    assert not OUT.exists(), 'Once-only edge probe already exists; inspect it.'
    OUT.mkdir()
    bridge_root = OUT.parent / 'false-family-bridge'
    bridge = a.read(bridge_root / 'result.json')
    assert a.read(OUT.parent/'feature-binding-replay/result.json')['sampled_saved_binding_reproduced_exactly']
    pairs = bridge['graph_bridges']; assert len(pairs)==7
    appearance = a.read(a.ROOT/'camera/appearance-tracks.json')
    track = next(t for t in appearance['records'] if t['id']==TID)
    rows = a.read(a.ROOT/'data/training-inputs.json')['records']
    nodes = dict(track['nodes'])
    K = np.asarray(a.read(a.ROOT/'intrinsic-contract.json')['raw_K_integer'], float)
    inputs = [a.ROOT/'camera/appearance-tracks.json', a.ROOT/'intrinsic-contract.json',
              a.H/'local_surface123_camera.py', bridge_root/'result.json',
              a.PROJECT/'src/shuolee_portrait/material_tracks.py',
              a.PROJECT/'src/shuolee_portrait/fixedk_correspondence.py']
    inputs += [a.ROOT/f'camera/features-{s:03d}.npz' for s in sorted(set(sum(pairs, [])))]
    hashes = {str(p):a.sha(p) for p in inputs}
    a.save(OUT/'plan.json', dict(registered=a.now(), hypothesis=
        'The initializer unions descriptor-only edges, unlike the existing canonical three-view cycle utility. '
        'Assess cycle support separately from fixed-K two-view geometric checks before changing any global camera or training.',
        scope='All seven graph bridges of the already documented false family, including same-material local bridges; no selection by their reprojection error.',
        pairs=pairs, threshold_px=1.5, minimum_FIT_pairs=30, seed=118, max_ransac_iterations=10000,
        settings_origin='1.5px and seed118 from frozen camera plan; 30 FIT pairs is a preregistered overdetermined support minimum, not selected against held errors.',
        split='Original whole appearance-family FIT/HELD labels frozen. Only original FIT families enter E estimation; every available HELD pair is reported.',
        contrast_A='Existing canonical closed three-view cycle builder on all30 saved family edges, without camera geometry.',
        contrast_B='One independently FIT-only E estimate per selected pair, full unchanged K. No old w2c initialization, K optimization, global SfM or BA.',
        stop='Insufficient/ambiguous pair models remain unavailable. No retries, threshold sweeps or held-based model choice. No partial graph is admitted as a complete camera initializer.',
        budget=dict(cpu_seconds=120, pair_models_max=7, new_feature_extraction=0,
                    global_camera_candidates=0, person_fits=0, gaussian_steps=0), source_hashes=hashes))
    a.save(OUT/'progress.json', dict(state='running', pid=os.getpid(), started=a.now()))
    began=time.monotonic();cv2.setNumThreads(4)
    edges = [[(i,nodes[i]), (j,nodes[j])] for i,j in [e['slots'] for e in bridge['accepted_edges']]]
    cycle = cycle_supported_subtracks([track], edges)
    assert all(t['held'] and t['parent_id']==TID for t in cycle['records'])
    membership={tuple(n):(t['id'],t['held']) for t in appearance['records'] for n in t['nodes']}
    held_ids=[t['id'] for t in appearance['records'] if t['held']]
    @lru_cache(maxsize=18)
    def features(s):
        return dict(np.load(a.ROOT/f'camera/features-{s:03d}.npz'))
    reports=[]
    for i,j in pairs:
        assert time.monotonic()-began<120
        fi,fj=features(i),features(j)
        matches=mutual_matches(fi['descriptors'],fj['descriptors'],ratio=.75)['matches']
        expected=next(p['count'] for p in appearance['pairs'] if p['i']==i and p['j']==j)
        assert len(matches)==expected, 'Saved original matching rule differs; do not continue.'
        fit=[];held=[];unassigned=0
        for ni,nj in matches:
            ma,mb=membership.get((i,int(ni))),membership.get((j,int(nj)))
            if ma is None or mb is None:
                unassigned+=1;continue
            assert ma==mb
            (held if ma[1] else fit).append((ma[0],int(ni),int(nj)))
        aa=np.array([fi['xy'][x] for _,x,_ in fit]).reshape(-1,2)
        bb=np.array([fj['xy'][y] for _,_,y in fit]).reshape(-1,2)
        model=fit_pair_model(aa,bb,K,K,fit_family_ids=[t[0] for t in fit],
                             held_family_ids=held_ids,threshold_px=1.5,minimum_fit=30,seed=118)
        checks=[]
        for ident,x,y in held:
            error = float(symmetric_epipolar_px(model['F'],fi['xy'][x:x+1],fj['xy'][y:y+1])[0]) if 'F' in model else None
            checks.append(dict(parent_id=ident, nodes=[[i,x],[j,y]],
                uv=[fi['xy'][x].tolist(),fj['xy'][y].tolist()], error_px=error,
                geometric_candidate=(error<1.5) if error is not None and model['status']=='two_view_geometric_candidate' else None,
                physical_identity_verified=False))
        target=next(t for t in checks if t['parent_id']==TID)
        reports.append(dict(slots=[i,j], source_frames=[rows[s]['source_frame_index'] for s in [i,j]],
            raw_mutual_matches=len(matches), excluded_unassigned_or_ambiguous_component=unassigned,
            fit_pairs=len(fit), held_pairs=len(held), model=serial(model), held_checks=serial(checks),
            known_family=serial(target)))
        a.save(OUT/'progress.json', dict(state='running',pid=os.getpid(),pairs_completed=len(reports),pairs_total=len(pairs)))
    for p,h in hashes.items():assert a.sha(Path(p))==h
    assert time.monotonic()-began<120
    result=dict(updated=a.now(),source_hashes_unchanged=True, cycle_only=serial(cycle), pairs=reports,
        all_held_errors_retained=True, held_observations_used_for_E=0,
        old_w2c_loaded=False, K_changed=False, global_camera_updated=False,
        physical_identity_accepted=False, geometric_subtracks_admitted=False,
        new_gaussian_steps=0, seconds=time.monotonic()-began)
    a.save(OUT/'result.json',result)
    a.save(OUT/'progress.json',dict(state='complete',pid=None,finished=a.now()))
    print(json.dumps(dict(cycle_only=[dict(nodes=t['nodes'],held=t['held']) for t in cycle['records']],
        pairs=[{k:r[k] for k in ['slots','source_frames','raw_mutual_matches','fit_pairs','held_pairs','known_family']} |
               dict(model_status=r['model']['status'],fit_inliers=r['model'].get('fit_inliers')) for r in reports],
        seconds=result['seconds'])))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            a.save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),auto_retry=False))
            a.save(OUT/'progress.json',dict(state='failed',pid=None,finished=a.now()))
        raise
