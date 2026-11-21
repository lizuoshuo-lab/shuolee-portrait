"""One frozen CPU-only four-person input guard preflight; no fitter calls."""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import official_smplifyx_entry123 as entry
import smplify_hand_identity124 as guard
from smplify_observation_validity124 import mask_outside_raster
from smplify_score_contract123 import relative_confidence
from run_official_gaussians123 import read, save, sha, now

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
OUT=ROOT/'hand-identity-guard124'
SUBJECTS=('grass','winterhat','sofa','indoor')


def prepare():
    assert not OUT.exists(), 'One registered candidate; do not overwrite.'
    visual=read(ROOT/'cross-hand-identity-audit/visual-review.json')
    assert visual['selected_native_images_reviewed']==24
    paths=[Path(__file__),Path(guard.__file__),Path(entry.__file__),
           Path(sys.modules[relative_confidence.__module__].__file__),
           Path(sys.modules[mask_outside_raster.__module__].__file__),
           ROOT/'cross-hand-identity-audit/plan.json',ROOT/'cross-hand-identity-audit/result.json',
           ROOT/'cross-hand-identity-audit/visual-review.json']
    paths += [ROOT/(s+'-input')/name for s in SUBJECTS for name in ['data/inputs.json','data/observations.npz','selection.json']]
    save(OUT/'plan.json',dict(updated=now(),state='registered_cpu_only',
        hypothesis='Separated body wrists with overlapping hand clusters and a clearly closer opposite-root association can identify an ambiguous hand chain. Maintain that uncertainty through subsequent detector collapse, rather than forcing a hidden wrist to the visible opposite hand.',
        modification='Confidence-only input adapter: ambiguous body wrist plus its21hand entries become zero. Coordinates and original scores are retained; elbow/shoulder, face/feet and all other entries unchanged. Original SMPLify-X solver, temporal objective and numeric weights unchanged.',
        entry_rule='Mean of the two active21-point hand diameters is s. Body-wrist separation>s, hand-centroid separation<0.5s, affected root distance to own body wrist>distance to opposite body wrist+0.5s, and opposite root distance to its own body wrist<0.5s. Both body/root observations must have positive normalized confidence.',
        recovery_rule='Chronological state clears only when centroid separation>s, both own root errors<0.5s and each own-root association beats its opposite by>0.5s. No coordinate interpolation or synthetic target. Unobserved identity remains unknown.',
        timing='Strict original timestamp order. Identity state does not impose a velocity/speed threshold; the existing fitting temporal loss still uses real dt.',
        fixed_constants=dict(association_margin_half_hand_span=guard.MARGIN),
        risks=['This is a heuristic developed from the reviewed examples, not independent visibility truth or a universally validated classifier.',
               'A wrong body-wrist anchor or inflated hand diameter can miss an error or quarantine a valid hand.',
               'Real hand contact following a previously ambiguous interval may remain withheld until separation; reduced data may worsen limb depth.',
               'Winterhat13 shows a correct hand chain but wrong body wrist without collision; this guard does not solve that case.'],
        historical_controls_not_repeated=['parent-confidence multiplication','root-only suppression'],
        budget=dict(candidates=1,subjects=4,frames=373,cpu_seconds=120,fitter_calls=0,gpu_calls=0,feature_extraction_calls=0),
        protected_visible_cases=dict(grass=[31,32],winterhat=[70,71],sofa=[21,22,25,26,48,49],indoor=[104,105]),
        required_positive_examples=dict(grass={'left':[74,75,76,77]},winterhat={'left':[105]},indoor={'right':[78,82,83]}),
        protection='Every sofa frame must remain unchanged in this conservative preflight, including real contact. Every original reference must be unchanged. If any control or protection fails, retain failure and do not fit or tune thresholds.',
        stopping='One CPU preflight only. Passing algebra/example protection is not pose acceptance and does not authorize default adoption. Actual onset/recovery review and a separate bounded fitting plan are required before any new fit.',
        source_hashes={str(p):sha(p) for p in paths},quality_accepted=False,gaussian_admitted=False))
    print('HAND_IDENTITY_GUARD124_CPU_REGISTERED',flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (OUT/'result.json').exists()
    plan=read(OUT/'plan.json');started=time.monotonic()
    for p,h in plan['source_hashes'].items():assert sha(p)==h,p
    checks=guard.verify();subjects={};passed=True
    dest=np.r_[entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    for subject in SUBJECTS:
        src=ROOT/(subject+'-input')
        rows=read(src/'data/inputs.json')['records']
        with np.load(src/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
        ids=[r['frame'] for r in rows];assert list(obs['frame_ids'].astype(str))==ids
        conf,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        points=np.zeros((len(rows),118,3),np.float32)
        points[:,dest,:2]=obs['keypoints'][:,source]
        points[:,dest,2]=conf[:,source]
        raster=np.concatenate([mask_outside_raster(points[i:i+1],r['wh'])[0] for i,r in enumerate(rows)])
        times=[r['time_seconds'] for r in rows]
        modified,records=guard.identity_guard(raster,times)
        flags=np.array([r['ambiguous'] for r in records])
        allowed=np.zeros(118,bool);allowed[[7,4]]=True;allowed[25:67]=True
        tests=dict(raw_coordinates_unchanged=np.array_equal(modified[...,:2],raster[...,:2]),
                   face_feet_body_unchanged=np.array_equal(modified[:,~allowed],raster[:,~allowed]),
                   confidences_never_increased=bool(np.all(modified[...,2]<=raster[...,2])))
        reference=read(src/'selection.json')['reference_frame']; ri=ids.index(reference)
        tests['reference_unchanged']=np.array_equal(modified[ri],raster[ri])
        tests['visible_examples_unchanged']=all(np.array_equal(modified[n-1],raster[n-1]) for n in plan['protected_visible_cases'][subject])
        if subject=='sofa':tests['all_real_contact_roster_unchanged']=np.array_equal(modified,raster)
        for side,frames in plan['required_positive_examples'].get(subject,{}).items():
            k=('left','right').index(side)
            tests[f'known_{side}_ambiguities_withheld']=all(flags[n-1,k] for n in frames)
        passed &= all(tests.values())
        for i,r in enumerate(records):r['frame']=ids[i]
        changes=np.any(modified[...,2]!=raster[...,2],axis=1)
        np.savez_compressed(OUT/(subject+'-input-preview.npz'),frame_ids=np.array(ids),time_seconds=np.array(times),
                            original_raster_checked=raster,guarded_preview=modified,ambiguous=flags)
        subjects[subject]=dict(frames=len(rows),reference=reference,checks=tests,
            changed_frames=(np.flatnonzero(changes)+1).tolist(),
            left_frames=(np.flatnonzero(flags[:,0])+1).tolist(),
            right_frames=(np.flatnonzero(flags[:,1])+1).tolist(),
            events=[],
            records=records)
        subjects[subject]['events']=[r for i,r in enumerate(records) if any(r['entered']) or (r['recovery_evidence'] and i>0 and any(records[i-1]['ambiguous']))]
        print(subject,'changed',int(changes.sum()),'checks',tests,flush=True)
        assert time.monotonic()-started<plan['budget']['cpu_seconds']
    assert sum(v['frames'] for v in subjects.values())==373
    for p,h in plan['source_hashes'].items():assert sha(p)==h,p
    save(OUT/'result.json',dict(updated=now(),state='cpu_input_preflight_passed' if passed else 'cpu_input_preflight_failed',
        passed=bool(passed),synthetic_checks=checks,subjects=subjects,elapsed_seconds=time.monotonic()-started,
        source_hashes_unchanged=True,original_observations_unchanged=True,existing_fits_unchanged=True,
        default_adopted=False,fitter_calls=0,gpu_calls=0,quality_accepted=False,gaussian_admitted=False))
    print('HAND_IDENTITY_GUARD124_CPU_COMPLETE',bool(passed),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.mode]()
