"""Reuse the original single-reference fitter for the two remaining people.

Only measured-raster validity is added at the observation interface. No new
solver, visibility estimator, temporal tuning, or old individual geometry.
"""
import argparse
from pathlib import Path
import numpy as np
import prepare_winterhat_sh0124 as original
import official_smplifyx_entry123 as entry
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster, verify
from run_official_gaussians123 import read, save, sha, now

HERE=Path(__file__).resolve().parent


def context(subject):
    assert subject in ('sofa','indoor')
    root=original.REC/'cross-subject-sh0-124'/(subject+'-input')
    selection=read(root/'selection.json')
    rows=read(root/'data/inputs.json')['records']
    with np.load(root/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    slot=[r['frame'] for r in rows].index(selection['reference_frame'])
    row=rows[slot]
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    target=np.zeros((1,118,3),np.float32)
    dst=np.r_[entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    target[0,dst,:2]=obs['keypoints'][slot,src]
    target[0,dst,2]=confidence[slot,src]
    return root,selection,row,target


def prepare(subject):
    root,selection,row,target=context(subject)
    assert not (root/'reference-plan.json').exists() and not (root/'reference-fit').exists()
    assert read(root/'mask-review/result.json')['actual_visual_review']
    changed,inside=mask_outside_raster(target,row['wh'])
    source_paths=[Path(__file__),Path(original.__file__),Path(entry.__file__),
        HERE/'smplify_observation_validity124.py',HERE/'smplify_score_contract123.py',
        HERE/'run_official_sequence123_topology.py',root/'selection.json',
        root/'data/inputs.json',root/'data/observations.npz',root/'length-contract.json']
    plan=dict(updated=now(),subject=subject,state='reference_prepared_not_started',
        route='Existing original SMPLify-X reference fit; no VHAP or network body initialization.',
        reference_frame=selection['reference_frame'],reference_selected_before_fit=True,
        source_roster=selection['source_roster'],checks=verify(),
        input_change='Only confidence outside explicit integer-center raster support is zero. All coordinates, valid weights, priors, full K and solver settings unchanged.',
        reference_zeroed_indices=np.flatnonzero((target[0,:,2]>0)&(~inside[0])).tolist(),
        reference_pre_guard_confidence=target[0,:,2].tolist(),
        reference_applied_confidence=changed[0,:,2].tolist(),
        limits=['Raster membership is not visibility. Furniture, image truncation and mask holes remain separately unresolved.',
                'Reference beta is a subject-shape initializer, not final quality acceptance or physical calibration.',
                'Full chronology must be fitted and reviewed before the eight selected Gaussian RGB frames are used.'],
        budget=dict(reference_fits=1,original_maxiters_per_stage=30,original_stages=5),
        no_overwrite=True,no_new_gpu_until_lock_available=True,gaussian_admitted=False,
        sources={str(p):sha(p) for p in source_paths})
    save(root/'reference-plan.json',plan)
    print('CORE_REFERENCE_PREPARED',subject,selection['reference_frame'],plan['reference_zeroed_indices'],flush=True)


def reference(subject):
    root,selection,row,expected=context(subject);plan=read(root/'reference-plan.json')
    for path,digest in plan['sources'].items():assert sha(path)==digest,path
    assert not (root/'reference-fit').exists()
    previous_fit,previous_root=entry.fit_one,original.ROOT
    def fit(**kwargs):
        assert np.array_equal(kwargs['keypoints'],expected)
        guarded,inside=mask_outside_raster(kwargs['keypoints'],row['wh'])
        # Mutate only this in-memory derived target so the entry's saved
        # observed_native_keypoints also records the actual fitting weights.
        kwargs['keypoints'][...]=guarded
        arrays,receipt=previous_fit(**kwargs)
        receipt['raster_validity']=dict(applied=True,zeroed_indices=plan['reference_zeroed_indices'],
            raw_coordinates_unchanged=True,original_observations_sha256=sha(root/'data/observations.npz'),
            physical_visibility_proven=False)
        return arrays,receipt
    entry.fit_one=fit;original.ROOT=root
    try:
        original.fit_reference()
        for path,digest in plan['sources'].items():assert sha(path)==digest,path
    finally:
        entry.fit_one,original.ROOT=previous_fit,previous_root


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','reference'])
    parser.add_argument('subject',choices=['sofa','indoor']);args=parser.parse_args()
    (prepare if args.mode=='prepare' else reference)(args.subject)
