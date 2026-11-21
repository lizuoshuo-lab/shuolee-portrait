"""Opt-in absence-of-detection ablation; never a full visibility classifier."""
import os
import time
from pathlib import Path
import numpy as np


def abstain_inner_face(mapped, associated_count):
    """Withhold face51 only for zero support; do not promote any other points.

    The caller must verify the stored detector/frame/image hash association.
    Missing detection is insufficient evidence, not proof of physical absence.
    One/multiple detections leave the previous input unchanged and unresolved.
    """
    x=np.asarray(mapped)
    if x.ndim!=3 or x.shape[1:]!=(118,3) or not np.isfinite(x).all():
        raise ValueError('Expected finite original118 x,y,confidence observations.')
    n=np.asarray(associated_count)
    if n.shape!=(len(x),) or n.dtype.kind not in 'iu' or np.any(n<0):
        raise ValueError('Supply an explicit nonnegative detection count per frame.')
    if np.any(x[...,2]<0) or np.any(x[...,2]>1):
        raise ValueError('Expected the unchanged relative confidence convention.')
    out=x.copy()
    out[n==0,67:118,2]=0
    return out


def preflight():
    from run_official_gaussians123 import read,save,sha,now
    from smplify_score_contract123 import relative_confidence
    from smplify_observation_validity124 import mask_outside_raster
    import audit_sofa_knee_objective124 as a
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=a.ROOT/'face-absence-input124'
    assert not out.exists()
    sfd=a.ROOT/'sfd-face-presence124/full167'
    review=read(sfd/'visual-review.json')
    assert review['actual_visual_review'] and not review['input_gate_admitted']
    records=read(sfd/'result.json')['records']
    lookup={(r['subject'],r['frame']):r for r in records}
    assert len(lookup)==len(records)==167
    sources=[Path(__file__),Path(a.entry.__file__),sfd/'result.json',sfd/'visual-review.json',
             a.OFFICIAL/'smplifyx/utils.py',Path(__file__).with_name('smplify_score_contract123.py'),
             Path(__file__).with_name('smplify_observation_validity124.py')]
    sources += [a.ROOT/(s+'-input')/'data'/f for s in ('sofa','indoor') for f in ('inputs.json','observations.npz')]
    frozen={str(p):sha(p) for p in sources}
    out.mkdir()
    save(out/'plan.json',dict(updated=now(),state='registered_input_only',
        hypothesis='Withholding the inner-face51 data term when an unchanged full-native SFD detector has no associated face may avoid fitting a face onto a posterior head. This is an ablation candidate, not a complete visibility gate.',
        rule='Zero the original118-layout indices67:118 confidence only when the frozen tracked-person-associated count is zero. Keep every coordinate, body/head5/hand/foot target and all positive/multiple-detection input weights exactly unchanged. Do not promote detections to true visible landmarks.',
        input_association='Exact subject,frame,image SHA; no inferred absent entry, ROI remapping, maximum-score choice or old detector fallback.',
        budget=dict(cpu_seconds=120,frames=167,new_detector_calls=0,fitter_calls=0,gpu_calls=0),
        risks=['A detector false negative could remove genuine facial evidence.','Ear/side-face false positives and partially hidden landmarks remain unresolved.','Body nose/eye/ear targets are deliberately unchanged in this single-group ablation.','Sofa shoulder failure cannot be explained by no-face detection because every sofa frame has face support.'],
        protections=['All original files and raw observations retained.','No camera, shape, pose, hand-root, solver, prior, temporal or optimization changes.','No full sequence or Gaussian admission from this CPU contract.'],
        default_applied=False,frozen=frozen))
    started=time.monotonic()
    utils,_,_,_=a.entry.load_official(a.OFFICIAL)
    common=dict(model_type='smplx',use_hands=True,use_face_contour=False,openpose_format='coco25')
    whole=utils.smpl_to_openpose(use_face=True,**common)
    bodyhands=utils.smpl_to_openpose(use_face=False,**common)
    assert len(whole)==118 and len(bodyhands)==67 and np.array_equal(whole[:67],bodyhands)
    subjects={}
    for subject in ('sofa','indoor'):
        src=a.ROOT/(subject+'-input/data');rows=read(src/'inputs.json')['records']
        obs=a.load_npz(src/'observations.npz')
        assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
        confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        mapped=np.zeros((len(rows),118,3),np.float32)
        dst,idx=np.r_[a.entry.COCO,19:25,25:67,67:118],np.r_[0:23,91:133,40:91]
        mapped[:,dst,:2]=obs['keypoints'][:,idx];mapped[:,dst,2]=confidence[:,idx]
        count=[]
        for i,row in enumerate(rows):
            r=lookup[subject,row['frame']]
            assert r['source_sha256']==row['source_sha256']==sha(row['image'])
            mapped[i:i+1],_=mask_outside_raster(mapped[i:i+1],row['wh'])
            count.append(len(r['associated_indices']))
        count=np.asarray(count,np.int64)
        revised=abstain_inner_face(mapped,count)
        assert np.array_equal(revised[...,:2],mapped[...,:2])
        assert np.array_equal(revised[:,:67],mapped[:,:67])
        assert np.array_equal(revised[count>0],mapped[count>0])
        assert np.all(revised[...,2]<=mapped[...,2])
        assert np.array_equal(abstain_inner_face(revised,count),revised)
        target=out/(subject+'-diagnostic-input.npz')
        np.savez_compressed(target,frame_ids=np.asarray([r['frame'] for r in rows]),
            original_mapped=mapped,candidate_mapped=revised,associated_count=count)
        subjects[subject]=dict(frames=len(rows),abstained_frames=[rows[i]['frame'] for i in np.flatnonzero(count==0)],
            changed_confidence_entries=int(np.count_nonzero(revised[...,2]!=mapped[...,2])),
            all_coordinates_unchanged=True,all_body_hand_foot_data_unchanged=True,
            supported_and_ambiguous_inputs_unchanged=True,diagnostic_input=str(target),sha256=sha(target))
    for path,digest in frozen.items():assert sha(path)==digest,path
    seconds=time.monotonic()-started;assert seconds<120
    next_action='The opt-in face51 absence adapter passed all167 immutable-input checks; it is not deployed or quality accepted. Read full167/visual-review.json before any use. Next register one bounded matched indoor diagnostic comparing the original versus candidate face51 data, using the same existing reference beta, original pose solver/camera initialization and unchanged priors; preserve all original points for evaluation and visible-body/face protections. Do not run111-frame or GS directly, sweep detection/temporal weights, restart failed sofa/winterhat sequences, or treat current fitted3D joints as independent truth. Positive/profile and ambiguous cases are still unresolved; lower-body occlusion is separate.'
    save(out/'preflight.json',dict(updated=now(),state='cpu_contract_passed_not_fitted',seconds=seconds,
        subjects=subjects,source_hashes_verified=True,original_mapper_prefix_verified=True,
        fitting_calls=0,new_detector_calls=0,gpu_calls=0,default_applied=False,
        quality_accepted=False,gaussian_admitted=False,next_action=next_action))
    print('FACE_ABSENCE_INPUT_PREFLIGHT',subjects,flush=True)


if __name__=='__main__':preflight()
