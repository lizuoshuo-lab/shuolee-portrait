"""CPU-only input contract candidate. No fitting/defaults/source observations change."""
import os
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read, save, sha, now
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster
import audit_sofa_knee_objective124 as a

OUT = a.ROOT / 'unique-wrist-input124'
PAIRS = ((7, 25, 20), (4, 46, 21))


def unique_wrist_observations(keypoints, mapper):
    """Use body wrist when already valid, otherwise retain the original hand root.

    This is a hypothesis about observation ownership, not physical ground truth.
    Coordinates are never replaced or averaged. Only the redundant root's
    confidence becomes zero. The other20 finger targets retain their confidence.
    """
    x = np.asarray(keypoints)
    if x.ndim != 3 or x.shape[1:] != (118, 3) or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,118,3] observations.')
    if np.any(x[..., 2] < 0) or np.any(x[..., 2] > 1):
        raise ValueError('Use the existing normalized confidence contract.')
    mapper = np.asarray(mapper)
    if mapper.shape != (118,):
        raise ValueError('Expected the original118-anchor mapper.')
    result = x.copy()
    counts = dict(body_primary=0, hand_fallback=0, missing=0, suppressed=0)
    for body, hand, joint in PAIRS:
        if mapper[body] != joint or mapper[hand] != joint:
            raise ValueError('Source semantics differ from the registered duplicate wrist.')
        primary = x[:, body, 2] > 0
        counts['body_primary'] += int(primary.sum())
        counts['hand_fallback'] += int((~primary & (x[:, hand, 2] > 0)).sum())
        counts['missing'] += int((~primary & (x[:, hand, 2] == 0)).sum())
        counts['suppressed'] += int((primary & (x[:, hand, 2] > 0)).sum())
        result[primary, hand, 2] = 0
    return result, counts


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists(), 'One preflight only; preserve evidence.'
    lib = Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/lib/python3.10/site-packages/rtmlib')
    paths = [Path(__file__), Path(a.entry.__file__), a.OFFICIAL/'smplifyx/utils.py',
             a.OFFICIAL/'cfg_files/fit_smplx.yaml', a.PROJECT/'src/shuolee_portrait/observations.py',
             lib/'visualization/skeleton/coco133.py', lib/'tools/pose_estimation/rtmpose.py',
             Path(__file__).with_name('smplify_score_contract123.py'),
             Path(__file__).with_name('smplify_observation_validity124.py'),
             a.ROOT/'hand-observation-coherence/control-verdict.json',
             a.ROOT/'hand-observation-coherence/control-plan.json',
             a.ROOT/'duplicate-wrist-anchor-audit/result.json']
    sources = {s:a.ROOT/(s+'-input') for s in ('winterhat','sofa','indoor')}
    sources['grass'] = Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
    paths += [p for src in sources.values() for p in (src/'data/inputs.json',src/'data/observations.npz')]
    frozen = {str(p):sha(p) for p in paths}
    OUT.mkdir()
    save(OUT/'plan.json', dict(updated=now(), state='registered_cpu_input_preflight',
        hypothesis='A deterministic unique wrist anchor avoids simultaneously fitting inconsistent body and hand-root targets at one exact3D joint. This is an unaccepted local observation contract, not an upstream default or proof of the true wrist location.',
        rule='For both wrists and every frame: body confidence>0 after the existing normalization/raster guard selects the unmodified body observation; suppress only duplicate hand-root confidence. Otherwise keep the unmodified hand root if valid. If neither is valid, no wrist observation. No residual threshold, coordinate synthesis, confidence winner or frame selection.',
        rationale='Keep the existing shoulder-elbow-body-wrist detector chain as the primary body anchor. This structural ownership choice is not evidence that body wrist localization is more accurate.',
        distinction_from_rejected_hand_confidence_control='Prior control multiplied all21 hand confidences by min(body wrist,hand root) and left conflicting roots active. This candidate changes at most two root confidences, with all40 finger targets unchanged.',
        budget=dict(cpu_seconds=120,subjects=4,frames=373,fitter_calls=0,gpu_calls=0),
        protections=['Original files and all2D coordinates bytewise unchanged.','All nonduplicate confidence entries unchanged.','Original temporal loss, prior, optimizer, pose, cameras and model unchanged.','Ignored duplicate observations remain separately available for evaluation.'],
        limits=['Two detector estimates of one joint are not intrinsically an invalid mapping.','No assertion that this is the unique cause of knee drift or ghosting.','No empirical pose-quality benefit established by an input preflight.','Do not suppress the duplicated target from audit/visual review to make quality appear better.'],
        next='Prepare one matched input-only fitting diagnostic with original parameter groups and fixed settings only after this preflight; separately register its implementation hash and body/face/hand/feet protections. Do not restart a rejected sequence or use the discarded target to pick frame-dependent winners.',
        default_applied=False, frozen=frozen))
    from rtmlib.visualization.skeleton.coco133 import coco133
    names = [coco133['keypoint_info'][i]['name'] for i in range(133)]
    assert [names[i] for i in (9,10,91,112)] == ['left_wrist','right_wrist','left_hand_root','right_hand_root']
    utils,_,_,_ = a.entry.load_official(a.OFFICIAL)
    mapper = utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    subjects = {}
    for subject, src in sources.items():
        rows = read(src/'data/inputs.json')['records']
        obs = a.load_npz(src/'data/observations.npz')
        assert obs['joint_names'].astype(str).tolist() == names
        assert obs['frame_ids'].astype(str).tolist() == [r['frame'] for r in rows]
        conf,_ = relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        x = np.zeros((len(rows),118,3),np.float32)
        dst,idx = np.r_[a.entry.COCO,19:25,25:67,67:118],np.r_[0:23,91:133,40:91]
        x[:,dst,:2],x[:,dst,2] = obs['keypoints'][:,idx],conf[:,idx]
        for slot,row in enumerate(rows):
            x[slot:slot+1],_ = mask_outside_raster(x[slot:slot+1],row['wh'])
        q,counts = unique_wrist_observations(x,mapper)
        keep = np.ones(118,bool);keep[[25,46]] = False
        assert np.array_equal(q[...,:2],x[...,:2]) and np.array_equal(q[:,keep],x[:,keep])
        assert np.all(q[...,2] <= x[...,2])
        assert all(not np.any((q[:,i,2]>0)&(q[:,j,2]>0)) for i,j,_ in PAIRS)
        subjects[subject] = dict(frames=len(rows),counts=counts,all_coordinates_unchanged=True,
            nonduplicate_observations_unchanged=True,all_original_observations_retained=True)
    x = np.zeros((3,118,3),np.float32);x[...,2]=1
    x[1,7,2]=0;x[2,[4,46],2]=0
    q,_=unique_wrist_observations(x,mapper)
    checks=dict(body_priority=q[0,7,2]==1 and q[0,25,2]==0,
        original_fallback=q[1,25,2]==1 and q[1,7,2]==0,
        both_missing=q[2,4,2]==0 and q[2,46,2]==0,
        fingers_preserved=np.array_equal(q[:,26:46],x[:,26:46]) and np.array_equal(q[:,47:67],x[:,47:67]))
    incorrect=np.array(mapper);incorrect[25]=19
    try:
        unique_wrist_observations(x,incorrect)
        raise AssertionError('Wrong semantic mapper accepted.')
    except ValueError:
        checks['incorrect_mapper_rejected']=True
    checks={k:bool(v) for k,v in checks.items()};assert all(checks.values())
    for path,digest in frozen.items():assert sha(path)==digest,path
    save(OUT/'preflight.json',dict(updated=now(),state='cpu_contract_passed_not_quality_tested',
        semantic_indices=dict(coco_left=[9,91],coco_right=[10,112],mapped_left=[7,25],mapped_right=[4,46],smplx=[20,21]),
        subjects=subjects,checks=checks,original_source_hashes_verified=True,
        fitter_calls=0,gpu_calls=0,default_applied=False,quality_accepted=False,gaussian_admitted=False))
    print(dict(subjects=subjects,checks=checks),flush=True)


if __name__ == '__main__':
    prepare()
