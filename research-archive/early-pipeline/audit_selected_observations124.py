"""Read-only observation-contract audit on already fitted selected frames."""
from pathlib import Path
from datetime import datetime, timezone
import time
import sys
import numpy as np
import audit_gaussianavatars_params123 as a
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster

ROOT = a.REC/'cross-subject-sh0-124'
OUT = ROOT/'selected-observation-audit124'
COCO = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
NAMES = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
         'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
         'left_knee','right_knee','left_ankle','right_ankle']
SUBJECTS = [('grass','grass-protected-refine124','grass-temporal-refinement'),
            ('winterhat','winterhat-selected-view-pose124','winterhat-temporal')]


def main():
    correction = '--correct-support' in sys.argv
    if correction:
        assert a.lp(OUT/'result.json').exists() and not a.lp(OUT/'result-completed.json').exists()
    else:
        assert not a.lp(OUT).exists(), 'This diagnostic is bounded; read its existing result.'
    start = time.monotonic()
    sources = [Path(__file__),Path(__file__).with_name('smplify_score_contract123.py'),
        Path(__file__).with_name('smplify_observation_validity124.py'),
        Path(__file__).with_name('select_quality_frames124.py'),
        a.REC/'smplifyx-reference-control/official/cfg_files/fit_smplx.yaml']
    for subject,fit,temporal in SUBJECTS:
        sources += [ROOT/f'{subject}-input/data/observations.npz',ROOT/f'{subject}-input/data/inputs.json',
            ROOT/fit/'selected-fit.npz',ROOT/fit/'conclusion.json',ROOT/fit/'selection-reviewed.json',
            ROOT/temporal/'official-fit/data/training-inputs.json']
    if correction: sources.append(OUT/'result.json')
    hashes = {str(p):a.sha(p) for p in sources}
    a.lp(OUT).mkdir(exist_ok=correction)
    a.save(OUT/('support-correction-plan.json' if correction else 'plan.json'),dict(updated=datetime.now(timezone.utc).isoformat(),
        state='registered',scope='Existing16selected frames, CPU read-only observation identity, mapping and confidence audit.',
        correction='Match the real selector positive squared float32 weight support. Original diagnostic used unsquared positive confidence; original result is retained.' if correction else None,
        budget=dict(CPU_seconds=120,candidates=1,GPU_fits=0,new_training_steps=0),source_hashes=hashes,
        prohibited=['No confidence threshold changes','No point removal','No modifying fits or selector gates',
                    'No new detector or feature extraction','No per-frame optimized-parameter mixing']))
    all_rows = []
    subjects = []
    for subject, fit, temporal in SUBJECTS:
        data = ROOT/f'{subject}-input/data'
        rows = a.read(data/'inputs.json')['records']
        measures = {r['frame']:r for r in a.read(ROOT/temporal/'official-fit/data/training-inputs.json')['records']}
        with np.load(a.lp(data/'observations.npz'),allow_pickle=False) as z: obs = dict(z)
        ids = obs['frame_ids'].astype(str).tolist()
        assert ids == [r['frame'] for r in rows]
        confidence,_ = relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        target = np.zeros((len(rows),118,3),np.float32)
        dst = np.r_[COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
        src = np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
        target[:,dst,:2] = obs['keypoints'][:,src]
        target[:,dst,2] = confidence[:,src]
        target = np.concatenate([mask_outside_raster(target[i:i+1],r['wh'])[0] for i,r in enumerate(rows)])
        with np.load(a.lp(ROOT/fit/'selected-fit.npz'),allow_pickle=False) as z:
            fitted = z['frame_ids'].astype(str).tolist()
            observed = z['observed_native_keypoints'].copy()
            projected = z['joints_projected_native'].copy()
        indices = [ids.index(f) for f in fitted]
        assert np.array_equal(target[indices],observed), subject+' input mapping changed'
        counts = dict(frames=len(fitted),max_body_error_on_ignored_joint=0,
            max_body_error_confidence_below_point1=0,body_gate_failed=0,
            frames_failing_only_ignored_hips=0)
        for i,f in enumerate(fitted):
            si=ids.index(f); height=measures[f]['initial_box_xyxy'][3]-measures[f]['initial_box_xyxy'][1]
            err=np.linalg.norm(projected[i]-observed[i,:,:2],axis=1)
            squared=observed[i,:,2]**2
            details=[]
            for k,(j,name) in enumerate(zip(COCO,NAMES)):
                details.append(dict(source_coco133_index=k,openpose25_index=int(j),name=name,
                    observed_xy=observed[i,j,:2].tolist(),projected_xy=projected[i,j].tolist(),
                    error_px=float(err[j]),error_over_height=float(err[j]/height),
                    raw_score=float(obs['scores'][si,k]),flip_disagreement_px=float(obs['flip_disagreement'][si,k]),
                    relative_confidence=float(observed[i,j,2]),squared_confidence=float(squared[j]),
                    optimized_by_original_joint_weights=int(j) not in (1,9,12),
                    anatomical_visibility='not_provided_by_detector_contract'))
            valid=[r for r in details if r['squared_confidence']>0]
            worst=max(valid,key=lambda r:r['error_px'])
            active=max((r for r in valid if r['optimized_by_original_joint_weights']),key=lambda r:r['error_px'])
            failed=worst['error_over_height']>.03
            saved=next(r for r in a.read(ROOT/fit/'selection-reviewed.json')['records'] if r['frame']==f)
            assert abs(worst['error_over_height']-saved['body_supported_max_over_height'])<1e-7
            assert failed==('body_reprojection_over_existing3percent_height' in saved['screen_flags'])
            counts['body_gate_failed']+=int(failed)
            counts['max_body_error_on_ignored_joint']+=int(not worst['optimized_by_original_joint_weights'])
            counts['max_body_error_confidence_below_point1']+=int(worst['relative_confidence']<.1)
            counts['frames_failing_only_ignored_hips']+=int(failed and active['error_over_height']<=.03)
            all_rows.append(dict(subject=subject,frame=f,height_px=height,source_fit_sha256=hashes[str(ROOT/fit/'selected-fit.npz')],
                current_gate_failed=failed,maximum=worst,optimized_joint_maximum=active,joints=details))
        subjects.append(dict(subject=subject,source_to_fit_mapping_exact=True,**counts))
    assert time.monotonic()-start<120
    assert all(a.sha(Path(p))==digest for p,digest in hashes.items())
    report=dict(updated=datetime.now(timezone.utc).isoformat(),state='read_only_completed',seconds=time.monotonic()-start,
        subjects=subjects,records=all_rows,source_hashes=hashes,source_hashes_unchanged=True,
        keypoint_semantic_order='COCO133 left/right anatomical identities map explicitly to original OpenPose25 indices; no frame-specific reindexing.',
        confidence='Relative score and flip agreement, not calibrated probability, observed visibility or physical correspondence.',
        training_objective='Original fitting uses squared confidence plus robust loss and ignores joints1,9,12. The real selector maximum uses positive squared float32 confidence and also inspects hips9,12. Exact agreement with saved selection metrics checked for all16frames.',
        correction_of='result.json wrongly reported the negligible-confidence winterhat96 wrist as the screen maximum. Its actual worst supported point is different; the existing selection itself was never changed.' if correction else None,
        scope_limit='Mapping equality proves correct data transfer, not detector anatomical correctness or true3D limb placement. Low-confidence outliers and ignored hips must be displayed, not silently deleted.',
        changed_training=False,changed_selection=False,physical_quality_accepted=False,new_gaussian_steps=0)
    a.save(OUT/('result-completed.json' if correction else 'result.json'),report)
    print(report['state'],report['seconds'])
    for subject in subjects: print(subject)
    for r in all_rows:
        x=r['maximum']; print(r['subject'],r['frame'],x['name'],round(x['error_px'],2),'confidence',round(x['relative_confidence'],6),'fit-active',x['optimized_by_original_joint_weights'])


if __name__=='__main__':main()
