"""Read saved RTMW confidence/visibility evidence. No detector, fitter or GPU."""
import os
import time
from pathlib import Path
import numpy as np
import audit_sofa_knee_objective124 as a
from run_official_gaussians123 import read, save, sha, now
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster

OUT = a.ROOT / 'observation-confidence-audit'
LIB = Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/lib/python3.10/site-packages/rtmlib')
SOURCES = {s:a.ROOT/(s+'-input') for s in ('winterhat','sofa','indoor')}
SOURCES['grass'] = Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
GROUPS = dict(body=np.arange(17), feet=np.arange(17,23), face=np.arange(40,91),
              left_hand=np.arange(91,112), right_hand=np.arange(112,133))


def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(min=float(x.min()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists(), 'Preserve this single registered audit.'
    selected={}
    paths=[Path(__file__),Path(__file__).with_name('smplify_score_contract123.py'),
           Path(__file__).with_name('smplify_observation_validity124.py'),
           a.PROJECT/'src/shuolee_portrait/observations.py',a.ASSETS/'rtmw/normalization.json',
           a.OFFICIAL/'smplifyx/fitting.py',a.OFFICIAL/'cfg_files/fit_smplx.yaml',
           a.OFFICIAL/'smplifyx/fit_single_frame.py',
           LIB/'tools/pose_estimation/rtmpose.py',LIB/'tools/pose_estimation/pre_processings.py',
           LIB/'tools/pose_estimation/post_processings.py',
           a.ROOT/'unique-wrist-input124/sofa16-control/gradient-audit-fp64-completed/conclusion.json']
    for subject,src in SOURCES.items():
        paths += [src/'data/inputs.json',src/'data/observations.npz']
        if subject in ('sofa','indoor'):
            plan=src/'observation-review/plan.json';paths += [plan,src/'observation-review/result.json']
            selected[subject]=read(plan)['frames']
        elif subject=='winterhat':
            plan=a.ROOT/'winterhat-input/selection.json';paths.append(plan)
            selected[subject]=read(plan)['fit_frames']
        else:
            # These three existing causal-review frames are diagnostic, not new training selection.
            selected[subject]=[f'frame_{i:06d}' for i in (32,44,45)]
    # Frame16 is the already preregistered stage/gradient failure, not a new fit.
    selected['sofa']=sorted(set(selected['sofa']+['frame_000016']))
    for subject,src in SOURCES.items():
        for row in read(src/'data/inputs.json')['records']:
            if row['frame'] in selected[subject]:paths += [Path(row['image']),Path(row['mask'])]
    frozen={str(p):sha(p) for p in paths}
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),
        hypothesis='The saved SimCC peak response and sequence-relative/flip weights are not calibrated anatomical visibility. Quantify their actual contribution and native sampling scale on existing observations before changing fitting inputs.',
        budget=dict(cpu_seconds=120,subjects=4,frames=373,new_detector_calls=0,optimizer_steps=0),
        selected_existing_review_frames=selected,
        invariants=['No original observation, confidence, fit, stage or mask is modified.',
                    'No residual-based filtering, score threshold, new detector or fit.',
                    'Retain invalid-raster labels in the audit while reflecting the existing raster guard.',
                    'Original OpenPose confidence itself is not assumed calibrated probability.'],
        outputs=['all-frame group weight and flip statistics','existing selected joint-level observations','source semantics and native SimCC bin width'],
        limits=['Visual visibility is a review label, not subpixel ground truth.',
                'Argmax bin spacing is a sampling limitation, not a measured error lower bound.',
                'Group weight sums do not equal parameter gradients or optimizer displacement.',
                'Cannot reconstruct probability/entropy from saved maxima without the full SimCC distributions.'],frozen=frozen))
    started=time.monotonic()
    subjects={}
    dst,src_idx=np.r_[a.entry.COCO,19:25,25:67,67:118],np.r_[0:23,91:133,40:91]
    for subject,source in SOURCES.items():
        assert time.monotonic()-started<120
        rows=read(source/'data/inputs.json')['records'];obs=a.load_npz(source/'data/observations.npz')
        assert list(obs['frame_ids'].astype(str))==[r['frame'] for r in rows]
        conf,q90=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        targets=np.zeros((len(rows),118,3),np.float32)
        targets[:,dst,:2]=obs['keypoints'][:,src_idx];targets[:,dst,2]=conf[:,src_idx]
        records=[]
        for i,row in enumerate(rows):
            t,_=mask_outside_raster(targets[i:i+1],row['wh']);c=t[0,:,2]
            # Exact RTMPose preprocessing: bbox padding1.25 and288:384 aspect ratio.
            wh=(obs['boxes'][i,2:]-obs['boxes'][i,:2])*1.25
            wh=np.array([wh[0],wh[0]/.75]) if wh[0]>wh[1]*.75 else np.array([wh[1]*.75,wh[1]])
            step=wh/np.array([288,384])/2
            person_h=float(obs['boxes'][i,3]-obs['boxes'][i,1])
            groups={name:dict(raw_score=stats(obs['scores'][i,idx]),
                             relative_confidence=stats(conf[i,idx]),
                             flip_px=stats(obs['flip_disagreement'][i,idx]),
                             sum_confidence_squared=float((conf[i,idx]**2).sum())) for name,idx in GROUPS.items()}
            weights=c.copy();weights[[1,9,12]]=0;weights[25:67]*=2;weights[67:]*=2
            effective=dict(body_feet=float((weights[:25]**2).sum()),hands=float((weights[25:67]**2).sum()),
                           face=float((weights[67:]**2).sum()))
            rec=dict(frame=row['frame'],time_seconds=row['time_seconds'],native_wh=row['wh'],
                     native_simcc_bin_width_px=step.tolist(),flip_scale_px=.012*person_h,
                     groups=groups,stage4_effective_weight_sum_without_common_data_factor=effective)
            if row['frame'] in selected[subject]:
                rec['original_review_frame']=True
                rec['observations']=[dict(index=int(j),name=str(obs['joint_names'][j]),xy=obs['keypoints'][i,j].tolist(),
                    raw_score=float(obs['scores'][i,j]),sequence_q90=float(q90[j]),confidence=float(conf[i,j]),
                    flip_px=float(obs['flip_disagreement'][i,j])) for j in np.r_[5:11,40:91,91:133]]
                rec['native_images']=dict(image=row['image'],mask=row['mask'])
            records.append(rec)
        subjects[subject]=dict(frames=len(rows),archive_keys=list(obs),raw_score=stats(obs['scores']),
            relative_confidence=stats(conf),native_simcc_bin_width_px=stats([r['native_simcc_bin_width_px'][0] for r in records]),
            records=records)
    for path,digest in frozen.items():assert sha(path)==digest,path
    save(OUT/'result.json',dict(updated=now(),state='numeric_audit_complete_visual_review_pending',
        elapsed_seconds=time.monotonic()-started,subjects=subjects,source_hashes_verified=True,
        new_detector_calls=0,new_fit_calls=0,optimizer_steps=0,default_applied=False,
        source_contract=dict(rtpose='RTMPose.postprocess uses argmax/2 and affine inverse; scores are the mean of x/y peak responses, with no softmax in this decoder.',
            stored='The saved score is min(original,flip score); only mean2D, min score and flip distance are retained, not full distributions or both coordinate vectors.',
            adapter='Per-point temporal q90 normalization and body-height-based flip discount; heuristic relative reliability only.',
            original_smplify='The upstream loss squares (joint_weight * supplied confidence), sums robust x/y residuals and scales by data_weight squared. It does not supply visibility or score calibration.',
            rgb='Saved normalization to_rgb=true; current observation reader supplies RGB and the RTMPose preprocessor preserves channel order. No channel-order defect found by code inspection.'),
        quality_accepted=False,gaussian_admitted=False))
    print('OBSERVATION_CONFIDENCE_AUDIT_COMPLETE', {s:{k:v for k,v in d.items() if k not in ('records','archive_keys')} for s,d in subjects.items()},flush=True)


if __name__=='__main__':main()
