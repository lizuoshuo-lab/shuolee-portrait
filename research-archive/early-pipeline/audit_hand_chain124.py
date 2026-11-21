"""Read-only duplicate-wrist reliability audit of frozen native observations."""
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read, save, sha, now
from smplify_score_contract123 import relative_confidence
import official_smplifyx_entry123 as entry

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')


def main():
    out=ROOT/'hand-observation-coherence'
    assert not out.exists(), 'Keep the first audit; do not regenerate it silently.'
    out.mkdir()
    official=ROOT.parent/'smplifyx-reference-control/official'
    utils,_,_,_=entry.load_official(official)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    assert mapping[7]==mapping[25] and mapping[4]==mapping[46]
    subjects={}
    for subject in ('winterhat','whiteT','grayT'):
        root=ROOT/(subject+'-input')
        with np.load(root/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
        confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        records=[]
        for i,name in enumerate(obs['frame_ids'].astype(str)):
            row={'frame':name}
            for side,parent,start in [('left',9,91),('right',10,112)]:
                C=confidence[i,start:start+21]
                row[side]=dict(body_wrist_confidence=float(confidence[i,parent]),hand_wrist_confidence=float(C[0]),
                    wrist_2d_disagreement_px=float(np.linalg.norm(obs['keypoints'][i,parent]-obs['keypoints'][i,start])),
                    body_wrist_flip_disagreement_px=float(obs['flip_disagreement'][i,parent]),
                    hand21_median_confidence=float(np.median(C)),hand21_max_confidence=float(C.max()),
                    original_last_stage_sum_squared_hand_weights=float(np.sum((2*C)**2)),
                    prospective_chain_gate=float(min(confidence[i,parent],C[0])))
            records.append(row)
        subjects[subject]=dict(frames=len(records),records=records,
            observations_sha256=sha(root/'data/observations.npz'),score_adapter_sha256=sha(Path(entry.__file__).with_name('smplify_score_contract123.py')))
    native=ROOT/'winterhat-temporal/official-fit/frames/frame_000096/winner-camera-space.npz'
    with np.load(native,allow_pickle=False) as z:j=z['joints_camera'][0]
    duplicate=float(max(np.max(abs(j[7]-j[25])),np.max(abs(j[4]-j[46]))))
    assert duplicate==0
    result=dict(updated=now(),mapping_duplicate_smpl_joint_ids={'left':[int(mapping[7]),int(mapping[25])],'right':[int(mapping[4]),int(mapping[46])]},
        saved_frame96_same_3d_wrist_max_difference=duplicate,subjects=subjects,
        no_observation_or_fitting_file_changed=True,training_started=False,
        interpretation='Body wrist and hand root map to the same3D joint, but native detector coordinates and reliability can disagree. Independently normalizing21 hand scores can leave hand supervision active after body-wrist evidence vanishes. This is a plausible remaining failure mechanism, not proof of the complete pose cause.',
        proposed_control='Only couple each hand chain to min(body-wrist confidence, hand-root confidence) before original fitting. All2D coordinates, body/face/feet weights, temporal term, original priors and optimizer remain fixed. This is a new local reliability rule, not an upstream default or visibility ground truth.')
    save(out/'result.json',result)
    old=read(ROOT/'winterhat-input/selection.json')
    plan=dict(updated=now(),state='registered_not_started',hypothesis='Occluded hand-chain observations can override a weak temporal prior even when their parent body wrist is unreliable.',
        intervention=result['proposed_control'],reference='Completed112-frame temporal candidate; never modify it or its running source.',
        preregistered_control_frames=old['fit_frames'],budget=dict(candidates=1,frames=len(old['fit_frames']),max_wall_seconds=900),
        initialization='Original zero/mean body pose; same frozen shared shape. No previous pose copied as initialization.',
        temporal_control='For each same frame use the completed baseline same-neighbour target, held fixed in this isolated loss-input control. Old neighbour estimates are controlled inputs, not3Dtruth or a new independent full-sequence fit.',
        protections=['All12 originally selected frames, no manual replacement or result-dependent filtering.','Check visible body/face/feet and hands as well as hidden-arm depth.','Do not stitch improved single-frame results into a sequence or launch Gaussians from this control.'],
        if_supported='Register one fresh complete-sequence fit using the same deterministic observation rule, then review all frames before Gaussian admission.',
        stopping='Stop on nonfinite values or significant new pose/image regression. No confidence threshold or temporal-weight sweep.',
        quality_accepted=False,fit_started=False)
    save(out/'control-plan.json',plan)
    print('HAND_COHERENCE_AUDIT_READY',subjects['winterhat']['records'][95],flush=True)


if __name__=='__main__':main()
