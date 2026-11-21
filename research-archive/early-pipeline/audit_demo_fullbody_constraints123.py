"""Inspect saved official fits and their existing limb constraints on CPU only.

Does not run a model, change a fit, infer visibility, or admit training.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

import numpy as np

H = Path(__file__).resolve().parent
BASE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT = BASE/'demo-vhap-initialization/fullbody-smplifyx/full-sequence'
OFFICIAL = BASE/'smplifyx-reference-control/official'
DATA = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()),
                range=float(np.ptp(a)), std=float(a.std()))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path.insert(0, str(H))
    import official_smplifyx_entry123 as entry
    from smplify_score_contract123 import relative_confidence
    _, expected_diff = entry.adapted_entry_source(OFFICIAL)
    config = entry.load_official_config(OFFICIAL)
    provenance = read(ROOT/'input-provenance.json')
    expected_entry = provenance.get('adapter_sha256', provenance.get('entry_sha256'))
    if expected_entry is not None:
        assert sha(H/'official_smplifyx_entry123.py') == expected_entry
    rows = read(DATA/'data/inputs.json')['records']
    by_id = {r['frame']: (i,r) for i,r in enumerate(rows)}
    with np.load(DATA/'data/observations.npz', allow_pickle=False) as z:
        obs = dict(z)
    assert list(obs['frame_ids'].astype(str)) == [r['frame'] for r in rows]
    confidence, _ = relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    destination = np.r_[entry.COCO, np.arange(19,25), np.arange(25,67), np.arange(67,118)]
    source = np.r_[np.arange(23), np.arange(91,133), np.arange(40,91)]
    targets = np.zeros((len(rows),118,3),np.float64)
    targets[:,destination,:2] = obs['keypoints'][:,source]
    targets[:,destination,2] = confidence[:,source]
    ready = [p for p in sorted((ROOT/'frames').iterdir())
             if (p/'result.json').exists() and (p/'winner-camera-space.npz').exists()]
    bones = {'right_upper_arm':(2,3),'right_forearm':(3,4),
             'left_upper_arm':(5,6),'left_forearm':(6,7),
             'right_thigh':(9,10),'right_shin':(10,11),
             'left_thigh':(12,13),'left_shin':(13,14)}
    bone_values = {k:[] for k in bones}
    hinge_ids = {'left_elbow':(5,6,7),'right_elbow':(2,3,4),
                 'left_knee':(12,13,14),'right_knee':(9,10,11)}
    frame_results = []
    beta = None
    for p in ready:
        result = read(p/'result.json')
        assert result['state']=='complete' and result['frame_id']==p.name
        assert result['upstream_loss_monitor_optimizer_unchanged'] and result['five_stages']
        assert result['interpenetration'] is False and result['freeze_betas'] is True
        assert (p/'input-output-adapter.diff').read_text() == expected_diff
        for filename, digest in result['source_hashes'].items():
            assert sha(OFFICIAL/filename)==digest
        with np.load(p/'winner-camera-space.npz', allow_pickle=False) as z:
            a = dict(z)
        slot,row = by_id[p.name]
        assert str(a['frame_id'].item())==p.name
        assert np.array_equal(a['K_integer_centers'],row['K_integer_centers'])
        if beta is None: beta=a['betas'].copy()
        assert np.array_equal(a['betas'],beta)
        j=a['joints_camera'][0].astype(np.float64)
        q=j@a['K_integer_centers'].T
        uv=q[:,:2]/q[:,2:]
        parity=float(np.max(np.abs(uv-a['joints_projected_native'][0])))
        assert parity<.001
        lengths={k:float(np.linalg.norm(j[b]-j[c])) for k,(b,c) in bones.items()}
        for k,v in lengths.items():bone_values[k].append(v)
        hinges={}
        for k,(b,c,d) in hinge_ids.items():
            u,v=j[b]-j[c],j[d]-j[c]
            hinges[k]=float(np.degrees(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
        aa=a['body_pose_axis_angle'][0].astype(np.float64)
        # Exact pinned SMPLifyAnglePrior indices after excluding the root.
        signed_components=aa[[52,55,9,12]]*np.array([1,-1,-1,-1])
        bending_terms=np.exp(signed_components)**2
        body_weight=float(config['body_pose_prior_weights'][-1])
        loss_2d=[]
        weights=np.ones(118);weights[[1,9,12]]=0
        weights[25:67]=float(config['hand_joints_weights'][-1])
        weights[67:]=float(config['face_joints_weights'][-1])
        delta=targets[slot,:,:2]-uv
        robust=(float(config['rho'])**2*delta**2)/(delta**2+float(config['rho'])**2)
        joint_loss=(weights*targets[slot,:,2])**2*robust.sum(axis=1)*(1000/row['wh'][1])**2
        for k in (2,3,4,5,6,7,10,11,13,14):
            loss_2d.append(dict(openpose_index=k,error_px=float(np.linalg.norm(delta[k])),
                confidence=float(targets[slot,k,2]),observed_xy=targets[slot,k,:2].tolist(),
                projected_xy=uv[k].tolist(),weighted_robust_loss=float(joint_loss[k])))
        frame_results.append(dict(frame_id=p.name,fit_sha256=sha(p/'winner-camera-space.npz'),
            result_sha256=sha(p/'result.json'),source_image_sha256=row['source_sha256'],
            saved_projection_parity_max_px=parity,bone_lengths_m=lengths,hinge_internal_angle_deg=hinges,
            bending_prior_component_values=bending_terms.tolist(),
            bending_prior_weight=3.17*body_weight,
            bending_prior_contribution=float(bending_terms.sum()*3.17*body_weight),
            vposer_latent_l2_contribution=float(np.square(a['pose_embedding'].astype(np.float64)).sum()*body_weight**2),
            body_observation_errors=loss_2d))
    progress=read(ROOT/'progress.json')
    assert 'torch' not in sys.modules
    answer=dict(state='saved_fit_limb_constraint_audit',updated=datetime.now(timezone.utc).isoformat(),
        source_progress=progress,completed_frames=len(ready),expected_frames=59,
        full_roster_complete=progress.get('state')=='complete' and len(ready)==59,
        source_script_sha256=sha(__file__),official_source_hashes=result['source_hashes'],
        input_sha256=sha(DATA/'data/inputs.json'),observations_sha256=sha(DATA/'data/observations.npz'),
        enabled=dict(shared_SMPLX_shape_and_articulated_skeleton=True,VPoser2_pose_prior=True,
            elbow_knee_soft_bending_prior=True,weighted_2D_body_hand_face_landmarks=True),
        disabled_or_absent=dict(self_collision=True,temporal_pose_loss=True,mask_silhouette_loss_in_fitting=True),
        interpretation='Bending prior is a soft exponential heuristic over four axis-angle components, not hard anatomical joint limits or a pixel-alignment guarantee. Actual anatomical visibility is not inferred from detector confidence.',
        bone_length_statistics_m={k:stats(v) for k,v in bone_values.items()},frames=frame_results,
        causality='Evidence confirms constraints exist and pose residuals remain. It does not uniquely identify the optimizer, 2D observations, K, shape or occlusion as their cause.',
        quality_accepted=False,model_forwards=0,GPU_calls=0,optimizer_updates=0)
    output=ROOT/'limb-constraint-audit.json'
    temp=output.with_suffix('.json.tmp')
    temp.write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n');os.replace(temp,output)
    print(json.dumps(dict(report=str(output),frames=len(ready),full_roster_complete=answer['full_roster_complete'],
        max_bone_length_range_m=max(v['range'] for v in answer['bone_length_statistics_m'].values()),
        constraints=answer['enabled'],missing=answer['disabled_or_absent'])))


if __name__=='__main__':main()
