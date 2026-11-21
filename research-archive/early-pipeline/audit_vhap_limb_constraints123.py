"""Read-only all-frame constraint inventory and rotation diagnostics."""
from datetime import datetime, timezone
import numpy as np
import audit_gaussianavatars_params123 as a


def stats(x):
    v = np.asarray(x, dtype=float)
    return dict(minimum=float(v.min()),median=float(np.median(v)),maximum=float(v.max()))


def rotations(x):
    p = np.asarray(x, dtype=float).reshape(-1,3)
    theta = np.linalg.norm(p,axis=1)
    k = p/np.maximum(theta[:,None],1e-30)
    s = np.zeros((len(p),3,3))
    s[:,0,1]=-k[:,2];s[:,0,2]=k[:,1];s[:,1,0]=k[:,2]
    s[:,1,2]=-k[:,0];s[:,2,0]=-k[:,1];s[:,2,1]=k[:,0]
    return np.eye(3)[None]+np.sin(theta)[:,None,None]*s+(1-np.cos(theta))[:,None,None]*(s@s)


def main():
    root = a.REC/'demo-vhap-initialization'
    run = root/'smplify-vhap-monocular-lr'
    cfg = a.read(run/'config.json')
    w = cfg['w']
    repo = a.T/'../../repos/VHAP_smplx' if hasattr(a,'T') else None
    from pathlib import Path
    source = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx/vhap/model/tracker_smplx.py')
    original_prior = a.REC/'smplifyx-reference-control/official/smplifyx/prior.py'
    text = a.lp(original_prior).read_text(encoding='utf-8')
    assert 'SMPLifyAnglePrior' in text and '55' in text and '58' in text
    inventory = dict(shared_articulated_smplx=True,body_representation='direct_axis_angle63',
        body_prior='reg_body * mean((I - R_j)^2), a soft zero-pose rotation penalty, not anatomical joint limits',
        vposer_active=cfg['model']['use_vposer'],hand_prior='PCA12 coefficient L2',
        hand_GMM_active=cfg['model']['use_hand_prior'],temporal='one-sided previous-frame raw parameter differences; previous value detached',
        smplify_elbow_knee_bending_prior_in_current_vhap=False,self_collision_loss_in_current_vhap=False,
        hard_anatomical_joint_limits_in_current_vhap=False,mask_and_photometric=True)
    outputs = {}
    for label,path in [('smplify_initial',run/'initial/tracked-smplx-params.npz'),
                       ('direct_control',root/'smplify-vhap-direct-body/epoch-30/tracked-smplx-params.npz'),
                       ('monocular_lr',run/'epoch-30/tracked-smplx-params.npz')]:
        with np.load(a.lp(path),allow_pickle=False) as z:
            d = dict(z)
        assert d['body_pose'].shape == (59,63)
        pose = d['body_pose'].astype(float)
        r = rotations(pose).reshape(59,21,3,3)
        assert np.max(np.abs(r@r.transpose(0,1,3,2)-np.eye(3))) < 1e-10
        prev = np.r_[0,np.arange(58)]
        temporal = {}
        keys = dict(body_pose='smooth_body',jaw_pose='smooth_jaw',leye_pose='smooth_eyes',reye_pose='smooth_eyes',left_hand_pose='smooth_hands',right_hand_pose='smooth_hands')
        for key,weight in keys.items():
            values = d[key].astype(float)
            temporal[key] = ((values-values[prev])**2).mean(axis=1)*w[weight]
        smooth = sum(temporal.values())
        relative = r[prev].transpose(0,1,3,2)@r
        geodesic = np.degrees(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
        signed = pose[:,[52,55,9,12]]*np.array([1,-1,-1,-1])
        # Same body-only indexing as the pinned upstream angle prior.
        bend = np.exp(signed)**2
        body_prior = ((r-np.eye(3))**2).mean(axis=(1,2,3))*w['reg_body']
        frames = []
        for i,f in enumerate(d['frame_ids']):
            frames.append(dict(frame=str(f),temporal_components={k:float(v[i]) for k,v in temporal.items()},
                total_temporal_joints=float(smooth[i]),largest_body_joint_rotation_change_deg=float(geodesic[i].max()),
                zero_pose_body_prior=float(body_prior[i]),smplify_soft_bending_components_unweighted=bend[i].tolist(),
                soft_bending_is_diagnostic_only=True))
        outputs[label] = dict(checkpoint=str(path),sha256=a.sha(path),frames=frames,
            temporal_component_summary={k:stats(v[1:]) for k,v in temporal.items()},
            dominant_temporal_component_counts={k:int((np.argmax(np.stack(list(temporal.values())),axis=0)[1:]==j).sum()) for j,k in enumerate(temporal)},
            body_prior_summary=stats(body_prior),body_rotation_change_deg=stats(geodesic[1:].max(axis=1)),
            soft_bending_sum_unweighted=stats(bend.sum(axis=1)),max_bending_frame=str(d['frame_ids'][int(bend.sum(axis=1).argmax())]))
    result = dict(updated=datetime.now(timezone.utc).isoformat(),state='complete_read_only',frames=59,model_forwards=0,
        optimizer_updates=0,inventory=inventory,source_hashes={str(p):a.sha(p) for p in (source,original_prior)},
        config_sha256=a.sha(run/'config.json'),runs=outputs,quality_accepted=False,
        limits=['Stored detector residuals and soft priors do not prove anatomical truth.',
                'Raw axis-angle temporal loss is not equivalent to rotation geodesic distance.',
                'Terms are audited separately; no change to priors, weights or fitting in this script.',
                'Absence of a bending/collision prior is an implementation gap, not proof of the unique cause of visible elbow failure.'])
    out = root/'vhap-limb-constraint-review.json'
    a.save(out,result)
    print(out)
    print(outputs['monocular_lr']['dominant_temporal_component_counts'])
    print(outputs['monocular_lr']['temporal_component_summary'])
    print(outputs['monocular_lr']['soft_bending_sum_unweighted'])


if __name__=='__main__':
    main()
