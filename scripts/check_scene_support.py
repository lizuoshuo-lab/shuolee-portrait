"""Known geometry and independent derivative checks; no captured-video experiment."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import scene_support as ss, runtime as rt
from shuolee_portrait.fixed_tracks import cameras_from_records, evaluate_fixed_tracks


def fixture(kind='rich'):
    rng=np.random.default_rng(117);rows=[]
    xyz=rng.uniform([-1.2,-.8,4.],[1.2,.8,7.],(32,3))
    if kind=='line':xyz=np.c_[np.linspace(-1.2,1.2,32),np.zeros(32),np.ones(32)*5.]
    K=np.array([[850.,0,640.5],[0,870.,480.5],[0,0,1.]])
    centers=np.array([[-.9,-.25,0.],[-.6,.2,.3],[-.2,-.3,-.2],[.15,.1,.4],
                      [.45,-.15,-.1],[.75,.3,.15],[1.,-.1,.45]])
    if kind in ('parallel','line'):centers=np.c_[np.linspace(-.9,.9,7),np.zeros((7,2))]
    tracks=[dict(track_id=f'point_{i:04d}',identity_basis='material_candidate',
        region='static_scene_candidate',triangulation_frames=['frame_0','frame_6'],observations=[]) for i in range(len(xyz))]
    for i,center in enumerate(centers):
        forward=np.array([0.,0.,5.5])-center;forward/=np.linalg.norm(forward)
        right=np.cross([0.,1.,0.],forward);right/=np.linalg.norm(right)
        up=np.cross(forward,right);R=np.stack([right,up,forward])
        if kind in ('parallel','line'):R=np.eye(3)
        E=np.eye(4);E[:3,:3]=R;E[:3,3]=-R@center
        row=dict(frame=f'frame_{i}',K=K.tolist(),w2c=E.tolist(),wh=[1280,960],source_sha256=f'{i+1:064x}')
        rows.append(row)
        projected=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],E[:3,3],K,np.zeros(5))[0].reshape(-1,2)-.5
        for j,uv in enumerate(projected):
            assert (uv>=0).all() and (uv<np.array(row['wh'])-1).all()
            tracks[j]['observations'].append(dict(frame=row['frame'],observation_id=f'p{j}v{i}',
                image_sha256=row['source_sha256'],uv=uv.tolist()))
    return rows,dict(schema='shuolee-fixed-tracks-v1',pixel_domain='undistorted_integer_centers',
                    camera_fit_observation_ids=None,tracks=tracks),xyz


def measure(rows,bank):
    return ss.analyze_scene_support(rows,bank,evaluate_fixed_tracks(rows,bank))


def run():
    cases={};rows,bank,xyz=fixture();original=copy.deepcopy((rows,bank))
    cams=cameras_from_records(rows);cam=cams['frame_3'];baseline=1.4;point=xyz[5]
    jac=ss.projection_jacobians(cam,point,baseline)
    def project(K,E,X):
        return cv2.projectPoints(np.asarray(X).reshape(1,3),cv2.Rodrigues(E[:3,:3])[0],E[:3,3],K,np.zeros(5))[0].ravel()-.5
    eps=1e-5;errors=[]
    for group,count in enumerate([4,6,3]):
        for j in range(count):
            values=[]
            for sign in [-1,1]:
                K=cam['K'].copy();E=cam['E'].copy();X=point.copy();h=sign*eps
                if group==0:
                    if j<2:K[j,j]*=1+.01*h
                    else:K[j-2,2]+=h
                elif group==1:
                    if j<3:
                        w=np.zeros(3);w[j]=np.pi/180*h
                        E[:3,:]=cv2.Rodrigues(w)[0]@E[:3,:]
                    else:E[j-3,3]+=.01*baseline*h
                else:X[j]+=baseline*h
                values.append(project(K,E,X))
            errors.append(float(np.max(np.abs((values[1]-values[0])/(2*eps)-jac[group][:,j]))))
    assert max(errors)<2e-7,max(errors)
    cases['opencv_central_difference']=dict(parameters=13,max_absolute_derivative_error=max(errors))

    rich=measure(rows,bank);info=rich['information'];joint=info['joint_k_and_pose']
    assert info['conditional_k']['numerical_rank']==4
    assert joint['shared_k_after_points_and_poses']['numerical_rank']==4
    assert joint['joint_structured_nullity']==0
    assert all(p['conditional_pose']['numerical_rank']==6 for p in info['per_frame_pose'])
    assert (rows,bank)==original
    cases['noncritical_known_geometry']=dict(conditional_K_rank=4,joint_K_rank=4,
        joint_parameter_count=joint['joint_parameters'],joint_nullity=0,
        scene_points=rich['scene_candidate_tracks'],physical_camera_verified=False)

    sparse=copy.deepcopy(bank);sparse['tracks']=sparse['tracks'][:3]
    under=measure(rows,sparse)['information']
    assert under['conditional_k']['numerical_rank']==4
    assert under['joint_k_and_pose']['shared_k_after_points_and_poses']['numerical_rank']<4
    assert under['joint_k_and_pose']['joint_structured_nullity']>0
    cases['fixed_extrinsics_can_hide_joint_ambiguity']=dict(scene_points=3,covered_frames=7,
        conditional_K_rank=under['conditional_k']['numerical_rank'],
        joint_K_rank=under['joint_k_and_pose']['shared_k_after_points_and_poses']['numerical_rank'],
        joint_nullity=under['joint_k_and_pose']['joint_structured_nullity'])

    parallel=measure(*fixture('parallel')[:2]);p=parallel['information']
    assert p['conditional_k']['nullity']>0
    assert p['joint_k_and_pose']['shared_k_after_points_and_poses']['nullity']>0
    assert p['conditional_k']['unit_pixel_l2_parameter_amplification'] is None
    assert max(e for t in parallel['tracks'] for e in t['held_error_px'])<1e-8
    line=measure(*fixture('line')[:2])
    assert all(f['conditional_pose']['nullity']>0 for f in line['information']['per_frame_pose'])
    cases['zero_residual_still_degenerate']=dict(parallel_conditional_rank=p['conditional_k']['numerical_rank'],
        parallel_joint_rank=p['joint_k_and_pose']['shared_k_after_points_and_poses']['numerical_rank'],
        line_pose_ranks=[f['conditional_pose']['numerical_rank'] for f in line['information']['per_frame_pose']],
        rank_deficiency_reported_without_false_zero_uncertainty=True)

    rotated=copy.deepcopy(rows);Q=cv2.Rodrigues(np.array([.25,-.17,.11]))[0];scale=3.7;b=np.array([10.,-7.,4.])
    for row in rotated:
        E=np.asarray(row['w2c']);R=E[:3,:3]@Q.T
        E[:3,3]=scale*E[:3,3]-R@b;E[:3,:3]=R;row['w2c']=E.tolist()
    transformed=measure(rotated,bank)
    for key in ['conditional_k']:
        assert np.allclose(info[key]['singular_values'],transformed['information'][key]['singular_values'],atol=1e-8,rtol=1e-8)
    assert np.allclose(joint['shared_k_after_points_and_poses']['singular_values'],
        transformed['information']['joint_k_and_pose']['shared_k_after_points_and_poses']['singular_values'],atol=1e-8,rtol=1e-8)
    assert transformed['information']['joint_k_and_pose']['joint_structured_nullity']==0
    bases,offsets,names=ss._layout(cams,rich['baseline_anchor_frame'])
    anchor=rich['baseline_anchor_frame'];E=cams[anchor]['E'];radial=cams[anchor]['center']-cams['frame_0']['center']
    assert np.max(np.abs(radial@(-E[:3,:3].T@bases[anchor][3:,3:])))<1e-12
    cases['similarity_gauge_and_scale_invariance']=dict(scale=scale,rotation_and_translation=True,
        joint_spectrum_unchanged=True,anchor_radial_translation_removed=True)

    missing=copy.deepcopy(rows);extra=copy.deepcopy(rows[3]);extra['frame']='empty_camera';missing.append(extra)
    result=measure(missing,bank);last=result['information']['per_frame_pose'][-1]
    assert result['frames_without_scene_observations']==['empty_camera']
    assert last['conditional_pose']['rows']==0 and last['conditional_pose']['numerical_rank']==0
    assert result['information']['joint_k_and_pose']['pose_nullity_with_fixed_K']>=6
    cases['missing_camera_support_preserved']=dict(missing_frames=result['frames_without_scene_observations'],
        nullity=result['information']['joint_k_and_pose']['pose_nullity_with_fixed_K'])

    altered=copy.deepcopy(bank);altered['tracks'][0]['observations'][3]['uv'][0]+=9.
    result=measure(rows,altered)
    assert len(result['tracks'])==32
    assert max(result['tracks'][0]['held_error_px'])>8.99
    assert result['information']['usable_tracks']==32 and not result['root_cause_resolved']
    cases['large_residual_not_pruned']=dict(retained_tracks=32,known_mismatch_px=9.,cause_not_assigned=True)

    different=copy.deepcopy(rows);different[2]['K'][0][0]+=.1
    result=measure(different,bank)
    assert result['state']=='shared_rectified_K_not_supported_by_manifest'
    assert result['information']['conditional_k'] is None and result['information']['joint_k_and_pose'] is None
    with patch.object(ss,'MAX_JOINT_ENTRIES',1):limited=measure(rows,bank)
    assert limited['state']=='joint_budget_exceeded_no_subsampling'
    assert limited['information']['usable_tracks']==32 and limited['information']['joint_k_and_pose'] is None
    cases['shared_K_contract_and_budget']=dict(different_K_not_averaged=True,no_subset_fallback=True)

    empty=copy.deepcopy(bank)
    for t in empty['tracks']:t['region']='face'
    result=measure(rows,empty)
    assert result['state']=='no_observable_scene_candidates'
    assert result['scene_candidate_tracks']==0 and len(result['frames_without_scene_observations'])==7
    assert all(f['conditional_depth_over_baseline']['median'] is None for f in result['frames'])
    coincident=copy.deepcopy(rows)
    for r in coincident:r['w2c']=np.eye(4).tolist()
    result2=measure(coincident,bank)
    assert result2['state']=='no_nonzero_camera_baseline' and result2['information'] is None
    json.dumps([rich,parallel,line,transformed,result,result2],allow_nan=False)
    cases['absent_scene_and_zero_baseline_are_unknown']=dict(no_empty_pass=True,null_depths=True,finite_json=True)
    return dict(passed=True,groups=len(cases),cases=cases,synthetic_only=True,
        real_video_reexecuted=False,physical_calibration_verified=False,training_started=False,
        source_measurements_modified=False,reference_diagnosis_reopened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();result=run();rt.save(args.report,result);print(json.dumps(result,indent=2))
