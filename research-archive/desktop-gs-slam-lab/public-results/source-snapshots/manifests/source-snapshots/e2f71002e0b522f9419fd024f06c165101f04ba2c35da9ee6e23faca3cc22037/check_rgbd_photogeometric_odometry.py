"""Synthetic-only CPU mathematical controls for RGB-D photo+geometry odometry.

No TUM data optimization, GPU, torch or renderer. Known poses construct toy
images and are used to evaluate outcomes; they never select optimizer steps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.linalg import expm

import rgbd_photogeometric_odometry as odom

ROOT=Path(__file__).resolve().parents[1]


def require(condition,message):
    if not condition:raise AssertionError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path,value):
    with path.open('x',encoding='utf-8')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def plane_rgbd(width,height,c2w,textured=True,brightness_offset=0.):
    K=np.array([[.75*width,0.,width/2-.5],[0.,.75*width,height/2-.5],[0.,0.,1.]])
    v,u=np.mgrid[:height,:width]
    rays=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)),axis=-1)
    world_directions=rays@c2w[:3,:3].T
    depth=(2.2-c2w[2,3])/world_directions[...,2]
    world=c2w[:3,3]+world_directions*depth[...,None]
    # A bilinear world-plane texture is exactly represented by target image
    # bilinear interpolation at the identity reference camera. No inverse crime
    # about real cameras/noise is claimed: this deliberately isolates algebra.
    gray=.4+.05*world[...,0]+.07*world[...,1]+.05*world[...,0]*world[...,1]if textured else np.full(depth.shape,.4)
    gray=gray+brightness_offset
    require(gray.min()>=0 and gray.max()<=1,'synthetic texture in [0,1] without clipping')
    rgb=np.repeat(gray[...,None],3,axis=-1)
    return depth,np.isfinite(depth)&(depth>0),K,rgb


def build_pyramid(pose,textured,config,brightness_offset=0.):
    return [odom.make_frame(*plane_rgbd(w,h,pose,textured,brightness_offset),config)for w,h in odom.icp.LEVELS]


def jacobian_controls(config):
    h,w=60,80;v,u=np.mgrid[:h,:w]
    # Nonlinear but bilinear-per-cell image: FD must stay in one pixel cell.
    image=.3+.002*u+.003*v+.00004*u*v+.02*np.sin(u*.11)*np.cos(v*.09)
    K=np.array([[60.,0.,39.5],[0.,60.,29.5],[0.,0.,1.]])
    uv=np.array([[12.27,13.68],[30.41,25.23],[44.76,33.34],[65.19,42.81]])
    z=np.array([1.7,2.2,2.6,3.1]);p=np.c_[(uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z]
    transform=odom.icp.se3_exp(np.array([.035,-.023,.012,.025,-.018,.011]))
    q=odom.icp.transform_points(odom.icp.inverse_rigid(transform),p)
    source=np.array([.2,.4,.5,.6]);_,analytic,queries=odom.intensity_residual_jacobian(p,K,image,source)
    distances=np.minimum(queries-np.floor(queries),np.ceil(queries)-queries)
    require(float(distances.min())>.1,'FD avoids cell boundaries')
    epsilon=1e-7;numeric=np.zeros_like(analytic)
    for axis in range(6):
        d=np.zeros(6);d[axis]=epsilon
        plus=odom.icp.transform_points(odom.icp.se3_exp(d)@transform,q)
        minus=odom.icp.transform_points(odom.icp.se3_exp(-d)@transform,q)
        rplus,_,uvplus=odom.intensity_residual_jacobian(plus,K,image,source)
        rminus,_,uvminus=odom.intensity_residual_jacobian(minus,K,image,source)
        require(np.array_equal(np.floor(uvplus),np.floor(queries))and np.array_equal(np.floor(uvminus),np.floor(queries)),'FD cell identity')
        numeric[:,axis]=(rplus-rminus)/(2*epsilon)
    error=float(np.max(np.abs(analytic-numeric)));require(error<1e-7,'analytic bilinear photo 6D Jacobian')
    # Separate geometric Jacobian probe at an off-identity transform.
    normals=np.array([[0.,0.,-1.],[0.,-.6,-.8],[.3,.4,-math.sqrt(.75)],[.8,0.,-.6]])
    targets=p+np.array([.003,-.004,.005]);_,geo=odom.icp.residual_and_jacobian(p,targets,normals)
    gnumeric=np.zeros_like(geo)
    for axis in range(6):
        d=np.zeros(6);d[axis]=epsilon
        pp=odom.icp.transform_points(odom.icp.se3_exp(d)@transform,q);pm=odom.icp.transform_points(odom.icp.se3_exp(-d)@transform,q)
        gnumeric[:,axis]=np.sum(normals*(pp-pm),axis=1)/(2*epsilon)
    geo_error=float(np.max(np.abs(geo-gnumeric)));require(geo_error<1e-7,'geometric six-dimensional Jacobian')
    constant=np.full_like(image,.4);res,jconstant,_=odom.intensity_residual_jacobian(p,K,constant,np.full(len(p),.4))
    require(np.count_nonzero(jconstant)==0 and np.max(np.abs(res))<1e-15,'constant texture contributes zero photo Jacobian')
    return {'status':'passed','epsilon':epsilon,'off_identity_base_T':transform.tolist(),'points_current':q.tolist(),
        'projected_uv':queries.tolist(),'minimum_cell_boundary_distance_pixels':float(distances.min()),
        'analytic_photo_6D':analytic.tolist(),'FD_photo_6D':numeric.tolist(),'maximum_photo_jacobian_error':error,
        'maximum_geo_jacobian_error':geo_error,'constant_texture_photo_jacobian_exact_zero':True}


def check_all():
    started=time.perf_counter();config=odom.OdometryConfig();odom.frozen_source_check()
    check=jacobian_controls(config)
    exponential_error=0.
    for twist in(np.zeros(6),np.array([.02,-.01,.03,.01,-.02,.015]),np.array([.2,.3,-.4,1e-10,-2e-10,3e-10])):
        algebra=np.zeros((4,4));algebra[:3,:3]=odom.icp.skew(twist[3:]);algebra[:3,3]=twist[:3]
        exponential_error=max(exponential_error,float(np.max(np.abs(odom.icp.se3_exp(twist)-expm(algebra)))))
    require(exponential_error<1e-12,'SE3 exponential independent scipy matrix exp')
    truth=odom.icp.se3_exp(np.array([.025,-.018,.012,.010,-.012,.009]))
    target=build_pyramid(np.eye(4),True,config);source=build_pyramid(truth,True,config)
    initial=odom.pair_correspondences(np.eye(4),source[-1],target[-1],config.geometry.distance_gates_m[-1],config)
    r_geo,j_geo,r_photo,j_photo,_=odom.terms(np.eye(4),initial,target[-1])
    geometry_only=replace(config,lambda_photo=0.)
    _,geo_stats=odom.weighted_system(r_geo,j_geo,r_photo,j_photo,geometry_only)
    _,joint_stats=odom.weighted_system(r_geo,j_geo,r_photo,j_photo,config)
    require(geo_stats['joint_rank']==3 and joint_stats['joint_rank']==6,'texture lifts front-plane rank3 to6')
    recovered=odom.register_pair(source,target,config)
    require(recovered['accepted'],'textured plane pair accepted '+str(recovered['failure_flags']))
    estimate=np.asarray(recovered['candidate_previous_from_current']);error=odom.icp.inverse_rigid(truth)@estimate
    translation=float(np.linalg.norm(error[:3,3]));rotation=odom.icp.rotation_degrees(error[:3,:3])
    require(translation<5e-5 and rotation<.005,'toy known pose recovery')
    constant_target=build_pyramid(np.eye(4),False,config);constant_source=build_pyramid(truth,False,config)
    flat=odom.register_pair(constant_source,constant_target,config)
    require(not flat['accepted']and flat['final_statistics']['joint_rank']==3 and 'final_unobservable_6d_pose'in flat['failure_flags'],'constant plane degeneracy retained')
    # Same geometry plus a declared brightness-constancy violation. No brightness
    # correction is fitted, and no weight/threshold sweep uses ground truth.
    bright=odom.register_pair(build_pyramid(truth,True,config,.2),target,config)
    require(not bright['accepted'],'large brightness mismatch must not be claimed successful')
    previous=odom.icp.se3_exp(np.array([.3,-.2,.1,.15,.1,-.2]))
    integrated=odom.integrate_relative(previous,recovered)
    rejected=odom.integrate_relative(integrated,bright)
    require(np.array_equal(integrated,rejected),'failed frame preserves previous pose')
    points_current=np.array([[.1,-.2,2.],[-.3,.4,2.5]])
    expected_world=odom.icp.transform_points(previous,odom.icp.transform_points(estimate,points_current))
    composed_world=odom.icp.transform_points(integrated,points_current)
    composition_error=float(np.max(np.abs(expected_world-composed_world)))
    wrong=estimate@previous;wrong_error=float(np.max(np.abs(wrong-integrated)))
    require(composition_error<1e-12 and wrong_error>.005,'noncommuting current->previous composition order')
    # The residual at the constructed true pose is inspected after recovery,
    # and never read inside register_pair/line search to choose an estimate.
    truth_pairs=odom.pair_correspondences(truth,source[-1],target[-1],config.geometry.distance_gates_m[-1],config)
    truth_terms=odom.terms(truth,truth_pairs,target[-1])
    return {'status':'passed','configuration':asdict(config),'jacobian_checks':check,
        'SE3_vs_scipy_expm_maximum_error':exponential_error,
        'textured_plane_observability':{'geometric_only':geo_stats,'joint':joint_stats},
        'textured_plane_recovery':{'true_T_previous_from_current':truth.tolist(),'translation_error_m':translation,
            'rotation_error_degrees':rotation,'true_pose_photo_rmse':float(np.sqrt(np.mean(truth_terms[2]**2))),
            'registration':recovered},'constant_texture_degeneracy':flat,'brightness_offset_02_control':bright,
        'composition':{'previous_c2w':previous.tolist(),'accepted_current_c2w':integrated.tolist(),
            'noncommuting_wrong_order_maximum_matrix_error':wrong_error,'correct_world_point_maximum_error':composition_error,
            'rejected_frame_current_c2w':rejected.tolist(),'all_rejected_frame_and_candidate_artifacts_retained':True},
        'runtime_seconds':time.perf_counter()-started,'CPU_only':True,'real_data_optimization_runs':0,'GPU_runs':0,
        'limitations':['Synthetic algebra/raster controls only, not evidence of TUM accuracy improvement.',
            'Bilinear world-plane texture is deliberately representable by the reference interpolation; this isolates algebra and is not a real-camera benchmark.',
            'Fixed gray=.299R+.587G+.114B in 0..1 is not radiometric calibration; no affine brightness/exposure/gamma/motion compensation is estimated.',
            'Scales .01m/.1 intensity, photo weight1 and normalized Huber delta1 are engineering choices, not noise covariance or GT-selected information.',
            'Final intensity RMSE gate .1 is an explicitly declared engineering rejection rule, unvalidated on real data.',
            'Photo derivatives are exact inside the bilinear cell; masks, nearest depth correspondences and cell boundaries remain piecewise nonsmooth.',
            'Rank/condition reflect the current accepted correspondences, not a global convergence guarantee or covariance.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',default='rgbd-photogeometric-cpu-controls-001');args=parser.parse_args()
    require(args.run and all(c in'abcdefghijklmnopqrstuvwxyz0123456789-'for c in args.run),'unique safe output name')
    output=ROOT/'outputs'/args.run;require(not output.exists(),'never overwrite previous controls')
    output.mkdir()
    source_hashes={name:sha(ROOT/'scripts'/name)for name in('rgbd_photogeometric_odometry.py','check_rgbd_photogeometric_odometry.py','track_rgbd_icp.py','tum_rgbd.py')}
    for name in('rgbd_photogeometric_odometry.py','check_rgbd_photogeometric_odometry.py'):shutil.copyfile(ROOT/'scripts'/name,output/name)
    json_write(output/'source_manifest.json',{'source_sha256':source_hashes,'real_data_loaded':False,'GPU_used':False})
    try:result=check_all()
    except BaseException as error:
        json_write(output/'failure.json',{'status':'failed','exception':type(error).__name__,'message':str(error),'source_sha256':source_hashes,'real_data_loaded':False,'GPU_used':False})
        raise
    result['source_sha256']=source_hashes;json_write(output/'metrics.json',result)
    print(json.dumps({'status':result['status'],'output':str(output),'photo_jacobian_maximum_error':result['jacobian_checks']['maximum_photo_jacobian_error'],
        'toy_translation_error_m':result['textured_plane_recovery']['translation_error_m'],'toy_rotation_error_degrees':result['textured_plane_recovery']['rotation_error_degrees'],
        'geo_rank':result['textured_plane_observability']['geometric_only']['joint_rank'],'joint_rank':result['textured_plane_observability']['joint']['joint_rank'],
        'brightness_failure_flags':result['brightness_offset_02_control']['failure_flags'],'source_sha256':source_hashes,'GPU_runs':0},indent=2))


if __name__=='__main__':main()
