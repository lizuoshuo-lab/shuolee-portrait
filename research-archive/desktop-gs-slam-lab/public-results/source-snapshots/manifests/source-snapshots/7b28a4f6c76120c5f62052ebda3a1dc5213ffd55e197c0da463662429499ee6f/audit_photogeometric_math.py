"""Additional CPU hand calculations and normalized-objective FD controls.

Uses scipy matrix exponential and a separate scalar four-corner interpolator
for finite differences. No real data, renderer, GPU or optimizer is executed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import rgbd_photogeometric_odometry as odom

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/rgbd-photogeometric-cpu-controls-003'


def require(condition,message):
    if not condition:raise AssertionError(message)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def hat(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def matrix_exponential(delta):
    A=np.zeros((4,4));A[:3,:3]=hat(delta[3:]);A[:3,3]=delta[:3]
    return expm(A)


def sample_scalar(image,uv):
    x,y=map(float,uv);u,v=math.floor(x),math.floor(y);a=x-u;b=y-v
    return ((1-a)*(1-b)*float(image[v,u])+a*(1-b)*float(image[v,u+1])+
            (1-a)*b*float(image[v+1,u])+a*b*float(image[v+1,u+1]))


def robust_scalar(r):
    x=abs(float(r))
    return .5*x*x if x<=1 else x-.5


def main():
    original=json.loads((OUT/'metrics.json').read_text())
    for name,digest_value in original['source_sha256'].items():require(digest(ROOT/'scripts'/name)==digest_value,'success003 source still fixed '+name)
    config=odom.OdometryConfig()
    sample=np.array([[.1,.4],[.2,.9]])
    value,gradient,inside=odom.bilinear(sample,np.array([[.3,.6]]))
    # Hand weights .28,.12,.42,.18 and corner differences .3,.7/.1,.5.
    require(inside[0]and abs(value[0]-.322)<1e-15 and np.max(np.abs(gradient[0]-[.54,.22]))<1e-15,'four-corner hand value/gradient')
    RGB=np.eye(3).reshape(1,3,3);gray=odom.grayscale(RGB).ravel()
    require(np.array_equal(gray,np.array([.299,.587,.114])),'fixed RGB weights')
    h,w=60,80;v,u=np.mgrid[:h,:w]
    image=.3+.002*u+.003*v+.00004*u*v+.02*np.sin(u*.11)*np.cos(v*.09)
    K=np.array([[60.,0.,39.5],[0.,60.,29.5],[0.,0.,1.]])
    T=np.asarray(original['jacobian_checks']['off_identity_base_T'])
    q=np.asarray(original['jacobian_checks']['points_current'])
    p=q@T[:3,:3].T+T[:3,3]
    source_intensity=np.array([.2,.4,.5,.6])
    normals=np.array([[0.,0.,-1.],[0.,-.6,-.8],[.3,.4,-math.sqrt(.75)],[.8,0.,-.6]])
    target=p+np.array([[-.002,.0025,.0018],[-.02,.025,.018],
                       [-.002,.0025,.0018],[-.02,.025,.018]])
    rg,jg=odom.icp.residual_and_jacobian(p,target,normals)
    rp,jp,queries=odom.intensity_residual_jacobian(p,K,image,source_intensity)
    # Derive [I,-skew(p)] explicitly and multiply two projection rows;
    # implementation uses its algebraically simplified cross-product form.
    independent_photo=[]
    for point,uv in zip(p,queries):
        x,y,z=point;pixel_x,pixel_y=np.floor(uv).astype(int)
        a=uv[0]-pixel_x;b=uv[1]-pixel_y
        i00,i10=image[pixel_y,pixel_x],image[pixel_y,pixel_x+1]
        i01,i11=image[pixel_y+1,pixel_x],image[pixel_y+1,pixel_x+1]
        grad=np.array([[(1-b)*(i10-i00)+b*(i11-i01),(1-a)*(i01-i00)+a*(i11-i10)]])
        Jproject=np.array([[60/z,0.,-60*x/z**2],[0.,60/z,-60*y/z**2]])
        Jpose=np.concatenate((np.eye(3),-hat(point)),axis=1)
        independent_photo.append((grad@Jproject@Jpose).ravel())
    chain_error=float(np.max(np.abs(np.asarray(independent_photo)-jp)))
    require(chain_error<1e-14,'explicit chain agrees with implementation')
    geo_normalized=rg/config.sigma_geo_m;photo_normalized=rp/config.sigma_intensity
    analytical=(jg.T@np.clip(geo_normalized,-1,1)/config.sigma_geo_m+
        config.lambda_photo*jp.T@np.clip(photo_normalized,-1,1)/config.sigma_intensity)/len(p)
    def independent_loss(delta):
        pose=matrix_exponential(delta)@T
        points=q@pose[:3,:3].T+pose[:3,3]
        g=np.sum(normals*(points-target),axis=1)/config.sigma_geo_m
        uv=np.c_[60*points[:,0]/points[:,2]+39.5,60*points[:,1]/points[:,2]+29.5]
        require(np.array_equal(np.floor(uv),np.floor(queries)),'objective FD same bilinear cell')
        i=(np.array([sample_scalar(image,x)for x in uv])-source_intensity)/config.sigma_intensity
        return (math.fsum(robust_scalar(x)for x in g)+config.lambda_photo*math.fsum(robust_scalar(x)for x in i))/len(p)
    eps=1e-7;numeric=[]
    for axis in range(6):
        d=np.zeros(6);d[axis]=eps
        numeric.append((independent_loss(d)-independent_loss(-d))/(2*eps))
    gradient_error=float(np.max(np.abs(analytical-numeric)))
    require(gradient_error<5e-7,'Huber normalized combined objective six-dimensional gradient')
    require(np.any(np.abs(geo_normalized)>1)and np.any(np.abs(geo_normalized)<1)and
            np.any(np.abs(photo_normalized)>1)and np.any(np.abs(photo_normalized)<1),'Huber quadratic and linear branches both exercised')
    require(abs(independent_loss(np.zeros(6))-odom.objective(rg,rp,config))<1e-14,'manual objective mean/scales')
    result={'status':'passed','audit_source_sha256':digest(Path(__file__)),'original_success_metrics_sha256':digest(OUT/'metrics.json'),
        'source_sha256':original['source_sha256'],'hand_bilinear':{'uv':[.3,.6],'corners':sample.tolist(),
            'weights':[.28,.12,.42,.18],'expected_intensity':.322,'actual_intensity':float(value[0]),
            'expected_gradient_du_dv':[.54,.22],'actual_gradient_du_dv':gradient[0].tolist()},
        'explicit_projection_pose_chain_maximum_error':chain_error,'normalized_geo_residual':geo_normalized.tolist(),
        'normalized_photo_residual':photo_normalized.tolist(),'Huber_delta':1.,'epsilon':eps,
        'analytic_combined_objective_6D':analytical.tolist(),'independent_FD_combined_objective_6D':numeric,
        'maximum_combined_objective_gradient_error':gradient_error,'both_huber_branches_exercised':True,
        'FD_exponential':'scipy.linalg.expm, separate scalar intensity interpolation/Huber summation',
        'no_real_data_loaded':True,'GPU_used':False,'limits':'Mathematical synthetic control only; no independent real-data improvement evidence.'}
    with (OUT/'supplementary_math_controls.json').open('x',encoding='utf-8')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'chain_error':chain_error,'normalized_objective_gradient_error':gradient_error,
                      'audit_source_sha256':result['audit_source_sha256'],'real_data_optimization_runs':0,'GPU_runs':0},indent=2))


if __name__=='__main__':main()
