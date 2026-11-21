"""Original COLMAP -> rectified training -> actual CUDA; read-only diagnostic."""
from pathlib import Path
import sys,json,time
import numpy as np,cv2
from PIL import Image
G=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')
sys.path.insert(0,str(G))
import shuolee_common as c
torch=c.runtime()
from scene.cameras import CameraProj2
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import read_points3D_binary,read_extrinsics_binary,read_intrinsics_binary
from gaussian_renderer import render
from types import SimpleNamespace
O=c.BASE/'current/camera-feature-root-cause-115';S=c.BASE/'current/full-scene-localization-074';P=c.BASE/'current/avatar-repository-pair-114'
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
def stats(x):
    x=np.asarray(x).ravel();x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x))) if len(x) else dict(n=0)
def project(x,e,k):
    v=x@e[:3,:3].T+e[:3,3];q=v@k.T
    return q[:,:2]/q[:,2:]-.5,v
@torch.no_grad()
def main():
    cv2.setNumThreads(4);assert not (O/'projection-result.json').exists()
    rows=c.read(P/'smplx/data/training-inputs.json')['records'];training=c.read(P/'smplx/gaussian-run/training-cameras.json')['records']
    raw=c.read(S/'cameras.json');gauge=c.read(S/'camera-gauge-comparison.json')
    Q=np.array(gauge['rotation']);t=np.array(gauge['translation']);s=gauge['scale']
    f,cx,cy,k=raw['intrinsics'];Kraw=np.array([[f,0,cx-.5],[0,f,cy-.5],[0,0,1.]])
    d=np.array([k,0,0,0,0.]);K=np.array(rows[0]['K']);Kcv=K.copy();Kcv[:2,2]-=.5
    rawxyz,_,_=read_points3D_binary(str(S/'refined_sparse/points3D.bin'))
    rawcams=read_intrinsics_binary(str(S/'refined_sparse/cameras.bin'))
    rawim=read_extrinsics_binary(str(S/'refined_sparse/images.bin'));byname={im.name:im for im in rawim.values()}
    assert len(rawcams)==1 and np.max(abs(next(iter(rawcams.values())).params-np.array(raw['intrinsics'])))<1e-10
    c.save(O/('projection-protocol.json' if not (O/'projection-protocol.json').exists() else 'projection-protocol-repaired.json'),dict(created=c.stamp(),source_code_sha256=c.sha(__file__),
        original='Native refined COLMAP binary cameras and points, independently read.',
        scope='All175 current actual114 CameraProj2/CUDA paths; raw3Dpoints plus9 independent colored-free Gaussian probes per frame.',
        tests=['Verify cameras.json equals native COLMAP binary poses/parameters.',
               'Apply original distorted projection, OpenCV undistortion and recorded image resize.',
               'Compare actual CUDA-rendered intensity centroids after the single world Sim3.',
               'Negative controls: wrong raw/rectified K, omitted Sim3, neighboring camera.'],
        thresholds=dict(raw_to_training_analytic_max_px=.001,cuda_probe_max_px=.03),
        no_training=True,no_source_updates=True,physical_calibration_truth_test=False))
    checks=[];st=time.monotonic();gputensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
    for i,(r,tr,sr) in enumerate(zip(rows,training,raw['records'])):
        e=np.array(r['w2c']);er=np.array(sr['w2c']);im=byname[sr['name']]
        eb=np.eye(4);eb[:3,:3]=im.qvec2rotmat();eb[:3,3]=im.tvec
        assert np.max(abs(eb-er))<1e-10
        W,H=tr['wh'];sx,sy=W/r['wh'][0],H/r['wh'][1];kt=np.array(tr['K'])
        assert np.max(abs(kt-np.diag([sx,sy,1.])@K))<1e-10
        cam=CameraProj2(i,e[:3,:3].T,e[:3,3],kt[0,0],kt[1,1],'black',W,None,H,r['image'],r['frame'],i,cx=kt[0,2],cy=kt[1,2],timestep=0)
        # Original SfM cloud, not reconstructed-person points.
        xx=rawxyz[::97];cp=xx@er[:3,:3].T+er[:3,3];xy=cp[:,:2]/cp[:,2:]
        ru=xy*(1+k*(xy*xy).sum(-1,keepdims=True))*f+Kraw[:2,2]
        rect=cv2.undistortPointsIter(ru[:,None],Kraw,d,None,Kcv,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12))[:,0]
        good=(cp[:,2]>.1)&(rect[:,0]>=5)&(rect[:,0]<r['wh'][0]-5)&(rect[:,1]>=5)&(rect[:,1]<r['wh'][1]-5)
        xx=xx[good];rect=rect[good];assert len(xx)>10
        aligned=s*xx@Q.T+t;uv,_=project(aligned,e,kt)
        expected=(rect+.5)*[sx,sy]-.5;analytic=np.linalg.norm(uv-expected,axis=1)
        wrongk=kt.copy();wrongk[0,2]=cx*sx;wrongk[1,2]=cy*sy
        wrongpr,_=project(aligned,e,wrongk)
        omitted,_=project(xx,e,kt)
        nextpr,_=project(aligned,np.array(rows[min(i+1,174)]['w2c']),kt)
        # Probes constructed in ORIGINAL world gauge then transformed once.
        dest=np.array([[a*W+.231,b*H+.173] for b in [.25,.5,.75] for a in [.25,.5,.75]])
        rc=(dest+.5)/[sx,sy]-.5
        camdir=np.c_[(rc-Kcv[:2,2])/np.diag(Kcv)[:2],np.ones(9)]
        z=np.array([2.,3.,4.,3.,4.,2.,4.,2.,3.]);pc=camdir*z[:,None]
        original=(pc-er[:3,3])@er[:3,:3];world=s*original@Q.T+t
        model=GaussianModel(0);model._xyz=gputensor(world)
        model._features_dc=torch.zeros((9,1,3),device='cuda');model._features_rest=torch.zeros((9,0,3),device='cuda')
        sigma=2.5*z*s/kt[0,0]
        model._scaling=gputensor(np.log(np.repeat(sigma[:,None],3,axis=1)))
        model._rotation=gputensor(np.tile([1.,0,0,0],(9,1)));model._opacity=gputensor(np.full((9,1),np.log(9.)))
        pkg=render(cam,model,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones((9,3),device='cuda'))
        raster=pkg['render'][0].cpu().numpy();centers=[]
        for u,v in dest:
            px,py=int(round(u)),int(round(v));x0,x1=px-20,px+21;y0,y1=py-20,py+21
            patch=raster[y0:y1,x0:x1];yy,xxx=np.mgrid[y0:y1,x0:x1];assert patch.sum()>0
            centers.append([(patch*xxx).sum()/patch.sum(),(patch*yy).sum()/patch.sum()])
        errors=np.linalg.norm(np.array(centers)-dest,axis=1)
        check=dict(raw_frame=i+1,original_SfM_samples=len(xx),raw_to_native_point_projection_px=stats(analytic),
            actual_CUDA_probe_error_px=errors.tolist(),camera_bin_matches_metadata=True,
            incorrect_raw_principal_point_px=stats(np.linalg.norm(wrongpr-expected,axis=1)),
            missing_world_transform_px=stats(np.linalg.norm(omitted-expected,axis=1)),
            neighboring_camera_px=stats(np.linalg.norm(nextpr-expected,axis=1)) if i<174 else None)
        checks.append(check)
        if i%35==0 or i==174:print(json.dumps(dict(stage='native_camera_projection',frames=i+1,maximum_centroid_px=float(errors.max()),seconds=time.monotonic()-st)),flush=True)
    # Detector slot mapping is separate from RGB frame mapping.
    obs=np.load(P/'smplx/data/observations.npz');timing=np.load(P/'smplx/inference/time-camera-inputs.npz')
    slots=np.array([r['official_prediction_slot'] for r in rows]);times=np.array([r['time_seconds'] for r in rows])
    raw_ids=np.array([r['source_frame_index'] for r in rows])
    detector_slot_frame_error=timing['nearest_source_indices'][slots]-raw_ids
    detector_time_error=np.abs(timing['source_pts_seconds'][timing['nearest_source_indices'][slots]]-times)
    vit=torch.load(P/'smplx/inference/vitpose-keypoints.pt',map_location='cpu',weights_only=True).numpy()[slots]
    result=dict(created=c.stamp(),state='projection_transfer_checked',camera_frames=175,cuda_probes=1575,
        actual_CUDA_probe_error_training_px=stats(np.array([x['actual_CUDA_probe_error_px'] for x in checks])),
        original_SfM_to_training_max_px=max(x['raw_to_native_point_projection_px']['maximum'] for x in checks),
        original_cloud_observations_checked=sum(x['original_SfM_samples'] for x in checks),
        native_camera_binary_matches_metadata=True,official_detector_slot_time_error_seconds=stats(detector_time_error),
        official_detector_slot_frame_error=stats(abs(detector_slot_frame_error)),
        detector_observation_disagreement_px=stats(np.linalg.norm(obs['keypoints'][:,:17]-vit[:,:,:2],axis=-1)),
        all_175_mapping_passed=bool(max(x['raw_to_native_point_projection_px']['maximum'] for x in checks)<.001 and max(max(x['actual_CUDA_probe_error_px']) for x in checks)<.03),
        physical_camera_accuracy_proven=False,training_started=False,source_updates=0,seconds=time.monotonic()-st)
    c.save(O/'projection-records.json',checks);c.save(O/'projection-result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
