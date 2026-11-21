"""Synthetic checks only; no asserted calibration of a real exported video."""
from pathlib import Path
import sys, json, copy
import numpy as np
import cv2
sys.path.insert(0, '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
from shuolee_portrait.camera_pixel_contract import rectified_contract, map_pixels, resize_matrix, rectify_extrinsics

spec=dict(input_pixel_domain='rectified_pinhole',pixel_convention='integer_centers',
  fixed_processing_across_frames=True,K_rectified=[[1661.,0,1997.3],[0,1658.,1498.1],[0,0,1]],
  rectified_wh=[4000,3000],downsampled_wh=[2000,1500],crop_xywh=[61,221,1800,1000],
  training_wh=[1440,800],rectification_R=np.eye(3).tolist())
rec=rectified_contract(spec);A=np.array(rec['rectified_to_training']);K=np.array(spec['K_rectified'])
rng=np.random.default_rng(120);xyz=rng.uniform([-1,-1,3],[1,1,7],(1000,3))
q=xyz@K.T;pixels=q[:,:2]/q[:,2,None]
mapped=map_pixels(pixels,A);projection=xyz@np.array(rec['K_integer_centers']).T;projection=projection[:,:2]/projection[:,2,None]
error=float(np.max(np.abs(mapped-projection)));assert error<1e-9
inverse=float(np.max(np.abs(map_pixels(mapped,rec['training_to_rectified'])-pixels)));assert inverse<1e-9
# Independent pixel-ramp image, using actual OpenCV resize and ROI slicing.
ys,xs=np.mgrid[:3000,:4000];im=np.stack([xs,ys],-1).astype(np.float32)
actual=cv2.resize(im,(2000,1500),interpolation=cv2.INTER_LINEAR)[221:1221,61:1861]
actual=cv2.resize(actual,(1440,800),interpolation=cv2.INTER_LINEAR)
dst=np.array([[15,33],[701,451],[1400,760]],float)
expected=map_pixels(dst,np.linalg.inv(A));actual_values=actual[dst[:,1].astype(int),dst[:,0].astype(int)]
ramp_error=float(abs(expected-actual_values).max());assert ramp_error<2e-4
# Scaled source-space ROI is equivalent; order alone is not a causal diagnosis.
equivalent=resize_matrix([3600,2000],[1440,800])@np.array([[1,0,-122],[0,1,-442],[0,0,1.]])
assert np.allclose(A,equivalent)
rejected=[]
for k,v in [('K_rectified',None),('crop_xywh',None),('crop_xywh',[61,221.2,1800,1000]),('fixed_processing_across_frames',False),('rectification_R',None),('input_pixel_domain','FISHEYE_8')]:
    bad=copy.deepcopy(spec);bad[k]=v
    try:rectified_contract(bad)
    except ValueError:rejected.append(k)
    else:raise AssertionError(k)
e=np.eye(4);e[:3,3]=[.1,.2,.3];r=cv2.Rodrigues(np.array([.05,-.02,.01]))[0]
assert np.allclose(rectify_extrinsics(e,r)[:3],r@e[:3])
result=dict(synthetic_only=True,real_dataset_calibrated=False,projection_points=1000,max_projection_error_px=error,
  roundtrip_error_px=inverse,opencv_resize_crop_ramp_error_px=ramp_error,scaled_crop_order_equivalence=True,
  invalid_contracts_rejected=rejected,training_started=False)
Path(__file__).with_name('camera-pixel-contract-tests120.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
