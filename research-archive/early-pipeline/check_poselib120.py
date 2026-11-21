"""Verify calibrated pose API in isolated small-package install; synthetic only."""
import sys,json,hashlib
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
import poselib
rng=np.random.default_rng(120)
K=np.array([[820.,0,773.25],[0,791.,463.75],[0,0,1.]])
cam=dict(model='PINHOLE',width=1600,height=900,params=[K[0,0],K[1,1],K[0,2],K[1,2]])
X=rng.uniform([-1.8,-1.2,4],[1.8,1.2,8],(240,3))
R=cv2.Rodrigues(np.array([.04,-.12,.02]))[0];t=np.array([.65,.06,.08])
def project(q):
    q=q@K.T;return q[:,:2]/q[:,2,None]
a=project(X);b=project(X@R.T+t)
# 20% mismatched physical points, with no new image/template/weight ingestion.
bad=b.copy();bad[:48]=bad[np.arange(48)[::-1]]
pose,info=poselib.estimate_relative_pose(a,bad,cam,cam,dict(max_epipolar_error=1.,max_iterations=10000,min_iterations=1000,seed=120),{})
absolute,abs_info=poselib.estimate_absolute_pose(bad,X,cam,dict(max_reproj_error=1.,max_iterations=10000,min_iterations=1000,seed=120),{})
angle=lambda r:float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))
rot=angle(pose.R@R.T);trans=float(np.degrees(np.arccos(np.clip(np.dot(pose.t,t)/(np.linalg.norm(pose.t)*np.linalg.norm(t)),-1,1))))
arot=angle(absolute.R@R.T);apos=float(np.linalg.norm(absolute.t-t))
assert rot<1e-4 and trans<1e-4 and arot<1e-4 and apos<1e-5,(rot,trans,arot,apos)
result=dict(package='poselib==2.0.5',isolated_install='/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5',
  synthetic_points=240,mismatched_pairs=48,relative_rotation_error_degrees=rot,translation_direction_error_degrees=trans,
  absolute_rotation_error_degrees=arot,absolute_translation_error=apos,relative_inliers=int(info['num_inliers']),absolute_inliers=int(abs_info['num_inliers']),
  K_optimized=False,real_video_evaluated=False,physical_correspondences_verified=False,training_started=False,
  note='Calibrated relative/absolute pose API works; this is not a full multi-view reconstruction or proof of real-video accuracy.')
Path(__file__).with_name('poselib-preflight120.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
