from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from datetime import datetime
R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\smplx-gvhmr-shared-world-fresh-025')
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
init=load(R/'free/initialization.npz');start=load(R/'free/starting-state.npz')
s0=float(init['initial_world_scale']);beta0=np.median(init['betas'],axis=0)
results={}
for arm in ['free','shared']:
    fit=load(R/arm/'fitted.npz');N=len(fit['frame_ids'])
    correction=(Rotation.from_rotvec(fit['body_pose'].reshape(-1,3))*Rotation.from_rotvec(start['body_pose'].reshape(-1,3)).inv()).as_rotvec().reshape(N,21,3)
    root=(Rotation.from_matrix(fit['root_rotation'])*Rotation.from_matrix(start['root_rotation']).inv()).as_rotvec()
    trans=(fit['world_translation']-start['world_translation'])/s0
    beta=fit['betas'].ravel()-beta0
    limits=np.full((21,1),.4)
    if arm=='shared':limits[:11]=.3
    ratios=abs(correction)/limits
    results[arm]=dict(shape_beta_delta=beta.tolist(),shape_components_near_bound=np.flatnonzero(abs(beta)>.95).tolist(),
        local_pose_max_fraction=float(ratios.max()),local_pose_components_near_bound=int((ratios>.95).sum()),
        root_max_fraction=float(abs(root).max()/.35),translation_max_fraction=float(abs(trans).max()/.30),
        scale_log_fraction=float(abs(np.log(float(fit['world_scale'])/s0))/.30),
        per_joint_pose_max_fraction=ratios.max(axis=(0,2)).tolist())
out=dict(created=datetime.now().astimezone().isoformat(timespec='seconds'),read_only=True,
         results=results,interpretation='A saturated parameter can limit fitting, but saturation alone does not prove relaxed shape is anatomically correct; silhouette is clothing and keypoints are detector observations.')
(R/'parameter-bounds-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results))
