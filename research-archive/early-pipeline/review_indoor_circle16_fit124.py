"""Reuse the existing native mesh review for all16registered indoor views."""
import inspect
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import review_indoor_selected_views124 as base
from run_official_gaussians123 import read,save,sha,now
import run_indoor_circle16_fit124 as run


def main():
    assert read(run.OUT/'process-exit.json')['exit_code']==0
    env=dict(base.__dict__,OUT=run.OUT,__file__=__file__)
    source=inspect.getsource(base.main)
    assert source.count('室内六帧联合124')==2
    source=source.replace('室内六帧联合124','室内一圈16帧联合124')
    exec(compile(source,'<same-all-view-native-review16>','exec'),env)
    env['main']()
    save(run.OUT/'review/reviewer-provenance.json',dict(updated=now(),base_source_sha256=sha(base.__file__),
        reviewer_sha256=sha(__file__),input_roster_only=True,optimizer_steps=0))
    current=dict(np.load(run.OUT/'selected-fit.npz',allow_pickle=False))
    old=dict(np.load(run.prior.original.ROOT/'indoor-selected-sh0-pipeline124/fit/fitted.npz',allow_pickle=False))
    # OpenPose midhip maps to the original SMPL-X pelvis, independently checked
    # against the previously exported shape-dependent pivot/world translation.
    error=float(np.max(abs(old['joints_camera'][:,8]-old['world_translation'])))
    assert error<1e-6
    R=Rotation.from_rotvec(current['global_orient']).as_matrix()
    ray=np.einsum('nji,nj->ni',R,-current['joints_camera'][:,8])
    yaw=np.degrees(np.arctan2(ray[:,0],ray[:,2]))%360
    elev=np.degrees(np.arctan2(ray[:,1],np.linalg.norm(ray[:,[0,2]],axis=1)))
    ids=current['frame_ids'].astype(str).tolist();order=np.argsort(yaw)
    gaps=[dict(start_frame=ids[int(order[i])],end_frame=ids[int(order[(i+1)%len(ids)])],
        degrees=float((yaw[order[(i+1)%len(ids)]]-yaw[order[i]])%360)) for i in range(len(ids))]
    gaps.sort(key=lambda r:r['degrees'],reverse=True)
    old_metrics={r['frame']:r for r in read(run.BASE/'result.json')['metrics']}
    new_metrics={r['frame']:r for r in read(run.OUT/'result.json')['metrics']}
    comparisons=[]
    for f in old['frame_ids'].astype(str):
        b=old_metrics[f];n=new_metrics[f]
        comparisons.append(dict(frame=f,old_arm_error_px=b['arm_error_px'],new_arm_error_px=n['arm_error_px'],
            old_face_rms_px=b['face_weighted_rms_px'],new_face_rms_px=n['face_weighted_rms_px']))
    save(run.OUT/'review/direction-coverage.json',dict(updated=now(),frames=len(ids),
        angles=[dict(frame=f,yaw_proxy_deg=float(yaw[i]),elevation_proxy_deg=float(elev[i])) for i,f in enumerate(ids)],
        circular_gaps=gaps,previous_six_max_gap_deg=125.40682616426503,
        mapped_midhip_vs_pelvis_oracle_m=error,common_six_comparison=comparisons,
        actual_angular_spacing_is_not_uniform=True,physical_calibration_claimed=False,
        limitation='Conditional fitted roots and pelvis; pose errors affect directions. Compare raw/source overlays, not angle numbers alone.',
        old_pose_reused_as_seed=False,quality_accepted=False))


if __name__=='__main__':main()
