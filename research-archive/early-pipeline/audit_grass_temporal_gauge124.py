"""Read-only root-frame transport audit before reusing temporal fitting."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_official_gaussians123 import read,save,sha,now


def angle(a,b):
    d=np.swapaxes(a,-1,-2)@b
    return np.degrees(np.arccos(np.clip((np.trace(d,axis1=-2,axis2=-1)-1)/2,-1,1)))


def main():
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/grass')
    out=root/'pose-cause-audit'
    source=Path(read(root/'plan.json')['source_fit'])
    assert not (out/'temporal-gauge.json').exists()
    digest=sha(source)
    save(out/'temporal-gauge-plan.json',dict(updated=now(),mode='read_only_cpu',
        hypothesis='The grass source E is not identity. A copied camera-relative root is different from a world-consistent neighbouring root.',
        source=str(source),source_sha256=digest,source_script_sha256=sha(__file__),
        historical_pose_is_analysis_only=True,no_new_pose_initialization=True,
        budget=dict(executions=1,cpu_seconds=120),no_gpu_or_fitting=True))
    with np.load(source,allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str);order=np.argsort(ids)
        ids=ids[order];E=z['w2c'][order];world=z['root_rotation'][order]
        camera=Rotation.from_rotvec(z['global_orient'][order].reshape(-1,3)).as_matrix()
    assert len(ids)==94 and len(set(ids))==94
    assert all(int(b[-6:])-int(a[-6:])==1 for a,b in zip(ids[:-1],ids[1:]))
    rot=E[:,:3,:3]
    parity=float(np.max(np.abs(rot@world-camera)))
    determinant_error=float(np.max(np.abs(np.linalg.det(rot)-1)))
    orthogonality_error=float(np.max(np.abs(np.swapaxes(rot,1,2)@rot-np.eye(3))))
    transported=rot[1:]@np.swapaxes(rot[:-1],1,2)@camera[:-1]
    same_world_in_current_camera=rot[1:]@world[:-1]
    transport_error=float(np.max(np.abs(transported-same_world_in_current_camera)))
    checks=dict(root_world_camera_parity=parity<1e-6,proper_camera_rotations=determinant_error<1e-6 and orthogonality_error<1e-6,
        transport_matches_same_world_rotation=transport_error<1e-6,
        source_unchanged=sha(source)==digest)
    assert all(checks.values()),checks
    pairs=[]
    for i in range(1,len(ids)):
        pairs.append(dict(previous=str(ids[i-1]),current=str(ids[i]),
            copied_vs_transported_root_deg=float(angle(camera[i-1],transported[i-1])),
            fitted_camera_root_change_deg=float(angle(camera[i-1],camera[i])),
            fitted_world_root_change_deg=float(angle(world[i-1],world[i]))))
    save(out/'temporal-gauge.json',dict(updated=now(),state='coordinate_identity_checked_only',
        frames=94,checks=checks,root_parity_max_abs=parity,transport_max_abs=transport_error,
        camera_rotation_determinant_error=determinant_error,camera_rotation_orthogonality_error=orthogonality_error,
        formula='R_root_current_camera = R_E_current @ R_E_neighbour.T @ R_root_neighbour_camera',
        local_body_rotations_need_no_world_rotation=True,
        pairs=pairs,max_copied_vs_transported_root_deg=max(p['copied_vs_transported_root_deg'] for p in pairs),
        source_fit_sha256=digest,old_pose_used_for_new_initialization=False,
        limitation='Historical failed poses are used only to verify the coordinate identity. This neither validates source E physically nor repairs posture; no training, pose replacement, or new camera candidate.',
        quality_accepted=False,gaussian_admitted=False))
    print('GRASS_TEMPORAL_GAUGE_CHECKED',parity,transport_error,max(p['copied_vs_transported_root_deg'] for p in pairs),flush=True)


if __name__=='__main__':main()
