"""Bound small rigid head changes at the frozen 022 fit in native pixels."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from official_smplifyx_entry123 import sha
from prepare_leafwoman_ear139 import ARCHIVE, OUT, atomic, read


def jacobian(points, center, K):
    q = points@K.T
    uv = q[:,:2]/q[:,2:]
    pixel_xyz = (K[:2]-uv[:,:,None]*K[2])/q[:,2,None,None]
    x,y,z = (points-center).T
    zero = np.zeros_like(x)
    twist = np.stack([zero,z,-y,-z,zero,x,y,-x,zero],axis=1).reshape(-1,3,3)
    return uv, np.concatenate([pixel_xyz@twist,pixel_xyz],axis=2)


def check():
    p = OUT/'pose-check-plan.json'
    out = OUT/'pose-check.json'
    assert not p.exists() and not out.exists()
    ear = read(OUT/'result.json')
    plan = read(OUT/'plan.json')
    assert ear['plan_sha256'] == sha(OUT/'plan.json') and plan['archive_sha256'] == sha(ARCHIVE)
    atomic(p, dict(ear_result_sha256=sha(OUT/'result.json'), baseline_sha256=sha(ARCHIVE),
        observations='Existing fitted 51 inner-face joints (indices 67:118), used only as reference projections, not photo truth.',
        variation='Linearize rigid head rotation and translation in fixed camera; no new parameters written.',
        constraint='Absolute change per x/y coordinate at every reference point ≤1px. This relaxes the stricter Euclidean ≤1px condition.',
        statistic='Four LP extrema of signed ear u/v movement. Rectangle radius bounds the first-order ear displacement. '
                  'Compare with each independent polygon distance after subtracting annotation tolerance.',
        verify='Check analytical Jacobian with central differences and verify nonlinear remainder at extrema. No constraint/parameter sweep.'))
    rows = []
    with np.load(ARCHIVE,allow_pickle=False) as data:
        ids = data['frame_ids'].tolist()
        K = data['K_integer_centers'].astype(float)
        for item in ear['rows']:
            frame,side = item['frame'],item['side']
            i = ids.index(frame)
            vertex = 6 if side == 'left' else 616
            face = data['joints_camera'][i,67:118].astype(float)
            xyz = np.vstack([face,data['vertices_camera'][i,vertex]])
            center = face.mean(0)
            uv, J = jacobian(xyz,center,K)

            def project(twist):
                R = Rotation.from_rotvec(twist[:3]).as_matrix()
                q = ((xyz-center)@R.T + center + twist[3:])@K.T
                return q[:,:2]/q[:,2:]

            eps = 1e-6
            finite = np.stack([(project(step)-project(-step))/(2*eps)
                               for step in eps*np.eye(6)],axis=-1)
            parity = float(np.max(np.abs(J-finite)))
            assert parity < 1e-5
            faces = J[:-1].reshape(-1,6)
            assert np.linalg.matrix_rank(faces) == 6
            A = np.r_[faces,-faces]
            extent = np.zeros(2)
            nonlinearity, angles, shifts = [], [], []
            for dim in range(2):
                for direction in (1,-1):
                    c = J[-1,dim]*direction
                    opt = linprog(-c,A_ub=A,b_ub=np.ones(len(A)),bounds=[(None,None)]*6,method='highs')
                    assert opt.success and np.max(np.abs(faces@opt.x)) <= 1+1e-7
                    extent[dim] = max(extent[dim], c@opt.x)
                    nonlinearity.append(float(np.max(np.abs(project(opt.x)-uv-J@opt.x))))
                    angles.append(float(np.rad2deg(np.linalg.norm(opt.x[:3]))))
                    shifts.append(float(np.linalg.norm(opt.x[3:])))
            bound = float(np.linalg.norm(extent))
            min_move = [max(0.,-r['vertex_signed_distance_px']-r['uncertainty_px']) for r in item['readings']]
            rows.append(dict(frame=frame, linear_ear_uv_abs_upper_px=extent.tolist(),linear_ear_radius_upper_px=bound,
                min_ear_move_by_reading_px=min_move,both_readings_outside_linear_upper_bound=min(min_move)>bound,
                max_linearization_error_at_extrema_px=max(nonlinearity), max_twist_angle_deg=max(angles),
                max_translation_model_m=max(shifts), jacobian_max_finite_difference_delta=parity))
    atomic(out, dict(plan_sha256=sha(p),rows=rows,no_candidate_written=True,scope='Local pose-only sensitivity at the saved geometry. Does not separate true shape, root, intrinsics, articulation, or other causes.'))
    for row in rows:
        print(row['frame'],round(row['linear_ear_radius_upper_px'],2),[round(d,2) for d in row['min_ear_move_by_reading_px']],round(row['max_linearization_error_at_extrema_px'],3))


if __name__=='__main__':
    check()
