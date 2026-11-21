"""One read-only check of a proposed metric-centre interface; never a trainer."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
import numpy as np

BASE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle')
OUT = BASE/'limb-and-reference-review/surface-interface-preflight'
SOURCE = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def geometry(v, faces):
    tri = v[faces]
    edge, other = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    width = np.linalg.norm(edge, axis=-1)
    normal = np.cross(edge, other)
    twice_area = np.linalg.norm(normal, axis=-1)
    assert (width > 0).all() and (twice_area > 0).all()
    a0, a1 = edge/width[:, None], normal/twice_area[:, None]
    a2 = -np.cross(a1, a0)
    a2 /= np.linalg.norm(a2, axis=-1, keepdims=True)
    R = np.stack((a0, a1, a2), axis=-1)
    height = abs(np.einsum('ni,ni->n', a2, other))
    # Preserve the CURRENT public adapter's (edge + altitude) / 2 definition.
    return tri.mean(1), R, (width+height)/2, twice_area/2


def metric_centres(x_ref, c_ref, R_ref, s_ref, c_t, R_t, s_t):
    """Proposed interface only: same motion, centres in reference-world metres."""
    Q = R_t @ R_ref.swapaxes(-1, -2)
    ratio = s_t/s_ref
    return c_t+ratio[:, None]*np.einsum('nij,nj->ni', Q, x_ref-c_ref)


def main():
    assert not OUT.exists(), 'Inspect the existing bounded preflight; do not repeat it.'
    OUT.mkdir()
    files = [BASE/'full94-shared-shape/fit/fitted.npz',
             SOURCE/'avatar_core/utils/graphics_utils.py', SOURCE/'avatar_core/scene/smplx_gaussian_model.py',
             SOURCE/'shuolee_portrait/training.py', SOURCE/'shuolee_portrait/free_clothing_transport.py',
             SOURCE/'shuolee_portrait/free_clothing_torch.py', SOURCE/'shuolee_portrait/framewise_native_training.py',
             SOURCE/'shuolee_portrait/framewise_clothed_surface.py']
    hashes = {str(p): sha(p) for p in files}
    save('plan.json', dict(created=datetime.now(timezone.utc).isoformat(),
        hypothesis='Reference-world metre centres can reproduce the current frozen triangle transport exactly without adopting the older alternative scale/SH/loss definitions.',
        inputs=hashes, reference_frame='frame_000051', frames=94, triangles=2048,
        budget=dict(CPU_seconds=60, learned_steps=0, optimizers=0, GPU_tasks=0),
        tolerances=dict(position_m=1e-10, covariance_m2=1e-12, reference_jacobian=1e-10, finite_difference_jacobian=1e-7),
        scope='Read-only algebra and actual frozen-mesh inputs. Not clothing initialization, training, pose accuracy or surface acceptance. Human representation choice remains pending.',
        stop='Any hash/identity/nonfinite mismatch, tolerance failure or CPU budget overrun. No retries or parameter search.'))
    start = time.perf_counter()
    with np.load(files[0], allow_pickle=False) as z:
        ids = z['frame_ids'].astype(str).tolist()
        vertices, faces = z['vertices_world'].astype(np.float64), z['faces']
    assert len(ids) == 94 and ids[50] == 'frame_000051'
    support = np.unique(np.linspace(0, len(faces)-1, 2048).astype(int))
    faces = faces[support]
    c0, R0, s0, area0 = geometry(vertices[50], faces)
    # Includes offsets beyond the present unit-local-position regularizer.
    local = np.sin(np.arange(len(support)*3).reshape(-1, 3)*.37)*5
    x0 = c0 + s0[:, None]*np.einsum('nij,nj->ni', R0, local)
    local_cov = np.broadcast_to(np.diag([.02, .03, .04])**2, (len(support), 3, 3))
    ref_cov = s0[:, None, None]**2*(R0@local_cov@R0.swapaxes(-1, -2))
    max_position = max_covariance = reference_jacobian = 0.
    scale_difference = []
    for i, frame in enumerate(ids):
        assert time.perf_counter()-start < 60
        ct, Rt, st, area = geometry(vertices[i], faces)
        Q, ratio = Rt@R0.swapaxes(-1, -2), st/s0
        old = ct + st[:, None]*np.einsum('nij,nj->ni', Rt, local)
        new = metric_centres(x0, c0, R0, s0, ct, Rt, st)
        original_cov = st[:, None, None]**2*(Rt@local_cov@Rt.swapaxes(-1, -2))
        metric_cov = ratio[:, None, None]**2*(Q@ref_cov@Q.swapaxes(-1, -2))
        max_position = max(max_position, float(abs(old-new).max()))
        max_covariance = max(max_covariance, float(abs(original_cov-metric_cov).max()))
        scale_difference.append(abs(np.sqrt(area/area0)/ratio-1))
        if i == 50:
            reference_jacobian = float(abs(ratio[:, None, None]*Q-np.eye(3)).max())
        if i == 62:
            epsilon = 1e-6
            numerical = np.empty_like(Q)
            for axis in range(3):
                plus, minus = x0.copy(), x0.copy()
                plus[:, axis] += epsilon
                minus[:, axis] -= epsilon
                numerical[:, :, axis] = (metric_centres(plus,c0,R0,s0,ct,Rt,st)-metric_centres(minus,c0,R0,s0,ct,Rt,st))/(2*epsilon)
            finite_error = float(abs(numerical-ratio[:, None, None]*Q).max())
    assert max_position < 1e-10 and max_covariance < 1e-12 and reference_jacobian < 1e-10 and finite_error < 1e-7
    assert hashes == {str(p): sha(p) for p in files}
    delta = np.concatenate(scale_difference)
    save('result.json', dict(state='read_only_metric_interface_passed_not_applied',
        updated=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
        frames=94, triangle_samples=len(support), reference_frame='frame_000051',
        position_max_abs_m=max_position, covariance_max_abs_m2=max_covariance,
        reference_jacobian_identity_max_abs=reference_jacobian, finite_difference_jacobian_max_abs=finite_error,
        alternate_area_scale_relative_difference=dict(median=float(np.median(delta)), p90=float(np.quantile(delta,.9)), maximum=float(delta.max())),
        reuse_findings=['Existing free-clothing transport uses sqrt(area ratio), not the current (edge+altitude)/2 ratio.',
                       'Existing free-clothing trainer changes RGB loss from L1+SSIM to L1 and rotates SH view directions into the reference frame.',
                       'Existing framewise silhouette voting principle is useful; its old field/trainer is not a drop-in single-variable replacement.'],
        minimal_prepared_interface='c_t+(s_t/s_ref)*R_t*R_ref.T*(x_ref-c_ref); current rotation/scale tables retained, metre coordinate gradient is identity in the reference frame.',
        clothing_surface_initialization_still_required=True, sources_unchanged=True,
        changed_production_code=False, optimization_steps=0, quality_accepted=False, human_choice_pending=True,
        elapsed_seconds=time.perf_counter()-start))
    print((OUT/'result.json').read_text())


if __name__ == '__main__':
    main()
