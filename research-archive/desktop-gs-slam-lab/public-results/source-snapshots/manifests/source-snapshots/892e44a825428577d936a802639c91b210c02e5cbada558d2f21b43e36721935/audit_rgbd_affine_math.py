"""Independent readonly checks of the affine8D derivatives and toy artifacts.

Uses a separately generated nonidentity case, scalar bilinear interpolation and
SciPy expm. No registration solver, dataset loader, TUM data, torch or GPU.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rgbd-affine-photometric-cpu-controls-001"
LOCKS = {"rgbd_affine_photometric_odometry.py": "15c7f160aa0f2134ea2a40d0fc0817c5642ca9e71f053e4cb0896235ac402349",
         "check_rgbd_affine_photometric_odometry.py": "dee2c5a2469a7889c0108f734b8b9f7a2b03990bdc6cbcefc739147b43c49a43"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for filename, expected in LOCKS.items():
    assert sha(ROOT / "scripts" / filename) == expected
import rgbd_affine_photometric_odometry as implementation


def exponential(xi):
    x, y, z = xi[3:]
    m = np.zeros((4, 4))
    m[:3, :3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
    m[:3, 3] = xi[:3]
    return expm(m)


def intensity_and_gradient(image, uv):
    values, gradients = [], []
    for u, v in uv:
        i, j = math.floor(u), math.floor(v)
        x, y = u-i, v-j
        q00, q10, q01, q11 = image[j, i], image[j, i+1], image[j+1, i], image[j+1, i+1]
        values.append((1-y)*((1-x)*q00+x*q10)+y*((1-x)*q01+x*q11))
        gradients.append([(1-y)*(q10-q00)+y*(q11-q01), (1-x)*(q01-q00)+x*(q11-q10)])
    return np.asarray(values), np.asarray(gradients)


def main():
    observed = {}
    metrics_path = OUT / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    assert metrics["status"] == "passed_synthetic_8d_affine_brightness_controls"
    observed[metrics_path.relative_to(ROOT).as_posix()] = sha(metrics_path)
    arrays_checked = 0
    for case in metrics["cases"]:
        result = ROOT / case["result_relative_path"]
        assert sha(result) == case["result_sha256"]
        manifest = result.parent / "input_manifest.json"
        assert sha(manifest) == case["input_manifest_sha256"]
        m = json.loads(manifest.read_text(encoding="utf-8-sig"))
        archive = result.parent / "synthetic_inputs.npz"
        assert sha(archive) == m["NPZ_sha256"]
        with np.load(archive, allow_pickle=False) as data:
            assert set(data.files) == set(m["arrays"])
            for key, row in m["arrays"].items():
                array = np.ascontiguousarray(data[key])
                assert list(array.shape) == row["shape"] and str(array.dtype) == row["dtype"]
                assert hashlib.sha256(array.tobytes()).hexdigest() == row["sha256"]
                arrays_checked += 1
        for path in (result, manifest, archive):
            observed[path.relative_to(ROOT).as_posix()] = sha(path)
    v, u = np.mgrid[:81, :99]
    image = .35+.002*u+.0013*v+.000017*u*v+.017*np.sin(.07*u)*np.cos(.09*v)
    K = np.array([[84., 0., 48.7], [0., 87., 40.2], [0., 0., 1.]])
    uv = np.array([[17.31, 15.62], [28.43, 26.29], [38.68, 37.21], [49.33, 47.77],
                   [59.61, 57.26], [69.29, 61.73], [79.72, 32.38], [86.44, 19.67]])
    z = np.array([1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 4.1])
    points = np.c_[(uv[:, 0]-K[0, 2])*z/K[0, 0], (uv[:, 1]-K[1, 2])*z/K[1, 1], z]
    T = exponential(np.array([.013, -.024, .015, -.017, .012, .022]))
    source = np.linalg.solve(T, np.c_[points, np.ones(8)].T).T[:, :3]
    normals = np.random.default_rng(54).normal(size=(8, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    a, b, epsilon = -.09, .027, 3e-7
    sampled, pixel_gradients = intensity_and_gradient(image, uv)
    offsets = np.array([.14, -.035, .002, -.19, .051, .11, -.009, .27])
    geo_offsets = np.array([.012, -.023, .007, .002, -.013, .019, .004, -.001])
    pair = {"source": source, "target": points-normals*geo_offsets[:, None], "normals": normals,
            "source_intensity": math.exp(a)*sampled+b-offsets}
    target = SimpleNamespace(gray=image, geometry=SimpleNamespace(K=K))
    config = implementation.AffineConfig()
    gain = math.exp(a)
    dxyz = gain*np.c_[pixel_gradients[:, 0]*K[0, 0]/z, pixel_gradients[:, 1]*K[1, 1]/z,
                     -pixel_gradients[:, 0]*K[0, 0]*points[:, 0]/z**2
                     -pixel_gradients[:, 1]*K[1, 1]*points[:, 1]/z**2]
    manual_geo = np.c_[normals, np.cross(points, normals), np.zeros((8, 2))]
    manual_photo = np.c_[dxyz, np.cross(points, dxyz), gain*sampled, np.ones(8)]
    def residual(transform, log_gain, bias):
        p = source @ transform[:3, :3].T + transform[:3, 3]
        q = np.c_[K[0, 0]*p[:, 0]/p[:, 2]+K[0, 2], K[1, 1]*p[:, 1]/p[:, 2]+K[1, 2]]
        I, _ = intensity_and_gradient(image, q)
        return np.sum((p-pair["target"])*normals, axis=1), math.exp(log_gain)*I+bias-pair["source_intensity"], q
    def huber_cost(rg, rp):
        total = 0.
        for residuals, scale in ((rg, .01), (rp, .1)):
            values = residuals/scale
            total += math.fsum(.5*x*x if abs(x)<=1 else abs(x)-.5 for x in values)/len(values)
        return total
    rg, jg, rp, jp, _ = implementation.affine_terms(T, a, b, pair, target)
    numeric_geo, numeric_photo, numeric_loss = np.zeros((8, 8)), np.zeros((8, 8)), np.zeros(8)
    for axis in range(8):
        delta = np.zeros(8); delta[axis] = epsilon
        plus = residual(exponential(delta[:6])@T, a+delta[6], b+delta[7])
        minus = residual(exponential(-delta[:6])@T, a-delta[6], b-delta[7])
        assert np.array_equal(np.floor(plus[2]), np.floor(uv)) and np.array_equal(np.floor(minus[2]), np.floor(uv))
        numeric_geo[:, axis] = (plus[0]-minus[0])/(2*epsilon)
        numeric_photo[:, axis] = (plus[1]-minus[1])/(2*epsilon)
        numeric_loss[axis] = (huber_cost(*plus[:2])-huber_cost(*minus[:2]))/(2*epsilon)
    analytic_loss = np.mean(np.clip(rg/.01, -1, 1)[:, None]*manual_geo/.01, axis=0)
    analytic_loss += np.mean(np.clip(rp/.1, -1, 1)[:, None]*manual_photo/.1, axis=0)
    errors = {"geo_library_vs_manual_max_abs": float(np.max(np.abs(jg-manual_geo))),
              "photo_library_vs_manual_max_abs": float(np.max(np.abs(jp-manual_photo))),
              "geo_manual_vs_FD_max_abs": float(np.max(np.abs(manual_geo-numeric_geo))),
              "photo_manual_vs_FD_max_abs": float(np.max(np.abs(manual_photo-numeric_photo))),
              "Huber_gradient_manual_vs_library_max_abs": float(np.max(np.abs(analytic_loss-implementation.objective_gradient(rg,jg,rp,jp,config)))),
              "Huber_gradient_manual_vs_FD_max_abs": float(np.max(np.abs(analytic_loss-numeric_loss)))}
    assert errors["geo_library_vs_manual_max_abs"] < 1e-12 and errors["photo_library_vs_manual_max_abs"] < 1e-12
    assert errors["geo_manual_vs_FD_max_abs"] < 1e-7 and errors["photo_manual_vs_FD_max_abs"] < 1e-7
    assert errors["Huber_gradient_manual_vs_library_max_abs"] < 1e-10 and errors["Huber_gradient_manual_vs_FD_max_abs"] < 1e-6
    implementation.frozen_source_check()
    for name, expected in LOCKS.items():
        assert sha(ROOT/"scripts"/name) == expected
        assert (OUT/name).read_bytes() == (ROOT/"scripts"/name).read_bytes()
    for rel, expected in observed.items(): assert sha(ROOT/rel) == expected
    result = {"status": "passed", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "audit_source_sha256": sha(Path(__file__)), "source_sha256": LOCKS,
              "synthetic_input_arrays_hash_checked": arrays_checked, "toy_artifact_files_rehashed": observed,
              "independent_generated_case": {"samples":8,"epsilon":epsilon,"log_gain":a,"bias":b,"nonidentity_transform":T.tolist(),
                  "all8_FD_stay_in_same_noninteger_cells":True,"geometry_nuisance_columns_exact_zero":bool(np.count_nonzero(jg[:,6:])==0)},
              "errors": errors, "optimization_calls":0, "real_TUM_data_or_GT_read":False, "GPU_calls":0,
              "scope":"Separately generated eight-sample scalar-bilinear/SciPy-expm derivative control and raw hashes of all four toy cases. Full solver trajectories were not rerun; toy success does not establish real-sequence benefit or the cause of historical rejections. Root also read complete new library/checker source; original frozen objective/scales and nuisance-model limitations remain explicit."}
    with (OUT/"root_independent_math_audit.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({"status":"passed","synthetic_arrays":arrays_checked,"errors":errors,"GPU_calls":0}))


if __name__ == "__main__":
    main()
