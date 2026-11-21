"""Reviewed preparation and one global-gauge frozen-map renderer control.

CPU preparation may read one completed GT-free checkpoint and test algebra.
GPU execution is separate, requires reviewed protocol SHA and pre-08:00 gate.
Exactly two renders; no GT, observations, optimizer, training, or retry.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "outputs/tum-desk-online-v4-prior0-full549-001/last_checkpoint.pt"
CHECKPOINT_SHA = "64cda836840fd28e450ba036a6d52b73914d2a85c00dda3ee402566a98444d82"
FRAME_ID = 548
SOURCE_LOCKS = {
    "scripts/run_local.py": "9f62cfc9cb4370ca966b6920cd318c700c1c575620d3fda84f5be39888ae0cc5",
    "scripts/train_tum_online_gs_v4.py": "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f",
    "scripts/evaluate_tum_v4_estimated_seen_maps.py": "7dfe57310c334d5e917dee5e12e0a68f9d83381b1096aeb00d0fc29bfa1bab74",
}
CONFIG = {
    "width": 160, "height": 120, "frame_id": FRAME_ID, "dtype": "float32",
    "gsplat_version": "1.5.3", "render_mode": "RGB+ED", "packed": False,
    "near_plane_m": 0.1, "far_plane_m": 6.0, "background_RGB": [0., 0., 0.],
    "correction_rotation_xyz_degrees_RzRyRx": [17., -11., 23.],
    "correction_translation_m": [0.23, -0.17, 0.09],
    "camera_update": "W_new = EXACT_saved_W_old @ inverse(world_left_Delta)",
    "mean_update": "mu_new = R_Delta @ mu_old + t_Delta, float64 arithmetic then float32 cast",
    "quaternion_update": "Hamilton active wxyz q_new = q_Delta product raw_q_old, float64 then float32 cast; renderer normalizes",
    "unchanged_parameters": ["log_scales", "color_logits", "opacity_logits"],
    "ED_support_alpha_strictly_greater": 1e-6, "coverage_alpha_strictly_greater": 0.5,
    "CPU_projection_depth_margin_m": [0.2, 5.9],
    "CPU_limits": {"proper_delta_abs": 1e-12, "saved_camera_rigid_abs": 2e-6,
        "inverse_algebra_rotation_abs": 1e-12, "inverse_algebra_translation_abs_m": 1e-12,
        "camera_mean_max_abs_m": 5e-6, "camera_covariance_max_abs_m2": 1e-7,
        "quaternion_covariance_max_abs_m2": 1e-7,
        "pinhole_mean_max_abs_pixel": 0.01, "pinhole_covariance_max_abs_pixel2": 0.05},
    "GPU_limits": {"RGB_max_abs": 0.001, "RGB_RMSE": 1e-5,
        "alpha_max_abs": 0.001, "alpha_RMSE": 1e-5,
        "finite_ED_mask_XOR_pixels": 0, "positive_supported_ED_mask_XOR_pixels": 4,
        "covered_ED_mask_XOR_pixels": 4,
        "common_supported_ED_max_abs_m": 0.01, "common_supported_ED_RMSE_m": 1e-4,
        "common_covered_ED_max_abs_m": 0.005, "common_covered_ED_RMSE_m": 5e-5},
    "deadline_Beijing": "2026-10-09T08:00:00+08:00",
}
LIMITS = [
    "Single global gauge action on every Gaussian and the one selected saved camera, not per-anchor correction, loop closure, BA or map repair.",
    "Constant RGB/opacity and scale parameter raw bytes are held; directional SH is absent and not transported.",
    "No GT or sensor observations are read. Original image means the original frozen-map render, not a raw observed RGB image.",
    "Original camera is the exact saved float32 w2c. It is never recomputed, fitted, SVD-corrected or replaced by GT.",
    "CPU float64 algebra and actual float32 payload diagnostics do not prove renderer invariance before the two GPU renders run.",
    "Float32 projection, depth sorting, tile visibility and alpha-threshold discontinuities can change numerical render outputs; failures are retained without threshold changes or retries.",
    "All-finite ED difference is reported. Acceptance ED errors use both-arm positive finite ED and predeclared common alpha support; empty masks cannot pass silently.",
    "There is no localization accuracy or rendering speed comparison, and no original SplaTAM backend comparison.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def array_record(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(value)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "raw_bytes": a.nbytes,
            "raw_bytes_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def source_records():
    result = dict(SOURCE_LOCKS)
    for name, expected in result.items():
        require(sha(ROOT / name) == expected, "Frozen reference source changed: " + name)
    result["scripts/check_gaussian_global_gauge_renderer.py"] = sha(__file__)
    return result


def gsplat_files():
    distribution = importlib.metadata.distribution("gsplat")
    require(distribution.version == CONFIG["gsplat_version"], "Fixed gsplat version changed")
    paths = [distribution.locate_file("gsplat/rendering.py"), distribution.locate_file("gsplat/cuda/_wrapper.py")]
    return {str(p.resolve()): sha(p) for p in paths}


def deadline():
    now = dt.datetime.now(dt.timezone.utc)
    require(now < dt.datetime.fromisoformat(CONFIG["deadline_Beijing"]), "No new GPU control at/after 08:00 Beijing")
    return now.isoformat()


def transform():
    x, y, z = np.deg2rad(np.asarray(CONFIG["correction_rotation_xyz_degrees_RzRyRx"]))
    rx = np.array([[1., 0., 0.], [0., np.cos(x), -np.sin(x)], [0., np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0., np.sin(y)], [0., 1., 0.], [-np.sin(y), 0., np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0.], [np.sin(z), np.cos(z), 0.], [0., 0., 1.]])
    d = np.eye(4)
    d[:3, :3], d[:3, 3] = rz @ ry @ rx, CONFIG["correction_translation_m"]
    inv = np.eye(4)
    inv[:3, :3], inv[:3, 3] = d[:3, :3].T, -d[:3, :3].T @ d[:3, 3]
    def multiply(a, b):
        w, i, j, k = a; v, x0, y0, z0 = b
        return np.array([w*v-i*x0-j*y0-k*z0, w*x0+i*v+j*z0-k*y0,
                         w*y0-i*z0+j*v+k*x0, w*z0+i*y0-j*x0+k*v])
    qx, qy, qz = [np.cos(x/2), np.sin(x/2), 0., 0.], [np.cos(y/2), 0., np.sin(y/2), 0.], [np.cos(z/2), 0., 0., np.sin(z/2)]
    return d, inv, multiply(qz, multiply(qy, qx))


def quaternion_matrices(values):
    q = np.asarray(values, dtype=np.float64)
    require(np.all(np.isfinite(q)) and np.all(np.linalg.norm(q, axis=-1) > 1e-12), "Finite nonzero quaternions required")
    w, x, y, z = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=-1).reshape(-1, 3, 3)


def load_input():
    require(sha(ROOT / CHECKPOINT) == CHECKPOINT_SHA, "Completed GT-free map checkpoint changed")
    c = torch.load(ROOT / CHECKPOINT, map_location="cpu", weights_only=True)
    require(c["status"] == "completed" and c["ground_truth_included"] is False,
            "Only completed causal GT-free checkpoint accepted")
    require(c["script_sha256"] == SOURCE_LOCKS["scripts/train_tum_online_gs_v4.py"] and
        c["frame_ids_with_stored_pose"] == list(range(549)) and c["current_local_index"] == 548, "Wrong completed desk map chronology")
    require(c["configuration"]["width"] == 160 and c["configuration"]["height"] == 120, "Fixed training resolution")
    params = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in c["parameters"].items()}
    require(set(params) == {"means", "quats", "log_scales", "color_logits", "opacity_logits"}, "Constant RGB map fields only")
    require(len(params["means"]) == 23838 and all(v.dtype == torch.float32 and not v.requires_grad and torch.isfinite(v).all() for v in params.values()), "Exact finite float32 frozen map")
    k, w = c["K"].detach().cpu().contiguous().clone(), c["estimated_w2c"][FRAME_ID].detach().cpu().contiguous().clone()
    require(k.shape == (1, 3, 3) and w.shape == (4, 4) and k.dtype == w.dtype == torch.float32, "Original saved K/camera payload")
    require(sha(ROOT / CHECKPOINT) == CHECKPOINT_SHA, "Checkpoint changed during CPU read")
    return params, k, w


def transformed_inputs(params, w):
    d, inverse, q = transform()
    result = {name: value.clone() for name, value in params.items()}
    result["means"] = torch.from_numpy((params["means"].numpy().astype(np.float64) @ d[:3,:3].T + d[:3,3]).astype(np.float32))
    old = params["quats"].numpy().astype(np.float64)
    a, b, c, e = q; u, x, y, z = old.T
    updated = np.stack([a*u-b*x-c*y-e*z, a*x+b*u+c*z-e*y,
                        a*y-b*z+c*u+e*x, a*z+b*y-c*x+e*u], axis=-1)
    result["quats"] = torch.from_numpy(updated.astype(np.float32))
    wnew = torch.from_numpy((w.numpy().astype(np.float64) @ inverse).astype(np.float32))
    return result, wnew, d


def covariance(params):
    r = quaternion_matrices(params["quats"].numpy())
    # The actual activated float32 axis scales, not float64 exp of logits.
    s = params["log_scales"].exp().numpy().astype(np.float64)
    return (r * (s*s)[:, None, :]) @ np.swapaxes(r, 1, 2)


def activated_constants(params):
    return {"scales":params["log_scales"].exp(), "RGB":params["color_logits"].sigmoid(),
            "opacity":params["opacity_logits"].sigmoid()}


def projections(p, s, k):
    x, y, z = p.T; fx, fy, cx, cy = k[0,0], k[1,1], k[0,2], k[1,2]
    uv = np.stack([fx*x/z+cx, fy*y/z+cy], axis=-1)
    j = np.zeros((len(p), 2, 3)); j[:,0,0], j[:,0,2] = fx/z, -fx*x/z**2
    j[:,1,1], j[:,1,2] = fy/z, -fy*y/z**2
    return uv, j @ s @ np.swapaxes(j, 1, 2)


def cpu_checks(params, k, w, updated, wnew, d):
    bounds, values = CONFIG["CPU_limits"], {}
    r = d[:3,:3]
    values["proper_delta_abs"] = float(max(np.max(np.abs(r.T@r-np.eye(3))), abs(np.linalg.det(r)-1.), np.max(np.abs(d[3]-[0,0,0,1]))))
    old_w, new_w = w.numpy().astype(np.float64), wnew.numpy().astype(np.float64)
    values["saved_camera_rigid_abs"] = float(max(np.max(np.abs(old_w[:3,:3].T@old_w[:3,:3]-np.eye(3))), abs(np.linalg.det(old_w[:3,:3])-1.), np.max(np.abs(old_w[3]-[0,0,0,1]))))
    old_c = np.linalg.inv(old_w)  # ONLY algebra; original render W stays exact saved bytes.
    new_c = d @ old_c
    identity = np.linalg.inv(new_c) @ d
    values["inverse_algebra_rotation_abs"] = float(np.max(np.abs(identity[:3,:3]-old_w[:3,:3])))
    values["inverse_algebra_translation_abs_m"] = float(np.max(np.abs(identity[:3,3]-old_w[:3,3])))
    old_p = params["means"].numpy().astype(np.float64) @ old_w[:3,:3].T + old_w[:3,3]
    new_p = updated["means"].numpy().astype(np.float64) @ new_w[:3,:3].T + new_w[:3,3]
    old_s, new_s = covariance(params), covariance(updated)
    values["quaternion_covariance_max_abs_m2"] = float(np.max(np.abs(new_s-r@old_s@r.T)))
    old_camera_s = old_w[:3,:3] @ old_s @ old_w[:3,:3].T
    new_camera_s = new_w[:3,:3] @ new_s @ new_w[:3,:3].T
    values["camera_mean_max_abs_m"] = float(np.max(np.abs(new_p-old_p)))
    values["camera_covariance_max_abs_m2"] = float(np.max(np.abs(new_camera_s-old_camera_s)))
    low, high = CONFIG["CPU_projection_depth_margin_m"]
    common = (old_p[:,2]>low)&(old_p[:,2]<high)&(new_p[:,2]>low)&(new_p[:,2]<high)
    require(common.any(), "No well-conditioned CPU projection points")
    old_uv, old_sigma = projections(old_p[common], old_camera_s[common], k.numpy()[0])
    new_uv, new_sigma = projections(new_p[common], new_camera_s[common], k.numpy()[0])
    values["pinhole_mean_max_abs_pixel"] = float(np.max(np.abs(new_uv-old_uv)))
    values["pinhole_covariance_max_abs_pixel2"] = float(np.max(np.abs(new_sigma-old_sigma)))
    checks = [{"name": name, "actual": value, "limit": bounds[name], "passed": bool(np.isfinite(value) and value<=bounds[name])} for name,value in values.items()]
    require(all(x["passed"] for x in checks), "CPU gauge algebra/payload check failed; retain evidence, do not infer rendering")
    omitted_covariance_error = float(np.max(np.abs(new_w[:3,:3] @ old_s @ new_w[:3,:3].T-old_camera_s)))
    inverse_d = np.linalg.inv(d)
    wrong_w = inverse_d @ old_w  # declared world-right camera action is deliberately reversed.
    wrong_p = updated["means"].numpy().astype(np.float64) @ wrong_w[:3,:3].T + wrong_w[:3,3]
    wrong_camera_error = float(np.max(np.abs(wrong_p-old_p)))
    require(omitted_covariance_error>1e-8 and wrong_camera_error>0.001, "Actual anisotropic/noncommuting map negatives must be distinguishable")
    unchanged = {name: array_record(params[name]) == array_record(updated[name]) for name in CONFIG["unchanged_parameters"]}
    require(all(unchanged.values()), "Constant RGB/opacity/scales RAW BYTES changed")
    activated_old, activated_new = activated_constants(params), activated_constants(updated)
    activated_equal = {name:array_record(v)==array_record(activated_new[name]) for name,v in activated_old.items()}
    require(all(activated_equal.values()), "Activated float32 RGB/opacity/scales raw bytes changed")
    require(not torch.cuda.is_initialized(), "CPU preparation unexpectedly initialized CUDA")
    return {"status": "passed_CPU_only_global_gauge_payload_checks", "all_checks": checks,
        "well_conditioned_projection_point_count": int(common.sum()), "unchanged_parameter_raw_bytes": unchanged,
        "unchanged_activated_constant_raw_bytes":activated_equal,
        "negative_controls_CPU_only": {"omit_covariance_rotation_max_abs_m2":omitted_covariance_error,
            "wrong_camera_action_max_abs_m":wrong_camera_error,
            "minimum_differences":{"omit_covariance_rotation_m2":1e-8,"wrong_camera_action_m":0.001}},
        "Delta_float64": d.tolist(), "old_saved_W_float32": w.tolist(), "new_W_float32": wnew.tolist(),
        "Cold_implied_only_for_algebra": old_c.tolist(), "Cnew_only_for_algebra": new_c.tolist(),
        "author_gpu_render_calls": 0, "optimizer_steps": 0, "GT_reads": 0, "CUDA_initialized": False}


def input_record(params, k, w, updated, wnew):
    return {"checkpoint": CHECKPOINT, "checkpoint_sha256": CHECKPOINT_SHA, "point_count": 23838,
        "frame_id": FRAME_ID, "parameters": {name:array_record(v) for name,v in params.items()},
        "transformed_parameters": {name:array_record(v) for name,v in updated.items()},
        "activated_constant_RGB_opacity_scales":{name:array_record(v) for name,v in activated_constants(params).items()},
        "fixed_K": array_record(k), "exact_saved_W": array_record(w), "transformed_W": array_record(wnew),
        "ground_truth_in_checkpoint": False, "GT_read": False}


def prepare(output):
    sources, external = source_records(), gsplat_files()
    params,k,w = load_input(); updated,wnew,d = transformed_inputs(params,w)
    # These are geometry/payload checks only; no rasterizer import or GPU call.
    cpu = cpu_checks(params,k,w,updated,wnew,d)
    protocol = {"status": "prepared_CPU_geometry_passed_GPU_not_authorized", "source_sha256": sources,
        "gsplat_python_source_sha256": external, "configuration": CONFIG, "fixed_input": input_record(params,k,w,updated,wnew),
        "CPU_checks": cpu, "expected_actual_run": "gaussian-global-gauge-renderer-controls-001",
        "expected_render_calls": 2, "GT_reads": 0, "optimizer_steps": 0, "GPU_render_calls": 0, "limits": LIMITS}
    write_new(output / "protocol.json", protocol)
    write_new(output / "cpu_checks.json", cpu)
    require(source_records()==sources and gsplat_files()==external and sha(ROOT/CHECKPOINT)==CHECKPOINT_SHA, "CPU inputs/source postguard")
    return {"status":protocol["status"], "protocol_sha256":sha(output/"protocol.json"), "CPU_checks_passed":len(cpu["all_checks"]), "GPU_render_calls":0}


def error_record(a, b, mask=None):
    difference = np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    if mask is not None: difference = difference[mask]
    if difference.size==0: return {"scalar_count":0,"max_abs":None,"RMSE":None}
    require(np.all(np.isfinite(difference)), "Finite error differences required")
    return {"scalar_count":int(difference.size),"max_abs":float(np.max(np.abs(difference))),"RMSE":float(np.sqrt(np.mean(difference*difference)))}


def compare(original, gauged):
    rgb0,ed0,a0 = [original[k] for k in ("RGB","ED","alpha")]
    rgb1,ed1,a1 = [gauged[k] for k in ("RGB","ED","alpha")]
    require(all(np.all(np.isfinite(v)) for v in (rgb0,rgb1,a0,a1)), "RGB and alpha must be finite")
    finite0,finite1 = np.isfinite(ed0),np.isfinite(ed1)
    support0 = finite0 & (ed0>0) & (a0>CONFIG["ED_support_alpha_strictly_greater"])
    support1 = finite1 & (ed1>0) & (a1>CONFIG["ED_support_alpha_strictly_greater"])
    covered0 = finite0 & (ed0>0) & (a0>CONFIG["coverage_alpha_strictly_greater"])
    covered1 = finite1 & (ed1>0) & (a1>CONFIG["coverage_alpha_strictly_greater"])
    result={"RGB":error_record(rgb0,rgb1),"alpha":error_record(a0,a1),
        "all_finite_ED_common":error_record(ed0,ed1,finite0&finite1),
        "positive_supported_ED_common":error_record(ed0,ed1,support0&support1),
        "covered_ED_common":error_record(ed0,ed1,covered0&covered1),
        "masks":{"finite_original":int(finite0.sum()),"finite_gauged":int(finite1.sum()),
            "finite_ED_XOR_pixels":int((finite0^finite1).sum()),
            "positive_supported_original":int(support0.sum()),"positive_supported_gauged":int(support1.sum()),
            "positive_supported_XOR_pixels":int((support0^support1).sum()),
            "positive_supported_common_pixels":int((support0&support1).sum()),
            "covered_original":int(covered0.sum()),"covered_gauged":int(covered1.sum()),
            "covered_XOR_pixels":int((covered0^covered1).sum()),"covered_common_pixels":int((covered0&covered1).sum())},
        "units":{"RGB":"RGB channel value","alpha":"opacity fraction","ED":"m"},
        "ED_scope":"All-finite common error reported without a tolerance gate. ED acceptance uses two nonempty predeclared positive common alpha masks; no sensor-valid mask or GT."}
    candidate={"RGB_max_abs":result["RGB"]["max_abs"],"RGB_RMSE":result["RGB"]["RMSE"],
        "alpha_max_abs":result["alpha"]["max_abs"],"alpha_RMSE":result["alpha"]["RMSE"],
        "finite_ED_mask_XOR_pixels":result["masks"]["finite_ED_XOR_pixels"],
        "positive_supported_ED_mask_XOR_pixels":result["masks"]["positive_supported_XOR_pixels"],
        "covered_ED_mask_XOR_pixels":result["masks"]["covered_XOR_pixels"],
        "common_supported_ED_max_abs_m":result["positive_supported_ED_common"]["max_abs"],
        "common_supported_ED_RMSE_m":result["positive_supported_ED_common"]["RMSE"],
        "common_covered_ED_max_abs_m":result["covered_ED_common"]["max_abs"],
        "common_covered_ED_RMSE_m":result["covered_ED_common"]["RMSE"]}
    result["all_numerical_checks"]=[{"name":name,"actual":v,"limit":CONFIG["GPU_limits"][name],
        "passed":bool(v is not None and np.isfinite(v) and v<=CONFIG["GPU_limits"][name])} for name,v in candidate.items()]
    return result


def execute(output, protocol_path, protocol_sha):
    require(protocol_path is not None and sha(protocol_path)==protocol_sha,"Exact reviewed protocol SHA required")
    protocol=json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    require(protocol["status"]=="prepared_CPU_geometry_passed_GPU_not_authorized" and protocol["configuration"]==CONFIG,
        "Protocol configuration changed or preparation was not completed")
    require(source_records()==protocol["source_sha256"] and gsplat_files()==protocol["gsplat_python_source_sha256"], "Reviewed source changed")
    require(output.name==protocol["expected_actual_run"], "Exact predeclared unique run only")
    params,k,w=load_input(); updated,wnew,d=transformed_inputs(params,w)
    require(input_record(params,k,w,updated,wnew)==protocol["fixed_input"], "Actual map/K/oldcamera/transformed payloads differ from reviewed input")
    cpu=cpu_checks(params,k,w,updated,wnew,d)
    started=deadline()
    state={"status":"starting_unique_global_gauge_renderer_control","started_utc":started,
        "source_sha256":protocol["source_sha256"],"protocol_sha256":protocol_sha,
        "configuration":CONFIG,"fixed_input_before":protocol["fixed_input"],"CPU_geometry_checks":cpu,
        "GT_reads":0,"optimizer_steps":0,"new_training_steps":0,"render_calls":0,
        "requested_arms":[{"name":"original_frozen_map_render","status":"not_attempted"},
                          {"name":"global_gauge_map_camera_render","status":"not_attempted"}],"limits":LIMITS}
    (output/"protocol_used.json").write_bytes(Path(protocol_path).read_bytes())
    try:
        deadline()  # directly before backend import / new CUDA initialization.
        from gsplat import rasterization
        require(torch.cuda.is_available(), "Local CUDA is required for the separately authorized renderer control")
        pairs=[]
        for pmat, camera in ((params,w),(updated,wnew)):
            pairs.append(({name:v.to("cuda") for name,v in pmat.items()},camera.to("cuda")))
        kgpu=k.to("cuda"); predictions=[]
        shared_constants=activated_constants(pairs[0][0])
        separately_activated_gauged=activated_constants(pairs[1][0])
        shared_records={name:array_record(v) for name,v in shared_constants.items()}
        require(shared_records=={name:array_record(v) for name,v in separately_activated_gauged.items()}, "Actual CUDA RGB/opacity/scales raw bytes differ")
        state["activated_constant_RGB_opacity_scales_GPU_raw_bytes_shared_by_both_renders"]=shared_records
        first_render_start=deadline()  # No first actual draw can start at/after the cutoff.
        state["first_render_started_utc"]=first_render_start
        with torch.no_grad():
            for index,(pmap,camera) in enumerate(pairs):
                state["requested_arms"][index]["status"]="rendering";state["render_calls"]+=1
                colors,alpha,_=rasterization(pmap["means"],torch.nn.functional.normalize(pmap["quats"],dim=-1),
                    shared_constants["scales"],shared_constants["opacity"],shared_constants["RGB"],
                    camera[None].contiguous(),kgpu,CONFIG["width"],CONFIG["height"],packed=False,render_mode="RGB+ED",
                    near_plane=CONFIG["near_plane_m"],far_plane=CONFIG["far_plane_m"],backgrounds=torch.zeros(1,3,device="cuda"))
                torch.cuda.synchronize()
                prediction={"RGB":colors[0,...,:3].detach().cpu().numpy(),"ED":colors[0,...,3].detach().cpu().numpy(),"alpha":alpha[0,...,0].detach().cpu().numpy()}
                require(prediction["RGB"].shape==(120,160,3) and prediction["ED"].shape==prediction["alpha"].shape==(120,160), "Exact raw render shapes")
                target=output/("original.npz" if index==0 else "global_gauge.npz")
                with target.open("xb") as stream:np.savez_compressed(stream,**prediction)
                state["requested_arms"][index].update(status="render_completed",prediction_path=target.name,
                    prediction_sha256=sha(target),raw_array_payloads={name:array_record(v) for name,v in prediction.items()})
                predictions.append(prediction)
        require(state["render_calls"]==2,"Exactly two actual renders, no retry")
        comparison=compare(*predictions);state["comparison"]=comparison
        write_new(output/"comparison.json",comparison)
        after=input_record(params,k,w,updated,wnew)
        require(after==state["fixed_input_before"] and source_records()==protocol["source_sha256"] and
            gsplat_files()==protocol["gsplat_python_source_sha256"] and sha(ROOT/CHECKPOINT)==CHECKPOINT_SHA, "Original inputs/source postguard")
        require(all(array_record(pairs[i][0][name])==array_record(p) for i,ps in enumerate((params,updated)) for name,p in ps.items()), "GPU map parameter bytes mutated by rendering")
        require(array_record(kgpu)==array_record(k) and all(array_record(pairs[i][1])==array_record(c) for i,c in enumerate((w,wnew))), "Camera or K bytes mutated")
        require(shared_records=={name:array_record(v) for name,v in shared_constants.items()}, "Shared activated constant bytes mutated")
        state.update(fixed_input_after=after,all_parameter_and_camera_bytes_unchanged_by_rendering=True)
        require(all(x["passed"] for x in comparison["all_numerical_checks"]), "Predeclared renderer numerical bounds failed; no threshold change or retry")
        from PIL import Image
        for index,pred in enumerate(predictions):
            # Display clipping is only for PNG; all numeric comparisons use unclipped raw arrays.
            Image.fromarray(np.rint(np.clip(pred["RGB"],0,1)*255).astype(np.uint8)).save(output/("original.png" if index==0 else "global_gauge.png"))
        state.update(status="completed_global_gauge_render_control_within_predeclared_float32_bounds",ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_new(output/"metrics.json",state)
        files={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(output.iterdir()) if p.is_file()}
        write_new(output/"completion.json",{"status":"completed","source_sha256":protocol["source_sha256"],"protocol_sha256":protocol_sha,"all_saved_files":files,"render_calls":2,"optimizer_steps":0,"GT_reads":0})
    except BaseException as error:
        state.update(status="failed_global_gauge_render_control_preserved_no_retry",exception_type=type(error).__name__,exception_message=str(error),traceback=traceback.format_exc())
        write_new(output/"failure.json",state)
        raise
    return {"status":state["status"],"render_calls":state["render_calls"],"optimizer_steps":0,"GT_reads":0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true");mode.add_argument("--execute",action="store_true")
    parser.add_argument("--protocol",type=Path);parser.add_argument("--protocol-sha");parser.add_argument("--run",required=True)
    args=parser.parse_args();require(args.run and all(ch in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in args.run),"Safe run name")
    output=ROOT/"outputs"/args.run;output.mkdir(exist_ok=False)
    with (output/"source_script.py").open("xb") as stream:stream.write(Path(__file__).read_bytes())
    try:
        result=prepare(output) if args.prepare else execute(output,args.protocol,args.protocol_sha)
    except BaseException as error:
        if not (output/"failure.json").exists():
            write_new(output/"failure.json",{"status":"failed_CPU_preparation_preserved" if args.prepare else "failed_pre_GPU_source_or_input_gate_preserved",
                "exception_type":type(error).__name__,"exception_message":str(error),"traceback":traceback.format_exc(),"GPU_render_calls":0})
        raise
    print(json.dumps(result))


if __name__=="__main__":main()
