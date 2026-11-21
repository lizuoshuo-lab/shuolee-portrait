"""Measure identical inputs and actual warmup-map bytes of full V4 weight pairs.

CPU artifact observation only: no GT, training, renderer or pose corrections.
Separate whole-pipeline arms are not assumed to share a byte-identical map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"manifests/tum-online-v4-full-sequence-protocol-001.json"
PROTOCOL_SHA = "ee635a6d65cc02dd26ff16e3f83721afa745a56b9f7a7464453833651a4963af"
INPUTS = ["image", "depth_m", "valid_depth_mask", "K", "initial_w2c", "prior_reference_w2c", "icp_prefix_c2w_float64"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def details(value):
    if value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all():
        raise ValueError("Context tensor must be finite detached CPU")
    array = value.numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "raw_C_order_bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def compare(scene):
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Unknown full-sequence protocol")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = [row for row in protocol["new_runs"] if row["scene"] == scene]
    if len(rows) != 2 or [row["configuration"]["pose_prior_weight"] for row in rows] != [0., .01]:
        raise ValueError("Unexpected reviewed scene weight pair")
    contexts, hashes = [], {}
    for row in rows:
        folder = ROOT/row["output_relative_path"]
        metrics = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        meta = json.loads((folder/"first_tracking_context.json").read_text(encoding="utf-8"))
        if not metrics["status"].startswith("completed") or metrics["configuration"] != row["configuration"] or meta != metrics["first_tracking_context"]:
            raise ValueError("A full paired arm is not completed/configuration-consistent")
        path = folder/"first_tracking_context.pt"
        if sha(path) != meta["checkpoint_sha256"]:
            raise ValueError("Context SHA mismatch")
        context = torch.load(path, map_location="cpu", weights_only=True)
        if context["configuration"] != row["configuration"] or context["ground_truth_included"] is not False or context["future_maps_or_trajectories_included"] is not False:
            raise ValueError("Context configuration/causal metadata")
        contexts.append(context)
        hashes[row["run"]] = {"metrics_sha256": sha(folder/"metrics.json"), "context_PT_sha256": sha(path), "context_JSON_sha256": sha(folder/"first_tracking_context.json")}
    left, right = contexts
    a, b = left["configuration"], right["configuration"]
    if {key for key in a if a[key] != b[key]} != {"run", "pose_prior_weight"}:
        raise ValueError("Paired arms differ outside run/weight")
    inputs, maps = {}, {}
    for key in INPUTS:
        before, after = details(left[key]), details(right[key])
        inputs[key] = {"prior0": before, "prior001": after, "shape_dtype_and_raw_bytes_equal": before == after}
        if before != after:
            raise ValueError("Paired fixed sensor/initial/reference differs: "+key)
    if set(left["parameters"]) != set(right["parameters"]):
        raise ValueError("Warmup parameter field mismatch")
    for key in left["parameters"]:
        p, q = left["parameters"][key], right["parameters"][key]
        a, b = details(p), details(q)
        if a["shape"] != b["shape"] or a["dtype"] != b["dtype"]:
            raise ValueError("Warmup map shape/dtype differ: "+key)
        maps[key] = {"prior0": a, "prior001": b, "raw_bytes_equal": a == b,
                     "max_absolute_element_difference": float(np.max(np.abs(p.numpy().astype(np.float64)-q.numpy().astype(np.float64))))}
    if left["source_frame_ids_per_gaussian"] != right["source_frame_ids_per_gaussian"] or set(left["source_frame_ids_per_gaussian"]) != {0}:
        raise ValueError("Warmup first-frame source/order metadata differ")
    if left["sensor_metadata"] != right["sensor_metadata"] or left["source"] != right["source"]:
        raise ValueError("Pair sensor/source metadata differ")
    return {"status": "passed", "scene": scene, "runs": [row["run"] for row in rows],
        "protocol_sha256": PROTOCOL_SHA, "checker_sha256": sha(Path(__file__)), "artifact_hashes": hashes,
        "seven_common_sensor_initial_reference_tensors_byte_identical": True, "input_tensors": inputs,
        "all_five_warmup_map_parameter_tensors_byte_identical": all(item["raw_bytes_equal"] for item in maps.values()),
        "warmup_map_parameters": maps, "warmup_point_count": len(left["source_frame_ids_per_gaussian"]),
        "first_map_source_ids_and_order_equal": True, "GT_read_or_online_pose_selection": False, "GPU_calls": 0,
        "scope": "Paired whole-pipeline artifact observation, not an identical-frozen-map causal experiment; CUDA reductions were not forced deterministic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["desk", "xyz"], required=True)
    args = parser.parse_args()
    result = compare(args.scene)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only comparison unexpectedly initialized CUDA")
    path = ROOT/"outputs/tum-online-v4-full-sequence-001"/(args.scene+"_warmup_context_pair_checks.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("status", "scene", "seven_common_sensor_initial_reference_tensors_byte_identical", "all_five_warmup_map_parameter_tensors_byte_identical", "warmup_point_count", "GPU_calls")}, indent=2))


if __name__ == "__main__":
    main()
