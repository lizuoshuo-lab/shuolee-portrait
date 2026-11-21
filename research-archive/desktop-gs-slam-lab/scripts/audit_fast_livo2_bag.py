"""Read a complete ROS1 bag on Windows; do not run or claim FAST-LIVO2.

Header, record and algorithm-adjusted clocks remain separate. The NTU Leica
stream is exported as position-only reference, never orientation ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
import yaml


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stamp_ns(msg) -> int | None:
    if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
        return None
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"count": 0}
    return {"count": int(a.size), "min": float(a.min()),
            "median": float(np.median(a)), "p95": float(np.quantile(a, .95)),
            "max": float(a.max()), "mean": float(a.mean())}


def nearest_ms(query, reference):
    q, r = np.asarray(query, dtype=np.int64), np.sort(np.asarray(reference, dtype=np.int64))
    if not q.size or not r.size:
        return []
    k = np.searchsorted(r, q)
    lo, hi = np.clip(k - 1, 0, r.size - 1), np.clip(k, 0, r.size - 1)
    return np.minimum(np.abs(q - r[lo]), np.abs(q - r[hi])) / 1e6


def imu_brackets(query, imu):
    """Sorted local bracketing is a data observation, not runtime buffer replay."""
    q, r = np.asarray(query, dtype=np.int64), np.sort(np.asarray(imu, dtype=np.int64))
    if not r.size:
        return {"queries": int(q.size), "exact": 0, "bracketed_nonexact": 0,
                "outside": int(q.size), "bracket_gap_ms": distribution([])}
    k = np.searchsorted(r, q)
    exact = (k < r.size) & (r[np.minimum(k, r.size - 1)] == q)
    valid = (~exact) & (k > 0) & (k < r.size)
    gaps = (r[k[valid]] - r[k[valid] - 1]) / 1e6
    return {"queries": int(q.size), "exact": int(exact.sum()),
            "bracketed_nonexact": int(valid.sum()),
            "outside": int(q.size - exact.sum() - valid.sum()),
            "bracket_gap_ms": distribution(gaps),
            "brackets_over_10_ms": int((gaps > 10).sum()),
            "scope": "Sorted offline header clock; original-order reversals remain separate. Does not emulate FAST-LIVO2 callback buffering."}


def ouster_scan_duration_ns(msg) -> int | None:
    fields = {field.name: field for field in getattr(msg, "fields", [])}
    field = fields.get("t")
    if field is None or int(field.datatype) != 6 or int(field.count) != 1:
        return None
    # sensor_msgs/PointField.UINT32; official Ouster preprocessing uses t in ns.
    height, width = int(msg.height), int(msg.width)
    if not height or not width:
        return None
    offset, step, row_step = int(field.offset), int(msg.point_step), int(msg.row_step)
    if offset + 4 > step or len(msg.data) < row_step * height:
        raise ValueError("Invalid PointCloud2 layout")
    dtype = np.dtype(">u4" if msg.is_bigendian else "<u4")
    times = np.ndarray((height, width), dtype=dtype, buffer=msg.data,
                       offset=offset, strides=(row_step, step))
    return int(times.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gt-topic", default="/leica/pose/relative")
    parser.add_argument("--camera-topic", default=None, help="Observed transport topic for timestamp inspection; does not change the estimator")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; previous observations are preserved")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    topics = {kind: config["common"][name] for kind, name in
              (("lidar", "lid_topic"), ("imu", "imu_topic"), ("camera", "img_topic"))}
    original_camera_topic = topics["camera"]
    if args.camera_topic:
        topics["camera"] = args.camera_topic
    offsets = config.get("time_offset", {})
    # These signs are from the pinned FAST-LIVO2 callbacks, not a generic rule.
    adjusted = {"lidar": float(offsets.get("lidar_time_offset", 0)),
                "imu": -float(offsets.get("imu_time_offset", 0)),
                "camera": float(offsets.get("img_time_offset", 0)) +
                          float(offsets.get("exposure_time_init", 0))}
    lidar_type = int(config["preprocess"]["lidar_type"])
    if lidar_type not in (1, 3):
        raise ValueError("Point-time contracts here cover only pinned Livox CustomMsg and NTU Ouster")
    if lidar_type == 1:
        # Pinned livox callback does not apply lidar_time_offset.
        adjusted["lidar"] = 0.0
    args.output.mkdir(parents=True)
    shutil.copy2(Path(__file__), args.output / "audit_source.py")
    versions = {name: importlib.metadata.version(name) for name in
                ("rosbags", "numpy", "matplotlib", "PyYAML")}
    bag_hash = sha(args.bag)
    reverse = {topic: kind for kind, topic in topics.items()}
    headers = {kind: [] for kind in topics}
    records = {kind: [] for kind in topics}
    scan_end = []
    scan_spans, livox_timebase_minus_header = [], []
    gt = []
    imu_norms = []
    missing_header = {kind: 0 for kind in topics}
    with (args.output / "sensor-timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "topic", "bag_record_ns", "sensor_header_ns", "algorithm_adjusted_ns", "point_span_ns"])
        with AnyReader([args.bag]) as reader:
            inventory = [{"topic": c.topic, "msgtype": c.msgtype, "count": c.msgcount}
                         for c in reader.connections]
            connections = [c for c in reader.connections if c.topic in reverse or c.topic == args.gt_topic]
            for connection, record, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, connection.msgtype)
                timestamp = stamp_ns(msg)
                if connection.topic == args.gt_topic:
                    if timestamp is None:
                        raise ValueError("GT topic has no header timestamp")
                    pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
                    point = pose.position
                    gt.append((timestamp, float(point.x), float(point.y), float(point.z)))
                    continue
                kind = reverse[connection.topic]
                records[kind].append(int(record))
                if timestamp is None:
                    missing_header[kind] += 1
                    continue
                headers[kind].append(timestamp)
                corrected = timestamp + round(adjusted[kind] * 1e9)
                span = None
                if kind == "lidar":
                    span = (ouster_scan_duration_ns(msg) if lidar_type == 3 else
                            max((int(point.offset_time) for point in msg.points), default=0))
                if kind == "lidar" and span is not None:
                    scan_end.append(corrected + span)
                    scan_spans.append(span)
                    if lidar_type == 1:
                        livox_timebase_minus_header.append(int(msg.timebase) - timestamp)
                if kind == "imu":
                    a, w = msg.linear_acceleration, msg.angular_velocity
                    imu_norms.append((np.linalg.norm([a.x, a.y, a.z]), np.linalg.norm([w.x, w.y, w.z])))
                writer.writerow([kind, connection.topic, record, timestamp, corrected, span])
    observed = all(len(values) > 1 for values in headers.values())
    report = {"status": ("complete_bag_read_only_timing_observation" if observed else
                          "complete_bag_read_but_required_sensor_topics_missing"), "bag": str(args.bag),
              "bag_bytes": args.bag.stat().st_size, "bag_sha256": bag_hash,
              "config_sha256": sha(args.config), "source_sha256": sha(Path(__file__)),
              "versions": versions, "connections": inventory, "topics": {},
              "configured_clock_adjustment_seconds": adjusted, "required_sensor_topics_observed": observed,
              "configured_camera_topic": original_camera_topic, "observed_camera_transport_topic": topics["camera"],
              "scope": "No estimator, calibration fit or hardware synchronization verification. GT positions only; no orientation GT, rotation-RPE, or FAST-LIVO2 trajectory accuracy."}
    for kind, values in headers.items():
        h, r = np.asarray(values, dtype=np.int64), np.asarray(records[kind], dtype=np.int64)
        delta = np.diff(h)
        report["topics"][kind] = {"topic": topics[kind], "header_messages": int(h.size),
            "missing_header_messages": missing_header[kind],
            "header_span_seconds": float((h[-1] - h[0]) / 1e9) if h.size > 1 else None,
            "header_negative_intervals": int(np.count_nonzero(delta < 0)),
            "header_duplicate_intervals": int(np.count_nonzero(delta == 0)),
            "positive_interval_ms": distribution(delta[delta > 0] / 1e6),
            "record_minus_header_ms": distribution((r - h) / 1e6) if r.size == h.size else None}
    corrected_headers = {kind: np.asarray(values, dtype=np.int64) + round(adjusted[kind] * 1e9)
                         for kind, values in headers.items()}
    imu_clock = np.sort(corrected_headers["imu"])
    report["camera_to_lidar_start_nearest_ms"] = distribution(nearest_ms(corrected_headers["camera"], corrected_headers["lidar"]))
    report["camera_to_lidar_end_nearest_ms"] = distribution(nearest_ms(corrected_headers["camera"], scan_end))
    for kind in ("camera", "lidar"):
        stamps = corrected_headers[kind]
        covered = np.count_nonzero((stamps >= imu_clock[0]) & (stamps <= imu_clock[-1])) if imu_clock.size else 0
        report[f"{kind}_timestamps_inside_imu_span"] = {"covered": int(covered), "total": int(stamps.size)}
        report[f"{kind}_local_imu_brackets"] = imu_brackets(stamps, imu_clock)
    report["lidar_scan_end_local_imu_brackets"] = imu_brackets(scan_end, imu_clock)
    report["raw_point_max_offset_ms"] = distribution(np.asarray(scan_spans) / 1e6)
    if livox_timebase_minus_header:
        report["livox_timebase_minus_header_ms"] = distribution(np.asarray(livox_timebase_minus_header) / 1e6)
        report["livox_point_clock_scope"] = "Pinned callback uses header plus point.offset_time; it does not use CustomMsg.timebase. Field difference is not independently measured physical clock offset."
    if imu_norms:
        norms = np.asarray(imu_norms)
        report["imu_acceleration_norm_native_units"] = distribution(norms[:, 0])
        report["imu_angular_velocity_norm_native_units"] = distribution(norms[:, 1])
        report["imu_units_scope"] = "Message type alone does not verify publisher units. Raw acceleration norm near one here; pinned IMU processor normalizes mean_acc.norm() to gravity. No SI conversion or unit calibration performed."
    report["gt_position_reference"] = {"topic": args.gt_topic, "samples": len(gt),
        "orientation_is_ground_truth": False, "reference_point": "Leica prism",
        "imu_to_prism_body_m": [-0.293656, -0.012288, -0.273095]}
    if gt:
        np.savetxt(args.output / "gt-prism-position.csv", gt, delimiter=",", fmt=["%d", "%.12g", "%.12g", "%.12g"],
                   header="timestamp_ns,x_m,y_m,z_m", comments="")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    for axis, kind in zip(axes, ("lidar", "camera", "imu")):
        h = corrected_headers[kind]
        if h.size > 1:
            axis.plot((h[1:] - h[0]) / 1e9, np.diff(h) / 1e6, linewidth=.65)
        axis.set(title=f"{kind}: adjusted sensor timestamp increments", ylabel="interval / ms", xlabel="time / s")
        axis.grid(alpha=.25)
    fig.suptitle("Recorded timing observations; not proof of hardware synchronization")
    fig.savefig(args.output / "timestamp-intervals.png", dpi=160)
    plt.close(fig)
    if sha(args.bag) != bag_hash:
        raise RuntimeError("Bag changed during inspection")
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "topics": report["topics"], "gt_samples": len(gt)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
