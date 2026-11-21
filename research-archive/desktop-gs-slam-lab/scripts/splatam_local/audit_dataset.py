"""Independent audit of the pinned SplaTAM TUM loader's input contract.

No torch/CUDA dependency, no modification of datasets.  Association follows the
upstream nearest-neighbour algorithm, including reuse of depth/GT samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

LAB = Path(__file__).resolve().parents[2]
PIN = "da6bbcd24c248dc884ac7f49d62e91b841b26ccc"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parse_inputs(sequence):
    sequence = Path(sequence).resolve()
    rgb = np.loadtxt(sequence / "rgb.txt", dtype=str, comments="#", ndmin=2)
    depth = np.loadtxt(sequence / "depth.txt", dtype=str, comments="#", ndmin=2)
    gt_name = "groundtruth.txt" if (sequence / "groundtruth.txt").is_file() else "pose.txt"
    # Preserve the pinned loader's skiprows=1 contract (normal TUM first row is a comment).
    gt = np.loadtxt(sequence / gt_name, dtype=str, comments="#", skiprows=1, ndmin=2)
    if rgb.shape[1] != 2 or depth.shape[1] != 2 or gt.shape[1] != 8:
        raise ValueError(f"Unexpected TUM columns: RGB {rgb.shape}, depth {depth.shape}, GT {gt.shape}")
    stamps = [arr[:, 0].astype(np.float64) for arr in (rgb, depth, gt)]
    for label, times in zip(("RGB", "depth", "GT"), stamps):
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Non-finite or non-increasing {label} timestamps")
    pose_values = gt[:, 1:].astype(np.float64)
    if not np.isfinite(pose_values).all() or np.any(np.linalg.norm(pose_values[:, 3:], axis=1) < 1e-12):
        raise ValueError("Invalid raw GT translation/quaternion")
    absolute_gt = np.repeat(np.eye(4)[None], len(gt), axis=0)
    absolute_gt[:, :3, :3] = Rotation.from_quat(pose_values[:, 3:]).as_matrix()
    absolute_gt[:, :3, 3] = pose_values[:, :3]
    return rgb, depth, gt, stamps, absolute_gt, gt_name


def associate(sequence):
    """Return all associations and the exact >1/32 s retained subset."""
    rgb, depth, gt, (tr, td, tp), absolute_gt, gt_name = parse_inputs(sequence)
    candidates = []
    for i, stamp in enumerate(tr):
        j = int(np.argmin(np.abs(td - stamp)))
        k = int(np.argmin(np.abs(tp - stamp)))
        if abs(td[j] - stamp) < 0.08 and abs(tp[k] - stamp) < 0.08:
            candidates.append((i, j, k))
    if not candidates:
        raise ValueError("No valid RGB-depth-GT associations")
    selected = [0]
    for candidate_index in range(1, len(candidates)):
        old_rgb = candidates[selected[-1]][0]
        new_rgb = candidates[candidate_index][0]
        if tr[new_rgb] - tr[old_rgb] > 1.0 / 32:
            selected.append(candidate_index)
    rows = []
    for retained_index, candidate_index in enumerate(selected):
        i, j, k = candidates[candidate_index]
        rows.append(dict(
            retained_index=retained_index, association_index=candidate_index,
            rgb_index=i, depth_index=j, gt_index=k,
            rgb_timestamp=float(tr[i]), depth_timestamp=float(td[j]), gt_timestamp=float(tp[k]),
            depth_minus_rgb_seconds=float(td[j] - tr[i]), gt_minus_rgb_seconds=float(tp[k] - tr[i]),
            rgb_path=str(rgb[i, 1]), depth_path=str(depth[j, 1]),
        ))
    return rows, absolute_gt, dict(rgb=len(rgb), depth=len(depth), groundtruth=len(gt),
                                  associated_before_32hz_filter=len(candidates), retained=len(rows), gt_name=gt_name)


def select_frames(rows, data):
    """Upstream slices after association; num_frames applies afterwards."""
    start = int(data.get("start", 0))
    end = int(data.get("end", -1))
    stride = int(data.get("stride", 1))
    if start < 0 or stride <= 0 or (end != -1 and end <= start):
        raise ValueError("Invalid start/end/stride")
    selected = rows[start:None if end == -1 else end:stride]
    limit = int(data.get("num_frames", -1))
    if limit != -1:
        if limit < 2 or limit > len(selected):
            raise ValueError("Invalid num_frames or request exceeds available frames")
        selected = selected[:limit]
    if len(selected) < 2:
        raise ValueError("Need at least two selected frames for trajectory evaluation")
    return selected


def relative_gt(rows, absolute_gt):
    poses = absolute_gt[np.asarray([row["gt_index"] for row in rows])]
    return np.linalg.inv(poses[0])[None] @ poses


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(sequence, output):
    sequence, output = Path(sequence).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, absolute_gt, counts = associate(sequence)
    rgb, depth, _, stamps, _, gt_name = parse_inputs(sequence)
    hashes = {name: dict(bytes=(sequence / name).stat().st_size, sha256=sha(sequence / name))
              for name in ("rgb.txt", "depth.txt", gt_name)}
    image_stats = {}
    for kind, listing in (("rgb", rgb), ("depth", depth)):
        dimensions, modes, total_bytes = set(), set(), 0
        valid_depth_fractions, positive_depth_min, positive_depth_max = [], [], []
        for relative_name in dict.fromkeys(listing[:, 1]):
            path = (sequence / str(relative_name)).resolve()
            if not path.is_relative_to(sequence):
                raise ValueError(f"Input path escapes sequence: {relative_name}")
            entry = dict(bytes=path.stat().st_size, sha256=sha(path))
            hashes[str(relative_name)] = entry
            total_bytes += entry["bytes"]
            with Image.open(path) as image:
                image.load()  # Decode all referenced files, not only inspect PNG headers.
                dimensions.add(image.size)
                modes.add(image.mode)
                if kind == "depth":
                    values = np.asarray(image)
                    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
                        raise ValueError(f"Unexpected depth dtype/shape: {relative_name}")
                    positive = values[values > 0]
                    valid_depth_fractions.append(float(positive.size / values.size))
                    if not positive.size:
                        raise ValueError(f"No valid depth pixels: {relative_name}")
                    positive_depth_min.append(float(positive.min() / 5000))
                    positive_depth_max.append(float(positive.max() / 5000))
        image_stats[kind] = dict(unique_referenced_files=len(set(listing[:, 1])),
                                 decoded_files=len(set(listing[:, 1])), dimensions=sorted(dimensions),
                                 modes=sorted(modes), total_bytes=total_bytes)
        if kind == "depth":
            image_stats[kind].update(depth_scale=5000, unit="metres", zero_is_invalid=True,
                                     minimum_positive_depth_m=min(positive_depth_min),
                                     maximum_positive_depth_m=max(positive_depth_max),
                                     valid_pixel_fraction_min=min(valid_depth_fractions),
                                     valid_pixel_fraction_mean=float(np.mean(valid_depth_fractions)))
    for row in rows:
        row["rgb_sha256"] = hashes[row["rgb_path"]]["sha256"]
        row["depth_sha256"] = hashes[row["depth_path"]]["sha256"]
    write_rows(output / "associations.csv", rows)
    save(output / "input-hashes.json", hashes)
    rel = relative_gt(rows, absolute_gt)
    loader = LAB / "repos/SplaTAM/datasets/gradslam_datasets/tum.py"
    provenance_manifest = (LAB / "manifests/datasets/tum_freiburg1_desk.json" if sequence.name.endswith("_desk")
                           else LAB / "manifests/tum-fr1-xyz-preparation.json")
    provenance = dict(publisher_checksum_available=False, checksum_scope="Local file identifier; no publisher SHA256 claim")
    if provenance_manifest.is_file():
        prior = json.loads(provenance_manifest.read_text(encoding="utf-8"))
        download = prior.get("download", {})
        archive_relative = prior.get("archive_relative_path", download.get("archive_relative_path"))
        expected_archive_hash = prior.get("archive_sha256", download.get("sha256_local_computed"))
        provenance.update(manifest=str(provenance_manifest.relative_to(LAB)), manifest_sha256=sha(provenance_manifest),
                          source_url=prior.get("source_url", download.get("final_url")),
                          license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                          attribution="TUM RGB-D Benchmark; Sturm et al., A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012")
        if archive_relative and expected_archive_hash:
            archive = LAB / archive_relative
            actual_archive_hash = sha(archive)
            if actual_archive_hash != expected_archive_hash:
                raise ValueError("Dataset archive hash differs from its acquisition receipt")
            provenance.update(archive=archive_relative, archive_bytes=archive.stat().st_size,
                              archive_sha256=actual_archive_hash, archive_matches_acquisition_receipt=True)
    metrics = dict(
        schema_version=1, status="verified", sequence=sequence.name,
        source_directory=str(sequence), splatam_source_pin=PIN,
        loader_sha256=sha(loader) if loader.exists() else None,
        official_dataset_url="https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download", provenance=provenance,
        counts=counts, decoded_image_stats=image_stats, hashed_input_files=len(hashes),
        association="Nearest depth and GT separately per RGB; each absolute dt <0.08 s; retain RGB interval >1/32 s",
        depth_reuse_after_filter=len(rows)-len({row["depth_index"] for row in rows}),
        gt_reuse_after_filter=len(rows)-len({row["gt_index"] for row in rows}),
        max_abs_rgb_depth_dt_s=max(abs(row["depth_minus_rgb_seconds"]) for row in rows),
        max_abs_rgb_gt_dt_s=max(abs(row["gt_minus_rgb_seconds"]) for row in rows),
        retained_rgb_duration_s=rows[-1]["rgb_timestamp"]-rows[0]["rgb_timestamp"],
        minimum_retained_rgb_interval_s=float(np.diff([row["rgb_timestamp"] for row in rows]).min()),
        groundtruth_translation_unit="metres", groundtruth_quaternion_order="qx qy qz qw",
        selected_gt_path_length_m=float(np.linalg.norm(np.diff(rel[:, :3, 3], axis=0), axis=1).sum()),
        selected_gt_start_end_displacement_m=float(np.linalg.norm(rel[-1, :3, 3])),
        associations_sha256=sha(output / "associations.csv"), input_hashes_sha256=sha(output / "input-hashes.json"),
        limits=["Association is the upstream benchmark loader convention, not physical clock calibration.",
                "GT nearest-neighbour selection is retained for parity; no interpolation is claimed.",
                "TUM Freiburg1 RGB is already registered to depth; 5000 is the official PNG depth scale."])
    save(output / "metrics.json", metrics)
    lines = [f"# SplaTAM TUM 数据独立审核：{sequence.name}", "",
             f"原始 RGB {counts['rgb']}、深度 {counts['depth']}、真值 {counts['groundtruth']} 行。",
             f"严格复现官方最近邻匹配：先得到 {counts['associated_before_32hz_filter']} 对，再按大于 1/32 秒保留 **{counts['retained']} 帧**。",
             f"全部 {len(hashes)} 个输入文件已计算 SHA256；RGB 与深度 PNG 均完整解码，未更改源数据。", "",
             f"RGB–depth 最大时间差 {metrics['max_abs_rgb_depth_dt_s']:.6f} 秒；RGB–GT 最大时间差 {metrics['max_abs_rgb_gt_dt_s']:.6f} 秒。",
             f"保留帧覆盖 {metrics['retained_rgb_duration_s']:.3f} 秒。GT 路程 {metrics['selected_gt_path_length_m']:.3f} 米。", "",
             "逐帧索引、RGB/depth/GT 时间戳、时间差和图像哈希保存在 associations.csv，完整输入哈希在 input-hashes.json。", "",
             "这里的时间匹配是公开基准加载规则，不能替代传感器时钟同步验证。GT 只用于此处关联和独立评测；正式跟踪必须关闭 use_gt_poses。",
             "同属 Freiburg1 的 desk 与 xyz 可复用公开相机内参，复用不等于重新完成标定。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=LAB / "outputs/splatam-dataset-audit-001")
    args = parser.parse_args()
    sequences = args.sequence or [LAB / "data/tum" / f"rgbd_dataset_freiburg1_{name}" for name in ("desk", "xyz")]
    args.output.mkdir(parents=True, exist_ok=True)
    results = [audit(sequence, args.output / sequence.name) for sequence in sequences]
    summary = dict(schema_version=1, status="all_available_sequences_verified", sequence_count=len(results),
                   sequences={item["sequence"]: item for item in results}, downloaded_bytes=0,
                   source_modified=False, auditor_sha256=sha(Path(__file__)))
    save(args.output / "metrics.json", summary)
    (args.output / "report.md").write_text("# SplaTAM 公开数据审核\n\n" + "\n\n".join(
        f"{item['sequence']}：原始 RGB {item['counts']['rgb']} / depth {item['counts']['depth']} / GT {item['counts']['groundtruth']}；官方关联过滤后 **{item['counts']['retained']} 帧**，全部图像解码和文件哈希通过。"
        for item in results) + "\n\n未下载新数据、未修改源文件。逐帧证据在各序列 associations.csv；5000 深度尺度来自 TUM Freiburg1 官方格式。\n", encoding="utf-8")
    print(json.dumps({item["sequence"]: item["counts"] for item in results}))


if __name__ == "__main__":
    main()
