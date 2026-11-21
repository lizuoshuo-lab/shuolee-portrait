"""Assemble CPU-only image evidence from an already delivered automatic SH0 run."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory, gettempdir

import numpy as np
from PIL import Image

from review_full_sequence123 import crop_box, joined, mesh_renderer, relocate
from run_official_gaussians123 import now, read, save, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_image(path, size, mode):
    with Image.open(path) as image:
        require(image.size == tuple(size), f"Unexpected native dimensions: {path}")
        return image.convert(mode)


def select_samples(rows, reference):
    ordered = sorted(rows, key=lambda row: (row["source_frame_index"], row["frame"]))
    first, last = (ordered[i]["source_frame_index"] for i in (0, -1))
    positions = [first + (last - first) * i / 4 for i in range(5)]
    # min keeps the first chronological row on a tie.
    selected = {min(ordered, key=lambda row: abs(row["source_frame_index"] - t))["frame"] for t in positions}
    return [row for row in ordered if row["frame"] in selected or row["frame"] == reference], positions


def review_run(out, output=None):
    out = relocate(out).resolve()
    publication_path = out / "delivery/publication.json"
    publication_hash = sha(publication_path)
    publication = read(publication_path)
    step = publication.get("step")
    require(publication.get("state") == "delivered_pending_quality_review"
            and publication.get("format_passed") is True and publication["run"] == out.name
            and type(step) is int and step > 0, "A completed automatic SH0 delivery is required.")
    snapshot = out / "gaussian-run" / f"step-{step:06d}"
    delivered = out / "delivery" / snapshot.name
    source_path = delivered / "source-receipt.json"
    source_hash = sha(source_path)
    require(source_hash == publication["source_receipt_sha256"]
            and sha(delivered / "point_cloud.ply") == publication["ply_sha256"]
            and sha(delivered / "cameras.json") == publication["camera_sha256"]
            and sha(delivered / "scene.zip") == publication["package_sha256"], "Delivered artifacts changed.")
    source = read(source_path)
    reference = publication["reference_frame"]
    require(source["terminal_verified"] is True and source["step"] == step
            and source["points"] == publication["points"] and source["reference_frame"] == reference
            and sha(snapshot / "world.ply") == source["source_ply_sha256"],
            "Publication does not match its terminal snapshot.")

    rows_path, meshes_path, result_path = (out / "data/training-inputs.json",
        snapshot / "current-meshes.npz", snapshot / "result.json")
    recorded = source["training_files_sha256"]
    evidence = {p.relative_to(out).as_posix(): sha(p) for p in (rows_path, meshes_path, result_path)}
    require(all(recorded.get(path) == digest for path, digest in evidence.items()),
            "Native inputs, mesh archive or snapshot result changed.")
    rows, result = read(rows_path)["records"], read(result_path)
    ids = [row["frame"] for row in rows]
    require(ids == source["actual_frames"] and len(ids) == len(set(ids)) == result["frames_rendered"]
            and len({row["source_frame_index"] for row in rows}) == len(ids)
            and reference in ids and ids[result["reference_slot"]] == reference == result["reference_frame"]
            and result["step"] == step and result["points"] == source["points"]
            and result["world_ply_sha256"] == source["source_ply_sha256"],
            "Native review frames or checkpoint reference differ from the published receipt.")
    with np.load(meshes_path, allow_pickle=False) as mesh_archive:
        mesh_ids = mesh_archive["frame_ids"].astype(str).tolist()
        vertices, faces = (mesh_archive[name] for name in ("vertices_camera", "faces"))
    require(mesh_ids == ids and vertices.ndim == 3 and vertices.shape[0] == len(ids)
            and vertices.shape[2] == 3 and np.isfinite(vertices).all()
            and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0
            and faces.max() < vertices.shape[1], "Invalid archived training mesh or topology.")
    slots = {frame: index for index, frame in enumerate(mesh_ids)}
    selected, positions = select_samples(rows, reference)
    mesh, renderer_path = mesh_renderer()

    output = relocate(output).resolve() if output else out / "post-delivery-visual-review"
    temp_root = Path(gettempdir()).resolve()
    require(output.parent == out or temp_root in output.parents,
            "Review output must be a new folder within this run or the system temp directory.")
    require(relocate("D:/00").resolve() not in (output, *output.parents)
            and all(part.casefold() != "viewer" for part in output.parts),
            "Review files must stay outside 00 and viewer directories.")
    require(output.parent.is_dir() and not output.exists(), "Review output parent missing or review already exists.")

    validation_path = delivered / "validation.json"
    validation = read(validation_path)
    require(validation["format_passed"] is True and validation["sha256"] == publication["ply_sha256"]
            and validation["source_sha256"] == source["source_ply_sha256"], "Published validation does not match its PLY.")
    orbit_root = delivered / "novel"
    orbit_cameras_path = orbit_root / "cameras.json"
    native, orbit = [], []
    with TemporaryDirectory(dir=output.parent, prefix=".sh0-review-") as temporary:
        stage = Path(temporary) / "review"
        stage.mkdir()
        for row in selected:
            frame, size = row["frame"], row["wh"]
            image_path, mask_path = (relocate(row[name]) for name in ("image", "mask"))
            gaussian_path = snapshot / f"{frame}_gaussian.png"
            require(sha(image_path) == row["source_sha256"] and sha(mask_path) == row["mask_sha256"],
                    f"Source image or mask changed: {frame}")
            gaussian_hash = sha(gaussian_path)
            pinned_rgb = recorded.get(gaussian_path.relative_to(out).as_posix())
            require(pinned_rgb is None or pinned_rgb == gaussian_hash, f"Archived Gaussian render changed: {frame}")
            original, gaussian = (read_image(path, size, "RGB") for path in (image_path, gaussian_path))
            mask = np.asarray(read_image(mask_path, size, "L")) >= 128
            pixels = int(mask.sum())
            delta = (np.asarray(gaussian)[mask].astype(np.float64) - np.asarray(original)[mask]) / 255.
            box = crop_box(row.get("initial_box_xyxy", [0, 0, *size]), size, padding=64)
            integer_K = np.asarray(row["K_integer_centers"], dtype=np.float64)
            corner_K = integer_K.copy()
            corner_K[:2, 2] += .5
            require(integer_K.shape == (3, 3) and np.isfinite(integer_K).all()
                    and np.array_equal(corner_K, row["K"]), f"Invalid native projection: {frame}")
            fit = dict(vertices_camera=vertices[slots[frame]:slots[frame] + 1], faces=faces,
                       K_integer_centers=integer_K)
            raw_crop = original.crop(box)
            overlay = mesh(raw_crop, np.asarray(box), fit)
            triplet = joined([raw_crop, overlay, gaussian.crop(box)])
            name = f"{frame}-original-trained-mesh-gaussian.png"
            triplet.save(stage / name)
            native.append(dict(frame=frame, source_frame_index=row["source_frame_index"],
                time_seconds=row.get("time_seconds"), reference_pose=frame == reference,
                image_sha256=row["source_sha256"], mask_sha256=row["mask_sha256"],
                archived_gaussian_sha256=gaussian_hash, archived_gaussian_pinned_by_receipt=pinned_rgb is not None,
                native_wh=size, crop_xyxy=box, file=name, sha256=sha(stage / name),
                foreground_pixels=pixels, foreground_rgb_mae=float(np.abs(delta).mean()) if pixels else None,
                foreground_rgb_rmse=float(np.sqrt(np.square(delta).mean())) if pixels else None))

        orbit_camera_hash = None
        if orbit_cameras_path.is_file():
            orbit_camera_hash = sha(orbit_cameras_path)
            cameras = read(orbit_cameras_path)
            camera_rows = cameras["records"]
            require(cameras["synthetic"] is True and cameras["ground_truth_available"] is False
                    and len(camera_rows) == validation["novel_synthetic_views"] and len(camera_rows) >= 4
                    and all(row["synthetic"] is True for row in camera_rows),
                    "Saved orbit differs from the frozen PLY export.")
            for i in (k * len(camera_rows) // 4 for k in range(4)):
                row = camera_rows[i]
                path = orbit_root / f"{row['frame']}.png"
                read_image(path, row["wh"], "RGB")
                name = f"orbit-{row['frame']}.png"
                shutil.copyfile(path, stage / name)
                orbit.append(dict(camera_index=i, frame=row["frame"], wh=row["wh"], file=name,
                                  sha256=sha(stage / name)))

        manifest = dict(schema="automatic-sh0-visual-review-v1", updated=now(), run=out.name, step=step,
            reference_frame=reference, publication_sha256=publication_hash, source_receipt_sha256=source_hash,
            published_ply_sha256=publication["ply_sha256"], snapshot_input_sha256=evidence,
            overlay_renderer_sha256=sha(renderer_path), export_validation_sha256=sha(validation_path),
            selection=dict(method="Nearest source frame index to first, quarter, half, three quarters and last; "
                                  "ties choose earlier, duplicates collapsed, published reference always included.",
                           interval_targets=positions, selected_frames=[row["frame"] for row in selected]),
            panel_order=["original", "trained mesh overlay", "archived Gaussian RGB"],
            crop_rule="Frozen initial box plus 64 native pixels, clipped to raster. All panels use the same integer crop.",
            rgb_error_rule="Native full-frame source foreground mask grayscale >=128. RGB errors are in [0,1].",
            captured_comparisons=native, synthetic_orbit=orbit, orbit_camera_archive_sha256=orbit_camera_hash,
            orbit_ground_truth_available=False, heldout_views_evaluated=False, quality_verdict="not_evaluated",
            cpu_only=True)
        save(stage / "manifest.json", manifest)
        require(sha(publication_path) == publication_hash, "Publication changed during review assembly.")
        require(not output.exists(), "Review output appeared during assembly.")
        stage.rename(output)
    return output / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output", help="New review folder. Defaults to post-delivery-visual-review inside the run.")
    args = parser.parse_args()
    print(json.dumps(dict(manifest=str(review_run(args.run_dir, args.output))), ensure_ascii=False))
