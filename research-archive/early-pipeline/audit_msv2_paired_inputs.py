"""Independent CPU audit of all native-parent / masked4x-child input groups."""

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from probe_msv2_face_fullframe import read, save, sha, host_path, VIDEO_SHA, INPUT_SHA


POINTS_SHA = "7e049a3b338912f28e87c181fbea9efc6ae1ffbfd0a522d6be7f874719f4f787"
IMPLEMENTATION_SHA = "8c2c123b015ad908a220a9374052a8cf768f8955a60a7c91935c16c52f32deea"
LAYOUT = "parent1x_mask_child4x_joint"


def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def coverage(mask, wh):
    """Independent cell-footprint integration, not Torch adaptive pooling."""
    width, height = wh
    if mask.shape == (height, width):
        return mask.copy()
    source_h, source_w = mask.shape
    starts = np.floor(np.arange(width) * source_w / width).astype(int)
    ends = np.ceil((np.arange(width) + 1) * source_w / width).astype(int)
    out = np.zeros((height, width), bool)
    for y in range(height):
        top, bottom = (
            math.floor(y * source_h / height),
            math.ceil((y + 1) * source_h / height),
        )
        prefix = np.pad(mask[top:bottom].sum(0, dtype=np.int64).cumsum(), (1, 0))
        out[y] = prefix[ends] - prefix[starts] > 0
    return out


def binary_mask(path, wh, expected_bbox=None):
    with Image.open(path) as image:
        assert image.mode == "L" and list(image.size) == wh
        assert sum(image.histogram()[1:255]) == 0
        if expected_bbox is not None:
            assert image.getbbox() == tuple(expected_bbox)
        return np.array(image) != 0


def audit(args):
    started = time.monotonic()
    run, record = Path(args.run), Path(args.record)
    variant, prep = run / "swinir4x", run / "prepared"
    sr = read(variant / "superresolution.json")
    assert sr["state"] == "completed", "Wait for all SR inputs"
    train = read(variant / "training-inputs.json")
    pixels = read(variant / "pixels.json")
    lineage = read(prep / "pixel-lineage.json")
    native_cameras_json = read(prep / "camera-inputs.json")["records"]
    raw = read(args.native_lineage)
    job = read(run / "job.json")
    assert (
        job["implementation_sha256"] == IMPLEMENTATION_SHA
        and job["input_sha256"] == INPUT_SHA
    )
    assert job["options"]["scales"] == [4] and job["options"]["sr_region"] == "face"
    assert job["options"]["seed"] == 0 and job["options"]["iterations"] == 18000
    assert job["options"]["importance_metric"] == "outdoor"
    assert raw["passed"] and raw["video_sha256"] == VIDEO_SHA and lineage["passed"]
    baseline = Path(args.baseline)
    original_rows = read(baseline / "original/training-inputs.json")["records"]
    old = baseline.parent / "video15-face-composite4x/swinir4x"
    old_sr = read(old / "superresolution.json")
    assert old_sr["state"] == "completed" and old_sr["mask_scope"] == "face"
    names = [r["image"] for r in lineage["records"]]
    assert len(set(names)) == 65
    for values, key in (
        (raw["records"], "image"),
        (native_cameras_json, "name"),
        (original_rows, "name"),
        (train["records"], "name"),
    ):
        assert [r[key] for r in values] == names
    assert len(sr["records"]) == len(pixels["items"]) == 65
    assert {p.name for p in (variant / "dataset/images").iterdir()} == set(names)
    for data in (sr, pixels, train):
        assert (
            data["view_layout"] == LAYOUT
            and data["parent_scale"] == 1
            and data["child_scale"] == 4
        )
        assert data["source_frames"] == data["view_count"] == 65
        assert (
            not data["additional_training_views"]
            and not data["full_frame_4x_generated"]
        )
        assert data["joint_source_update"]
    assets = sr["parser_detector_assets"]
    assert assets["head_labels"] == list(range(1, 14)) and assets[
        "excluded_labels"
    ] == [0, 14, 15, 16, 17, 18]
    assert (
        assets["parser"]["weight_sha256"]
        == "18c44e06b77e47da86fe2b48fac78e0ccbd0b5241c829f9487ebf7fd4a6389e6"
    )
    assert (
        sr["weight_sha256"]
        == "99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a"
    )
    assert (
        sr["precision"] == "float32" and sr["tile"] == 256 and sr["tile_overlap"] == 32
    )
    protected = {}

    def protect(path, expected=None):
        path = host_path(path)
        key = str(path)
        digest = protected.get(key)
        if digest is None:
            digest = sha(path)
            protected[key] = digest
        assert expected is None or digest == expected, "SHA mismatch: " + key
        return digest

    old_audit = read(Path(args.old_record) / "全65帧输入核查.json")
    assert old_audit["passed"]
    for path, digest in old_audit["protected_SHA256"].items():
        protect(path, digest)
    for path in (
        run / "job.json",
        prep / "pixel-lineage.json",
        prep / "camera-inputs.json",
        variant / "superresolution.json",
        variant / "training-inputs.json",
        variant / "pixels.json",
    ):
        protect(path)
    protect(args.native_lineage)
    protect(host_path(raw["source"]), VIDEO_SHA)
    for filename in ("images.bin", "points3D.bin"):
        source = protect(prep / "dataset/sparse/0" / filename)
        assert (
            protect(variant / "dataset/sparse/0" / filename)
            == source
            == protect(baseline / "original/dataset/sparse/0" / filename)
        )
    assert protect(prep / "dataset/sparse/0/points3D.bin") == POINTS_SHA
    assert train["initial_sparse_sha256"] == POINTS_SHA
    sys.path.insert(0, args.candidate_source)
    from msv2.data import _read_cameras, _read_images, _camera, _w2c

    camera_bin = _read_cameras(variant / "dataset/sparse/0")
    images_bin = {r.name: r for r in _read_images(variant / "dataset/sparse/0")}
    assert len(images_bin) == 65 and set(images_bin) == set(names)
    protect(variant / "dataset/sparse/0/cameras.bin")
    native_bin = _read_cameras(prep / "dataset/sparse/0")
    protect(prep / "dataset/sparse/0/cameras.bin")
    assert native_bin == camera_bin, "Registered cameras must remain native"
    by_sr = {r["name"]: r for r in sr["records"]}
    by_old = {r["name"]: r for r in old_sr["records"]}
    by_pixel = {r["name"]: r for r in pixels["items"]}
    rows = []
    for i, (row, source, source_camera, raw_row, base_row) in enumerate(
        zip(
            train["records"],
            lineage["records"],
            native_cameras_json,
            raw["records"],
            original_rows,
        )
    ):
        name = names[i]
        assert (
            row["name"]
            == source["image"]
            == source_camera["name"]
            == raw_row["image"]
            == base_row["name"]
            == name
        )
        item, old_item = by_sr[name], by_old[name]
        assert row["wh"] == source["wh"] == base_row["wh"] == [1520, 2704]
        assert row["K"] == source_camera["K"] == base_row["K"]
        assert row["w2c"] == source_camera["w2c"] == base_row["w2c"]
        assert row["split"] == base_row["split"] == ("test" if i % 8 == 0 else "train")
        assert (
            row["source_frame"] == row["frame"] == source["frame"]
            and row["source_group_split"]
            and row["view_type"] == "full"
        )
        pose = images_bin[name]
        assert (
            row["colmap_image_id"] == pose.id
            and row["colmap_camera_id"] == pose.camera_id
        )
        assert row["K"] == _camera(camera_bin[pose.camera_id]) and row["w2c"] == _w2c(
            pose
        )
        native_path = prep / "dataset/images" / name
        raw_path = Path(args.native_lineage).parent / "dataset/images" / name
        parent_path = variant / "dataset/images" / name
        parent_digest = protect(parent_path, source["sha256"])
        assert (
            protect(native_path, source["sha256"])
            == parent_digest
            == item["native_sha256"]
            == item["output_sha256"]
        )
        assert (
            parent_digest
            == row["source_sha256"]
            == row["native_sha256"]
            == by_pixel[name]["image_sha256"]
        )
        protect(raw_path, raw_row["sha256"])
        with Image.open(parent_path) as image:
            assert image.mode == "RGB" and image.size == (1520, 2704)
            native = np.array(image)
        with Image.open(raw_path) as image:
            assert np.array_equal(np.array(image.convert("RGB")), native)
        assert (
            array_sha(native)
            == raw_row["rgb_sha256"]
            == source["rgb_sha256"]
            == item["composition"]["source_rgb_sha256"]
        )
        assert not item["composition"]["full_frame_4x_training_image_generated"]
        assert (
            item["composition"]["sampling_group_count"] == 1
            and item["composition"]["children_are_auxiliary"]
        )
        union = np.zeros((2704, 1520), bool)
        face_masks = {}
        assert item["face_count"] == len(item["faces"])
        for face in item["faces"]:
            path = variant / "face-masks" / face["mask_name"]
            protect(path, face["mask_sha256"])
            mask = binary_mask(path, [1520, 2704], face["bbox_native"])
            assert int(mask.sum()) == face["mask_pixels"]
            number = int(Path(face["mask_name"]).stem.rsplit("_face_", 1)[1])
            face_masks[number] = mask
            union |= mask
        protect(row["face_mask"], row["face_mask_sha256"])
        assert np.array_equal(binary_mask(row["face_mask"], [1520, 2704]), union)
        assert array_sha(union) == item["composition"]["union_native_mask_sha256"]
        assert int(union.sum()) == item["composition"]["union_native_mask_pixels"]
        old_union = np.zeros_like(union)
        for face in old_item["faces"]:
            path = old / "face-masks" / face["mask_name"]
            protect(path, face["mask_sha256"])
            old_union |= binary_mask(path, [1520, 2704], face["bbox_native"])
        assert np.array_equal(union, old_union), (
            "Fresh2D mask differs from validated input: " + name
        )
        old_target = old / "dataset/images" / name
        protect(old_target, old_item["output_sha256"])
        owned = np.zeros_like(union)
        child_checks = []
        assert len(item["children"]) == len(row["children"])
        with Image.open(old_target) as old_image:
            old_image.load()
            for saved, child in zip(item["children"], row["children"]):
                assert {k: child[k] for k in saved} == saved
                x0, y0, x1, y1 = roi = saved["roi_native"]
                assert all(type(v) is int and v % 8 == 0 for v in roi)
                assert saved["wh"] == [(x1 - x0) * 4, (y1 - y0) * 4]
                for key in ("image", "mask"):
                    protect(variant / saved[key], saved[key + "_sha256"])
                    assert Path(saved[key]).parent == Path("children")
                high_mask = binary_mask(variant / saved["mask"], saved["wh"])
                provenance = saved["provenance"]
                mask = face_masks[provenance["region_number"]]
                support = mask & ~owned
                owned |= mask
                expected = np.zeros_like(high_mask)
                valid_x1, valid_y1 = min(x1, 1520), min(y1, 2704)
                expected[: 4 * (valid_y1 - y0), : 4 * (valid_x1 - x0)] = (
                    support[y0:valid_y1, x0:valid_x1].repeat(4, 0).repeat(4, 1)
                )
                assert np.array_equal(high_mask, expected)
                assert (
                    int(high_mask.sum())
                    == saved["mask_pixels"]
                    == int(support.sum()) * 16
                )
                assert array_sha(high_mask) == provenance["mask_sha256"]
                with Image.open(variant / saved["image"]) as image:
                    assert image.mode == "RGB" and list(image.size) == saved["wh"]
                    high = np.array(image)
                assert array_sha(high) == provenance["rgb_sha256"]
                reference = np.array(
                    old_image.crop((4 * x0, 4 * y0, 4 * valid_x1, 4 * valid_y1))
                )
                if x1 > 1520 or y1 > 2704:
                    reference = np.pad(
                        reference,
                        ((0, 4 * (y1 - valid_y1)), (0, 4 * (x1 - valid_x1)), (0, 0)),
                        mode="edge",
                    )
                different = np.any(high != reference, axis=2)
                supported_mismatch = int(different[high_mask].sum())
                outside_mismatch = int(different[~high_mask].sum())
                max_delta = int(
                    np.abs(high.astype(np.int16) - reference.astype(np.int16)).max()
                )
                assert supported_mismatch == 0, (
                    "Supervised child pixels differ from validated face input: " + name
                )
                assert outside_mismatch == 0, (
                    "Child context differs from whole-reference crop: " + name
                )
                calibration = child["camera"]
                k = np.asarray(row["K"])
                full_k = np.diag([4.0, 4.0, 1.0]) @ k
                child_k = (
                    np.array(
                        [[1.0, 0.0, -4 * x0], [0.0, 1.0, -4 * y0], [0.0, 0.0, 1.0]]
                    )
                    @ full_k
                )
                assert np.array_equal(calibration["K"], child_k)
                assert np.array_equal(calibration["full_K"], full_k)
                assert calibration["w2c"] == row["w2c"]
                assert (
                    calibration["full_wh"] == [6080, 10816]
                    and calibration["wh"] == saved["wh"]
                )
                assert calibration["_msv2_roi"] == dict(
                    full_wh=[6080, 10816], viewport_xy=[4 * x0, 4 * y0], wh=saved["wh"]
                )
                assert calibration["full_tan_fovx"] == 1520 / (
                    2 * k[0, 0]
                ) and calibration["full_tan_fovy"] == 2704 / (2 * k[1, 1])
                assert all(v % 16 == 0 for v in (4 * x0, 4 * y0, *saved["wh"]))
                receipt = provenance["sr_receipt"]
                assert (
                    receipt["source_rgb_sha256"] == array_sha(native)
                    and receipt["no_source_downsampling"]
                )
                a, b, c, d = receipt["context_bbox_native"]
                assert receipt["context_rgb_sha256"] == array_sha(native[b:d, a:c])
                bx0, by0, bx1, by1 = receipt["bbox_native"]
                assert receipt["native_head_rgb_sha256"] == array_sha(
                    native[by0:by1, bx0:bx1]
                )
                child_checks.append(
                    dict(
                        image=saved["image"],
                        roi_native=roi,
                        wh=saved["wh"],
                        inside_4x_pixels=int(high_mask.sum()),
                        outside_4x_pixels=int((~high_mask).sum()),
                        validated_input_inside_mismatch=supported_mismatch,
                        validated_input_outside_mismatch=outside_mismatch,
                        validated_input_max_channel_delta=max_delta,
                        K_child_exact=True,
                        full_FoV_preserved=True,
                        viewport_tile16_aligned=True,
                        original_context_rgb_sha256=receipt["context_rgb_sha256"],
                    )
                )
        assert np.array_equal(owned, union)
        warmup = coverage(union, [760, 1352])
        assert not union.any() or warmup.any()
        rows.append(
            dict(
                frame=row["frame"],
                source_index=raw_row["source_index"],
                time_seconds=raw_row["time_seconds"],
                split=row["split"],
                parent_wh=[1520, 2704],
                parent_png_sha256=parent_digest,
                parent_RGB_unchanged=True,
                parent_inside_mask_pixels=int(union.sum()),
                parent_outside_mask_pixels=int((~union).sum()),
                warmup_parent_wh=[760, 1352],
                warmup_inside_coverage_pixels=int(warmup.sum()),
                warmup_outside_coverage_pixels=int((~warmup).sum()),
                child_count=len(child_checks),
                children=child_checks,
                K_native_exact=True,
                w2c_unchanged=True,
                source_group_sampling=True,
                independent_child_cameras=False,
            )
        )
        if (i + 1) % 10 == 0 or i == 64:
            print("PAIRED_INPUT_AUDITED", i + 1, name, flush=True)
        del native, union, old_union, owned, warmup
    changed = {
        p: dict(before=d, after=sha(p)) for p, d in protected.items() if sha(p) != d
    }
    assert not changed, changed
    assert "torch" not in sys.modules
    result = dict(
        schema="msv2-full65-parent-child-independent-input-audit-v1",
        passed=True,
        created=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic() - started,
        phase="independent CPU input provenance, calibration and pixel audit only",
        run=str(run),
        implementation_sha256=IMPLEMENTATION_SHA,
        source_video_sha256=VIDEO_SHA,
        input_sha256=INPUT_SHA,
        source_groups=65,
        registered_parent_views=65,
        intrinsic_camera_models=len(camera_bin),
        auxiliary_child_ROIs=sum(r["child_count"] for r in rows),
        children_in_independent_sampling=False,
        parent_wh=[1520, 2704],
        train_groups=sum(r["split"] == "train" for r in rows),
        test_groups=sum(r["split"] == "test" for r in rows),
        parent_pixels_byte_preserved=True,
        parent_K_native_exact=True,
        parent_w2c_exact=True,
        child_K_full_FoV_viewport_exact=True,
        mask_owned_child_pixels_exact=True,
        all_child_pixels_equal_validated_input_crops=True,
        initial_sparse_sha256=POINTS_SHA,
        initial_points=58509,
        old_learned_model_files_hashed_only=True,
        old_learned_state_reused=False,
        protected_file_count=len(protected),
        protected_SHA256=protected,
        all_protected_SHA256_unchanged=True,
        total_parent_outside_pixels=sum(r["parent_outside_mask_pixels"] for r in rows),
        total_child_inside4_pixels=sum(
            c["inside_4x_pixels"] for r in rows for c in r["children"]
        ),
        torch_imported=False,
        GPU_used=False,
        training_interrupted=False,
        training_modified=False,
        quality_accepted=False,
        limitation="Input identity does not validate segmentation or reconstruction quality. Warmup counts are independently derived from the runtime coverage rule; no claim of RGB interpolation-footprint isolation.",
        records=rows,
    )
    save(record / "全65帧子母输入核查.json", result)
    print(
        "ALL65_PAIRED_INPUT_AUDIT_PASSED", result["seconds"], len(protected), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="/opt/gs-slam-lab/msv2-video15-20261105/runs/video15-parent1-child4",
    )
    parser.add_argument(
        "--record", default="/mnt/d/00/记录/2026-11-06_MSV2_video15_子母图联合训练"
    )
    parser.add_argument(
        "--old-record",
        default="/mnt/d/00/记录/2026-11-06_MSV2_video15_面部mask整图合成",
    )
    parser.add_argument(
        "--native-lineage",
        default="/mnt/d/GS-SLAM-Data/msv2-video15-20261105/pixel-lineage.json",
    )
    parser.add_argument(
        "--baseline", default="/opt/gs-slam-lab/msv2-video15-20261105/runs/video15"
    )
    parser.add_argument("--candidate-source", default="/mnt/d/Projects/msv2-paired")
    args = parser.parse_args()
    try:
        audit(args)
    except Exception as error:
        save(
            Path(args.record) / "全65帧子母输入核查错误.json",
            dict(
                passed=False,
                error_type=type(error).__name__,
                error=str(error),
                GPU_used=False,
                training_modified=False,
                quality_accepted=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
