"""Read-only CPU pixel budget and camera audit for native-parent/ROI4 inputs.

Existing validated full4 targets are not loaded or copied. This only reads
their input receipts plus native parents/masks and writes budget metadata.
It does not generate new targets, invoke SR, render, fit, or train a model.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    sys.path.insert(0, str(args.package))
    from msv2.paired_data import child_roi, child_camera
    import msv2.paired_data as paired_module

    source = Path(args.run).resolve()
    record = Path(args.record).resolve()
    assert not record.is_relative_to(source) and not source.is_relative_to(record)
    sr_path = source / "swinir4x/superresolution.json"
    parent_path = source / "original/training-inputs.json"
    old_training_path = source / "swinir4x/training-inputs.json"
    sr = read(sr_path)
    parents = read(parent_path)["records"]
    old_training = read(old_training_path)["records"]
    assert sr["state"] == "completed" and sr["view_layout"] == "full4x_mask_composite"
    assert sr["mask_scope"] == "face"
    by_name = {r["name"]: r for r in sr["records"]}
    previous = {r["name"]: r for r in old_training}
    assert len(by_name) == len(sr["records"]) == len(parents) == len(previous)
    assert set(by_name) == set(previous) == {r["name"] for r in parents}
    protected = {}

    def protect(path, expected=None):
        path = Path(path)
        digest = sha(path)
        assert expected is None or digest == expected, "Source SHA mismatch: " + str(
            path
        )
        assert str(path) not in protected or protected[str(path)] == digest
        protected[str(path)] = digest
        return digest

    for path in (
        sr_path,
        parent_path,
        old_training_path,
        source / "job.json",
        source / "prepared/camera-inputs.json",
        Path(paired_module.__file__),
        args.package / "msv2/head.py",
    ):
        protect(path)
    rows = []
    totals = {
        str(level): dict(
            parent_render_pixels=0,
            child_render_pixels=0,
            full4_render_pixels=0,
            parent_outside_mask_pixels=0,
            child_inside_mask_pixels=0,
            auxiliary_children=0,
        )
        for level in (1, 2)
    }
    for parent in parents:
        name = parent["name"]
        item = by_name[name]
        old = previous[name]
        w, h = parent["wh"]
        expected = parent["native_sha256"]
        assert parent["source_sha256"] == item["native_sha256"] == expected
        assert old["native_sha256"] == expected and old["split"] == parent["split"]
        assert old["wh"] == [w * 4, h * 4] and old["w2c"] == parent["w2c"]
        np.testing.assert_allclose(
            old["K"],
            np.diag([4.0, 4.0, 1.0]) @ np.asarray(parent["K"]),
            rtol=0,
            atol=1e-10,
        )
        native = Path(parent["native_image"])
        protect(native, expected)
        protect(parent["image"], expected)
        with Image.open(native) as image:
            assert (
                image.format == "PNG"
                and image.mode == "RGB"
                and list(image.size) == [w, h]
            )
            rgb_sha = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        assert item["face_extraction"]["source_rgb_sha256"] == rgb_sha
        assert item["composition"]["source_rgb_sha256"] == rgb_sha
        owned = np.zeros((h, w), bool)
        children = []
        assert item["face_count"] == len(item["faces"])
        for face_no, face in enumerate(item["faces"], 1):
            mask_name = face["mask_name"]
            assert Path(mask_name).name == mask_name
            mask_path = source / "swinir4x/face-masks" / mask_name
            protect(mask_path, face["mask_sha256"])
            with Image.open(mask_path) as image:
                assert image.mode == "L" and list(image.size) == [w, h]
                assert image.getbbox() == tuple(face["bbox_native"])
                assert sum(image.histogram()[1:255]) == 0
                mask = np.asarray(image) != 0
            assert int(mask.sum()) == face["mask_pixels"]
            support = mask & ~owned
            count = int(support.sum())
            owned |= mask
            if not count:
                continue
            roi = child_roi(face["bbox_native"], [w, h])
            x0, y0, x1, y1 = roi
            cameras = {}
            for level in (1, 2):
                parent_wh = [round(w / level), round(h / level)]
                full_wh = [w * 4 // level, h * 4 // level]
                camera = child_camera(
                    parent["K"],
                    parent["w2c"],
                    [w, h],
                    roi,
                    warmup_factor=level,
                    current_parent_wh=parent_wh,
                    current_full_wh=full_wh,
                )
                assert camera["w2c"] == parent["w2c"]
                assert camera["full_wh"] == full_wh and camera["warmup_tile16_aligned"]
                assert camera["wh"] == [(x1 - x0) * 4 // level, (y1 - y0) * 4 // level]
                cameras[str(level)] = camera
                t = totals[str(level)]
                t["child_render_pixels"] += camera["wh"][0] * camera["wh"][1]
                t["child_inside_mask_pixels"] += count * (4 // level) ** 2
                t["auxiliary_children"] += 1
            children.append(
                dict(
                    face_number=face_no,
                    bbox_native=face["bbox_native"],
                    roi_native=roi,
                    mask_sha256=face["mask_sha256"],
                    native_mask_pixels=int(mask.sum()),
                    owned_native_pixels=count,
                    padding_native=[0, 0, max(0, x1 - w), max(0, y1 - h)],
                    padding_loss_mask_false=True,
                    cameras=cameras,
                )
            )
        assert (
            hashlib.sha256(owned.tobytes()).hexdigest()
            == item["composition"]["union_mask_native_sha256"]
        )
        assert int(owned.sum()) == item["composition"]["union_mask_native_pixels"]
        levels = {}
        for level in (1, 2):
            pwh = [round(w / level), round(h / level)]
            current_union = np.asarray(
                Image.fromarray(owned).resize(tuple(pwh), Image.Resampling.NEAREST)
            )
            mother = pwh[0] * pwh[1]
            outside = mother - int(current_union.sum())
            full4 = (w * 4 // level) * (h * 4 // level)
            child_render = sum(
                c["cameras"][str(level)]["wh"][0] * c["cameras"][str(level)]["wh"][1]
                for c in children
            )
            inside = sum(c["owned_native_pixels"] * (4 // level) ** 2 for c in children)
            t = totals[str(level)]
            t["parent_render_pixels"] += mother
            t["full4_render_pixels"] += full4
            t["parent_outside_mask_pixels"] += outside
            levels[str(level)] = dict(
                parent_current_wh=pwh,
                parent_render_pixels=mother,
                child_render_pixels=child_render,
                new_total_render_pixels=mother + child_render,
                old_full4_render_pixels=full4,
                render_fraction_of_full4=(mother + child_render) / full4,
                parent_outside_mask_pixels=outside,
                child_inside_mask_pixels=inside,
                supervision_pixels_before_ssim_safe_windows=outside + inside,
                SSIM_safe_center_count_not_computed=True,
            )
        rows.append(
            dict(
                frame=parent["frame"],
                name=name,
                split=parent["split"],
                native_wh=[w, h],
                native_png_sha256=expected,
                source_rgb_sha256=rgb_sha,
                parent_K=parent["K"],
                w2c=parent["w2c"],
                face_count=item["face_count"],
                auxiliary_child_count=len(children),
                source_sampling_groups=1,
                children=children,
                levels=levels,
            )
        )
    for t in totals.values():
        t["new_total_render_pixels"] = (
            t["parent_render_pixels"] + t["child_render_pixels"]
        )
        t["render_fraction_of_full4"] = (
            t["new_total_render_pixels"] / t["full4_render_pixels"]
        )
        t["render_pixel_reduction_fraction"] = 1 - t["render_fraction_of_full4"]
        t["supervision_pixels_before_ssim_safe_windows"] = (
            t["parent_outside_mask_pixels"] + t["child_inside_mask_pixels"]
        )
    assert all(sha(path) == digest for path, digest in protected.items())
    assert "torch" not in sys.modules
    result = dict(
        schema="msv2-paired-input-pixel-budget-v1",
        created=datetime.now(timezone.utc).isoformat(),
        state="CPU_metadata_and_mask_audit_passed",
        source_run=str(source),
        source_frames=len(rows),
        source_sampling_groups=len(rows),
        train_source_frames=sum(r["split"] == "train" for r in rows),
        test_source_frames=sum(r["split"] == "test" for r in rows),
        outside_processing_wh_native=[
            list(wh) for wh in sorted({tuple(r["native_wh"]) for r in rows})
        ],
        outside_processed_at_native1=True,
        outside_resized_to_full4=False,
        old_full4_targets_read_or_reused=False,
        new_training_targets_written=False,
        new_cameras_written_to_training_run=False,
        old_Gaussians_loaded=False,
        GPU_used=False,
        torch_imported=False,
        SR_called=False,
        renderer_called=False,
        optimizer_updates=0,
        model_updates=0,
        parent_png_bytes_preserved_by_plan=True,
        child_targets_are_auxiliary=True,
        source_group_split=True,
        totals_by_level=totals,
        records=rows,
        protected_SHA256=protected,
        protected_files_unchanged=True,
        quality_accepted=False,
        limitation="This audits existing native masks and computes proposed ROI pixels/cameras only. It does not prove segmentation quality, native ROI rendering/gradients, SSIM-safe center counts, runtime speed/memory or new training completion.",
    )
    record.mkdir(parents=True, exist_ok=True)
    output = record / "paired-pixel-budget.json"
    assert not output.exists(), (
        "Preserve existing budget evidence; use a separate record"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "PAIRED_PIXEL_BUDGET_CPU_PASSED",
        json.dumps(
            dict(
                source_frames=len(rows),
                outside_wh=result["outside_processing_wh_native"],
                totals=totals,
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "record", "package"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
