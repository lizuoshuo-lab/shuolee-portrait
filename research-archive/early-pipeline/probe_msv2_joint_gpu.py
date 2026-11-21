"""Bounded fresh-SfM joint GPU gate; no saved Gaussian/checkpoint or full train."""

import argparse
import copy
import gc
import hashlib
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

from probe_msv2_face_fullframe import read, save, sha, now


BASE = Path("/opt/gs-slam-lab/msv2-video15-20261105/runs/video15")
OLD = BASE.parent / "video15-face-composite4x"
POINTS_SHA = "7e049a3b338912f28e87c181fbea9efc6ae1ffbfd0a522d6be7f874719f4f787"


def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def prepare(record):
    from msv2.paired_data import child_roi, child_camera

    record = Path(record)
    if record.exists():
        raise FileExistsError("Preserve the existing GPU gate record: " + str(record))
    record.mkdir(parents=True)
    run = record / "paired"
    frame = "frame_000033"
    native_row = copy.deepcopy(
        next(
            r
            for r in read(BASE / "original/training-inputs.json")["records"]
            if r["frame"] == frame
        )
    )
    native_path = Path(native_row["image"])
    sr = read(OLD / "swinir4x/superresolution.json")
    item = next(r for r in sr["records"] if r["name"] == frame + ".png")
    assert sr["state"] == "completed" and sr["mask_scope"] == "face"
    assert sr["parser_detector_assets"]["head_labels"] == list(range(1, 14))
    assert item["face_count"] == 1 and len(item["faces"]) == 1
    mask_path = OLD / "swinir4x/face-masks" / item["faces"][0]["mask_name"]
    target_path = OLD / "swinir4x/dataset/images" / item["name"]
    protected = {}

    def protect(path, expected=None):
        path = Path(path)
        digest = sha(path)
        assert expected is None or digest == expected
        protected[str(path)] = digest
        return digest

    protect(BASE / "job.json")
    protect(BASE / "fast.config")
    protect(BASE / "original/training-inputs.json")
    protect(BASE / "original/dataset/sparse/0/points3D.bin", POINTS_SHA)
    protect(native_path, native_row["source_sha256"])
    protect(OLD / "job.json")
    protect(OLD / "swinir4x/superresolution.json")
    protect(mask_path, item["faces"][0]["mask_sha256"])
    protect(target_path, item["output_sha256"])
    for name in ("original", "interpolation2x", "swinir4x"):
        protect(BASE / name / "model/point_cloud/iteration_18000/point_cloud.ply")
    protect(
        BASE.parent
        / "video15-head4x/swinir4x/model/point_cloud/iteration_18000/point_cloud.ply"
    )
    with Image.open(native_path) as image:
        native = np.array(image)
    assert array_sha(native) == item["composition"]["source_rgb_sha256"]
    with Image.open(mask_path) as image:
        assert image.mode == "L" and image.size == tuple(native_row["wh"])
        mask = np.array(image) != 0
        assert image.getbbox() == tuple(item["faces"][0]["bbox_native"])
    assert array_sha(mask) == item["composition"]["union_mask_native_sha256"]
    roi = child_roi(item["faces"][0]["bbox_native"], native_row["wh"])
    x0, y0, x1, y1 = roi
    assert x1 <= native.shape[1] and y1 <= native.shape[0], (
        "This one-frame gate expects an internal ROI"
    )
    child_calibration = child_camera(
        native_row["K"], native_row["w2c"], native_row["wh"], roi
    )
    with Image.open(target_path) as image:
        assert image.size == (native.shape[1] * 4, native.shape[0] * 4)
        high = np.array(image.crop(tuple(v * 4 for v in roi)))
    high_mask = mask[y0:y1, x0:x1].repeat(4, 0).repeat(4, 1)
    assert int(high_mask.sum()) == int(mask.sum()) * 16
    parent = run / "dataset/images" / (frame + ".png")
    parent.parent.mkdir(parents=True)
    shutil.copy2(native_path, parent)
    assert sha(parent) == sha(native_path)
    for level in (1, 2):
        cache = run / f"rgb-cache-{level}"
        cache.mkdir()
        with Image.open(native_path) as image:
            wh = tuple(round(v / level) for v in image.size)
            np.save(cache / (frame + ".npy"), np.array(image.resize(wh)))
    (run / "children").mkdir()
    Image.fromarray(high).save(run / "children/child.png")
    Image.fromarray(high_mask.astype(np.uint8) * 255).save(
        run / "children/child-mask.png"
    )
    mask_output = run / "face-mask.png"
    shutil.copy2(mask_path, mask_output)
    native_row.update(
        image=str(parent),
        native_image=str(native_path),
        face_mask=str(mask_output),
        face_mask_sha256=sha(mask_output),
        children=[
            dict(
                image="children/child.png",
                mask="children/child-mask.png",
                image_sha256=sha(run / "children/child.png"),
                mask_sha256=sha(run / "children/child-mask.png"),
                roi_native=roi,
                wh=[high.shape[1], high.shape[0]],
                camera=child_calibration,
            )
        ],
    )
    save(
        record / "input.json",
        dict(
            frame=frame,
            diagnostic_source_split=native_row["split"],
            source_input_only_reuse=True,
            old_learned_state_reused=False,
            old_model_files_hashed_only=True,
            source_rgb_sha256=array_sha(native),
            face_native_pixels=int(mask.sum()),
            row=native_row,
            pure_input_sources=dict(
                native=str(native_path),
                old_full4_target=str(target_path),
                face_mask=str(mask_path),
            ),
            protected_SHA256=protected,
        ),
    )
    return native_row, run, protected


def parameter_state(g, gradients=False):
    result = {}
    for name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
    ):
        tensor = getattr(g, name)
        assert bool(tensor.isfinite().all())
        value = tensor.detach().cpu().numpy()
        info = dict(shape=list(tensor.shape), sha256=array_sha(value), finite=True)
        if gradients:
            grad = tensor.grad
            if tensor.numel():
                assert grad is not None and bool(grad.isfinite().all()), name
            info["gradient"] = dict(
                present=grad is not None,
                finite=grad is None or bool(grad.isfinite().all()),
                norm=0.0 if grad is None else float(grad.norm()),
                nonzero=0 if grad is None else int(grad.count_nonzero()),
            )
        result[name] = info
    return result


def execute(record):
    import torch
    from msv2.engine import Engine, LazyCamera, verify_source
    from msv2.paired import JointSupervision
    from msv2.paired_data import child_camera

    row, run, protected = prepare(record)
    record = Path(record)
    e = Engine(BASE / "job.json")
    verify_source(e.repo)
    sys.path.insert(0, str(BASE / "source"))
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import read_points3D_binary
    from utils.graphics_utils import BasicPointCloud
    from gaussian_renderer import render_imp
    from fused_ssim import fused_ssim

    args, data, opt, pipe = e.options(1)
    assert (
        args.iterations == 18000
        and args.imp_metric == "outdoor"
        and opt.lambda_dssim == 0.2
    )
    save(record / "resolved-native-options.json", vars(args))
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    xyz, colors, _ = read_points3D_binary(
        str(BASE / "original/dataset/sparse/0/points3D.bin")
    )
    assert len(xyz) == 58509 and np.isfinite(xyz).all()
    all_rows = read(BASE / "original/training-inputs.json")["records"]
    centers = np.asarray(
        [np.linalg.inv(r["w2c"])[:3, 3] for r in all_rows if r["split"] == "train"]
    )
    extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    results = []
    started = time.monotonic()
    receipt = dict(
        schema="msv2-fresh-SfM-joint-GPU-gate-v1",
        started=now(),
        passed=False,
        scope="one held-out frame33 bounded implementation diagnostic, not model training or quality acceptance",
        initial_sparse_sha256=POINTS_SHA,
        initial_points=58509,
        old_learned_state_reused=False,
        source_input_only_reuse=True,
        quality_accepted=False,
        full_training_executed=False,
        levels=[],
    )
    try:
        for level in (2, 1):
            torch.manual_seed(0)
            np.random.seed(0)
            torch.cuda.reset_peak_memory_stats()
            g = GaussianModel(0)
            g.create_from_pcd(
                BasicPointCloud(xyz, colors / 255.0, np.zeros_like(xyz)), extent
            )
            g.training_setup(opt)
            g.init_culling(len(centers))
            g.update_learning_rate(1)
            before = parameter_state(g)
            c = LazyCamera(row, 0, level)
            joint = JointSupervision(render_imp, e.repo, run)
            output = joint.render(
                c, g, pipe, torch.zeros(3, device="cuda"), culling=g._culling[:, 0]
            )
            output["render"].retain_grad()
            for view in output["paired_children"]:
                view["prediction"].retain_grad()
            for saved in joint.cached[2]:
                assert (
                    saved["camera"]._msv2_shared_viewspace_points
                    is output["viewspace_points"]
                )
            target = c.original_image
            result = joint.loss(
                c, output, target, 0.2, lambda a, b: (a - b).abs().mean(), fused_ssim
            )
            assert bool(torch.isfinite(result["loss"]))
            result["loss"].backward()
            torch.cuda.synchronize()
            stats = result["stats"]
            pg = output["render"].grad
            assert pg is not None and bool(pg.isfinite().all())
            face = output["face_mask"]
            wrong_parent = float(pg[:, face].abs().max()) if bool(face.any()) else 0.0
            assert wrong_parent == 0.0, "Parent gradients entered face mask"
            child_checks = []
            for view in output["paired_children"]:
                grad = view["prediction"].grad
                assert grad is not None and bool(grad.isfinite().all())
                outside = ~view["mask"]
                wrong_child = (
                    float(grad[:, outside].abs().max()) if bool(outside.any()) else 0.0
                )
                assert wrong_child == 0.0, "Child gradients entered outside-mask pixels"
                child_checks.append(
                    dict(
                        wh=list(view["prediction"].shape[1:])[::-1],
                        inside_pixels=int(view["mask"].sum()),
                        outside_gradient_max=wrong_child,
                        inside_gradient_norm=float(grad[:, view["mask"]].norm()),
                        safe_ssim_centers=view["support_cache"].center_pixels,
                    )
                )
            dummy = output["viewspace_points"]
            assert dummy.grad is not None and bool(dummy.grad.isfinite().all())
            backward_state = parameter_state(g, gradients=True)
            assert all(
                before[k]["sha256"] == backward_state[k]["sha256"] for k in before
            )
            assert backward_state["_xyz"]["gradient"]["nonzero"] > 0
            sampled = child_camera(
                row["K"],
                row["w2c"],
                row["wh"],
                row["children"][0]["roi_native"],
                warmup_factor=level,
            )
            assert sampled["_msv2_roi"] == joint.cached[2][0]["camera"]._msv2_roi
            item = dict(
                level=level,
                passed=True,
                parent_wh=[c.width, c.height],
                child_calibration=sampled,
                loss=float(result["loss"]),
                l1=float(result["l1"]),
                ssim=float(result["ssim"]),
                loss_stats=stats,
                parent_face_gradient_max=wrong_parent,
                parent_outside_gradient_norm=float(pg[:, ~face].norm()),
                children=child_checks,
                shared_dummy_identity=True,
                shared_dummy_units="complete-camera NDC; no extra factor4",
                dummy_gradient_norm_xy=float(dummy.grad[:, :2].norm()),
                dummy_gradient_finite=True,
                backward_calls=1,
                optimizer_step_calls=0,
                zero_update_parameters_unchanged=True,
                parameter_before=before,
                parameter_after_backward=backward_state,
                optimizer_class=type(g.optimizer).__name__,
                points_before=len(g.get_xyz),
            )
            if level == 1:
                g.optimizer.step(output["radii"] > 0, len(output["radii"]))
                torch.cuda.synchronize()
                after_step = parameter_state(g)
                assert len(g.get_xyz) == 58509
                changed = [
                    k for k in before if before[k]["sha256"] != after_step[k]["sha256"]
                ]
                assert changed, "One real native optimizer step changed no parameters"
                item.update(
                    optimizer_step_calls=1,
                    parameter_after_step=after_step,
                    changed_parameters=changed,
                    points_after=len(g.get_xyz),
                    one_step_all_finite=True,
                )
                g.optimizer.zero_grad(set_to_none=True)
            item["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            item["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            results.append(item)
            receipt["levels"] = results
            save(record / "receipt.json", receipt)
            print(
                "JOINT_LEVEL_PASSED",
                level,
                item["loss"],
                item["optimizer_step_calls"],
                flush=True,
            )
            del output, result, joint, c, g, target, pg, face, dummy, grad, view, saved
            LazyCamera.cached_tensor = None
            LazyCamera.cached_id = None
            LazyCamera.last = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        changes = {
            p: dict(before=d, after=sha(p)) for p, d in protected.items() if sha(p) != d
        }
        assert not changes, changes
        receipt.update(
            passed=True,
            finished=now(),
            seconds=time.monotonic() - started,
            protected_files_unchanged=True,
            protected_SHA256=protected,
            backward_calls_total=2,
            optimizer_steps_total=1,
            saved_Gaussian_models=0,
            GPU_allocated_after_release_bytes=torch.cuda.memory_allocated(),
            gate_state_discarded_not_reused_for_training=True,
        )
        print("FRESH_SFM_JOINT_GPU_GATE_PASSED", flush=True)
    except Exception as error:
        receipt.update(
            error_type=type(error).__name__, error=str(error), finished=now()
        )
        raise
    finally:
        save(record / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument(
        "--record",
        default="/mnt/d/00/记录/2026-11-06_MSV2_video15_子母图联合训练/GPU_joint_probe",
    )
    args = parser.parse_args()
    execute(args.record)


if __name__ == "__main__":
    main()
