"""Zero-update native full4-crop versus direct-head4 rendering diagnostic.

--check-only and --help never import torch or initialize CUDA. GPU execution
is explicitly left to the caller after the active baseline releases the GPU.
This compares the native off-axis footprint approximation, not training quality.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys

COMMIT = "5ec9202c2db7d900728bfc09733e85467cff885e"
ARCHIVE_SHA = "7ea4ae14faf45d18b62dbd85cdd64129e3739868f4ddc2a9e74c080d7b1cf3b4"
VARIANTS = {1: "original", 4: "swinir4x"}


def host_path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt":
        if value.startswith("/mnt/") and len(value) > 6:
            return Path(value[5].upper() + ":" + value[6:])
        if value.startswith("/opt/"):
            distro = os.environ.get("MSV2_WSL_DISTRO", "Ubuntu-20.04")
            return Path("//wsl.localhost/" + distro + value)
    elif len(value) > 1 and value[1] == ":":
        return Path("/mnt/" + value[0].lower() + value[2:])
    return Path(value)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_rows(rows, key, label):
    result = {}
    for row in rows:
        name = row[key]
        require(name not in result, "Duplicate " + label + ": " + name)
        result[name] = row
    return result


def frame_name(value):
    return f"frame_{int(value):06d}" if value.isdigit() else value


def document_path(document_file, value):
    path = host_path(value)
    return (path if path.is_absolute() else document_file.parent / path).resolve()


def crop_projection_error(full_K, crop_K, origin):
    """Check only the calibrated pixel-center mapping, without scene geometry."""
    maximum = 0.0
    for ray in ((0.0, 0.0, 1.0), (-0.2, 0.15, 1.0), (0.3, -0.1, 2.0)):
        projected = []
        for K in (full_K, crop_K):
            q = [sum(row[j] * ray[j] for j in range(3)) for row in K]
            projected.append([q[0] / q[2], q[1] / q[2]])
        for axis in range(2):
            maximum = max(
                maximum, abs(projected[0][axis] - 4 * origin[axis] - projected[1][axis])
            )
    require(maximum < 1e-9, "Crop intrinsics changed projected pixel centers")
    return maximum


def pixel_mask_bbox(item, document_file, native, pixel, protected):
    """Bind a 2D mask to the baseline's original pixels; no model imports."""
    from PIL import Image

    frame = native["frame"]
    require(
        item.get("source_wh") == native["wh"] == pixel["wh"],
        "Head-mask source dimensions differ from baseline: " + frame,
    )
    require(
        item.get("source_png_sha256")
        == pixel.get("source_file_sha256", pixel["sha256"]),
        "Head-mask source PNG differs from baseline source lineage: " + frame,
    )
    require(
        item.get("source_rgb_sha256") == pixel["rgb_sha256"],
        "Head-mask source RGB differs from baseline source lineage: " + frame,
    )
    source = document_path(document_file, item["source_image"])
    require(
        sha(source) == item["source_png_sha256"],
        "Head-mask source PNG changed: " + frame,
    )
    for path in dict.fromkeys([source, host_path(native["native_image"])]):
        with Image.open(path) as image:
            require(list(image.size) == native["wh"], "Source PNG dimensions changed")
            rgb_sha = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        require(
            rgb_sha == item["source_rgb_sha256"],
            "Decoded source RGB no longer matches head mask: " + frame,
        )
        protected.append(path)
    parts = item.get("parts")
    if parts is not None:
        require(
            isinstance(parts, list) and len(parts) == 1 and isinstance(parts[0], dict),
            "This mask diagnostic requires one head region per selected frame: "
            + frame,
        )
        bbox = parts[0]["bbox_native"]
    else:
        require(item.get("head_count", 1) == 1, "Ambiguous multi-head mask bbox")
        bbox = item.get("mask_bbox")
    require(
        item.get("mask_bbox", bbox) == bbox,
        "Head region and mask bbox disagree: " + frame,
    )
    mask_file = document_path(document_file, item["mask_file"])
    require(sha(mask_file) == item["mask_sha256"], "Head-mask PNG changed: " + frame)
    with Image.open(mask_file) as mask:
        require(
            mask.format == "PNG"
            and mask.mode == "L"
            and list(mask.size) == native["wh"],
            "Expected a full native-resolution grayscale mask PNG: " + frame,
        )
        histogram = mask.histogram()
        require(
            sum(histogram[1:255]) == 0 and histogram[255] > 0,
            "Head-mask PNG must contain nonempty binary0/255 pixels: " + frame,
        )
        actual_bbox = mask.getbbox()
        require(
            actual_bbox is not None and list(actual_bbox) == bbox,
            "Head-mask PNG getbbox does not match the first head region: " + frame,
        )
        if parts and parts[0].get("native_mask_array_sha256"):
            array_sha = hashlib.sha256(
                bytes(1 if x else 0 for x in mask.tobytes())
            ).hexdigest()
            require(
                array_sha == parts[0]["native_mask_array_sha256"],
                "Head-mask native array differs from its region receipt: " + frame,
            )
    protected.append(mask_file)
    return bbox, dict(
        source="2d_head_pixel_extraction_mask",
        source_png_sha256=item["source_png_sha256"],
        source_rgb_sha256=item["source_rgb_sha256"],
        mask_file=str(mask_file),
        mask_sha256=item["mask_sha256"],
        mask_native_wh=native["wh"],
        getbbox_verified=True,
    )


def validate(args):
    """Read only CPU metadata and bytes; import neither torch nor MSV2 modules."""
    job_file = host_path(args.baseline_job).resolve()
    if job_file.is_dir():
        job_file /= "job.json"
    job = read(job_file)
    require(job.get("schema") == "msv2-job-v1", "Unsupported baseline job schema")
    run = host_path(job["run_dir"]).resolve()
    require(job_file.parent == run, "Baseline job path/run_dir do not agree")
    variant = run / VARIANTS[args.scale]
    execution_file = variant / "execution.json"
    require(
        execution_file.is_file(), "Baseline has no execution receipt: " + str(variant)
    )
    execution = read(execution_file)
    require(
        execution.get("state") == "training_completed",
        f"Baseline {VARIANTS[args.scale]} is not completed: state={execution.get('state')!r}. "
        "Finish that variant first, or choose a completed --scale 1/4.",
    )
    iterations = job["options"]["iterations"]
    require(
        execution.get("iterations") == iterations, "Baseline iteration receipt mismatch"
    )
    model_file = (
        variant / "model/point_cloud" / f"iteration_{iterations}/point_cloud.ply"
    )
    require(model_file.is_file(), "Completed baseline PLY is missing")
    require(
        sha(model_file) == execution["ply_sha256"],
        "Completed baseline PLY SHA256 mismatch",
    )
    parity_file = variant / "ply-parity.json"
    require(
        read(parity_file).get("passed") is True,
        "Baseline native PLY readback did not pass",
    )
    # The original job freezes its own pipeline package, independent of Git updates.
    package = run / "pipeline-source/msv2"
    digest = hashlib.sha256()
    own_sources = sorted(package.glob("*.py"))
    require(bool(own_sources), "Baseline frozen pipeline package is missing")
    for path in own_sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    require(
        digest.hexdigest() == job["implementation_sha256"],
        "Frozen pipeline fingerprint mismatch",
    )
    plan = read(run / "plan.json")
    require(plan.get("commit") == COMMIT, "Baseline upstream commit mismatch")
    protected = [job_file, execution_file, model_file, parity_file, *own_sources]
    for check in plan["source_checks"]:
        relative = PurePosixPath(check["path"].replace("\\", "/"))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Invalid frozen source path",
        )
        path = run / "source" / Path(*relative.parts)
        require(
            sha(path) == check["sha256"],
            "Frozen upstream source changed: " + str(relative),
        )
        protected.append(path)
    repo = host_path(job["source_dir"])
    receipt_file = repo / ".msv2-source.json"
    receipt = read(receipt_file)
    require(
        receipt.get("commit") == COMMIT
        and receipt.get("archive_sha256") == ARCHIVE_SHA,
        "Baseline CUDA source receipt is not the pinned archive",
    )
    protected.append(receipt_file)
    kernels = {}
    for filename in ("forward.cu", "auxiliary.h"):
        relative = "submodules/diff-gaussian-rasterization/cuda_rasterizer/" + filename
        path = repo / relative
        require(
            sha(path) == receipt["files"][relative],
            "Native rasterizer source changed: " + filename,
        )
        protected.append(path)
        kernels[filename] = dict(path=str(path), sha256=sha(path))
    one_file = run / "original/training-inputs.json"
    four_file = run / "swinir4x/training-inputs.json"
    require(
        one_file.is_file() and four_file.is_file(),
        "Actual baseline1/full4 camera inputs are not ready",
    )
    one = unique_rows(read(one_file)["records"], "frame", "native1 frame")
    four = unique_rows(read(four_file)["records"], "frame", "full4 frame")
    require(set(one) == set(four), "Baseline1/full4 frame identities differ")
    boxes_file = host_path(args.boxes).resolve()
    document = read(boxes_file)
    pixel_mask_document = (
        document.get("schema") == "msv2-head-pixel-extraction-probe-v1"
    )
    detected_rows = {}
    if not pixel_mask_document:
        detector_file = boxes_file.parent.parent / "face-locations.json"
        require(
            detector_file.is_file()
            and sha(detector_file) == document.get("source_face_locations_sha256"),
            "BBox detector/source provenance is unavailable or changed; retain the original head_bbox_probe/probe.json layout",
        )
        detector = read(detector_file)
        require(
            detector.get("source_video_sha256") == document.get("source_video_sha256"),
            "BBox and current detector source videos differ",
        )
        detected_rows = unique_rows(
            detector["records"], "frame", "detector source frame"
        )
        protected.append(detector_file)
    lineage_file = run / "prepared/pixel-lineage.json"
    lineage = read(lineage_file)
    require(
        lineage.get("passed") is True, "Baseline source pixel lineage is unverified"
    )
    pixels = unique_rows(lineage["records"], "frame", "baseline source pixels")
    box_rows = document.get(
        "all_frame_computed_boxes", document.get("records", document.get("frames"))
    )
    require(
        isinstance(box_rows, list) and all(isinstance(r, dict) for r in box_rows),
        "Expected per-frame bbox records",
    )
    boxes = unique_rows(box_rows, "frame", "head bbox frame")
    frames = [frame_name(value) for value in args.frames]
    require(len(frames) == len(set(frames)), "Requested frames must be unique")
    selected = []
    for frame in frames:
        require(
            frame in one and frame in four and frame in boxes,
            "Requested frame is unavailable: " + frame,
        )
        native, full, item = one[frame], four[frame], boxes[frame]
        require(frame in pixels, "Baseline source pixel lineage is missing: " + frame)
        if not pixel_mask_document:
            require(
                frame in detected_rows
                and detected_rows[frame]["source_rgb_sha256"]
                == pixels[frame]["rgb_sha256"],
                "BBox source decoded RGB differs from the completed baseline input: "
                + frame,
            )
        native_image = host_path(native["native_image"])
        require(
            sha(native_image) == native["native_sha256"] == pixels[frame]["sha256"],
            "Baseline original RGB PNG changed: " + frame,
        )
        protected.append(native_image)
        require(
            full["wh"] == [4 * x for x in native["wh"]],
            "Baseline4 dimensions are not truly4x",
        )
        for i in range(3):
            for j in range(3):
                expected = native["K"][i][j] * (4 if i < 2 else 1)
                require(
                    math.isfinite(full["K"][i][j])
                    and abs(full["K"][i][j] - expected) < 1e-9,
                    "Baseline full4 K does not match native4x",
                )
        require(native["w2c"] == full["w2c"], "Baseline1/full4 poses differ")
        provenance = dict(source="legacy_head_bbox_probe")
        if pixel_mask_document:
            bbox, provenance = pixel_mask_bbox(
                item, boxes_file, native, pixels[frame], protected
            )
        else:
            bbox = item.get("head_box_xyxy", item.get("bbox_native"))
        require(
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(type(x) is int for x in bbox),
            "Head bbox must contain four integer exclusive edges",
        )
        x0, y0, x1, y1 = bbox
        require(
            0 <= x0 < x1 <= native["wh"][0] and 0 <= y0 < y1 <= native["wh"][1],
            "Head bbox lies outside native image",
        )
        K = [list(row) for row in full["K"]]
        K[0][2] -= 4 * x0
        K[1][2] -= 4 * y0
        projection_error = crop_projection_error(full["K"], K, (x0, y0))
        selected.append(
            dict(
                frame=frame,
                bbox_native=bbox,
                bbox_provenance=provenance,
                intrinsics_crop_pixel_mapping_max_error_px=projection_error,
                full4=full,
                head4=dict(
                    frame=frame + "_head_probe",
                    wh=[4 * (x1 - x0), 4 * (y1 - y0)],
                    K=K,
                    w2c=full["w2c"],
                ),
            )
        )
    protected.extend(
        [
            one_file,
            four_file,
            boxes_file,
            lineage_file,
            Path(__file__).resolve(),
        ]
    )
    immutable = {str(path): sha(path) for path in dict.fromkeys(protected)}
    record = host_path(args.record).resolve()
    require(
        not record.is_relative_to(run),
        "Diagnostic output must be outside the baseline run",
    )
    return dict(
        job=job,
        job_file=job_file,
        run=run,
        execution=execution,
        model_file=model_file,
        model_sha256=execution["ply_sha256"],
        selected=selected,
        immutable=immutable,
        kernels=kernels,
        bbox_schema=document.get("schema"),
        pixel_mask_document=pixel_mask_document,
        output=record / f"native-head-render-scale{args.scale}",
    )


def summary(value, args):
    return dict(
        schema="msv2-native-head-render-probe-v1",
        baseline_job=str(value["job_file"]),
        model_scale=args.scale,
        model_PLY=str(value["model_file"]),
        model_sha256=value["model_sha256"],
        model_points=value["execution"]["points"],
        model_iterations=value["execution"]["iterations"],
        frames=[r["frame"] for r in value["selected"]],
        selected_baseline_completed=True,
        frozen_source_checks_verified=True,
        bbox_and_baseline_source_RGB_binding_verified=True,
        intrinsics_crop_pixel_mapping_max_error_px=max(
            row["intrinsics_crop_pixel_mapping_max_error_px"]
            for row in value["selected"]
        ),
        bbox_schema=value["bbox_schema"],
        native_head_pixel_mask_SHA_and_getbbox_verified=value["pixel_mask_document"],
        bbox_provenance=[
            dict(
                frame=row["frame"],
                bbox_native=row["bbox_native"],
                **row["bbox_provenance"],
            )
            for row in value["selected"]
        ],
        protected_sha256_before=value["immutable"],
        native_kernel_sources=value["kernels"],
        method="One completed frozen model; native full4 rendering then exact ROI crop versus direct cropK4/WH camera rendering, same w2c/fx/fy. No RGB target or image cache is attached to either geometry camera.",
        approximation="Native computeCov2D clamps x/z,y/z using each camera tanFoV. Direct off-axis head footprint can differ from full4-crop. RGB equality is not assumed or required for diagnostic completion.",
        metrics_domain="Native rendered normalized RGB clamped to[0,1], black background, SH degree3, scaling_modifier1. PNGs use nearest uint8 rounding; differences are not pose/geometry accuracy.",
        optimizer_updates=0,
        backward_calls=0,
        training_setup_called=False,
        model_initializes_new_experiment=False,
        diagnostic_implementation_sha256=sha(Path(__file__).resolve()),
        diagnostic_only=True,
        quality_accepted=False,
    )


def gpu_probe(value, args):
    # This function is never entered by --check-only or --help.
    require(
        os.name != "nt", "Run GPU diagnostics in the baseline Linux CUDA environment"
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    sys.path.insert(0, str(value["run"] / "source"))
    import numpy as np
    from PIL import Image
    import torch
    from types import SimpleNamespace
    from utils.graphics_utils import getWorld2View2
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    import diff_gaussian_rasterization

    require(
        torch.cuda.is_available(), "A supported CUDA runtime is required for GPU mode"
    )
    torch.set_num_threads(value["job"]["options"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    memory_fraction = min(0.80, value["job"]["options"]["gpu_memory_fraction"])
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.cuda.reset_peak_memory_stats()

    class GeometryCamera:
        """The frozen LazyCamera geometry, without target/cache construction."""

        def __init__(self, row):
            self.image_name = row["frame"]
            self.image_width, self.image_height = self.width, self.height = row["wh"]
            self.K = np.asarray(row["K"], dtype=np.float64)
            w2c = np.asarray(row["w2c"], dtype=np.float64)
            self.R, self.T = w2c[:3, :3].T, w2c[:3, 3]
            self.FoVx = 2 * math.atan(self.width / (2 * self.K[0, 0]))
            self.FoVy = 2 * math.atan(self.height / (2 * self.K[1, 1]))
            self.znear, self.zfar = 0.01, 100.0
            self.world_view_transform = torch.tensor(
                getWorld2View2(self.R, self.T)
            ).T.cuda()
            p = np.zeros((4, 4))
            p[0, 0], p[1, 1] = (
                2 * self.K[0, 0] / self.width,
                2 * self.K[1, 1] / self.height,
            )
            p[0, 2], p[1, 2] = (
                2 * self.K[0, 2] / self.width - 1,
                2 * self.K[1, 2] / self.height - 1,
            )
            p[2, 2] = self.zfar / (self.zfar - self.znear)
            p[2, 3] = -self.zfar * self.znear / (self.zfar - self.znear)
            p[3, 2] = 1
            self.projection_matrix = torch.tensor(
                p, dtype=torch.float32, device="cuda"
            ).T.contiguous()
            self.full_proj_transform = self.world_view_transform[None].bmm(
                self.projection_matrix[None]
            )[0]
            self.camera_center = self.world_view_transform.inverse()[3, :3]

    def save_png(path, tensor):
        array = np.rint(
            tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)
        Image.fromarray(array).save(path)
        return array

    folder = value["output"]
    folder.mkdir(parents=True, exist_ok=False)
    result = summary(value, args)
    result.update(
        state="running",
        started=datetime.now(timezone.utc).isoformat(),
        GPU_used=True,
        torch_imported=True,
        records=[],
    )
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    bg = torch.zeros(3, device="cuda")
    parameter_names = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    )
    try:
        with torch.no_grad():
            model = GaussianModel(3)
            model.load_ply(str(value["model_file"]))
            model.active_sh_degree = 3
            require(
                model.get_xyz.shape[0] == value["execution"]["points"],
                "Loaded point count differs",
            )
            require(
                getattr(model, "optimizer", None) is None,
                "A diagnostic must not create an optimizer",
            )
            versions = {name: getattr(model, name)._version for name in parameter_names}
            for row in value["selected"]:
                destination = folder / row["frame"]
                destination.mkdir()
                full_camera, head_camera = (
                    GeometryCamera(row["full4"]),
                    GeometryCamera(row["head4"]),
                )
                require(
                    not hasattr(head_camera, "rgb_path")
                    and not hasattr(head_camera, "original_image"),
                    "Head geometry camera must not attach fake RGB targets/cache",
                )
                x0, y0, x1, y1 = row["bbox_native"]
                whole = render(full_camera, model, pipe, bg)["render"].clamp(0, 1)
                reference = whole[:, y0 * 4 : y1 * 4, x0 * 4 : x1 * 4].clone()
                del whole
                torch.cuda.empty_cache()
                direct = render(head_camera, model, pipe, bg)["render"].clamp(0, 1)
                require(
                    direct.shape == reference.shape
                    and torch.isfinite(direct).all()
                    and torch.isfinite(reference).all(),
                    "Invalid diagnostic render or crop dimensions",
                )
                difference = (reference - direct).abs()
                mse = float((reference - direct).square().mean())
                one = save_png(destination / "full4_then_crop.png", reference)
                two = save_png(destination / "head4_direct.png", direct)
                save_png(destination / "abs_diff.png", difference)
                save_png(destination / "abs_diff_x8.png", (difference * 8).clamp(0, 1))
                byte_diff = np.abs(one.astype(np.int16) - two.astype(np.int16))
                metrics = dict(
                    frame=row["frame"],
                    bbox_native=row["bbox_native"],
                    full4_wh=row["full4"]["wh"],
                    head4_wh=row["head4"]["wh"],
                    full4_K=row["full4"]["K"],
                    head4_K=row["head4"]["K"],
                    w2c=row["full4"]["w2c"],
                    float_L1=float(difference.mean()),
                    float_max_abs=float(difference.max()),
                    float_RMSE=math.sqrt(mse),
                    PSNR_dB=(-10 * math.log10(mse) if mse > 0 else None),
                    exact_float_equal=(mse == 0),
                    fraction_channels_abs_gt_1_255=float(
                        (difference > 1 / 255).float().mean()
                    ),
                    PNG_max_byte_error=int(byte_diff.max()),
                    PNG_mean_byte_error=float(byte_diff.mean()),
                    PNG_differing_pixel_fraction=float(
                        np.any(byte_diff != 0, axis=2).mean()
                    ),
                    difference_display="abs_diff.png true abs RGB; abs_diff_x8.png8x gain clipped for visibility",
                    camera_geometry_target_free=True,
                    model_parameter_versions_unchanged=all(
                        getattr(model, n)._version == v for n, v in versions.items()
                    ),
                )
                require(
                    metrics["model_parameter_versions_unchanged"],
                    "Model parameters changed during rendering",
                )
                result["records"].append(metrics)
                print(
                    "ZERO_UPDATE_HEAD_RENDER",
                    json.dumps(metrics, ensure_ascii=False),
                    flush=True,
                )
                del reference, direct, difference, full_camera, head_camera
                torch.cuda.empty_cache()
        after = {path: sha(path) for path in value["immutable"]}
        require(
            after == value["immutable"], "Baseline PLY/job/frozen source inputs changed"
        )
        extension = Path(diff_gaussian_rasterization._C.__file__)
        result.update(
            state="completed_diagnostic",
            finished=datetime.now(timezone.utc).isoformat(),
            protected_sha256_after=after,
            protected_files_unchanged=True,
            compiled_rasterizer=dict(path=str(extension), sha256=sha(extension)),
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            allocator_fraction=memory_fraction,
            device=torch.cuda.get_device_name(0),
        )
    except Exception as error:
        result.update(
            state="failed_diagnostic",
            error=str(error),
            finished=datetime.now(timezone.utc).isoformat(),
        )
        raise
    finally:
        (folder / "probe.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print("ZERO_UPDATE_HEAD_RENDER_READY", folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-job",
        required=True,
        help="Completed model's original job.json, or its run directory",
    )
    parser.add_argument(
        "--boxes",
        required=True,
        help="head_pixel_masks_probe/probe.json or legacy head_bbox_probe/probe.json; native integer bbox coordinates",
    )
    parser.add_argument(
        "--record",
        required=True,
        help="Independent record directory outside the baseline run",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=(1, 4),
        default=4,
        help="Read-only completed model to compare; default4",
    )
    parser.add_argument("--frames", nargs="+", default=["1", "33", "65"])
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate CPU inputs/hashes only; no torch/CUDA import, no output writes",
    )
    args = parser.parse_args()
    try:
        value = validate(args)
        if args.check_only:
            require(
                "torch" not in sys.modules, "CPU-only check unexpectedly imported torch"
            )
            result = summary(value, args)
            result.update(
                state="cpu_inputs_verified",
                GPU_used=False,
                torch_imported=False,
                output_written=False,
                planned_output=str(value["output"]),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            gpu_probe(value, args)
    except Exception as error:
        print("HEAD_RENDER_PROBE_ERROR: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
