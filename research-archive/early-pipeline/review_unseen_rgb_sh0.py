"""Compare frozen SH0 models on same-video RGB excluded from Gaussian training."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

from review_full_sequence123 import crop_box, joined, relocate
from run_official_gaussians123 import ASSETS, PROJECT, now, read, save, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evenly_spaced_rows(rows):
    """Nearest unused row to up to four uniformly spaced source indices."""
    remaining = sorted(rows, key=lambda row: (row["source_frame_index"], row["frame"]))
    if not remaining:
        return []
    targets = np.linspace(remaining[0]["source_frame_index"], remaining[-1]["source_frame_index"],
                          min(4, len(remaining)))
    selected = []
    for target in targets:
        i = min(range(len(remaining)), key=lambda i: abs(remaining[i]["source_frame_index"] - target))
        selected.append(remaining.pop(i))
    return sorted(selected, key=lambda row: row["source_frame_index"])


def equivalent_camera(E, pose, reference_pose, export_basis):
    """Map the fixed exported body from its fitted pelvis frame to the target camera."""
    return E @ pose @ np.linalg.inv(reference_pose) @ np.linalg.inv(export_basis)


def verify_delivery(out):
    pins = {}

    def verify(path, expected=None):
        actual = sha(path)
        require(expected is None or actual == expected, f"Artifact hash changed: {path}")
        pins[path] = actual
        return path

    publication_path = verify(out / "delivery/publication.json")
    publication = read(publication_path)
    step = publication.get("step")
    require(publication.get("state") == "delivered_pending_quality_review"
            and publication.get("format_passed") is True and publication["run"] == out.name
            and type(step) is int and step > 0, "A completed automatic SH0 delivery is required.")
    snapshot = out / "gaussian-run" / f"step-{step:06d}"
    delivery = out / "delivery" / snapshot.name
    for name, key in (("source-receipt.json", "source_receipt_sha256"), ("point_cloud.ply", "ply_sha256"),
                      ("cameras.json", "camera_sha256"), ("scene.zip", "package_sha256")):
        verify(delivery / name, publication[key])
    source = read(delivery / "source-receipt.json")
    reference = publication["reference_frame"]
    require(source["terminal_verified"] is True and source["step"] == step
            and source["points"] == publication["points"] and source["reference_frame"] == reference,
            "Publication and terminal source identify different models.")
    recorded = source["training_files_sha256"]
    for path in ("plan.json", "progress.json", "process-exit.json", "gaussian-run/result.json",
                 "data/training-inputs.json"):
        verify(out / path, recorded[path])
    for name in ("result.json", "current-meshes.npz", "binding.npz", "checkpoint.pt"):
        path = snapshot / name
        verify(path, recorded[path.relative_to(out).as_posix()])
    verify(snapshot / "world.ply", source["source_ply_sha256"])
    verify(snapshot / "bound-local.ply")
    progress, exit_status, terminal, final = (read(path) for path in (out / "progress.json",
        out / "process-exit.json", out / "gaussian-run/result.json", snapshot / "result.json"))
    require(progress["state"] == terminal["state"] == source["training_state"] == "completed_pending_quality_review"
            and progress["pid"] is None and progress["step"] == terminal["steps"] == final["step"] == step
            and exit_status["exit_code"] == 0 and exit_status["training_function_returned"] is True
            and exit_status["terminal_result_sha256"] == pins[out / "gaussian-run/result.json"]
            and final["reference_frame"] == reference and final["reference_slot"] == 0
            and progress["points"] == terminal["points"] == final["points"] == source["points"]
            and final["active_sh_degree"] == 0 and final["world_ply_sha256"] == source["source_ply_sha256"],
            "Terminal run, PLY or reference snapshot differs from the delivery.")
    validation = read(verify(delivery / "validation.json"))
    require(validation["format_passed"] is True and validation["sha256"] == publication["ply_sha256"]
            and validation["source_sha256"] == source["source_ply_sha256"]
            and validation["serialized_camera_sha256"] == publication["camera_sha256"]
            and validation["reference_frame"] == reference and validation["scale"] == 1.
            and validation["origin_world"] == [0., 0., 0.], "Expected a verified rigid reference export.")
    rotation = np.asarray(validation["rotation"])
    require(rotation.shape == (3, 3) and np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
            and abs(np.linalg.det(rotation) - 1.) < 1e-6, "Invalid native-to-export orientation.")
    export_basis = np.eye(4)
    export_basis[:3, :3] = rotation
    return publication, source, final, snapshot, delivery, export_basis, pins, verify


def review_run(out):
    out = relocate(out).resolve()
    target = out / "unseen-rgb-review"
    require(not target.exists(), "A prior review already exists.")
    publication, source, final, snapshot, delivery, export_basis, pins, verify = verify_delivery(out)
    plan = read(out / "plan.json")
    planned = {relocate(path).resolve(): digest for path, digest in plan["source_hashes"].items()}

    def verify_planned(path):
        path = path.resolve()
        require(path in planned, f"Unpinned training input: {path}")
        return verify(path, planned[path])

    selection = read(verify_planned(out / "pose-training-selection.json"))
    train_rows = read(out / "data/training-inputs.json")["records"]
    train = {row["frame"] for row in train_rows}
    protected = set(selection["retained_holdouts"])
    ref = source["reference_frame"]
    require(plan["pose_selection_sha256"] == pins[out / "pose-training-selection.json"]
            and train == set(selection["selected_frames"]) == set(plan["selected_frames"])
            and train.isdisjoint(protected) and [row["frame"] for row in train_rows] == source["actual_frames"]
            and len(train_rows) == len(train) == final["frames_rendered"] and train_rows[0]["frame"] == ref,
            "Training rows differ from the original frozen split.")
    fit_dir = relocate(selection["fit_dir"]).resolve()
    fit_result = verify_planned(fit_dir / "result.json")
    fit_plan_path = verify_planned(fit_dir / "plan.json")
    fit_plan = read(fit_plan_path)
    require(relocate(plan["source_fit"]).resolve() == fit_result
            and pins[fit_result] == selection["source_fit_sha256"]
            and pins[fit_plan_path] == selection["source_plan_sha256"]
            and read(fit_result)["plan_sha256"] == pins[fit_plan_path]
            and protected == set(fit_plan["retained_holdouts"]), "Frozen fit or holdout split changed.")
    contacts_path = verify_planned(fit_dir / "contacts.json")
    verify_planned(fit_dir / "shared-shape.json")
    require(pins[contacts_path] == fit_plan["contacts_sha256"], "Frozen contact list changed.")
    train_indices = {row["source_frame_index"] for row in train_rows}
    train_pixels = {row["source_sha256"] for row in train_rows}
    candidates = []
    for row in read(contacts_path)["records"]:
        frame = row["frame"]
        decision = selection["reasons"].get(frame, {})
        if (frame not in train and frame not in protected and decision.get("status") == "technically_valid"
                and row["source_frame_index"] not in train_indices and row["source_sha256"] not in train_pixels):
            require(row["source_frame_index"] == decision["source_index"], f"Frozen source index changed: {frame}")
            candidates.append(row)
    selected = evenly_spaced_rows(candidates)
    frames = [row["frame"] for row in selected]
    records = []
    report = dict(schema="same-source-gaussian-unseen-rgb-v1", updated=now(), run=out.name,
        step=publication["step"], reference_frame=ref, candidates=len(candidates), selected_frames=frames,
        selection_rule="Nearest unused row to up to four uniform source-index targets, earlier frame on ties.",
        scope="Same-video RGB excluded from Gaussian training. Poses and source observations were used during pose "
              "preprocessing and roster selection. Protected holdouts remain excluded.",
        physical_extrinsics_known=False, independent_3d_ground_truth=False, heldout_quality_accepted=False,
        optimizer_updates=0, panel_order=["source RGB", "posed Gaussian", "fixed reference PLY"],
        metrics_rule="Native full raster. Source mask >=128/255 and rendered alpha >=0.5. RGB absolute error "
                     "computed inside source mask in [0,1] before JPEG compression.", records=records)

    with TemporaryDirectory(dir=out, prefix=".unseen-rgb-") as temporary:
        stage = Path(temporary) / "review"
        stage.mkdir()
        if frames:
            report.update(render_comparisons(out, fit_dir, fit_plan, train_rows[0], frames, snapshot,
                                             delivery, export_basis, verify, verify_planned, stage, records))
        report["source_hashes"] = {str(path): digest for path, digest in pins.items()}
        require(all(sha(path) == digest for path, digest in pins.items()), "A review input changed during replay.")
        require(not target.exists(), "Review output appeared during replay.")
        save(stage / "result.json", report)
        stage.rename(target)
    return target / "result.json"


def render_comparisons(out, fit_dir, fit_plan, reference, frames, snapshot, delivery,
                       export_basis, verify, verify_planned, stage, records):
    import fcntl

    from prepare_reviewed_sh0 import pack_poses
    from run_official_sequence123 import root_to_world
    from train_demo_body_upstream123 import UP, module

    fit, rows, poses, pose_paths = pack_poses(fit_dir, [reference["frame"], *frames])
    receipts = {item["frame"]: item for item in read(fit_dir / "result.json")["frames"]}
    for row, pose_path in zip(rows, pose_paths, strict=True):
        verify(pose_path, receipts[row["frame"]]["sha256"])
        verify(relocate(row["image"]), row["source_sha256"])
        verify(relocate(row["mask"]), row["mask_sha256"])
    require(all(row["wh"] == reference["wh"] and np.array_equal(row["K"], reference["K"])
                and np.array_equal(row["K_integer_centers"], fit_plan["K_integer_centers"])
                and np.array_equal(row["w2c"], reference["w2c"]) for row in rows)
            and rows[0]["source_sha256"] == reference["source_sha256"],
            "The fitted frames do not share the delivered native camera contract.")
    with np.load(verify_planned(out / "fit/fitted.npz"), allow_pickle=False) as trained:
        require(np.array_equal(trained["betas"], fit["betas"]) and np.array_equal(trained["faces"], fit["faces"])
                and str(trained["frame_ids"][0]) == reference["frame"]
                and np.array_equal(trained["K_integer_centers"][0], fit_plan["K_integer_centers"]),
                "Replay fit and trained mesh use different topology, shared beta or K.")
    with np.load(verify_planned(out / "fit/surface-cache.npz"), allow_pickle=False) as cache:
        clothing, protected = cache["clothing_vertices"], cache["protected_face_vertices"]
        require(np.array_equal(cache["faces"], fit["faces"]) and not np.any(clothing & protected),
                "Frozen clothing policy changed.")
    with np.load(snapshot / "binding.npz", allow_pickle=False) as binding:
        triangle_ids, offset = binding["triangle_ids"], binding["static_offset"]
        require(np.isfinite(offset).all(), "Invalid trained canonical offset.")
    with np.load(snapshot / "current-meshes.npz", allow_pickle=False) as saved:
        require(str(saved["frame_ids"][0]) == reference["frame"] and np.array_equal(saved["faces"], fit["faces"]),
                "Trained mesh archive has a different reference or topology.")
        saved_ref = saved["vertices_camera"][0]

    asset = verify_planned(ASSETS / "smplx/SMPLX_NEUTRAL.npz")
    model_path = verify_planned(UP / "scene/gaussian_model.py")
    for name in ("src/avatar_core/scene/smplx_gaussian_model.py", "src/shuolee_portrait/training.py",
                 "src/avatar_core/gaussian_renderer/__init__.py"):
        verify_planned(PROJECT / name)
    for name in ("frozen_skeleton_surface123.py", "facial_geometry_boundary123.py", "body_root_sh_colors123.py"):
        verify_planned(Path(__file__).with_name(name))
    verify(snapshot / f"{reference['frame']}_gaussian.png")

    with Path("/tmp/shuolee-native122-gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from frozen_skeleton_surface123 import FrozenSkeletonSurface
        from facial_geometry_boundary123 import clothing_only_offset

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        sys.path.insert(0, str(PROJECT / "src"))
        from shuolee_portrait import runtime as rt
        rt.configure(ASSETS)
        from shuolee_portrait.training import camera, image
        from scene.smplx_gaussian_model import SMPLXGaussianModel
        from gaussian_renderer import render
        from body_root_sh_colors123 import render_root_frame
        from verify_demo_head_export123 import serialized_camera

        bridge = FrozenSkeletonSurface(fit, np.zeros_like(offset), asset, device="cpu")
        replay_error = float(np.abs(bridge.base_vertices.numpy() - fit["vertices_camera"]).max())
        delta = clothing_only_offset(torch.as_tensor(offset), torch.as_tensor(clothing), torch.as_tensor(protected))
        meshes = bridge.vertices(delta).numpy()
        ref_mesh_error = float(np.abs(meshes[0] - saved_ref).max())
        require(replay_error < 2e-6 and ref_mesh_error < 2e-6, "Original SMPL-X pose or offset replay differs.")
        meshes[0] = saved_ref

        model = bridge.model
        reg = model.J_regressor[0].double().numpy()
        pivot = (reg @ model.v_template.double().numpy(),
                 np.einsum("v,vck->ck", reg, model.shapedirs.double().numpy()),
                 np.einsum("v,vck->ck", reg, model.expr_dirs.double().numpy()))
        roots = np.repeat(np.eye(4)[None], len(rows), axis=0)
        for i, (pose, E) in enumerate(zip(poses, fit["w2c"], strict=True)):
            roots[i, :3, :3], roots[i, :3, 3], _ = root_to_world(pose, E, *pivot)
        require(np.allclose(roots[:, :3, 3], bridge.joints().numpy()[:, 0], rtol=0, atol=2e-6),
                "Fitted pelvis transform differs from the original SMPL-X joint.")

        official = module(model_path, "unseen_rgb_frozen_sh0")

        class Bound(official.GaussianModel):
            select_mesh_by_timestep = SMPLXGaussianModel.select_mesh_by_timestep
            get_covariance = SMPLXGaussianModel.get_covariance

        posed, native, frozen = Bound(0), official.GaussianModel(0), official.GaussianModel(0)
        posed.static_offset = None
        posed.vertices_sequence = torch.as_tensor(meshes, dtype=torch.float32, device="cuda")
        posed.faces = torch.as_tensor(fit["faces"], dtype=torch.long, device="cuda")
        posed.load_ply(str(snapshot / "bound-local.ply"))
        native.load_ply(str(snapshot / "world.ply"))
        frozen.load_ply(str(delivery / "point_cloud.ply"))
        require(np.array_equal(posed.binding.cpu().numpy(), triangle_ids)
                and len(posed._xyz) == len(native._xyz) == len(frozen._xyz) == read(snapshot / "result.json")["points"]
                and all(model.active_sh_degree == 0 and model.optimizer is None for model in (posed, native, frozen)),
                "Frozen Gaussian state differs.")
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        black = torch.zeros(3, device="cuda")
        root_basis = torch.as_tensor(roots[0, :3, :3] @ roots[:, :3, :3].transpose(0, 2, 1),
                                     dtype=torch.float32, device="cuda")

        with torch.no_grad():
            posed.select_mesh_by_timestep(0)
            ref_camera, _ = camera(rows[0], 0)
            ref_posed = render_root_frame(render, ref_camera, posed, pipe, black, root_basis[0])["render"]
            ref_native = render(ref_camera, native, pipe, black)["render"]
            serialized = serialized_camera(delivery, rows[0], export_basis[:3, :3])
            ref_frozen_camera, _ = camera(serialized, 0)
            ref_frozen = render(ref_frozen_camera, frozen, pipe, black)["render"]
            reference_parity = {}
            for label, left, right in (("posed_to_native", ref_posed, ref_native),
                                       ("native_to_delivered", ref_native, ref_frozen)):
                difference = (left - right).abs()
                error = dict(mean_abs=float(difference.mean()), max_abs=float(difference.max()))
                require(error["mean_abs"] < 2e-6 and error["max_abs"] < .005, f"Reference {label} differs: {error}")
                reference_parity[label] = error
            with Image.open(snapshot / f"{reference['frame']}_gaussian.png") as archived:
                png_difference = np.abs(np.asarray(image(ref_posed), dtype=np.int16) -
                                        np.asarray(archived.convert("RGB"), dtype=np.int16))
            reference_parity.update(archived_mean_8bit=float(png_difference.mean()),
                                    archived_max_8bit=int(png_difference.max()))
            require(reference_parity["archived_mean_8bit"] < .001
                    and reference_parity["archived_max_8bit"] <= 1, f"Reference replay differs: {reference_parity}")
            del ref_native, ref_posed, ref_frozen, difference, left, right

            for i, row in enumerate(rows[1:], start=1):
                frame = row["frame"]
                depth = float(meshes[i, :, 2].min())
                info = dict(frame=frame, source_frame_index=row["source_frame_index"],
                            time_seconds=row.get("time_seconds"), min_depth_m=depth,
                            image_sha256=row["source_sha256"], mask_sha256=row["mask_sha256"])
                if depth <= .05:
                    records.append(dict(info, rendered=False, reason="Offset mesh crosses the near plane."))
                    continue
                E = equivalent_camera(fit["w2c"][i], roots[i], roots[0], export_basis)
                posed_cam, posed_receipt = camera(row, i, pose_index=i)
                frozen_cam, frozen_receipt = camera(dict(row, w2c=E.tolist()), i)
                posed.select_mesh_by_timestep(i)
                a = render_root_frame(render, posed_cam, posed, pipe, black, root_basis[i])["render"]
                b = render(frozen_cam, frozen, pipe, black)["render"]
                with Image.open(relocate(row["image"])) as raw, Image.open(relocate(row["mask"])) as matte:
                    require(raw.size == matte.size == tuple(row["wh"]), f"Native raster changed: {frame}")
                    original, mask = raw.convert("RGB"), np.asarray(matte.convert("L")) >= 128
                rgb = np.asarray(original, dtype=np.float32) / 255.
                scores = {}
                for label, pixels, cam, gaussian in (("posed", a, posed_cam, posed), ("frozen", b, frozen_cam, frozen)):
                    alpha = render(cam, gaussian, pipe, black,
                                   override_color=torch.ones_like(gaussian.get_xyz))["render"][0]
                    visible = alpha.cpu().numpy() >= .5
                    color = pixels.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    overlap, foreground = int((visible & mask).sum()), int(mask.sum())
                    union = int((visible | mask).sum())
                    scores[label] = dict(
                        foreground_rgb_mae=float(np.abs(color[mask] - rgb[mask]).mean()) if foreground else None,
                        alpha_half_mask_iou=overlap / union if union else None,
                        alpha_half_mask_coverage=overlap / foreground if foreground else None,
                        alpha_half_spill_pixels=int((visible & ~mask).sum()))
                box = crop_box(row.get("initial_box_xyxy", [0, 0, *row["wh"]]), row["wh"], padding=64)
                path = stage / f"{frame}-source-posed-frozen.jpg"
                joined([panel.crop(box) for panel in (original, image(a), image(b))]).save(path, quality=95, subsampling=0)
                records.append(dict(info, rendered=True, crop_xyxy=box, metrics=scores,
                    fitted_camera=posed_receipt, pelvis_relative_camera=frozen_receipt, image=path.name,
                    comparison_sha256=sha(path)))
        require(all(model.optimizer is None for model in (posed, native, frozen)), "A replay optimizer was created.")
    return dict(smplx_replay_max_m=replay_error, reference_offset_mesh_replay_max_m=ref_mesh_error,
                reference_render_replay=reference_parity,
                equivalent_camera_rule="E_target @ P_target @ inverse(P_reference) @ inverse(A_export), "
                                       "with original fitted SMPL-X pelvis pivots and the saved rigid export basis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivered_run")
    args = parser.parse_args()
    print(json.dumps(dict(report=str(review_run(args.delivered_run))), ensure_ascii=False))
