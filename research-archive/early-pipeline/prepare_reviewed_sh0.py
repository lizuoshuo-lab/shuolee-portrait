"""Prepare bound SH0 training and run the automatic delivery workflow."""
import argparse
from pathlib import Path
import traceback

import numpy as np

import cross_subject_sh0124 as gs
from pose_training_gate151 import POSE_FIELDS, load_admitted_selection, require, verify_prepared
from review_full_sequence123 import relocate
from run_official_sequence123 import root_to_world, save_npz


def load_automatic_selection(selection_path):
    from automatic_sh0_selection import select_automatic_fit

    frozen = gs.read(selection_path)
    fit_dir = relocate(frozen["fit_dir"])
    expected = select_automatic_fit(fit_dir, frozen["arc"], screen_pose="pose_screen" in frozen)
    expected["fit_dir"] = frozen["fit_dir"]
    require(expected == frozen, "Automatic selection inputs changed.")
    return fit_dir, frozen


def pack_poses(fit_dir, frames):
    plan, result = (gs.read(fit_dir / name) for name in ("plan.json", "result.json"))
    contact_path = fit_dir / "contacts.json"
    require(gs.sha(contact_path) == plan["contacts_sha256"], "The original source contacts changed.")
    contacts = {row["frame"]: row for row in gs.read(contact_path)["records"]}
    receipts = {row["frame"]: row for row in result["frames"]}
    shape_path = fit_dir / "shared-shape.json"
    shape = gs.read(shape_path)
    require(gs.sha(shape_path) == (result.get("shared_shape_sha256") or plan.get("neutral_shape_sha256")),
            "The shared shape receipt changed.")
    beta = np.asarray(shape["betas"], dtype=np.float32)
    require(beta.shape == (1, 10) and np.isfinite(beta).all() and shape["model_unit_scale"] == 1.,
            "Expected a shared body shape at unit scale.")
    require(not set(frames) & set(plan["retained_holdouts"]), "The roster includes protected holdouts.")
    rows, poses, paths = [], [], []
    for frame in frames:
        row, receipt = dict(contacts[frame]), receipts[frame]
        pose_path = relocate(receipt["output"])
        require(receipt["freeze_betas"] and receipt["source_index"] == row["source_frame_index"]
                and gs.sha(pose_path) == receipt["sha256"], f"The saved pose changed for {frame}.")
        for key, digest in (("image", "source_sha256"), ("mask", "mask_sha256")):
            path = relocate(row[key])
            require(gs.sha(path) == row[digest], f"The source {key} changed for {frame}.")
            row[key] = str(path)
        with np.load(pose_path, allow_pickle=False) as data:
            pose = dict(data)
        K = pose["K_integer_centers"].copy()
        K[:2, 2] += .5
        require(np.array_equal(pose["betas"], beta) and np.array_equal(pose["K_integer_centers"], row["K_integer_centers"])
                and np.array_equal(K, row["K"]) and np.array_equal(row["w2c"], np.eye(4))
                and np.all(pose["vertices_camera"][..., 2] > .05), f"Invalid shape, native camera or depth for {frame}.")
        rows.append(row)
        poses.append(pose)
        paths.append(pose_path)

    faces = poses[0]["faces"]
    require(all(np.array_equal(pose["faces"], faces) for pose in poses), "SMPL-X topology differs across frames.")
    fit = {key: np.concatenate([pose[key] for pose in poses]) for key in (*POSE_FIELDS, "pose_embedding", "vertices_camera")}
    fit.update(frame_ids=np.asarray(frames), reference_frame_id=np.asarray(frames[0]), betas=beta, shared_betas=beta,
               faces=faces, static_offset=np.zeros((1, fit["vertices_camera"].shape[1], 3), np.float32),
               vertices_world=fit["vertices_camera"].astype(np.float32), w2c=np.asarray([row["w2c"] for row in rows]),
               K_integer_centers=np.asarray([row["K_integer_centers"] for row in rows]),
               source_frames=np.asarray([row["source_frame_index"] for row in rows]),
               image_sha256=np.asarray([row["source_sha256"] for row in rows]),
               mask_sha256=np.asarray([row["mask_sha256"] for row in rows]))
    return fit, rows, poses, paths


def prepare(selection_path, out, *, automatic=False):
    loader = load_automatic_selection if automatic else load_admitted_selection
    fit_dir, selection = loader(selection_path)
    selected = selection["selected_frames"]
    require(selection["target_frames"] in (24, 36) and not (out / "plan.json").exists(),
            "Expected a new 24- or 36-target run.")
    if automatic:
        reference = selection["reference_frame"]
        selection_evidence = gs.HERE / "automatic_sh0_selection.py"
    else:
        selection_evidence = fit_dir / "review/result.json"
        review = {row["frame"]: row for row in gs.read(selection_evidence)["records"]}
        frontal = [review[frame] for frame in selected if review[frame]["face_weighted_rms_px"] is not None]
        require(frontal, "A visible reviewed face is required for the reference pose.")
        reference = min(frontal, key=lambda row: (abs((row["yaw_proxy_deg"] + 180) % 360 - 180) > 25,
                        row["face_weighted_rms_px"], -row["face_sharpness"], row["frame"]))["frame"]
    require(reference in selected, "The reference pose must belong to the selected frames.")
    order = [reference] + [frame for frame in selected if frame != reference]
    fit, rows, poses, pose_paths = pack_poses(fit_dir, order)

    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface

    torch.set_num_threads(4)
    asset = gs.ASSETS / "smplx/SMPLX_NEUTRAL.npz"
    bridge = FrozenSkeletonSurface(fit, fit["static_offset"][0], asset)
    replay = float((bridge.base_vertices - torch.from_numpy(fit["vertices_world"])).abs().max())
    require(replay < 2e-6, f"Original SMPL-X pose replay differs by {replay} m.")
    model = bridge.model
    reg = model.J_regressor[0].double().numpy()
    pivot = (reg @ model.v_template.double().numpy(), np.einsum("v,vck->ck", reg, model.shapedirs.double().numpy()),
             np.einsum("v,vck->ck", reg, model.expr_dirs.double().numpy()))
    roots = [root_to_world(pose, E, *pivot) for pose, E in zip(poses, fit["w2c"], strict=True)]
    for key, values in zip(("root_rotation", "world_translation", "root_pivot_model"), zip(*roots, strict=True), strict=True):
        fit[key] = np.asarray(values)

    policy_path = gs.BASE / "fit/surface-cache.npz"
    with np.load(policy_path, allow_pickle=False) as policy:
        require(np.array_equal(fit["faces"], policy["faces"]), "Clothing policy topology differs.")
        regions = {key: policy[key] for key in ("clothing_vertices", "protected_face_vertices")}
    (out / "fit").mkdir(parents=True, exist_ok=False)
    (out / "data").mkdir()
    archive, cache = (out / "fit" / name for name in ("fitted.npz", "surface-cache.npz"))
    inputs = out / "data/training-inputs.json"
    save_npz(archive, **fit)
    save_npz(cache, frame_ids=fit["frame_ids"], faces=fit["faces"], base_vertices=bridge.base_vertices.numpy(),
             linear=bridge.linear.numpy(), skeletal_joints=bridge.base_skeleton.numpy(), **regions)
    gs.save(inputs, dict(records=rows, camera_kind="camera_relative_fixedK_body_motion", quality_accepted=False))
    interface = out / "smplify-gaussian-interface-receipt.json"
    n = len(rows)
    gs.save(interface, dict(frames=n, reference_frame=reference, fit_sha256=gs.sha(archive),
                           training_inputs_sha256=gs.sha(inputs), surface_cache_sha256=gs.sha(cache), replay_max_abs_m=replay))
    selected_path = out / "selection.json"
    gs.save(selected_path, dict(selected_frames=selected, retained_holdouts=selection["retained_holdouts"]))
    adapter = out / "adapter.py"
    code = gs.generate_source(out, n, reference, automatic=automatic)
    for before, after in (("assert len(training_slots)==8", f"assert len(training_slots)=={n}"),
                          ("training_frames=8", f"training_frames={n}")):
        require(code.count(before) == 1, f"Frozen SH0 adapter changed: {before}")
        code = code.replace(before, after)
    adapter.write_text(code, encoding="utf-8")

    baseline = gs.read(gs.BASE / "plan.json")
    profile = {key: baseline[key] for key in ("upstream_commit", "upstream_options", "options", "shared_laplacian_weight",
               "laplacian_cache", "max_points_safety_stop", "initial_points", "point_target", "model_sh_degree")}
    hashes = [Path(__file__), Path(gs.__file__), selection_path, selection_evidence, fit_dir / "plan.json", fit_dir / "result.json",
              fit_dir / "contacts.json", fit_dir / "shared-shape.json", gs.BASE / "adapter.py", gs.BASE / "plan.json", policy_path, gs.UP / "train.py",
              gs.UP / "scene/gaussian_model.py", gs.UP / "arguments/__init__.py", asset, Path(profile["laplacian_cache"]),
              adapter, archive, cache, inputs, interface, selected_path, *pose_paths]
    hashes.extend(gs.HERE / name for name in ("pose_training_gate151.py", "smplify_direct_surface123.py",
                  "frozen_skeleton_surface123.py", "facial_geometry_boundary123.py", "body_root_sh_colors123.py"))
    hashes.extend(gs.PROJECT / name for name in ("src/avatar_core/scene/smplx_gaussian_model.py",
                  "src/shuolee_portrait/training.py", "src/avatar_core/gaussian_renderer/__init__.py"))
    if automatic:
        schedule = dict(iterations=18000, position_lr_max_steps=18000, densify_from_iter=300,
                        densification_interval=60, densify_until_iter=18000, opacity_reset_interval=1800)
        profile["schedule_overrides"] = schedule
        profile["options"] = dict(profile["options"], **schedule)
        policy = gs.HERE.parent / "shuolee-coordination/director-workflow-policy-20261110.json"
        hashes.append(policy)
        if "pose_screen" in selection:
            hashes.extend(gs.HERE / name for name in ("screen_official_body_contract124.py", "select_quality_frames124.py"))
    profile.update(updated=gs.now(), state="prepared" if automatic else "prepared_input_review_pending",
                   automatic_delivery=automatic, subject=out.name, label=out.name,
                   source_fit=str(fit_dir / "result.json"), reference_frame=reference, frames=n, training_frames=n,
                   evaluation_frames=n, selected_frames=selected, pose_selection_sha256=gs.sha(selection_path),
                   from_scratch=True, gaussian_parameters_reused=False, optimizer_state_reused=False,
                   comparison_budget=18000, snapshots=[1000, 10000, 18000], quality_accepted=False,
                   model_promoted=False, end_to_end_15min_accepted=False,
                   source_hashes={str(path): gs.sha(path) for path in hashes})
    gs.save(out / "plan.json", profile)
    require(verify_prepared(selection_path, out, selection_loader=loader) == selected,
            "Prepared frame order does not match the frozen roster.")
    return n, reference, replay


def run_automatic(fit_dir, out, arc):
    from automatic_sh0_selection import select_automatic_fit

    out.mkdir(exist_ok=False)
    stage = "automatic_selection"
    try:
        selection = out / "pose-training-selection.json"
        roster = select_automatic_fit(fit_dir, arc)
        gs.save(selection, roster)
        require(roster["selected_frames"], "No non-holdout pose passed the existing body and feet screen.")
        stage = "prepare"
        print("AUTOMATIC_SH0_PREPARED", *prepare(selection, out, automatic=True), flush=True)
        stage = "training"
        gs.train(out.name, pose_selection=selection, automatic=True)
        terminal_path = out / "gaussian-run/result.json"
        terminal = gs.read(terminal_path)
        require(terminal["state"] == "completed_pending_quality_review" and terminal["steps"] == 18000,
                "The native trainer did not complete its scheduled steps.")
        gs.save(out / "process-exit.json", dict(exit_code=0, terminal_result_sha256=gs.sha(terminal_path),
                                                stage=stage, training_function_returned=True))
        stage = "delivery"
        from deliver_automatic_sh0 import deliver
        delivered = deliver(out)
        print("AUTOMATIC_SH0_DELIVERED", delivered, flush=True)
    except Exception as exc:
        gs.save(out / "failure.json", dict(stage=stage, error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "run", "deliver"))
    parser.add_argument("input", help="Reviewed selection for prepare/train or fitted input directory for run")
    parser.add_argument("run", help="New run name within the existing cross-subject output directory")
    parser.add_argument("--arc", choices=("half", "full"))
    args = parser.parse_args()
    require(args.run not in ("", ".", "..") and Path(args.run).name == args.run, "Expected a single run name.")
    if args.mode == "prepare":
        print("REVIEWED_SH0_PREPARED", *prepare(relocate(args.input), gs.ROOT / args.run), flush=True)
    elif args.mode == "train":
        gs.train(args.run, pose_selection=relocate(args.input))
    elif args.mode == "run":
        require(args.arc is not None, "Specify the source capture arc.")
        run_automatic(relocate(args.input), gs.ROOT / args.run, args.arc)
    else:
        from deliver_automatic_sh0 import deliver
        print("AUTOMATIC_SH0_DELIVERED", deliver(gs.ROOT / args.run), flush=True)
