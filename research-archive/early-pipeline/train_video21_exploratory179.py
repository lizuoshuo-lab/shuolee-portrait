"""User-authorized 021 preview from the frozen fit, up to its step-1000 snapshot."""
import copy
import fcntl
import os
from pathlib import Path
import sys

import numpy as np

import cross_subject_sh0124 as gs
from pose_training_gate151 import POSE_FIELDS, require
from review_full_sequence123 import relocate
from run_official_sequence123 import root_to_world, save_npz
from select_quality_frames124 import select


SOURCE = gs.ROOT / "fixed-flow175-video21-front24"
OUT = gs.ROOT / "video21-exploratory179-sh0"
VIDEO = Path("/mnt/d/00/工作/数据集/video21.mp4")
SNAPSHOT = 1000


def save_surface_cache(bridge, pose):
    with np.load(gs.BASE / "fit/surface-cache.npz", allow_pickle=False) as data:
        require(np.array_equal(data["faces"], pose["faces"]), "Clothing policy topology differs.")
        path = OUT / "fit/surface-cache.npz"
        save_npz(path, frame_ids=pose["frame_ids"], faces=pose["faces"],
                 base_vertices=bridge.base_vertices.numpy(), linear=bridge.linear.numpy(),
                 skeletal_joints=bridge.base_skeleton.numpy(),
                 clothing_vertices=data["clothing_vertices"], protected_face_vertices=data["protected_face_vertices"])
    return path


def prepare():
    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface

    plan_path = SOURCE / "plan.json"
    fit_path = SOURCE / "result.json"
    review_path = SOURCE / "review/result.json"
    gate_path = SOURCE / "review/pose-training-selection.json"
    contact_path = SOURCE / "contacts.json"
    plan, fitted, review, gate = map(gs.read, (plan_path, fit_path, review_path, gate_path))
    fit_hash = gs.sha(fit_path)
    require(not OUT.exists() and fitted["fitting_complete"] and fitted["plan_sha256"] == gs.sha(plan_path)
            and fit_hash == review["source_fit_sha256"] == gate["source_fit_sha256"]
            and gs.sha(review_path) == gate["fit_review_sha256"]
            and gs.sha(contact_path) == plan["contacts_sha256"]
            and gs.sha(VIDEO) == plan["source_video_sha256"],
            "The source video, fit and review must match the frozen 021 run.")
    require(not gate["full_roster_ready"] and not gate["selected_frames"]
            and gate["target_frames"] == len(plan["source_view_bins"]) == 16,
            "Use the recorded 021 failure only in this dedicated preview.")
    by_name = {row["frame"]: row for row in select(review["records"], target=16)["records"]}
    held = set(plan["retained_holdouts"])
    rank = lambda name: (by_name[name]["quality_score"], name)
    selected = [min((f for f in bucket if f in by_name and f not in held), key=rank)
                for bucket in plan["source_view_bins"]]
    reference = min(selected, key=lambda name: (abs(by_name[name]["yaw_proxy_deg"]) > 25, rank(name)))
    order = [reference] + [name for name in selected if name != reference]
    contacts = {row["frame"]: row for row in gs.read(contact_path)["records"]}
    receipts = {row["frame"]: row for row in fitted["frames"]}
    beta = np.asarray(fitted["fixed_shape"], np.float32)
    require(beta.shape == (1, 10) and fitted["model_unit_scale"] == 1., "Expected one frozen body shape and scale.")
    rows, arrays, pose_paths = [], [], []
    for name in order:
        row = dict(contacts[name])
        row["image"], row["mask"] = (str(relocate(row[key])) for key in ("image", "mask"))
        receipt = receipts[name]
        path = relocate(receipt["output"])
        require(receipt["freeze_betas"] and receipt["source_index"] == row["source_frame_index"]
                and gs.sha(path) == receipt["sha256"]
                and gs.sha(row["image"]) == row["source_sha256"]
                and gs.sha(row["mask"]) == row["mask_sha256"], f"Changed source input for {name}.")
        with np.load(path, allow_pickle=False) as data:
            archive = dict(data)
        intrinsic = archive["K_integer_centers"].copy()
        intrinsic[:2, 2] += .5
        require(np.array_equal(archive["betas"], beta) and np.array_equal(intrinsic, row["K"])
                and np.array_equal(archive["K_integer_centers"], row["K_integer_centers"])
                and np.array_equal(row["w2c"], np.eye(4)), f"Changed scale or camera contract for {name}.")
        rows.append(row)
        arrays.append(archive)
        pose_paths.append(path)

    pose = {key: np.concatenate([a[key] for a in arrays]) for key in (*POSE_FIELDS, "pose_embedding", "vertices_camera")}
    faces = arrays[0]["faces"]
    require(all(np.array_equal(a["faces"], faces) for a in arrays), "Inconsistent SMPL-X topology.")
    n = len(rows)
    pose.update(frame_ids=np.asarray(order), reference_frame_id=np.asarray(reference), betas=beta, shared_betas=beta,
                faces=faces, w2c=np.asarray([r["w2c"] for r in rows]),
                K_integer_centers=np.asarray([r["K_integer_centers"] for r in rows]),
                vertices_world=pose["vertices_camera"].astype(np.float32),
                static_offset=np.zeros((1, 10475, 3), np.float32),
                source_frames=np.asarray([r["source_frame_index"] for r in rows]),
                image_sha256=np.asarray([r["source_sha256"] for r in rows]),
                mask_sha256=np.asarray([r["mask_sha256"] for r in rows]))
    torch.set_num_threads(4)
    asset = gs.ASSETS / "smplx/SMPLX_NEUTRAL.npz"
    bridge = FrozenSkeletonSurface(pose, pose["static_offset"][0], asset)
    replay = float((bridge.base_vertices - torch.from_numpy(pose["vertices_world"])).abs().max())
    require(replay < 2e-6, f"Pose-to-mesh replay error {replay} exceeds the existing tolerance.")
    body = bridge.model
    regressor = body.J_regressor[0].double().numpy()
    rt = regressor @ body.v_template.double().numpy()
    rs = np.einsum("v,vck->ck", regressor, body.shapedirs.double().numpy())
    re = np.einsum("v,vck->ck", regressor, body.expr_dirs.double().numpy())
    roots = [root_to_world(a, e, rt, rs, re) for a, e in zip(arrays, pose["w2c"], strict=True)]
    for key, values in zip(("root_rotation", "world_translation", "root_pivot_model"), zip(*roots), strict=True):
        pose[key] = np.asarray(values)

    (OUT / "fit").mkdir(parents=True, exist_ok=False)
    (OUT / "data").mkdir()
    archive = OUT / "fit/fitted.npz"
    inputs = OUT / "data/training-inputs.json"
    save_npz(archive, **pose)
    cache = save_surface_cache(bridge, pose)
    gs.save(inputs, dict(records=rows, camera_kind="camera_relative_fixedK_body_motion", quality_accepted=False))
    interface = OUT / "smplify-gaussian-interface-receipt.json"
    gs.save(interface, dict(frames=n, reference_frame=reference, fit_sha256=gs.sha(archive),
                            training_inputs_sha256=gs.sha(inputs), surface_cache_sha256=gs.sha(cache),
                            replay_max_abs_m=replay, quality_accepted=False))
    selection = OUT / "selection.json"
    gs.save(selection, dict(selected_frames=selected, retained_holdouts=sorted(held), source_fit_sha256=fit_hash,
                            source_pose_gate_sha256=gs.sha(gate_path), quality_accepted=False,
                            rule="Original numeric rank picks one candidate per fixed source bin for this preview."))
    adapter = OUT / "adapter.py"
    code = gs.generate_source(OUT, n, reference)
    for before, after in (("assert len(training_slots)==8", f"assert len(training_slots)=={n}"),
                          ("training_frames=8", f"training_frames={n}")):
        require(code.count(before) == 1, f"Frozen training adapter changed: {before}")
        code = code.replace(before, after)
    adapter.write_text(code, encoding="utf-8")
    profile = copy.deepcopy(gs.read(gs.BASE / "plan.json"))
    sources = [Path(__file__), Path(gs.__file__), plan_path, fit_path, review_path, gate_path, contact_path, VIDEO,
               gs.BASE / "adapter.py", gs.BASE / "plan.json", gs.UP / "train.py", gs.UP / "scene/gaussian_model.py",
               gs.UP / "arguments/__init__.py", asset, Path(profile["laplacian_cache"]),
               gs.BASE / "fit/surface-cache.npz", archive, inputs, interface, selection, adapter, cache, *pose_paths]
    profile.update(updated=gs.now(), state="prepared", subject=OUT.name, record_id="021",
                   label="021 共享体型 SH0 试训", source_fit=str(fit_path), reference_frame=reference,
                   frames=n, evaluation_frames=n, training_frames=n, selected_frames=selected,
                   from_scratch=True, quality_accepted=False, model_promoted=False, source_fit_quality_accepted=False,
                   approval_scope="021-only exploratory training requested after the pose gate rejected all views.",
                   user_authorization="先训练啊，然后分析", comparison_budget=SNAPSHOT, snapshots=[SNAPSHOT],
                   source_hashes={str(p): gs.sha(p) for p in sources})
    gs.save(OUT / "plan.json", profile)
    print("VIDEO21_PREVIEW_PREPARED", n, reference, replay, flush=True)


def launch_prepared():
    gs.save(OUT / "launch.json", dict(updated=gs.now(), pid=os.getpid(), command=sys.argv,
             proc_start_identity=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
             single_gpu_lock=True, exploratory_only=True))
    scope = dict(vars(gs), OUT=OUT, LIMIT=SNAPSHOT, SNAPSHOTS=[SNAPSHOT], suspended_baseline=lambda: None)
    adapter = OUT / "adapter.py"
    exec(compile(adapter.read_text(encoding="utf-8"), str(adapter), "exec"), scope)
    scope["train"]()


def run():
    with open("/tmp/shuolee-native122-gpu.lock", "a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        launch_prepared()


if __name__ == "__main__":
    run()
