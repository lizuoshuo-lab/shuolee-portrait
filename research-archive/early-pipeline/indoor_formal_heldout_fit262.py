"""Isolated CPU SMPLify-X pose inference for 008's three registered evaluation views."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from difflib import unified_diff
import importlib.util
import inspect
import os
from pathlib import Path
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from update_leafwoman129_state import ROOT


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


COMMON_PATH = ROOT / "grass-winter-formal-heldout-fit262.py"
common = module(COMMON_PATH, "formal_eval262_contract")
legacy = common.legacy
np, fixed, adapter, entry = common.np, common.fixed, common.adapter, common.original
read, save, sha, require, relocate = legacy.read, legacy.save, legacy.sha, legacy.require, legacy.relocate
from smplify_face_support124 import abstain_inner_face

RUN = ROOT / "indoor-automatic-sh0-254"
PIPELINE = ROOT / "indoor-automatic-sh0-254-pipeline"
FIT = ROOT / "indoor-hand-support225"
OUT = RUN.with_name(RUN.name + "-formal-heldout-fit262")
STAGE = OUT / "shared-cpu"
ARCHIVED_WORKER = ROOT / "fixed-flow192-indoor-knee-visibility/recovery-depth/source/run_leafwoman_front24_cpu145.py"
HOLDOUTS = ["frame_000015", "frame_000043", "frame_000071", "frame_000098"]
UNAVAILABLE = {"frame_000071": "Original index 561 lies outside both frozen capture arcs. "
               "The original video shows a side/rear profile, with no registered camera-initialization branch for strict replay."}
FRAMES = [frame for frame in HOLDOUTS if frame not in UNAVAILABLE]
WITNESSES = {"front": (ROOT / "indoor-upright206/shared-cpu/frame_000002/upstream-adapter.diff",
                       "f32dbfc98a4d47e10f89d5b32b4a33dc5ac2088fcb50bc22065edc9eff5f90f3"),
             "rear": (ROOT / "indoor-knee205/fit/shared-cpu/frame_000040/upstream-adapter.diff",
                      "4e2cccf50fb1a936edf042411589583c49c922bbfae510c19e2bd76e2d63883a")}


def source_with_upright(root, base_source, patch):
    source, diff = base_source(root)
    before, after = patch
    require(source.count(before) == 1, "Frozen upright camera-stage seed marker changed")
    changed = source.replace(before, after)
    return changed, diff + "".join(unified_diff(source.splitlines(True), changed.splitlines(True),
                                                fromfile="existing-input-adapter", tofile="upright-camera-start"))


def check_prefix_source(plan, pin):
    """Verify the same effective official-entry patch and orientation snapshot used by 225."""
    for side, (witness, digest) in WITNESSES.items():
        if side == "front":
            source, diff = source_with_upright(adapter.original.OFFICIAL, entry.adapted_entry_source,
                                              plan["upright_input_patch"])
        else:
            source, diff = entry.adapted_entry_source(adapter.original.OFFICIAL)
        marker = "            results.append({'loss': final_loss_val,"
        require(source.count(marker) == 1, "Original orientation-copy hook changed")
        copied = source.replace(marker, "            result = {key: value.copy() for key, value in result.items()}\n" + marker)
        diff += "".join(unified_diff(source.splitlines(True), copied.splitlines(True),
                                     fromfile="input-output-adapter", tofile="orientation-snapshot"))
        require(diff == pin(witness, digest).read_text(encoding="utf-8"),
                f"Live effective {side} recipe differs from the exact archived input adapter")


def observations(plan, rule, native, manifest, archive, source_archive):
    rows = read(manifest)["records"]
    names = [r["frame"] for r in rows]
    ids = {frame: i for i, frame in enumerate(names)}
    require(len(rows) == len(ids) == len({row["source_frame_index"] for row in rows}) == 111,
            "Duplicate or incomplete native source rows")
    training = np.asarray([ids[f] for f in plan["source_frames"]])
    held = np.asarray([ids[f] for f in HOLDOUTS])
    with np.load(source_archive, allow_pickle=False) as data:
        require(data["frame_ids"].tolist() == names, "Raw observation order changed")
        xy, scores, boxes, flip = (data[k] for k in ("keypoints", "scores", "boxes", "flip_disagreement"))
    with np.load(archive, allow_pickle=False) as saved:
        K, q90, expected = (saved[k] for k in ("K_integer_centers", "confidence_q90", "observed_native_keypoints"))
        require(saved["frame_ids"].tolist() == plan["source_frames"] and np.array_equal(K, plan["K_integer_centers"]),
                "Frozen candidate observation order or K changed")
    confidence, replayed_q90 = legacy.relative_confidence(scores[training], boxes[training], flip[training])
    require(np.array_equal(q90, replayed_q90), "Frozen training-only q90 does not replay")
    counts = fixed.face_counts(rule, rows)

    def map_rows(indices, weights):
        weights = fixed.abstain_reviewed_source_visibility(rule, indices + 1, weights)
        mapped = np.zeros((len(indices), 118, 3), np.float32)
        mapped[:, adapter.TARGET_IDX, :2] = xy[indices][:, adapter.SOURCE_IDX]
        mapped[:, adapter.TARGET_IDX, 2] = weights[:, adapter.SOURCE_IDX]
        mapped, _ = legacy.mask_outside_raster(mapped, plan["native_wh"])
        mapped = abstain_inner_face(mapped, counts[indices])
        feet = fixed.abstain_reviewed_ankle_feet(rule, native["observation_manifest_sha256"], rows, indices, mapped)
        return mapped, feet

    checked, _ = map_rows(training, confidence)
    require(np.array_equal(checked[..., :2], expected[..., :2]), "Frozen candidate XY does not replay")
    changed = checked[..., 2] != expected[..., 2]
    changed_rows = {plan["source_frames"][i]: np.flatnonzero(row).tolist() for i, row in enumerate(changed) if row.any()}
    allowed = ([7, *range(25, 46)], [4, *range(46, 67)])
    require(set(changed_rows) == set(plan["refit_frames"])
            and int(changed.sum()) == plan["changed_confidence_entries"] == 132
            and all(joints in allowed for joints in changed_rows.values())
            and np.all(expected[..., 2][changed] == 0), "Unexpected difference from the documented 225 far-hand policy")
    held_scores, held_boxes, held_flip = (x[held].astype(np.float64) for x in (scores, boxes, flip))
    height = held_boxes[:, 3] - held_boxes[:, 1]
    require(np.isfinite(held_scores).all() and np.isfinite(held_boxes).all() and np.isfinite(held_flip).all()
            and np.all(height > 0) and np.all(held_scores >= 0) and np.all(held_flip >= 0),
            "Invalid archived holdout confidence inputs")
    weights = np.clip(held_scores / q90, 0, 1) * np.exp(-.25 * (held_flip / (.012 * height[:, None])) ** 2)
    mapped, feet = map_rows(held, weights.astype(np.float32))
    require(counts[held].tolist() == [1, 0, 1, 1]
            and all(set(groups) == {"left_foot", "right_foot"} for groups in feet.values())
            and set(feet) == set(HOLDOUTS) and not np.any(mapped[:, [11, 14, *range(19, 25)], 2]),
            "Frozen face or foot visibility changed")
    return [dict(rows[i]) for i in held], mapped, K, q90, changed_rows


def prepare():
    publication, _, _, _, _, _, delivery_pins, _ = legacy.verify_delivery(RUN)
    pins = {str(path): digest for path, digest in delivery_pins.items()}

    def pin(path, expected=None):
        path = relocate(path).resolve(strict=True)
        digest = sha(path)
        require(expected is None or expected == digest, f"Input hash changed: {path}")
        pins[str(path)] = digest
        return path

    state = read(pin(PIPELINE / "pipeline.json"))
    require(state["state"] == "delivered_pending_quality_review" and state["delivery"] == publication
            and relocate(state["fit_dir"]).resolve() == FIT and relocate(state["output"]).resolve() == RUN,
            "254 pipeline does not bind the delivered model and original fit")
    model_plan = read(RUN / "plan.json")
    selected = read(pin(RUN / "pose-training-selection.json", model_plan["pose_selection_sha256"]))
    result_path = pin(FIT / "result.json", state["fit_result_sha256"])
    result = read(result_path)
    plan_path = pin(FIT / "plan.json", result["plan_sha256"])
    plan = read(plan_path)
    require(result["fitting_complete"] and sha(result_path) == selected["source_fit_sha256"]
            and sha(plan_path) == selected["source_plan_sha256"]
            and relocate(selected["fit_dir"]).resolve() == FIT
            and HOLDOUTS == plan["retained_holdouts"] == selected["retained_holdouts"]
            and set(HOLDOUTS).isdisjoint(set(plan["source_frames"]) | set(plan["shape_anchors"]) |
                                         set(selected["selected_frames"]))
            and [r["frame"] for r in result["frames"]] == plan["source_frames"], "Protected source split changed")
    shape = pin(FIT / "shared-shape.json", result["shared_shape_sha256"])
    shape_data = read(shape)
    beta = np.asarray(shape_data["betas"], dtype=np.float32)
    shared = plan["shared_body_constraint"]
    require(beta.shape == (1, 10) and np.isfinite(beta).all() and shape_data["model_unit_scale"] == 1.
            and np.array_equal(beta, result["fixed_shape"]) and shared["body_weight"] == 30.
            and shared["root_weight"] == 0. and not shared.get("root_targets_camera"),
            "Original shared beta or the one body constraint changed")
    worker_digest = next(digest for path, digest in plan["pins"].items()
                         if relocate(path).name == ARCHIVED_WORKER.name)
    pin(ARCHIVED_WORKER, worker_digest)
    pin(entry.__file__, plan["fit_entry_sha256"])
    runner_path = pin(fixed.__file__)
    for path, digest in plan["pins"].items():
        if relocate(path).name not in ("run_fixed_smplify175.py", "run_leafwoman_front24_cpu145.py"):
            pin(path, digest)
    for name, digest in shared["loss_source_sha256"].items():
        pin(legacy.SUPPORT / name, digest)
    for path in (Path(__file__), COMMON_PATH, common.REUSED_DRIVER, adapter.__file__,
                 FIT / "shared-cpu/plan.json", legacy.SUPPORT / "smplify_face_support124.py"):
        pin(path)
    pin(FIT / "fit_control.py", plan["worker_sha256"])
    check_prefix_source(plan, pin)
    native_path = pin(ROOT / "indoor-native/result.json", plan["native_receipt_sha256"])
    native = read(native_path)
    pin(state["source_video"], plan["source_video_sha256"])
    require(native["observations_ready"] and native["video_sha256"] == state["source_video_sha256"] == plan["source_video_sha256"]
            and native["observations_sha256"] == plan["source_observations_sha256"], "Native source receipt changed")
    arcs_path = pin(ROOT / "indoor-native/capture-arcs.json",
                    "e2bb4a800ed3434ea944d1852b25412d1b65c9f71c9cae91679c0f151fe68255")
    arcs = read(arcs_path)
    require(arcs["video_sha256"] == native["video_sha256"] and arcs["native_receipt_sha256"] == sha(native_path),
            "Capture arc does not belong to this video")
    rule = read(pin(ROOT / "indoor-knee205/source-admission.json", plan["source_review_sha256"]))
    for key in ("face_support_receipt", "joint_group_support_receipt"):
        pin(rule[key]["path"], rule[key]["sha256"])
        if "review_path" in rule[key]:
            pin(rule[key]["review_path"], rule[key]["review_sha256"])
    data = relocate(native["data_root"]) / "data"
    manifest = pin(data / "inputs.json", native["observation_manifest_sha256"])
    source_archive = pin(data / "observations.npz", native["observations_sha256"])
    archive = pin(FIT / "fit-observations.npz", plan["baseline_observation_sha256"])
    pin(FIT / "contacts.json", plan["contacts_sha256"])
    rows, observed, K, q90, changed_rows = observations(plan, rule, native, manifest, archive, source_archive)
    require([r["source_frame_index"] for r in rows] == arcs["held_source_indices"] == [112, 336, 561, 777],
            "Protected source indices changed")
    roles, identities, contacts, included = {}, [], [], []
    for slot, row in enumerate(rows):
        frame, index = row["frame"], row["source_frame_index"]
        role = [side for side in ("front", "rear")
                if any(first <= index <= last for first, last in arcs[f"{side}_source_ranges"])]
        expected = "front" if frame in rule["front_frame_ids"] else "rear" if frame in rule["rear_frame_ids"] else None
        require(role == ([expected] if expected else []) and (not role) == (frame in UNAVAILABLE),
                f"Frozen arc role changed for {frame}")
        require(row["wh"] == plan["native_wh"] and np.array_equal(row["K_integer_centers"], K)
                and np.array_equal(row["w2c"], np.eye(4)), f"Camera contract changed for {frame}")
        image = pin(row["image"], row["source_sha256"])
        pin(row["mask"], row["mask_sha256"])
        identities.append(dict(frame=frame, source_index=index, source_image_sha256=row["source_sha256"],
                               fresh_mask_sha256=row["mask_sha256"], role=expected,
                               unavailable_reason=UNAVAILABLE.get(frame)))
        if role:
            roles[frame] = role[0]
            row["paths"] = {"body": dict(path=str(image), sha256=row["source_sha256"], native_integer_crop=True,
                                         resized=False, crop_original_xyxy=[0, 0, *row["wh"]])}
            contacts.append(row)
            included.append(slot)

    reference = next(r for r in result["frames"] if r["frame"] == publication["reference_frame"])
    with np.load(pin(reference["output"], reference["sha256"]), allow_pickle=False) as fitted:
        require(np.array_equal(beta, fitted["betas"]) and np.array_equal(K, fitted["K_integer_centers"]),
                "254 reference uses different beta or K")
    require(list(roles) == FRAMES and roles == dict(zip(FRAMES, ("front", "rear", "front"), strict=True))
            and reference["original_optimizer_and_monitor"] and reference["freeze_betas"],
            "Reference recipe or registered side assignment changed")
    require(not OUT.exists() or {path.name for path in OUT.iterdir()} == {"prepare-failure.json"},
            "The isolated registration directory already contains evaluation work")
    OUT.mkdir(exist_ok=True)
    archive = OUT / "fit-observations.npz"
    legacy.save_npz(archive, frame_ids=np.asarray(FRAMES), observed_native_keypoints=observed[included],
                    K_integer_centers=K, confidence_q90=q90)
    contact_file = OUT / "contacts.json"
    save(contact_file, dict(records=contacts))
    shutil.copyfile(shape, OUT / "shared-shape.json")
    registration = OUT / "preregistration.json"
    save(registration, dict(updated=legacy.now(), evaluation_only=True, delivered_run=str(RUN), source_pipeline=str(PIPELINE),
        protected_training_holdouts=HOLDOUTS, unavailable_frames=UNAVAILABLE, frames=identities,
        verified_training_observations=len(plan["source_frames"]), q90_frozen=True,
        archived_training_hand_patch=changed_rows, training_pins=pins, gaussian_optimizer_updates_allowed=0,
        runtime_provenance=dict(historical_runner_sha256=plan["runner_sha256"], runtime_runner_sha256=sha(runner_path),
            archived_worker_sha256=sha(ARCHIVED_WORKER), effective_front_rear_entry_diff_bitexact=True,
            runtime_device="windows_cpu", camera_body_stage_placement_refined=False),
        scope="Strict 254 recipe for three protected frames. Holdout RGB and fitted observations are for evaluation only. "
              "Front uses the archived optical upright seed; rear uses the archived default seed."))
    new_plan = dict(runner_sha256=sha(runner_path), worker_sha256=sha(ARCHIVED_WORKER),
        native_wh=plan["native_wh"], K_integer_centers=K.tolist(), shared_body_constraint=shared,
        source_frames=FRAMES, native_indices=[r["source_frame_index"] for r in contacts],
        retained_holdouts=HOLDOUTS, protected_training_holdouts=HOLDOUTS, unavailable_frames=UNAVAILABLE,
        roles=roles, upright_input_patch=plan["upright_input_patch"], baseline_observation_sha256=sha(archive),
        contacts_sha256=sha(contact_file), automatic_delivery=True, evaluation_only=True,
        preregistration_sha256=sha(registration), reference_receipt=reference, shared_shape_sha256=sha(shape))
    save(OUT / "plan.json", new_plan)
    save(STAGE / "plan.json", dict(new_plan, shared_shape=dict(vector=beta.tolist()), shards=[[i] for i in range(len(FRAMES))]))
    print("INDOOR_FORMAL_262_REGISTERED", registration, flush=True)


def worker(i):
    plan = read(STAGE / "plan.json")
    require(sha(ARCHIVED_WORKER) == plan["worker_sha256"] and sha(fixed.__file__) == plan["runner_sha256"],
            "Isolated worker or current runtime changed")
    original = module(ARCHIVED_WORKER, "archived_indoor_145")
    source = inspect.getsource(original.fit_frames)
    marker = "        assert np.array_equal(arrays['K_integer_centers'], K) and (arrays['vertices_camera'][..., 2] > .05).all()"
    require(source.count(marker) == 1, "Archived depth receipt marker changed")
    source = source.replace(marker, "        assert np.array_equal(arrays['K_integer_centers'], K)\n"
                            "        receipt['diagnostic_min_depth'] = float(arrays['vertices_camera'][..., 2].min())")
    scope = dict(vars(original))
    exec(compile(source, "<archived-008-depth-receipt>", "exec"), scope)
    original.fit_frames = scope["fit_frames"]
    adapter.original = original
    if plan["roles"][FRAMES[i]] == "front":
        from functools import partial
        entry.adapted_entry_source = partial(source_with_upright, base_source=entry.adapted_entry_source,
                                             patch=plan["upright_input_patch"])
    fixed.fit_shard(STAGE, i)


def fit():
    plan = read(OUT / "plan.json")
    registration = OUT / "preregistration.json"
    prereg = read(registration)
    require(not (OUT / "result.json").exists() and plan["preregistration_sha256"] == sha(registration)
            and plan["source_frames"] == FRAMES and plan["protected_training_holdouts"] == HOLDOUTS
            and all(sha(Path(path)) == digest for path, digest in prereg["training_pins"].items()),
            "Prepared 008 preregistration changed or already completed")
    require(sha(OUT / "fit-observations.npz") == plan["baseline_observation_sha256"]
            and sha(OUT / "contacts.json") == plan["contacts_sha256"]
            and sha(OUT / "shared-shape.json") == plan["shared_shape_sha256"]
            and all(read(STAGE / "plan.json")[key] == value for key, value in plan.items()),
            "Prepared evaluation files changed")
    require(not any((STAGE / frame).exists() for frame in FRAMES), "Existing evaluation work needs inspection")
    with ProcessPoolExecutor(max_workers=len(FRAMES)) as pool:
        list(pool.map(worker, range(len(FRAMES))))
    receipts = [read(STAGE / frame / "receipt.json") for frame in FRAMES]
    reference = plan["reference_receipt"]
    contacts = read(OUT / "contacts.json")["records"]
    with np.load(OUT / "fit-observations.npz", allow_pickle=False) as prepared, \
            np.load(relocate(reference["output"]), allow_pickle=False) as ref:
        diagnostics = []
        for i, (row, receipt) in enumerate(zip(contacts, receipts, strict=True)):
            frame = row["frame"]
            role = plan["roles"][frame]
            path = relocate(receipt["output"]).resolve()
            require(path.is_relative_to(STAGE) and receipt["frame"] == frame and receipt["source_index"] == row["source_frame_index"]
                    and sha(path) == receipt["sha256"] and sha(path.with_name("upstream-adapter.diff")) == WITNESSES[role][1]
                    and receipt["freeze_betas"] and receipt["native_crop_bitexact"] and receipt["original_optimizer_and_monitor"]
                    and receipt["independent_orientation_snapshots"] and receipt["shared_body_loss_calls"] > 0
                    and all(receipt[k] == reference[k] for k in ("official_commit", "source_hashes", "shared_body_target_sha256",
                                                                 "shared_body_weight", "shared_root_weight")),
                    f"Executed optimizer or camera-prefix source differs: {frame}")
            with np.load(path, allow_pickle=False) as pose:
                observed, projected = pose["observed_native_keypoints"], pose["joints_projected_native"]
                require(np.array_equal(observed, prepared["observed_native_keypoints"][i:i + 1])
                        and all(np.array_equal(pose[k], ref[k]) for k in ("betas", "faces", "K_integer_centers")),
                        f"Executed frozen pose inputs differ: {frame}")
                depth = float(pose["vertices_camera"][..., 2].min())
                height = row["initial_box_xyxy"][3] - row["initial_box_xyxy"][1]
                weights = np.ones(118)
                weights[list(legacy.IGNORED)] = 0.
                diagnostic = legacy.measure(observed[0], projected[0], height, weights)
                require(not np.any(observed[0, 19:25, 2]), "Unobserved foot support entered evaluation")
                diagnostics.append(dict(frame=frame, min_depth_m=depth, depth_valid=bool(np.isfinite(depth) and depth > .05),
                    body=diagnostic, feet_max_px=None, feet_gate_failed=None, camera_prefix_source_sha256=WITNESSES[role][1],
                    role=role))
    require(all(sha(Path(path)) == digest for path, digest in prereg["training_pins"].items()),
            "A protected source changed during evaluation")
    result = OUT / "result.json"
    save(result, dict(updated=legacy.now(), state="evaluation_fit_completed_pending_render", evaluation_only=True,
        frames=receipts, diagnostics=diagnostics, protected_training_holdouts=HOLDOUTS, unavailable_frames=UNAVAILABLE,
        plan_sha256=sha(OUT / "plan.json"), shared_shape_sha256=sha(OUT / "shared-shape.json"),
        preregistration_sha256=sha(registration), gaussian_optimizer_updates=0, quality_accepted=False))
    print("INDOOR_FORMAL_262_COMPLETE", result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "fit"))
    args = parser.parse_args()
    try:
        (prepare if args.mode == "prepare" else fit)()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        failure = OUT / f"{args.mode}-failure.json"
        if not failure.exists():
            save(failure, dict(updated=legacy.now(), evaluation_only=True, protected_training_holdouts=HOLDOUTS,
                               unavailable_frames=UNAVAILABLE, error=repr(error), gaussian_optimizer_updates=0))
        raise
