"""Fit one shared 021 seated body pose across the frozen 16-view preview roster."""
import fcntl
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from PIL import Image

import control_smplifyx_root_seed123 as seed_interface
import run_grass_selected_views124 as shared
import train_video21_exploratory179 as preview
from pose_training_gate151 import require
from review_full_sequence123 import relocate
from run_official_sequence123 import save_npz


ROOT = shared.ROOT
OUT = ROOT / "video21-shared-seat179"
SEED = ROOT / "seated021-fit148/right-hand-surface170"
ROOT_SEEDS = ROOT / "seated021-fit148/world-root-coherent173"
ANCHOR = "frame_000051"


def run():
    import torch

    read, save, sha = shared.read, shared.save, shared.sha
    selection_path = preview.OUT / "selection.json"
    selection = read(selection_path)
    frames = selection["selected_frames"]
    fit_plan_path = preview.SOURCE / "plan.json"
    fit_plan = read(fit_plan_path)
    heldouts = fit_plan["retained_holdouts"]
    require(len(frames) == len(set(frames)) == 16 and not set(frames + [ANCHOR]) & set(heldouts),
            "Use the fixed 16-view roster and preserve the original holdouts.")
    preview_rows = preview.OUT / "data/training-inputs.json"
    contact = {row["frame"]: row for row in read(preview_rows)["records"]}
    rows = [contact[name] for name in frames]
    observations = preview.SOURCE / "fit-observations.npz"
    require(sha(observations) == fit_plan["baseline_observation_sha256"], "Frozen 021 observations changed.")
    with np.load(observations, allow_pickle=False) as archive:
        indices = [archive["frame_ids"].tolist().index(name) for name in frames]
        target = archive["observed_native_keypoints"][indices]
        K = archive["K_integer_centers"]
    for row in rows:
        require(np.array_equal(K, row["K_integer_centers"]) and sha(row["image"]) == row["source_sha256"]
                and sha(row["mask"]) == row["mask_sha256"], f"Source pixels or K changed for {row['frame']}.")
    fit = read(preview.SOURCE / "result.json")
    require(fit["plan_sha256"] == sha(fit_plan_path) and sha(preview.SOURCE / "result.json") == selection["source_fit_sha256"],
            "Preview roster was not generated from this fit.")
    beta = np.asarray(fit["fixed_shape"], np.float32)

    prior_plan_path = SEED / "plan.json"
    prior_plan = read(prior_plan_path)
    prior_control = SEED / "control/result.json"
    prior = read(SEED / "result.json")
    old_review = SEED / "review/visual-decision.json"
    require(prior["completed"] and prior["plan_sha256"] == sha(prior_plan_path)
            and prior["fit_result_sha256"] == sha(prior_control)
            and read(old_review)["actual_visual_review"], "Missing the original reviewed 170 seed.")
    row = next(row for row in read(prior_control)["frames"] if row["frame"] == ANCHOR)
    seed_path = relocate(row["output"])
    anchor_index = fit_plan["native_indices"][fit_plan["source_frames"].index(ANCHOR)]
    require(prior_plan["video_sha256"] == fit_plan["source_video_sha256"]
            and row["source_index"] == anchor_index and sha(seed_path) == row["sha256"],
            "Seed frame does not match 021's original video and source index.")
    with np.load(seed_path, allow_pickle=False) as archive:
        require(np.array_equal(archive["K_integer_centers"], K), "170 seed belongs to a different camera K.")
        latent = archive["pose_embedding"].copy()
    require(latent.shape == (1, 32) and np.isfinite(latent).all(), "Expected the saved 170 VPoser latent.")

    roots_path = ROOT_SEEDS / "plan.json"
    roots_plan = read(roots_path)
    require(roots_plan["video_sha256"] == fit_plan["source_video_sha256"]
            and np.array_equal(roots_plan["K_integer_centers"], K), "173 root seeds use different source pixels.")
    root_indices = dict(zip(roots_plan["source_frames"], roots_plan["native_indices"], strict=True))
    require(all(root_indices[r["frame"]] == r["source_frame_index"] for r in rows), "173 root frame identity differs.")
    roots = np.asarray([roots_plan["initial_roots"][name]["body_fit_initial_root"][0] for name in frames], np.float32)
    require(roots.shape == (16, 3) and np.isfinite(roots).all(), "Invalid 173 camera-relative orientation seeds.")
    reference = read(preview.OUT / "plan.json")["reference_frame"]
    bound = [Path(__file__), Path(shared.__file__), Path(shared.adapter.__file__), Path(shared.entry.__file__),
             Path(seed_interface.__file__), selection_path, preview_rows, fit_plan_path, preview.SOURCE / "result.json",
             observations, prior_plan_path, SEED / "result.json", prior_control, old_review, seed_path, roots_path]

    OUT.mkdir(exist_ok=False)
    save(OUT / "plan.json", dict(updated=shared.now(), frames=frames, holdouts=heldouts, seed_frame=ANCHOR,
         seed_source_index=anchor_index, seed_pose_sha256=sha(seed_path), reference_frame=reference,
         beta=beta.tolist(), native_K=K.tolist(), initial_roots=roots.tolist(), source_hashes={str(p): sha(p) for p in bound},
         shared_local_pose=True, source_pose_policy="170 body latent initialization; 173 camera-relative root initialization.",
         per_view_free=["root orientation", "translation", "hands", "expression", "jaw", "eyes"],
         observation_weights_unchanged=True, budget_seconds=600, gaussian_started=False, quality_accepted=False))
    save(OUT / "launch.json", dict(updated=shared.now(), pid=os.getpid(), command=sys.argv,
         start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]))
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        controls = shared.adapter.verify(shared.OFFICIAL)
        body, vposer = shared.make_models(len(frames))
        entry, delta = shared.adapter.make_entry(len(frames), roots)
        (OUT / "selected-view-interface.diff").write_text(delta, encoding="utf-8")
        config = entry.__globals__["load_official_config"]
        shared.entry._root_seed_original_source = shared.adapter.source
        entry.__globals__.update(adapted_entry_source=seed_interface.adapted_source,
            load_official_config=lambda root: dict(config(root), _seed_pose_embedding=latent,
                                                   _seed_global_orient=roots, _seed_root_enabled=True))
        save(OUT / "preflight.json", controls)
        print("VIDEO21_SHARED_SEAT_FIT_STARTED", len(frames), ANCHOR, flush=True)
        image_row = contact[reference]
        with Image.open(image_row["image"]) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
        arrays, receipt = entry(official_root=shared.OFFICIAL, image=image, keypoints=target,
            K_integer_centers=K, body_model=body, vposer2=vposer, result_path=OUT / "official-result.pkl",
            shared_betas=beta, freeze_betas=True, device="cuda", diff_path=OUT / "executed-interface.diff")
        require(arrays["pose_embedding"].shape == (1, 32) and arrays["body_pose_axis_angle"].shape == (1, 63)
                and np.array_equal(arrays["betas"], np.repeat(beta, len(frames), axis=0))
                and np.array_equal(arrays["K_integer_centers"], K), "Shared pose, shape or K invariant failed.")
        require(np.array_equal(shared.entry._root_seed_trace["body_fit_initial_root"], roots),
                "The verified root seeds were not consumed.")
        archive = OUT / "selected-fit.npz"
        save_npz(archive, **arrays, frame_ids=np.asarray(frames), observed_native_keypoints=target)
        receipt.update(seconds=time.monotonic()-started, frames=frames, seed_frame=ANCHOR,
            initialization="One 170 saved body latent, 16 173 camera-relative root seeds.",
            shared_local_pose=True, shared_betas_frozen=True, fit_sha256=sha(archive),
            actual_visual_review=False, gaussian_started=False, quality_accepted=False)
        save(OUT / "result.json", receipt)
        print("VIDEO21_SHARED_SEAT_FIT_COMPLETE", receipt["seconds"], flush=True)
    except BaseException as exc:
        save(OUT / "failure.json", dict(error=repr(exc), seconds=time.monotonic()-started, quality_accepted=False))
        raise


if __name__ == "__main__":
    with open("/tmp/shuolee-native122-gpu.lock", "a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("600 second fit budget")))
        signal.alarm(600)
        try:
            run()
        finally:
            signal.alarm(0)
