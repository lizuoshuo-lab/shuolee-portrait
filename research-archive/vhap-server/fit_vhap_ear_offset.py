"""Fit one shared ear-local FLAME offset with calibrated native observations."""

from argparse import ArgumentParser
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from fit_guided_vhap import camera_gauge
from fit_native_skin_profile import camera, file_hash
from prepare_video_avatar import setup_vhap_cameras
from probe_vhap_profile_translation import describe, visible_points
from review_smplx_head_neck import project


EAR_IDS = (6, 616)
EAR_REGIONS = ("left_ear", "right_ear")
MAX_OFFSET = .006


def offset_field(tracker, masks_path):
    with masks_path.open("rb") as handle:
        masks = pickle.load(handle, encoding="latin1")
    mesh = tracker.flame
    template = mesh.v_template.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    edges = np.sort(np.concatenate([faces[:, [i, (i + 1) % 3]] for i in range(3)]), axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(template[edges[:, 0]] - template[edges[:, 1]], axis=1)
    graph = coo_matrix((lengths, edges.T), shape=(len(template), len(template))).tocsr()
    direction = np.zeros_like(template)
    for region, anchor in zip(EAR_REGIONS, EAR_IDS, strict=True):
        boundary = np.asarray(masks[region], dtype=int)
        if anchor not in boundary:
            raise ValueError(f"Anatomical ear anchor {anchor} is not in {region}")
        near = dijkstra(graph, directed=False, indices=anchor, limit=.028)
        phase = np.minimum(near[boundary] / .028, 1.)
        direction[boundary, 2] = .5 * (1. + np.cos(np.pi * phase))
    # Hold skeletal joints and facial barycentric supports fixed.
    regressors = mesh.J_regressor.detach().cpu().numpy()
    direction[np.any(regressors != 0., axis=0)] = 0.
    embedding = mesh.full_lmk_faces_idx.detach().cpu().numpy().reshape(-1)[17:68]
    direction[np.unique(faces[embedding])] = 0.
    if not np.array_equal(direction[list(EAR_IDS), 2], np.ones(2)):
        raise ValueError("Ear offset would disturb fixed facial or skeletal supports")
    return torch.as_tensor(direction[None], device=tracker.device)


@torch.no_grad()
def run(args):
    initialization = args.sequence / "initialization.json"
    context = json.loads(initialization.read_text())
    records = context["frames"]
    filenames = [row["name"] for row in records]
    scale = float(context["scene_scale"])
    archive = np.load(args.registered, allow_pickle=False)
    if (archive["filenames"].tolist() != filenames
            or archive["checkpoint_sha256"].item() != file_hash(args.checkpoint)
            or archive["initialization_sha256"].item() != file_hash(initialization)):
        raise ValueError("Expected exact registered VHAP run and camera manifest")
    setup_vhap_cameras(camera_gauge(records, scale)[2])
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "native"),
        "--no-async-func",
    ])
    tracker = GlobalTracker(cfg)
    checkpoint = np.load(args.checkpoint, allow_pickle=False)
    if (tracker.n_timesteps != len(records)
            or checkpoint["timestep_id"].tolist() != list(tracker.dataset.timestep_ids)):
        raise ValueError("VHAP loader frame order mismatch")
    tracker.load_from_tracked_flame_params(args.checkpoint)
    start = tracker.static_offset.clone()
    direction = offset_field(tracker, args.flame_masks)
    transform, origin, faces = (archive[key] for key in ("tracker_to_world_rotation", "scene_origin", "faces"))
    if not np.array_equal(faces, tracker.flame.faces.cpu().numpy()):
        raise ValueError("FLAME triangulation has changed")
    affected = direction[0, :, 2].cpu().numpy() > 0.
    indices = torch.arange(tracker.n_timesteps, device=tracker.device)

    def evaluate(amount):
        tracker.static_offset.copy_(start + amount * direction)
        batches = [tracker.forward_flame(ids)[:3:2] for ids in indices.split(cfg.batch_size)]
        return tuple(torch.cat(tensors).cpu().numpy() for tensors in zip(*batches, strict=True))

    base, base_lm = evaluate(0.)
    if not np.allclose(base, archive["vertices_tracker"], atol=1e-6, rtol=0.):
        raise RuntimeError("Stock VHAP replay does not match the registered source")
    moved, _ = evaluate(MAX_OFFSET)
    world = base @ transform.T * scale + origin
    shifts = (moved[:, EAR_IDS] - base[:, EAR_IDS]) @ transform.T * scale
    detections = np.load(args.rtmw, allow_pickle=False)
    if (detections["names"].tolist() != filenames
            or detections["joint_names"][3:5].tolist() != list(EAR_REGIONS)):
        raise ValueError("RTMW ear labels or source frame order changed")
    control = json.loads(args.translation_control.read_text())
    train, held = [], []
    for i, (row, mesh) in enumerate(zip(records, world, strict=True)):
        sign = int(np.sign(row["baseline_yaw_deg"]))
        if abs(row["baseline_yaw_deg"]) < 55.:
            continue
        K, R, C = camera(row)
        cam = (mesh - C) @ R
        eye = int(np.argmin(cam[list(EAR_IDS), 2]))
        site = EAR_IDS[eye]
        score = detections["scores"][i, 3 + eye]
        flip = detections["flip_disagreement"][i, 3 + eye]
        if score <= 0. or flip > 3. or not visible_points(cam, cam[[site]], faces, K)[0]:
            continue
        suites = control["holdout_side"][str(sign)]
        target = detections["keypoints"][i, 3 + eye]
        entry = (row["name"], sign, EAR_REGIONS[eye], cam[site], shifts[i, eye] @ R, K, target, float(flip))
        if row["name"] in suites["fit"]:
            train.append(entry)
        elif row["name"] in suites["hold"]:
            held.append(entry)

    def residual(ratio, group):
        return np.array([project((point + ratio * delta)[None], K)[0][0] - target
                         for _, _, _, point, delta, K, target, _ in group])

    if any(not [x for x in dataset if x[1] == side] for dataset in (train, held) for side in (-1, 1)):
        raise ValueError("Both anatomical sides need visible fitting and held ear observations")

    def loss(ratio):
        value = np.linalg.norm(residual(ratio, train), axis=1)
        return np.mean(4. * (np.sqrt(1. + (value / 2.)**2) - 1.))

    # FLAME replay is linear in this field because the joint regression is fixed.
    # Bound the solve by triangle quality before reading held-out error.
    selected_faces = faces[affected[faces].any(axis=1)]

    def face_normals(vertices):
        p, q, s = vertices[:, selected_faces].transpose(2, 0, 1, 3)
        return np.cross(q - p, s - p)

    base_normals = face_normals(base)
    base_lengths = np.linalg.norm(base_normals, axis=-1)

    def geometry(vertices):
        normals = face_normals(vertices)
        area = np.linalg.norm(normals, axis=-1) / np.maximum(base_lengths, 1e-12)
        opposed = int((np.sum(base_normals * normals, axis=-1) <= 0.).sum())
        return float(area.min()), float(area.max()), opposed

    def feasible(vertices):
        lo, hi, opposed = geometry(vertices)
        return not opposed and lo >= .65 and hi <= 1.60

    limit = 1.
    if not feasible(moved):
        lo, hi = 0., 1.
        for _ in range(14):
            mid = (lo + hi) * .5
            if feasible(base + mid * (moved - base)):
                lo = mid
            else:
                hi = mid
        limit = lo * .995
    fit = minimize_scalar(loss, bounds=(0., limit), method="bounded", options={"xatol": 1e-6})
    ratio = float(fit.x)
    amount = ratio * MAX_OFFSET
    current, current_lm = evaluate(amount)
    area_lo, area_hi, opposed = geometry(current)
    protected = float(np.max(abs(current_lm[:, 17:68] - base_lm[:, 17:68])))
    unaffected = float(np.max(abs(current[:, ~affected] - base[:, ~affected])))
    if not feasible(current) or max(protected, unaffected) > 2e-6:
        raise RuntimeError(f"Ear mesh guard failed: {opposed=}, area={area_lo:.3f}..{area_hi:.3f}, {protected=}, {unaffected=}")
    report = {
        "source": str(args.checkpoint), "frames": len(records), "ear_vertices": list(EAR_IDS),
        "shared_forward_offset_flame_m": amount, "limit_reached": ratio >= .999 * limit,
        "initial_max_offset_flame_m": MAX_OFFSET, "geometry_limit_flame_m": limit * MAX_OFFSET,
        "maximum_facial_landmark_delta_tracker": protected, "maximum_outside_vertex_delta_tracker": unaffected,
        "area_ratio": [area_lo, area_hi], "opposed_triangles": opposed,
        "notes": "Held RTMW ear point proxy only; true tragus and glasses geometry still need independent support.",
        "fit_frames": [row[0] for row in train], "held": {},
    }
    for side in (-1, 1):
        group = [x for x in held if x[1] == side]
        report["held"][str(side)] = {"frames": [x[0] for x in group],
                                      "before": describe(residual(0., group)),
                                      "after": describe(residual(ratio, group))}
    for entry, baseline, candidate in zip(held, residual(0., held), residual(ratio, held), strict=True):
        if entry[0] == "00850.jpg":
            report["00850"] = {"ear": entry[2], "flip_disagreement": entry[-1],
                               "before_px": float(np.linalg.norm(baseline)), "after_px": float(np.linalg.norm(candidate))}
    report["status"] = "candidate_only"
    (args.output / "ear-fit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tracker.timestep = len(records)
    tracker.evaluate(make_visualization=True, epoch=0)
    tracker.tb_writer.close()
    grid = tracker.out_dir / "eval_0/image_grid"
    if {p.name for p in grid.glob("*.jpg")} != {f"frame_{i:05d}.jpg" for i in range(len(records))}:
        raise RuntimeError("Incomplete native VHAP grid")
    report.update(grid=str(grid), candidate_checkpoint=str(tracker.out_dir / "tracked_flame_params_0.npz"))
    (args.output / "ear-fit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "checkpoint", "registered", "rtmw", "flame-masks", "translation-control", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())
