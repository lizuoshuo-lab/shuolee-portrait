"""Share the lower-neck transform from a registered side pose."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation
import torch
import tyro
from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker

from export_head_neck_frames import boundary_loops
from fit_guided_vhap import camera_gauge
from fit_native_skin_profile import camera, project
from fit_vhap_neck_sequence import contour_edges, face_area_check
from prepare_video_avatar import setup_vhap_cameras


def clamp_length(vectors, maximum):
    length = np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=1e-12)
    return vectors * np.minimum(1., maximum / length)


@torch.no_grad()
def load_neck_track(args):
    source = (args.sequence / "initialization.json").read_bytes()
    manifest = json.loads(source)
    records = manifest["frames"]
    names = [row["name"] for row in records]
    initialization_hash = sha256(source).hexdigest()
    pose = None
    if args.pose_report is not None:
        pose = json.loads(args.pose_report.read_bytes())
        rim = json.loads(args.rim_report.read_bytes())
        checkpoint = Path(pose["source"]["checkpoint"])
        if (Path(rim["current_checkpoint"]) != checkpoint
                or pose["source"]["initialization_sha256"] != initialization_hash):
            raise ValueError("Current neck pose and rim must derive from the same fit")
    else:
        cold = json.loads(args.cold_report.read_bytes())
        checkpoint = Path(cold["checkpoint"])
        if cold["frame_names"] != names or cold["seeded_count"] != len(names) or cold["trained_steps"] <= 0:
            raise ValueError("Fresh pose input must contain the complete VHAP run")
    with np.load(args.registered, allow_pickle=False) as tape:
        if (tape["filenames"].tolist() != names
                or tape["initialization_sha256"].item() != initialization_hash
                or tape["checkpoint_sha256"].item() != sha256(checkpoint.read_bytes()).hexdigest()):
            raise ValueError("The neck anchor must use the exact registered sequence")
        faces = tape["faces"]
        world_basis = tape["tracker_to_world_rotation"]
        origin, scale = tape["scene_origin"], float(tape["scene_scale"])
    args.output.mkdir()
    setup_vhap_cameras(camera_gauge(records, scale)[2])
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(args.sequence.parent), "--data.sequence", args.sequence.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(args.output / "native"),
        "--no-async-func",
    ])
    tracker = GlobalTracker(cfg)
    tracker.load_from_tracked_flame_params(Path(pose["checkpoint"]) if pose is not None else checkpoint)
    if pose is not None:
        with np.load(rim["checkpoint"], allow_pickle=False) as archive:
            tracker.static_offset.copy_(torch.from_numpy(archive["static_offset"]).to(tracker.static_offset))
    if not np.array_equal(tracker.flame.faces.cpu().numpy(), faces) or tracker.n_timesteps != len(names):
        raise ValueError("Stock FLAME topology or sequence changed")
    return tracker, records, faces, world_basis, origin, scale


@torch.no_grad()
def replay_neck_track(tracker):
    ids = torch.arange(tracker.n_timesteps, device=tracker.device)
    frames = [tracker.forward_flame(batch)[:2] for batch in ids.split(tracker.cfg.batch_size)]
    return [torch.cat(part).cpu().numpy() for part in zip(*frames, strict=True)]


@torch.no_grad()
def run(args):
    tracker, records, faces, world_basis, origin, scale = load_neck_track(args)
    names = [row["name"] for row in records]
    anchor = names.index(args.anchor)
    current, canonical = replay_neck_track(tracker)
    root_joint, neck_joint = np.einsum("jv,nvc->jnc", tracker.flame.J_regressor[:2].cpu().numpy(), canonical)
    arm = neck_joint - root_joint
    root = Rotation.from_rotvec(tracker.rotation.cpu().numpy())
    neck = Rotation.from_rotvec(tracker.neck_pose.cpu().numpy())
    translation = tracker.translation.cpu().numpy().copy()
    root_prior = root[anchor] * Rotation.from_rotvec(clamp_length(
        (root[anchor].inv() * root).as_rotvec(), np.radians(args.angle_slack_deg)))
    head = root * neck
    neck_update = root_prior.inv() * head
    root_translation = translation + root.apply(arm) - root_prior.apply(arm)
    for key, value in (("rotation", root_prior.as_rotvec()), ("neck_pose", neck_update.as_rotvec()),
                       ("translation", root_translation)):
        target = getattr(tracker, key)
        target.copy_(torch.from_numpy(value).to(target))
    rotated, _ = replay_neck_track(tracker)
    tracker.tb_writer.close()

    # Transport the observed anchor surface to each bounded lower-joint
    # position. A geodesic transition leaves the original head and expression intact.
    base = root_joint + root_translation
    attached = base[anchor] + clamp_length(base - base[anchor], args.base_slack_mm * .001)
    edges, _, protected, regions = contour_edges(faces, args.flame_masks)
    protected = np.unique(np.concatenate(
        [protected] + [regions[key] for key in ("lips", "left_eyeball", "right_eyeball")]))
    loop, = [value for value in boundary_loops(faces) if len(value) == 30]
    lengths = np.linalg.norm(canonical[anchor, edges[:, 0]] - canonical[anchor, edges[:, 1]], axis=1)
    graph = coo_matrix((lengths, edges.T), shape=(current.shape[1],) * 2).tocsr()
    rim_distance = dijkstra(graph, directed=False, indices=loop, min_only=True)
    face_distance = dijkstra(graph, directed=False, indices=protected, min_only=True)

    def smooth(value):
        value = np.clip(value, 0., 1.)
        return value ** 2 * (3. - 2. * value)

    weight = smooth(1. - rim_distance / .105) * smooth(face_distance / .030)
    turn = (root_prior * root_prior[anchor].inv()).as_matrix()
    donor = current[anchor].astype(np.float64)
    donor_fit = None
    donor_offset = np.zeros_like(donor)
    if args.anchor_matte is not None:
        matte = cv2.imread(str(args.anchor_matte), cv2.IMREAD_GRAYSCALE)
        if matte is None or not (matte >= 128).any():
            raise ValueError("Expected the native foreground matte for the anchor camera")
        exterior = cv2.distanceTransform((matte < 128).astype(np.uint8), cv2.DIST_L2, 5)
        local = (donor - donor[loop].mean(axis=0)) @ root[anchor].as_matrix()
        rear = smooth(-local[:, 2] / .040)
        rear *= smooth(1. - rim_distance / .130) * smooth(face_distance / .025)
        direction = rear[:, None] * root[anchor].apply([0., 0., 1.])
        ids = rear > .03
        k, r, c = camera(records[anchor])

        def overflow(amount):
            points = (donor[ids] + amount * direction[ids]) @ world_basis.T * scale + origin
            xy, _ = project(points, k, r, c)
            return map_coordinates(exterior, xy.T[::-1], order=1, mode="nearest")

        def cost(amount):
            return float(np.mean(np.maximum(overflow(amount) - .5, 0.) ** 2) + (amount / .06) ** 2 * .4)

        optimum = minimize_scalar(cost, bounds=(0., .065), method="bounded", options={"xatol": 1e-5})
        amount = float(optimum.x) if optimum.fun < cost(0.) else 0.
        donor_offset = amount * direction
        donor_fit = {
            "matte_sha256": sha256(args.anchor_matte.read_bytes()).hexdigest(),
            "max_requested_posterior_inset_flame_mm": amount * 1000.,
            "max_mesh_overflow_native_px_before": float(overflow(0.).max()),
            "max_mesh_overflow_native_px_after": float(overflow(amount).max()),
        }
    target = (donor - base[anchor])[None] @ turn.transpose(0, 2, 1) + attached[:, None]
    offset = (target - current) * weight[None, :, None]
    changed = (weight > 0) | np.any(donor_offset != 0, axis=1)
    edited_faces = faces[changed[faces].any(axis=1)]

    def safe(geometry):
        return geometry[0] >= .40 and geometry[1] <= 2.1 and geometry[2] == 0

    def transfer(start, update):
        def at(fraction):
            vertices = (start + fraction * update).astype(current.dtype)
            return vertices, face_area_check(current, vertices, edited_faces)

        final, geometry = at(1.)
        if safe(geometry):
            return final, geometry, 1.
        lo, hi = 0., 1.
        for _ in range(16):
            mid = (lo + hi) / 2.
            if safe(at(mid)[1]):
                lo = mid
            else:
                hi = mid
        fraction = lo * .99
        final, geometry = at(fraction)
        return final, geometry, fraction

    seated, geometry, fraction = transfer(current, offset)
    if donor_fit is not None:
        seated, geometry, shape_fraction = transfer(seated, donor_offset[None] @ turn.transpose(0, 2, 1))
        donor_fit["applied_shape_fraction"] = shape_fraction
        donor_fit["max_mesh_overflow_native_px_after"] = float(
            overflow(shape_fraction * donor_fit["max_requested_posterior_inset_flame_mm"] * .001).max())
    if not safe(geometry):
        raise RuntimeError("Could not transfer the neck shape without folding the mesh")
    if args.anchor_matte is None:
        np.testing.assert_array_equal(seated[anchor], current[anchor])
    np.testing.assert_array_equal(seated[:, protected], current[:, protected])

    def span(points):
        return float(np.max(np.linalg.norm(points - points[anchor], axis=2)) * 1000.)

    drift = []
    for record, before, after in zip(records, current, seated, strict=True):
        k, r, c = camera(record)
        uv0, _ = project(before[protected] @ world_basis.T * scale + origin, k, r, c)
        uv1, _ = project(after[protected] @ world_basis.T * scale + origin, k, r, c)
        drift.append(float(np.quantile(np.linalg.norm(uv1 - uv0, axis=1), .95)))
    report = {
        "anchor": args.anchor, "frames": len(names), "reference_geometry_retained": args.anchor_matte is None,
        "anchor_fit": donor_fit,
        "max_root_angles_before_after_deg": [
            float(np.degrees((root[anchor].inv() * root).magnitude()).max()),
            float(np.degrees((root_prior[anchor].inv() * root_prior).magnitude()).max())],
        "rim_max_distance_to_anchor_before_rotation_seated_flame_mm": [
            span(current[:, loop]), span(rotated[:, loop]), span(seated[:, loop])],
        "max_protected_face_p95_native_px": max(drift),
        "protected_face_p95_native_px_by_frame": dict(zip(names, drift, strict=True)),
        "triangle_area_ratio_min_max_and_flips": geometry,
        "applied_transfer_fraction": fraction,
        "world_frame_unchanged": True,
    }
    (args.output / "transfer.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "neck-anchor-geometry.npz", frames=names, old=current, rotated=rotated,
                        vertices=seated.astype(current.dtype), faces=faces, rim=loop, basis=world_basis,
                        origin=origin, scale=scale, root=base.astype(current.dtype))
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("by_frame")}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "registered", "pose-report", "rim-report", "flame-masks", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--anchor", required=True, help="Exact registered side-frame image name.")
    parser.add_argument("--anchor-matte", type=Path, help="Native integer-center person matte of the anchor frame.")
    parser.add_argument("--angle-slack-deg", type=float, default=1.5)
    parser.add_argument("--base-slack-mm", type=float, default=4.)
    args = parser.parse_args()
    if not (0 <= args.angle_slack_deg <= 15 and 0 <= args.base_slack_mm <= 30):
        parser.error("Use nonnegative lower-neck slack near the physical attachment.")
    run(args)
