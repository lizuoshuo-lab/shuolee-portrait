"""CPU audit of a delivered mesh-bound Gaussian snapshot and fitted mesh."""

import argparse
import json

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from review_full_sequence123 import relocate
from run_official_gaussians123 import now, read, save, sha


REPORT_NAME = "gaussian-mesh-surface-audit.json"
REPLAY_TOLERANCE_M = 5e-6
OPACITY_MINIMUM = .1
PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def columns(ply, names):
    values = np.column_stack([ply[name] for name in names]).astype(np.float64)
    require(np.isfinite(values).all(), "Gaussian PLY contains a nonfinite field.")
    return values


def percentiles(values):
    if values.size == 0:
        return None
    return dict(zip((str(q) for q in PERCENTILES), np.percentile(values, PERCENTILES).tolist()))


def binding_geometry(triangles, local_xyz, world_xyz, scales, rotations):
    """Replay upstream triangle binding, then project center and covariance onto its normal."""
    center = triangles.mean(axis=1)
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    length0 = np.linalg.norm(edge0, axis=1)
    require(np.all(length0 > 1e-10), "Bound triangle has a degenerate first edge.")
    axis0 = edge0 / length0[:, None]
    normal = np.cross(axis0, edge1)
    normal_length = np.linalg.norm(normal, axis=1)
    require(np.all(normal_length > 1e-10), "Bound triangle has a degenerate normal.")
    normal /= normal_length[:, None]
    axis2 = np.cross(axis0, normal)
    axis2 /= np.linalg.norm(axis2, axis=1, keepdims=True)
    height = np.abs(np.sum(axis2 * edge1, axis=1))
    scale = np.maximum((length0 + height) * .5, 1e-8)
    basis = np.stack((axis0, normal, axis2), axis=2)
    replayed = center + scale[:, None] * (basis @ local_xyz[..., None])[..., 0]
    signed_distance = np.sum((world_xyz - center) * normal, axis=1)
    # sqrt(n^T Sigma n), with Sigma = R diag(s^2) R^T.
    normal_components = (normal[:, None, :] @ rotations)[:, 0, :]
    sigma_normal = np.linalg.norm(normal_components * scales, axis=1)
    return replayed, signed_distance, sigma_normal


def audit_snapshot(out, step, source=None):
    """Return metrics without writing any files. A source receipt binds the delivered variant."""
    out = relocate(out).resolve()
    require(type(step) is int and step > 0, "A positive snapshot step is required.")
    snapshot = out / "gaussian-run" / f"step-{step:06d}"
    files = [snapshot / name for name in ("result.json", "world.ply", "bound-local.ply",
                                           "binding.npz", "current-meshes.npz")]
    result_path, world_path, local_path, binding_path, meshes_path = files
    plan_path = out / "plan.json"
    bridge_path = out / "smplify-gaussian-interface-receipt.json"
    cache_path = out / "fit/surface-cache.npz"
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in (*files, plan_path, bridge_path, cache_path)}
    result, plan, bridge = (read(p) for p in (result_path, plan_path, bridge_path))
    reference, ref_slot = result["reference_frame"], result["reference_slot"]
    require(result["step"] == step and result["world_ply_sha256"] == hashes[world_path.relative_to(out).as_posix()]
            and bridge["surface_cache_sha256"] == hashes["fit/surface-cache.npz"]
            and reference == bridge["reference_frame"] == plan["reference_frame"],
            "Snapshot or fitted mesh does not match its receipt.")
    if source is not None:
        recorded = source["training_files_sha256"]
        pinned = {relocate(p).resolve(): value for p, value in plan["source_hashes"].items()}
        require(source["terminal_verified"] is True and source["step"] == step
                and source["reference_frame"] == reference and source["points"] == result["points"]
                and source["source_ply_sha256"] == result["world_ply_sha256"]
                and pinned.get(cache_path) == hashes["fit/surface-cache.npz"]
                and pinned.get(bridge_path) == hashes[bridge_path.relative_to(out).as_posix()]
                and all(recorded.get(p.relative_to(out).as_posix()) == hashes[p.relative_to(out).as_posix()]
                        for p in (plan_path, result_path, binding_path, meshes_path)),
                "Delivered source receipt differs from the audit inputs.")

    with np.load(meshes_path, allow_pickle=False) as saved, np.load(binding_path, allow_pickle=False) as bound, \
            np.load(cache_path, allow_pickle=False) as cache:
        frames, vertices, faces = (saved[k] for k in ("frame_ids", "vertices_camera", "faces"))
        triangle_ids, offset = bound["triangle_ids"], bound["static_offset"]
        base, protected, clothing = (cache[k] for k in ("base_vertices", "protected_face_vertices", "clothing_vertices"))
        require(np.array_equal(frames, bound["frame_ids"]) and np.array_equal(frames, cache["frame_ids"])
                and np.array_equal(faces, cache["faces"]) and frames.ndim == 1
                and len(set(frames.tolist())) == len(frames) == result["frames_rendered"] == bridge["frames"]
                and 0 <= ref_slot < len(frames) and str(frames[ref_slot]) == reference,
                "Inconsistent mesh topology or reference-pose roster.")
    require(vertices.ndim == 3 and vertices.shape[0] == len(frames) and vertices.shape[2] == 3
            and vertices.shape == base.shape and vertices.shape[1:] == offset.shape
            and all(np.isfinite(v).all() for v in (vertices, base, offset)),
            "Invalid current mesh, fitted mesh or canonical offsets.")
    require(protected.dtype == clothing.dtype == np.bool_
            and protected.shape == clothing.shape == (vertices.shape[1],)
            and faces.ndim == 2 and faces.shape[1] == 3 and faces.size > 0
            and np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0 and faces.max() < vertices.shape[1],
            "Invalid cached vertex flags or mesh topology.")

    local = PlyData.read(local_path)["vertex"].data
    world = PlyData.read(world_path)["vertex"].data
    require(triangle_ids.ndim == 1 and np.issubdtype(triangle_ids.dtype, np.integer)
            and len(local) == len(world) == len(triangle_ids) == result["points"] > 0
            and triangle_ids.min() >= 0 and triangle_ids.max() < len(faces)
            and np.array_equal(local["binding_0"], triangle_ids), "Invalid bound point order or triangle indices.")
    xyz_local = columns(local, ("x", "y", "z"))
    xyz_world = columns(world, ("x", "y", "z"))
    quat = columns(world, [f"rot_{i}" for i in range(4)])
    require(np.all(np.linalg.norm(quat, axis=1) > 1e-12), "Invalid world quaternion.")
    rotations = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    scales = np.exp(columns(world, [f"scale_{i}" for i in range(3)]))
    require(np.isfinite(scales).all() and np.all(scales > 0), "Invalid world Gaussian scales.")
    opacity = np.exp(-np.logaddexp(0, -columns(world, ("opacity",))[:, 0]))
    triangles = vertices[ref_slot, faces[triangle_ids]].astype(np.float64)
    replayed, distance, normal_sigma = binding_geometry(triangles, xyz_local, xyz_world, scales, rotations)
    replay_error = float(np.abs(replayed - xyz_world).max())
    require(replay_error < REPLAY_TOLERANCE_M and np.isfinite(normal_sigma).all(),
            f"Saved binding replay failed: {replay_error:.9g} m.")

    # Same policy masks used by the trainer. Give protected boundary faces priority.
    vertex_regions = {"protected_head": protected, "clothing": clothing & ~protected,
                      "other": ~(clothing | protected)}
    head_faces = protected[faces].any(axis=1)
    clothing_faces = vertex_regions["clothing"][faces].any(axis=1) & ~head_faces
    face_regions = {"all": np.ones(len(faces), dtype=bool), "protected_head": head_faces,
                    "clothing": clothing_faces, "other": ~(head_faces | clothing_faces)}
    subsets = {"all": np.ones(len(world), dtype=bool), "opacity_at_least_0.1": opacity >= OPACITY_MINIMUM}
    gaussian_groups = {}
    for region, face_keep in face_regions.items():
        groups = {}
        face_count = int(face_keep.sum())
        for subset, keep in subsets.items():
            keep = keep & face_keep[triangle_ids]
            covered = len(np.unique(triangle_ids[keep]))
            groups[subset] = dict(
                points=int(keep.sum()), region_faces=face_count, faces_with_binding=covered,
                fraction_faces_with_binding=covered / face_count if face_count else None,
                signed_center_to_bound_face_plane_mm=percentiles(distance[keep] * 1000),
                absolute_center_to_bound_face_plane_mm=percentiles(np.abs(distance[keep]) * 1000),
                normal_sigma_mm=percentiles(normal_sigma[keep] * 1000),
                base_opacity=percentiles(opacity[keep]))
        gaussian_groups[region] = groups

    displacement = np.linalg.norm(vertices.astype(np.float64) - base, axis=2) * 1000
    canonical_mm = np.linalg.norm(offset.astype(np.float64), axis=1) * 1000
    vertex_regions["all"] = np.ones(vertices.shape[1], dtype=bool)
    displacement_groups = {region: dict(vertices=int(keep.sum()),
        reference_pose_displacement_mm=percentiles(displacement[ref_slot, keep]),
        all_poses_displacement_mm=percentiles(displacement[:, keep]),
        canonical_offset_mm=percentiles(canonical_mm[keep])) for region, keep in vertex_regions.items()}
    return dict(schema="automatic-sh0-surface-audit-v1", updated=now(), step=step, reference_frame=reference,
                points=len(world), mesh_faces=len(faces), mesh_vertices=vertices.shape[1], fitted_pose_count=len(frames),
                bound_local_to_world_replay_max_abs_m=replay_error, replay_tolerance_m=REPLAY_TOLERANCE_M,
                opacity_audit_threshold=OPACITY_MINIMUM, gaussian_groups=gaussian_groups, mesh_displacement=displacement_groups,
                cache_mask_overlap_vertices=int((clothing & protected).sum()),
                region_rule="Disjoint masks. Protected vertices take priority; faces touching any protected vertex are protected, "
                            "then faces touching any active clothing vertex are clothing; remaining faces are other.",
                measurement_scope="Native reference-pose checkpoint gauge. Center distances use the stored bound triangle plane "
                                  "and saved winding. Normal sigma is one standard deviation along that plane normal.",
                limitations=["Triangle-plane distance does not measure distance to the finite triangle, the nearest mesh surface or physical ground truth.",
                             "Opacity is base Gaussian alpha. The 0.1 subset measures neither visibility nor accumulated transmittance.",
                             "Binding coverage counts assigned triangle IDs, with no claims about rendered coverage.",
                             "Mesh displacement compares identical fitted poses before and after the shared offset in the model's metre scale."],
                source_hashes=hashes, quality_accepted=False, delivery_verified=False, cpu_only=True)


def audit_delivered(out):
    out = relocate(out).resolve()
    publication_path = out / "delivery/publication.json"
    publication_sha = sha(publication_path)
    publication = read(publication_path)
    require(publication.get("state") == "delivered_pending_quality_review"
            and publication.get("format_passed") is True and publication["run"] == out.name
            and type(publication["step"]) is int and publication["step"] > 0,
            "A completed automatic SH0 publication is required.")
    folder = out / "delivery" / f"step-{publication['step']:06d}"
    delivered = {name: sha(folder / name) for name in ("source-receipt.json", "point_cloud.ply", "cameras.json", "scene.zip")}
    require(delivered["source-receipt.json"] == publication["source_receipt_sha256"]
            and delivered["point_cloud.ply"] == publication["ply_sha256"]
            and delivered["cameras.json"] == publication["camera_sha256"]
            and delivered["scene.zip"] == publication["package_sha256"], "Delivered publication hashes changed.")
    source = read(folder / "source-receipt.json")
    require(source["reference_frame"] == publication["reference_frame"] and source["points"] == publication["points"],
            "Delivered source and publication identify different models.")
    report = audit_snapshot(out, publication["step"], source)
    require(sha(publication_path) == publication_sha, "Publication changed while the audit was running.")
    report.update(delivery_verified=True, publication_sha256=publication_sha, delivery_hashes=delivered)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--dry-run", action="store_true", help="Verify and report without saving.")
    args = parser.parse_args()
    out = relocate(args.run_dir).resolve()
    report = audit_delivered(out)
    if not args.dry_run:
        save(out / REPORT_NAME, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
