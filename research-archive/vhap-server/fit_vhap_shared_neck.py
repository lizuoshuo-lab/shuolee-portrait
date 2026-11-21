"""Fit a shared lower neck in the registered scene from multiple real views."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from export_head_neck_frames import boundary_loops
from fit_native_skin_profile import camera, project
from fit_vhap_neck_sequence import contour_edges, face_area_check, load_observations, neck_metrics, profile_values
from transfer_vhap_neck_anchor import clamp_length, load_neck_track, replay_neck_track


def tolerance(values, millimetres):
    return clamp_length(values, millimetres * .001)


def audit_contours(observations, projected, names, yaw, limit_px):
    """Audit all observed held profile rows, including model crossings that disappear."""
    areas = {}
    by_point = []
    for observation, x_pred in zip(observations, projected, strict=True):
        index, _, x, y, _, region, provider = observation
        key = (region, int(np.sign(yaw[index])))
        error = float(abs(x_pred - x)) if np.isfinite(x_pred) else None
        areas.setdefault(key, []).append(error)
        by_point.append({"frame": names[index], "anatomy": region, "yaw_sign": key[1],
                         "native_xy": [x, y], "source": provider, "error_px": error})
    regions = {}
    for (region, sign), data in sorted(areas.items()):
        finite = [value for value in data if value is not None]
        regions[f"{region}/{sign}"] = {"total": len(data), "missing": len(data) - len(finite),
                                       "over_limit": sum(value > limit_px for value in finite),
                                       "max_px": max(finite, default=None)}
    coverage = all((region, sign) in areas for region in ("posterior", "anterior") for sign in (-1, 1))
    errors = [item["error_px"] for item in by_point if item["error_px"] is not None]
    missing = len(by_point) - len(errors)
    over = sum(error > limit_px for error in errors)
    return {"limit_px": limit_px, "total": len(by_point), "missing": missing, "over_limit": over,
            "max_native_px": max(errors, default=None), "both_sides_covered": coverage,
            "passed": coverage and missing == 0 and over == 0,
            "regions": regions, "observations": by_point}


def run(args):
    tracker, records, faces, basis, origin, scale = load_neck_track(args)
    current, canonical = replay_neck_track(tracker)
    root = Rotation.from_rotvec(tracker.rotation.detach().cpu().numpy())
    tracker.tb_writer.close()
    names = [record["name"] for record in records]
    yaw = np.asarray([record["baseline_yaw_deg"] for record in records])
    split = json.loads((args.split or args.pose_report).read_bytes())
    held = {names.index(name) for name in split["held_frames"]}
    fit_ids = [index for index in range(len(records)) if index not in held]
    # Initialize from the observed sequence in the common gauge. No pose is
    # assigned the role of a template, including in the geodesic computation.
    groups = [np.asarray([i for i in fit_ids if selected[i]]) for selected in
              (yaw <= -55., np.abs(yaw) < 55., yaw >= 55.)]
    if min(len(group) for group in groups) < 3 or not held:
        raise ValueError("Shared neck fitting requires frontal, two-sided and held observations")
    template = np.median(np.stack([np.median(current[group], axis=0) for group in groups]), axis=0)
    neutral = np.median(canonical[fit_ids], axis=0)
    orientation = Rotation.from_matrix(np.stack([root[group].mean().as_matrix() for group in groups])).mean()
    edges, eligible, protect, regions = contour_edges(faces, args.flame_masks)
    protect = np.unique(np.concatenate([protect] + [regions[key] for key in ("lips", "left_eyeball", "right_eyeball")]))
    ring, = [loop for loop in boundary_loops(faces) if len(loop) == 30]
    lengths = np.linalg.norm(neutral[edges[:, 0]] - neutral[edges[:, 1]], axis=1)
    graph = coo_matrix((lengths, edges.T), shape=(len(neutral), len(neutral))).tocsr()
    rim_d = dijkstra(graph, directed=False, indices=ring, min_only=True)
    face_d = dijkstra(graph, directed=False, indices=protect, min_only=True)

    def smooth(x):
        x = np.clip(x, 0., 1.)
        return x*x*(3.-2.*x)

    weight = smooth(1. - rim_d / .135) * smooth(face_d / .030)
    if not np.array_equal(weight[ring], np.ones(len(ring))):
        raise ValueError("The shared attachment must cover the complete unprotected lower rim")
    residual = tolerance((current - template) * .12, args.slack_mm)
    center = template[ring].mean(axis=0)
    local = orientation.inv().apply(template - center)
    posterior = smooth(-local[:, 2] / .035) * smooth(rim_d / .040)
    lower = smooth(1. - rim_d / .035)
    rear_lower = smooth(-local[:, 2] / .035) * lower
    active_faces = faces[(weight[faces] > 0).any(axis=1)]
    views = [camera(record) for record in records]
    fringe_base = None
    if args.rim_fringe_from:
        with np.load(args.rim_fringe_from, allow_pickle=False) as saved:
            if (saved["frames"].tolist() != names or not np.array_equal(saved["faces"], faces)
                    or not np.array_equal(saved["old"], current)
                    or not np.allclose(saved["basis"], basis, atol=1e-8, rtol=0.)
                    or not np.allclose(saved["origin"], origin, atol=1e-8, rtol=0.)
                    or not np.isclose(float(saved["scale"]), scale, atol=1e-8, rtol=0.)):
                raise ValueError("Rim fringe input differs from the registered mesh sequence")
            fringe_base = saved["vertices"], saved["shared"]
        skin_distance = dijkstra(graph, directed=False, indices=np.unique(edges[eligible]), min_only=True)
        fringe_weight = smooth(skin_distance / .018) * smooth(face_d / .030)

    def rim_offset(parameters):
        rear_inset, rear_height, lateral_inset = parameters * .001
        side = -lateral_inset * np.clip(local[:, 0] / .040, -1., 1.) * lower
        return np.column_stack((side, rear_height * rear_lower, rear_inset * rear_lower))

    def mesh(parameters):
        if fringe_base is not None:
            delta = orientation.apply(rim_offset(parameters)) * fringe_weight[:, None]
            vertices, shared = fringe_base
            return vertices + delta, shared + delta
        shift = orientation.apply(parameters[:3] * .001)
        turn = orientation * Rotation.from_euler("x", parameters[3], degrees=True)
        shape = np.array([1.+parameters[4]/100., 1., 1.+parameters[5]/100.])
        section = local * shape
        if args.fit_posterior_curvature:
            # A common bend of the posterior surface above the attachment ring.
            section[:, 2] += parameters[6] * .001 * posterior
        if args.fit_rim_envelope:
            section += rim_offset(parameters[-3:])
        shared = center + shift + turn.apply(section)
        return current + weight[None, :, None] * (shared + residual - current), shared

    def project_neck(vertices, points):
        indexes = sorted({p[0] for p in points})
        xy = np.zeros((len(records), vertices.shape[1], 2))
        depth = np.zeros(xy.shape[:2])
        world = vertices @ basis.T * scale + origin
        for i in indexes:
            xy[i], depth[i] = project(world[i], *views[i])
        return profile_values(xy, depth, world, points, edges, eligible)

    raw = load_observations(args, records)
    observations = [p[:4] + ("check" if p[0] in held else p[4],) + p[5:] for p in raw
                    if p[5] in ("posterior", "anterior")]
    # Freeze the evaluation denominator before checking any mesh projection.
    evaluation = [point for point in observations if point[0] in held]
    baseline = project_neck(current, observations)
    measurable = np.isfinite(baseline)
    observations = [point for point, good in zip(observations, measurable, strict=True) if good]
    baseline = baseline[measurable]
    fit = [p for p in observations if p[4] == "fit"]
    sides = [len({p[0] for p in fit if p[5] == "posterior" and np.sign(yaw[p[0]]) == sign}) for sign in (-1, 1)]
    envelope_fallback = min(sides) < 2 and bool(args.cold_report) and args.fit_rim_envelope
    if min(sides) < 2 and not envelope_fallback:
        raise ValueError("Too few visible anatomical fitting contours on one side")
    if envelope_fallback:
        print(f"Limited anatomical profile coverage {sides}; using the registered rim envelope. "
              "The independent contour coverage remains an acceptance requirement.", flush=True)
    target = np.asarray([p[2] for p in fit])
    grouping = Counter((p[0], p[5]) for p in fit)
    category = Counter((np.sign(yaw[i]), region) for i, region in grouping)
    weights = np.array([1. / np.sqrt(grouping[p[0], p[5]] * category[np.sign(yaw[p[0]]), p[5]])
                        for p in fit])
    contacts = {}
    if args.body_contact is not None:
        with np.load(args.body_contact, allow_pickle=False) as cache:
            if (cache["initialization_sha256"].item() != sha256((args.sequence / "initialization.json").read_bytes()).hexdigest()
                    or cache["scene_sha256"].item() != sha256(args.body_scene.read_bytes()).hexdigest()):
                raise ValueError("Collar points do not match this exact camera and scene")
            xyz, uv, labels = cache["world"], cache["uv"], cache["names"]
            z, owners = cache["z"], cache["owner_id"]
        for index, name in enumerate(names):
            mask = labels == name
            if np.count_nonzero(mask) < 4:
                continue
            projected, depth = project(xyz[mask], *views[index])
            np.testing.assert_allclose(projected, uv[mask], rtol=0., atol=.35)
            np.testing.assert_allclose(depth, z[mask], rtol=0., atol=1e-3)
            _, owner_index = np.unique(owners[mask], return_inverse=True)
            owner_counts = np.bincount(owner_index)
            # Duplicate samples and densely covered collar viewpoints carry no extra weight.
            by_owner = 1. / np.sqrt(owner_counts[owner_index] * len(owner_counts))
            contacts[index] = (projected, depth, by_owner)
    collar_fit = [i for i in fit_ids if i in contacts]
    collar_check = [i for i in held if i in contacts]
    if args.body_contact is not None and (len(collar_fit) < 3 or len(collar_check) < 2):
        raise ValueError("Body layering needs both fitting and held collar views")

    def collar_distance(vertices, indices):
        ring_xyz = vertices[:, ring] @ basis.T * scale + origin
        result = []
        for index in indices:
            xy, depth, by_owner = contacts[index]
            samples = ring_xyz[index]
            segment = np.roll(samples, axis=0, shift=-1) - samples
            dense = (samples[:, None] + segment[:, None]*np.arange(0., 1., .25)[None, :, None]).reshape(-1, 3)
            given, predicted_z = project(dense, *views[index])
            gap = cdist(given, xy)
            # Prefer a physically nearby visible arc to the projected opposite edge.
            depth_mm = 1000. * np.abs(predicted_z[:, None] - depth) / scale
            distance = np.hypot(gap, np.maximum(depth_mm - 12., 0.) * .45)
            result.append((distance.min(axis=0), by_owner))
        return result

    envelope = {}
    envelope_ids = np.flatnonzero((lower > 0.) & (weight > 0.))
    if args.fit_rim_envelope:
        import cv2
        from fit_side_frames import alpha_maps

        size = tuple(json.loads((args.scene / "source-frames.json").read_bytes())["native_size"])
        k = views[0][0]
        _, _, _, maps, raw_size = alpha_maps(args.scene, size, k, [names[group[0]] for group in groups])
        for index in [*fit_ids, *sorted(held)]:
            np.testing.assert_allclose(k, views[index][0], rtol=0., atol=1e-6)
            matte = cv2.imread(str(args.scene / "alpha_maps" / names[index]), cv2.IMREAD_GRAYSCALE)
            if matte is None or matte.shape[::-1] != raw_size:
                raise ValueError(f"No registered person matte for {names[index]}")
            pixels, _ = project(current[index, envelope_ids] @ basis.T * scale + origin, *views[index])
            x, y = np.floor(pixels.min(0) - 110.).astype(int).clip(0, size)
            X, Y = np.ceil(pixels.max(0) + 110.).astype(int).clip(0, size)
            uv = maps[0][y:Y, x:X], maps[1][y:Y, x:X]
            alpha = cv2.remap(matte, *uv, cv2.INTER_LINEAR)
            envelope[index] = (np.array([x, y]), cv2.distanceTransform(
                (alpha < 38).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))

    def envelope_distance(vertices, indices):
        world = vertices[:, envelope_ids] @ basis.T * scale + origin
        distances = []
        for index in indices:
            corner, distance = envelope[index]
            pixels, _ = project(world[index], *views[index])
            xy = pixels - corner
            if ((xy < 0.).any() or (xy >= np.array(distance.shape[::-1]) - 1.).any()):
                raise ValueError("Lower rim projected outside the frozen evaluation crop")
            distances.append(map_coordinates(distance, xy[:, ::-1].T, order=1))
        return np.concatenate(distances)

    def objective(parameters):
        vertices, _ = mesh(parameters)
        value = project_neck(vertices, fit)
        # A missing anatomical crossing carries a penalty. It cannot improve
        # the objective by quietly switching from skin to the collar mesh.
        error = np.where(np.isfinite(value), value-target, 18.)
        collar = [d*w / np.sqrt(len(collar_fit)) for d, w in collar_distance(vertices, collar_fit)]
        components = [weights*error, *collar, parameters * .12]
        if args.fit_rim_envelope:
            for group in groups:
                exterior = envelope_distance(vertices, group)
                components.append(3. * np.maximum(exterior - 1., 0.) / np.sqrt(len(exterior)))
        return np.concatenate(components)

    limits = [22., 25., 30., 10., 20., 22.] + ([8.] if args.fit_posterior_curvature else [])
    if args.fit_rim_envelope:
        limits += [12., 12., 10.]
    if fringe_base is not None:
        limits = [12., 12., 10.]
    bound = np.array(limits)
    steps = np.full_like(bound, .25)
    def jacobian(parameters):
        return np.stack([(objective(parameters+d) - objective(parameters-d)) / (2.*d.sum())
                         for d in np.diag(steps)], axis=1)

    fit_result = least_squares(objective, np.zeros_like(bound), jac=jacobian, bounds=(-bound, bound),
                               loss="soft_l1", f_scale=2., max_nfev=32)
    fitted, shared = mesh(fit_result.x)

    def valid_mesh(fraction):
        base = current if fringe_base is None else fringe_base[0]
        mixed = (base + fraction * (fitted - base)).astype(current.dtype)
        shape = face_area_check(current, mixed, active_faces)
        return mixed, shape

    def safe(shape):
        return shape[0] >= .40 and shape[1] <= 2.1 and shape[2] == 0

    accepted, geometry = valid_mesh(1.)
    fraction = 1.
    if not safe(geometry):
        left, right = 0., 1.
        for _ in range(16):
            middle = (left + right)*.5
            if safe(valid_mesh(middle)[1]):
                left = middle
            else:
                right = middle
        fraction = left*.99
        accepted, geometry = valid_mesh(fraction)
    if not safe(geometry):
        raise RuntimeError("Shared neck would fold the mesh")
    np.testing.assert_array_equal(accepted[:, protect], current[:, protect])
    if fringe_base is not None:
        np.testing.assert_array_equal(accepted[:, edges[eligible].ravel()], fringe_base[0][:, edges[eligible].ravel()])
    after = project_neck(accepted, observations)
    sides_check = {}
    for sign in (-1, 1):
        signed = [p[:4]+("exclude",)+p[5:] if np.sign(yaw[p[0]]) != sign or p[0] not in held else p
                  for p in observations]
        sides_check[str(sign)] = neck_metrics(baseline, after, signed)
    held_collar = [float(np.median(np.concatenate([value for value, _ in collar_distance(mesh, collar_check)])))
                   for mesh in (current, accepted)] if collar_check else None
    rim_spread = [float(1000.*np.linalg.norm(frame[:, ring]-np.median(frame[:, ring], axis=0), axis=2).max())
                  for frame in (current, accepted)]
    report = {"frames": len(names), "template": "balanced robust sequence median", "donor_frame": None,
              "fit_frames": [names[i] for i in fit_ids], "held_frames": [names[i] for i in sorted(held)],
              "posterior_fit_frames_by_yaw_sign": sides,
              "uses_rim_envelope_for_missing_anatomical_support": envelope_fallback,
              "collar_fit_frames": [names[i] for i in collar_fit], "collar_held_frames": [names[i] for i in collar_check],
              "collar_weighting": "equal view weight, equal owner weight within each view" if contacts else None,
              "maximum_rim_deviation_from_median_flame_mm": rim_spread,
              "neck_checks_by_yaw_sign": sides_check, "held_collar_combined_residual_before_after": held_collar,
              "geometry": geometry, "safe_fraction": fraction, "fitted_parameters": fit_result.x.tolist(),
              "fit_posterior_curvature": args.fit_posterior_curvature,
              "fit_rim_envelope": args.fit_rim_envelope,
              "held_native_contour_limit": audit_contours(
                  evaluation, project_neck(accepted, evaluation), names, yaw, 3.),
              "fit_jacobian_singular_values": np.linalg.svd(fit_result.jac, compute_uv=False).tolist(),
              "protected_face_identical": True,
              "held_posterior_points": [{"frame": names[p[0]], "uv": [p[2], p[3]], "before": abs(a-p[2]),
                                         "after": abs(b-p[2]) if np.isfinite(b) else None}
                                        for p, a, b in zip(observations, baseline, after, strict=True)
                                        if p[0] in held and p[4] == "check" and p[5] == "posterior"]}
    if args.fit_rim_envelope:
        outside = envelope_distance(accepted, sorted(held))
        report["held_background_exterior"] = {"max_sampled_rim_px": float(outside.max()),
                                              "samples_over_3px": int((outside > 3.).sum())}
    if args.rim_fringe_from:
        report["rim_fringe_base_sha256"] = sha256(args.rim_fringe_from.read_bytes()).hexdigest()
    (args.output / "shared-neck.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.output / "shared-neck-geometry.npz", frames=names, old=current, vertices=accepted,
                        shared=shared, faces=faces, rim=ring, basis=basis, origin=origin, scale=scale)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("fit_frames", "held_frames", "held_posterior_points", "held_native_contour_limit")}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("sequence", "registered", "flame-masks", "output", "scene", "targets", "native-fallback"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    for key in ("pose-report", "rim-report", "cold-report", "split", "body-contact", "body-scene"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--fit-posterior-curvature", action="store_true")
    parser.add_argument("--fit-rim-envelope", action="store_true")
    parser.add_argument("--rim-fringe-from", type=Path)
    parser.add_argument("--slack-mm", type=float, default=3.)
    args = parser.parse_args()
    if not 0 <= args.slack_mm <= 10.:
        parser.error("Shared lower-neck slack must be between 0 and 10 FLAME millimetres")
    if args.rim_fringe_from and not args.fit_rim_envelope:
        parser.error("A saved fringe input requires --fit-rim-envelope")
    if args.cold_report and (args.pose_report or args.rim_report or not args.split):
        parser.error("Fresh VHAP input requires a split and no separate fitted pose or rim")
    if not args.cold_report and not (args.pose_report and args.rim_report):
        parser.error("Provide the fresh VHAP report or a matched pose and rim")
    if bool(args.body_contact) != bool(args.body_scene) or not (args.body_contact or args.fit_rim_envelope):
        parser.error("Provide paired body contacts or native rim-envelope supervision")
    run(args)
