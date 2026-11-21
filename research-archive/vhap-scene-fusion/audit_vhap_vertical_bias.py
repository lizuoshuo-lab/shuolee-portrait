"""Audit signed vertical residuals without modifying the Demo2 mesh or cameras."""

from argparse import ArgumentParser
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from finish_vhap_shoulder_seam import shoulder_anchors


def project(v, camera, center, intrinsics):
    c = ((v - center[:, None]) @ camera) @ np.swapaxes(intrinsics, -1, -2)
    if (c[..., 2] <= 0).any():
        raise ValueError("Nonpositive depth in audit")
    return c[..., :2] / c[..., 2:]


def dy_stats(dy):
    v = np.asarray(dy).reshape(-1)
    return None if not len(v) else dict(count=len(v), median_px=float(np.median(v)),
                                        p10_p90_px=np.percentile(v, [10, 90]).tolist(),
                                        median_absolute_px=float(np.median(np.abs(v))), high_fraction=float((v < 0).mean()))


def embedded_landmarks(vertices, reference, assets):
    flame_data = assets / "code/VHAP/asset/flame"
    faces = np.array([[int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                      for line in (flame_data / "head_template_mesh.obj").read_text().splitlines()
                      if line.startswith("f ")])
    e = np.load(flame_data / "landmark_embedding_with_eyes.npy", allow_pickle=True).item()
    mapping = np.load(assets / "models/smplx/correspondences/SMPL-X__FLAME_vertex_ids.npy", allow_pickle=False)
    endpoints = reference["source_vertex_endpoints"]
    ids = {int(a): i for i, (a, b) in enumerate(endpoints) if a == b}
    support = mapping[faces[e["full_lmk_faces_idx"].reshape(-1)[17:68]]]
    indices = np.array([[ids[int(i)] for i in row] for row in support])
    rows = {tuple(f) for f in np.sort(reference["faces"], axis=1)}
    if not all(tuple(f) in rows for f in np.sort(indices, axis=1)):
        raise ValueError("Landmark face IDs do not match the compact mesh topology")
    mix = e["full_lmk_bary_coords"].reshape(-1, 3)[17:68]
    return np.einsum("ntsc,ts->ntc", vertices[:, indices], mix)


def top_systematic(vertices, camera, center, crop, mattes):
    uv = project(vertices, camera, center, crop)
    result = []
    for p, matte in zip(uv, mattes, strict=True):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((matte >= 128).astype(np.uint8))
        if count < 2:
            raise ValueError("No visible foreground")
        target = labels == (1 + stats[1:, cv2.CC_STAT_AREA].argmax())
        ys, xs = np.where(target)
        crown = xs[ys <= ys.min() + 7]
        x_min, x_max = int(np.percentile(crown, 15)), int(np.percentile(crown, 85))
        hull = cv2.convexHull(np.rint(p).astype(np.int32))
        region = cv2.fillConvexPoly(np.zeros(matte.shape, np.uint8), hull, 1).astype(bool)
        x = np.arange(x_min, x_max + 1)
        valid = region[:, x].any(axis=0)
        result.append(float(np.median(region[:, x[valid]].argmax(0) - target[:, x[valid]].argmax(0))))
    return np.array(result)


def shoulder_surface(mesh, anchors, camera, center, crop, mattes):
    points = project(anchors, camera, center, crop)
    xy = project(mesh["vertices_world"], camera, center, crop)
    cloth = mesh["faces"][mesh["cloth_faces"]]
    res = np.full(len(mattes), np.nan)
    for i, matte in enumerate(mattes):
        x = np.arange(matte.shape[1])
        roots = points[i, :, 0]
        width = abs(roots[0] - roots[1])
        span = (abs(x - roots.mean()) >= .32*width) & (abs(x[:, None] - roots).min(1) < .14*width)
        pred = np.zeros_like(matte)
        for tri in np.rint(xy[i, cloth]*4).astype(np.int32):
            cv2.fillConvexPoly(pred, tri, 1, shift=2)
        _, labels, cc, _ = cv2.connectedComponentsWithStats((matte >= 128).astype(np.uint8))
        target = labels == (1 + cc[1:, cv2.CC_STAT_AREA].argmax())
        old_y, new_y = target.argmax(0), pred.argmax(0)
        floor = points[i, :, 1].min() - 105
        use = span & target.any(0) & pred.any(0) & (old_y >= floor) & (new_y >= floor)
        if use.sum() >= 20:
            res[i] = np.median(new_y[use] - old_y[use])
    return res


def run(args):
    p = args.run
    base = np.load(p / "native-vhap-head-neck/head-neck-fitted.npz", allow_pickle=False)
    head = np.load(p / "scalp-outline-head-neck/head-neck-fitted.npz", allow_pickle=False)
    final = np.load(p / "shoulder-seam-refined/head-neck-lower-seam.npz", allow_pickle=False)
    ref = np.load(p / "head-neck-only/head-neck-frames.npz", allow_pickle=False)
    body = np.load(p / "head-neck-smplx/smplx-frames.npz", allow_pickle=False)
    obs = np.load(p / "head-neck-smplx/body-observations.npz", allow_pickle=False)
    torso = np.load(p / "shoulder-seam-refined/review/head-neck-shoulders.npz", allow_pickle=False)
    with ZipFile(p / "head-neck-smplx/head-neck-87frames.zip") as source:
        records = json.loads(source.read("initialization.json"))["frames"]
        landmarks = np.load(BytesIO(source.read("landmark2d/STAR.npz")), allow_pickle=False)["face_landmark_2d"]
        mattes = [cv2.imdecode(np.frombuffer(source.read(f"alpha_maps/{i:08d}.jpg"), np.uint8), 0) for i in range(len(records))]
    names = [r["name"] for r in records]
    for m in (base, head, final, ref, body, obs, torso):
        if m["names"].tolist() != names:
            raise ValueError("Audit arrays disagree on source frames")
    for key in ("native_K", "native_camera_to_world_rotation", "native_camera_center", "scene_scale"):
        if any(not np.array_equal(head[key], m[key]) for m in (base, final, ref, body, torso)):
            raise ValueError("Audit arrays disagree on cameras")
    K = head["native_K"]
    cameras, centers, H = (head[k] for k in ("native_camera_to_world_rotation", "native_camera_center", "virtual_crop_homographies"))
    K_crop = H @ K
    lv = landmarks[:, 17:68]
    h = np.concatenate((lv[..., :2]*mattes[0].shape[0] - .5, np.ones((*lv.shape[:2], 1))), axis=-1)
    n = h @ np.linalg.inv(H).transpose(0, 2, 1)
    truth = n[..., :2]/n[..., 2:]
    valid = (lv[..., 2] > 0) & np.asarray([r["has_landmarks"] for r in records])[:, None]
    if not valid.any():
        raise ValueError("No comparable trusted facial landmarks")
    yaw = np.abs([r["baseline_yaw_deg"] for r in records])
    output = {"residual": "projected y minus observed y; negative means above", "faces": {}, "crown": {}, "shoulders": {}}
    vrot = np.asarray([r["virtual_from_native_cv"] for r in records])
    vk = np.asarray([r["K_integer_pixel_centers"] for r in records])
    v = torso["vertices_world"]
    q = project(v, cameras, centers, K_crop)
    pvir = project(v, cameras @ vrot.transpose(0, 2, 1), centers, vk)
    output["projection_max_crop_pixel_delta"] = float(np.abs(q - pvir).max())
    for name, mesh in (("before_hair", base), ("after_hair", head), ("after_neck_seam", final)):
        v = mesh["vertices_world"]
        lm = embedded_landmarks(v, ref, args.assets)
        dy = project(lm, cameras, centers, K)[..., 1] - truth[..., 1]
        output["faces"][name] = {k: dy_stats(dy[:, i][valid[:, i]]) for k, i in
                                  (("brow", slice(0, 10)), ("nose", slice(10, 19)),
                                   ("eyes", slice(19, 31)), ("mouth", slice(31, 51)))}
        output["faces"][name]["overall"] = dy_stats(dy[valid])
        top = top_systematic(v, cameras, centers, K_crop, mattes)
        output["crown"][name] = {"all": dy_stats(top), "front": dy_stats(top[yaw < 30]),
                                  "pure_profile": dy_stats(top[yaw >= 80]),
                                  "selected": {names[i]: float(top[i]) for i in (1, 51, 83, 84, 85, 86)}}
    anchor, basis, delta, shoulder_valid, _ = shoulder_anchors(body, obs, records)
    shoulder_observed = obs["keypoints"][:, [5, 6]]
    final_anchor = anchor + delta
    for n, pts in (("before", anchor), ("after", final_anchor)):
        d = project(pts, cameras, centers, K)[..., 1] - shoulder_observed[..., 1]
        output["shoulders"][n] = {"all": dy_stats(d[shoulder_valid]),
                                  "front": dy_stats(d[shoulder_valid & (yaw < 30)[:, None]]),
                                  "oblique": dy_stats(d[shoulder_valid & ((yaw >= 30) & (yaw < 80))[:, None]]),
                                  "pure_profile": dy_stats(d[shoulder_valid & (yaw >= 80)[:, None]])}

    silhouette = shoulder_surface(torso, final_anchor, cameras, centers, K_crop, mattes)
    output["shoulder_surface_crop_pixels"] = {"front": dy_stats(silhouette[np.isfinite(silhouette) & (yaw < 30)]),
                                              "oblique": dy_stats(silhouette[np.isfinite(silhouette) & (yaw >= 30) & (yaw < 60)])}
    edges = ref["source_vertex_endpoints"]
    ears = [int(np.flatnonzero((edges == k).all(axis=1)).item()) for k in (6, 616)]
    ear_obs = obs["keypoints"][:, [3, 4]]
    h = np.concatenate((ear_obs, np.ones((len(records), 2, 1))), axis=-1) @ H.transpose(0, 2, 1)
    target_y = h[..., 1] / h[..., 2]
    visible = (obs["scores"][:, [3, 4]] >= .5) & (obs["flip_disagreement"][:, [3, 4]] < 7)
    side = yaw > 45
    depth = ((final["vertices_world"][:, ears] - centers[:, None]) @ cameras)[..., 2]
    visible[np.where(side)[0], depth[side].argmax(axis=1)] = False
    output["ear_proxy_crop_pixels"] = {}
    for key, mesh in (("smplx_before_vhap", ref), ("before_hair", base), ("after_neck_seam", final)):
        dy = project(mesh["vertices_world"][:, ears], cameras, centers, K_crop)[..., 1] - target_y
        output["ear_proxy_crop_pixels"][key] = {"front": dy_stats(dy[visible & (yaw < 30)[:, None]]),
                                                "pure_profile": dy_stats(dy[visible & (yaw >= 80)[:, None]])}

    # A read-only counterfactual: shared translation along the body-up direction.
    up = np.mean(basis[:, :, 1], axis=0)
    up /= np.linalg.norm(up)
    targets = [truth[..., 1], shoulder_observed[..., 1]]
    observations = [valid, shoulder_valid]
    points = [lm, final_anchor]
    split = np.arange(len(records)) % 5 == 1
    def residual(amount, ids):
        return [(project(v + amount*up, cameras, centers, K)[..., 1] - target)[mask & ids[:, None]]
                for v, target, mask in zip(points, targets, observations, strict=True)]
    best = minimize_scalar(lambda amount: sum(np.median(np.abs(e)) for e in residual(amount, ~split)),
                           bounds=(-.05, .05), method="bounded").x
    output["common_shift_control"] = {"candidate_world_units_positive_up": float(best), "check_before": [dy_stats(v) for v in residual(0., split)],
                                       "check_after": [dy_stats(v) for v in residual(best, split)]}
    with args.output.open("x", encoding="utf8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("run", "assets", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())
