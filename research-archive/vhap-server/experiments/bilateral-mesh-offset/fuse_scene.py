"""Pose bound GaussianAvatars in the native COLMAP camera and fuse with its scene."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.flame_gaussian_model import FlameGaussianModel
from scene.gaussian_model import GaussianModel

from refine_color import make_example, refine_dc

# Rigid alignment uses the stable inner face and leaves the mouth out of the fit.
FIT_LANDMARKS = np.arange(17, 48)
MOUTH_LANDMARKS = np.arange(48, 68)
SCALE_LANDMARKS = np.array([27, 28, 29, 30, 31, 35, 36, 39, 42, 45])


def read_scene_cameras(scene, background):
    sparse = scene / "orig_undist/sparse"
    intrinsics = read_intrinsics_binary(sparse / "cameras.bin")
    if len(intrinsics) != 1:
        raise ValueError("Expected a single COLMAP camera.")
    camera = next(iter(intrinsics.values()))
    if camera.model != "PINHOLE":
        raise ValueError("The input raster must be COLMAP PINHOLE.")
    width, height = camera.width, camera.height
    fx, fy, cx, cy = camera.params
    named_extrinsics = {Path(image.name).stem: image
                        for image in read_extrinsics_binary(sparse / "images.bin").values()}
    # COLMAP first pixel center is (0.5, 0.5); OpenCV's first center is (0, 0).
    K = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.0]])
    cameras = {}
    for record in json.loads((background / "cameras.json").read_text()):
        if (record["width"], record["height"]) != (width, height) or not np.allclose(
            [record["fx"], record["fy"]], [fx, fy], atol=1e-7, rtol=0
        ):
            raise ValueError("Scene raster and background camera intrinsics disagree.")
        name = Path(record["img_name"]).stem
        if name in cameras:
            raise ValueError(f"Ambiguous background frame: {name}")
        R = np.asarray(record["rotation"], float)
        C = np.asarray(record["position"], float)
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError(f"Invalid scene camera rotation: {name}")
        native = named_extrinsics[name]
        if native.camera_id != camera.id or not np.allclose(native.qvec2rotmat(), R.T, atol=1e-7, rtol=0):
            raise ValueError(f"Native and background camera orientations disagree: {name}")
        if not np.allclose(native.tvec, -R.T @ C, atol=1e-7, rtol=0):
            raise ValueError(f"Native and background camera translations disagree: {name}")
        cameras[name] = (R, C)
    return K, (width, height), cameras


def project(points, K):
    if (points[:, 2] <= 0).any():
        raise ValueError("Head projects behind the scene camera.")
    uv = points @ K.T
    return uv[:, :2] / uv[:, 2:3]


def make_rasterizer(K, size, R, C):
    width, height = size
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ C
    near, far = 0.01, 500.0
    projection = np.array([
        [2 * K[0, 0] / width, 0, (2 * K[0, 2] + 1) / width - 1, 0],
        [0, 2 * K[1, 1] / height, (2 * K[1, 2] + 1) / height - 1, 0],
        [0, 0, far / (far - near), -far * near / (far - near)],
        [0, 0, 1, 0],
    ])
    gpu = lambda values: torch.as_tensor(values, dtype=torch.float32, device="cuda")
    view = gpu(w2c.T)
    settings = GaussianRasterizationSettings(
        image_width=width, image_height=height,
        tanfovx=width / (2 * K[0, 0]), tanfovy=height / (2 * K[1, 1]),
        bg=gpu([0, 0, 0]), scale_modifier=1.0, viewmatrix=view,
        projmatrix=view @ gpu(projection.T), sh_degree=3, campos=gpu(C),
        prefiltered=False, debug=False,
    )
    return GaussianRasterizer(raster_settings=settings)


def render(rasterizer, gaussians, alpha_only=False):
    xyz, features, opacity, scales, rotations = gaussians
    result = rasterizer(
        means3D=xyz, means2D=torch.zeros_like(xyz), shs=None if alpha_only else features,
        colors_precomp=torch.ones_like(xyz) if alpha_only else None,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None,
    )
    return result[0][0] if alpha_only else result[0]


def check_principal_point(rasterizer_factory=make_rasterizer):
    K = np.array([[100., 0., 27.25], [0., 105., 35.75], [0., 0., 1.]])
    rasterizer = rasterizer_factory(K, (80, 64), np.eye(3), np.zeros(3))
    gpu = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda")
    alpha = render(rasterizer, (
        gpu([[0, 0, 2]]), gpu(np.zeros((1, 16, 3))), gpu([[0.9]]),
        gpu([[0.06, 0.06, 0.06]]), gpu([[1, 0, 0, 0]]),
    ), alpha_only=True)
    a = alpha.squeeze().cpu().numpy()
    yy, xx = np.indices(a.shape)
    actual = np.array([(a * xx).sum(), (a * yy).sum()]) / a.sum()
    error = float(np.linalg.norm(actual - K[:2, 2]))
    if not error < 0.05:
        raise RuntimeError(f"Off-axis CUDA projection failed: {actual=} {error=}")
    return error


def local_head(actor, track, index):
    p = actor.flame_param
    posed = {
        "shape": p["shape"], "static_offset": p["static_offset"],
        "expr": track["expr"][index:index + 1],
        "rotation": torch.zeros((1, 3), device="cuda"),
        "neck": track["neck_pose"][index:index + 1],
        "jaw": track["jaw_pose"][index:index + 1],
        "eyes": track["eyes_pose"][index:index + 1],
        "translation": torch.zeros((1, 3), device="cuda"),
    }
    actor.update_mesh_by_param_dict(posed)
    landmarks = actor.flame_model(
        posed["shape"][None], posed["expr"], posed["rotation"], posed["neck"],
        posed["jaw"], posed["eyes"], posed["translation"],
        zero_centered_at_root_node=False, return_landmarks=True,
        static_offset=posed["static_offset"],
    )[1][0]
    # GaussianAvatars appends two eyeball centers after the 68 STAR landmarks.
    return landmarks[:68].cpu().numpy().astype(np.float64)


def align_rigid(landmarks, observed, K):
    fit = FIT_LANDMARKS
    ok, rotation, translation = cv2.solvePnP(
        landmarks[fit], observed[fit], K, None, flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        raise RuntimeError("FLAME pose estimation failed.")
    rotation, translation = cv2.solvePnPRefineLM(
        landmarks[fit], observed[fit], K, None, rotation, translation,
    )
    R = cv2.Rodrigues(rotation)[0]
    t = translation[:, 0]
    local = landmarks @ R.T + t
    return R, t, local, np.linalg.norm(project(local, K) - observed, axis=1)


def estimate_scale(centers, offsets):
    """Fit one scale across matched views of the static part of the face."""
    dC = centers - centers.mean(axis=0)
    dU = offsets - offsets.mean(axis=0)
    scale = -float(np.sum(dC * dU) / np.sum(dU ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Scene scale must be positive: {scale}")
    anchors = (centers + scale * offsets).mean(axis=0)
    return scale, anchors


def world_gaussians(actor, R, t, scale, keep):
    rotate = torch.as_tensor(R, dtype=torch.float32, device="cuda")
    shift = torch.as_tensor(t, dtype=torch.float32, device="cuda")
    xyz = scale * (actor.get_xyz[keep] @ rotate.T) + shift
    q = actor.get_rotation[keep].cpu().numpy()
    actor_rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    q = (Rotation.from_matrix(R) * actor_rotation).as_quat()[:, [3, 0, 1, 2]]
    # DC is invariant under placement. The fixed world scene retains its trained SH.
    features = actor.get_features[keep].clone()
    features[:, 1:] = 0
    return (xyz, features, actor.get_opacity[keep], actor.get_scaling[keep] * scale,
            torch.as_tensor(q, device="cuda", dtype=torch.float32))


def ply_rotation(rotation):
    """Choose a near-unit quaternion whose normalization reproduces the rendered orientation."""
    target = rotation.detach().contiguous()
    normalize = torch.nn.functional.normalize
    best = target.clone()
    error = (normalize(best, dim=-1) - target).square().sum(-1)
    # Multiplication by a scalar preserves the rotation. At most eight float32 epsilons
    # compensate for rounding when the standard 3DGS reader normalizes it a second time.
    eps = torch.finfo(target.dtype).eps
    for i in range(1, 9):
        for scale in (1 - i*eps, 1 + i*eps):
            candidate = target * scale
            delta = (normalize(candidate, dim=-1) - target).square().sum(-1)
            best = torch.where((delta < error)[:, None], candidate, best)
            error = torch.minimum(error, delta)
    return best


def write_ply(path, source, head, scene_dc=None):
    scene = PlyData.read(source)["vertex"].data
    if scene_dc is not None:
        scene = scene.copy()
        values = scene_dc[:, 0].cpu().numpy()
        for i in range(3):
            scene[f"f_dc_{i}"] = values[:, i]
    points = np.zeros(len(head[0]), dtype=scene.dtype)
    xyz, features, opacity, sizes = (part.cpu().numpy() for part in head[:4])
    rotation = ply_rotation(head[4]).cpu().numpy()
    for i, name in enumerate(("x", "y", "z")):
        points[name] = xyz[:, i]
    for i in range(3):
        points[f"f_dc_{i}"] = features[:, 0, i]
        points[f"scale_{i}"] = np.log(sizes[:, i])
    points["opacity"] = torch.logit(torch.from_numpy(opacity[:, 0]).clamp(1e-7, 1 - 1e-7)).numpy()
    for i in range(4):
        points[f"rot_{i}"] = rotation[:, i]
    combined = np.concatenate((scene, points))
    PlyData([PlyElement.describe(combined, "vertex")], text=False).write(path)
    return len(scene), len(points)


def read_gaussians(path):
    model = GaussianModel(3)
    model.load_ply(str(path))
    return (model.get_xyz, model.get_features, model.get_opacity,
            model.get_scaling, model.get_rotation)


def to_bgr(rgb):
    return cv2.cvtColor((rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def save_jpg(path, image):
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Could not save {path}")


def prepare_scene(args):
    K, size, cameras = read_scene_cameras(args.scene, args.background)
    detection = np.load(args.landmarks)
    if str(detection["tracker_sha256"]) != sha256(args.tracker.read_bytes()).hexdigest():
        raise ValueError("Landmarks and tracker checkpoint have different provenance.")
    tracks = np.load(args.tracker)
    names = detection["names"].astype(str).tolist()
    if list(tracks["timestep_id"].astype(int)) != list(range(len(names))):
        raise ValueError("VHAP tracker timestamps do not match the ordered scene-time manifest.")
    if len(set(names)) != len(names) or args.export_frame not in names:
        raise ValueError("Scene frame names must be unique and include the export frame.")
    for name, digest in zip(names, detection["native_sha256"], strict=True):
        path = args.scene / "orig_undist/images" / name
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Scene image changed after landmark detection: {name}")
    observed = detection["xy"]
    if observed.shape != (len(names), 68, 2) or not np.isfinite(observed).all():
        raise ValueError("Expected 68 finite landmarks per scene frame.")
    track = {key: torch.as_tensor(tracks[key], device="cuda", dtype=torch.float32)
             for key in ("expr", "neck_pose", "jaw_pose", "eyes_pose")}
    if any(len(values) != len(names) for values in track.values()):
        raise ValueError("VHAP articulation data is incomplete.")
    actor = FlameGaussianModel(3)
    actor.load_ply(str(args.bound), has_target=False)
    if actor.binding is None or not len(actor.binding):
        raise ValueError("The actor PLY has no FLAME triangle bindings.")
    keep = torch.ones_like(actor.binding, dtype=torch.bool)
    if args.flame_boundary_mask is not None:
        # Bottom rim splats learned the shirt in the avatar crop. The scene already owns it.
        with args.flame_boundary_mask.open("rb") as mask_file:
            vertices = pickle.load(mask_file, encoding="latin1")["boundary"]
        boundary = torch.zeros(len(actor.flame_model.v_template), device="cuda", dtype=torch.bool)
        boundary[torch.as_tensor(vertices, device="cuda", dtype=torch.long)] = True
        faces = actor.flame_model.faces[actor.binding.to(torch.long)]
        keep = boundary[faces].sum(dim=1) < 2
    bgfile = getattr(args, "background_ply", None) or args.background / "point_cloud/iteration_18000/point_cloud.ply"
    bg = read_gaussians(bgfile)
    return K, size, cameras, names, observed, track, actor, keep, bgfile, bg


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "point_cloud.ply").exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    projection_error = check_principal_point()
    print(f"CUDA off-axis projection: {projection_error:.5f} px", flush=True)
    K, size, cameras, names, observed, track, actor, keep, bgfile, bg = prepare_scene(args)

    frames, camera_centers, offsets = [], [], []
    for i, name in enumerate(names):
        R, C = cameras[Path(name).stem]
        lm = local_head(actor, track, i)
        R_obj, t_obj, local, err = align_rigid(lm, observed[i], K)
        frames.append((R, C, R_obj, t_obj, local, err))
        camera_centers.append(C)
        offsets.append(local[SCALE_LANDMARKS] @ R.T)
    centers, offsets = np.array(camera_centers)[:, None], np.array(offsets)
    held = np.arange(len(frames)) % 4 == 2
    scale, anchors = estimate_scale(centers[~held], offsets[~held])
    print(f"Scene units per actor unit: {scale:.6f}", flush=True)

    result = {"K_integer_centers": K.tolist(), "image_size": size, "scale": scale,
              "cuda_offaxis_error_px": projection_error,
              "removed_boundary_splats": int((~keep).sum()), "frames": []}
    posed_heads = []
    for i, (name, frame) in enumerate(zip(names, frames)):
        R, C, R_obj, t_obj, _, _ = frame
        local_head(actor, track, i)
        world_R = R @ R_obj
        world_t = C + scale * (R @ t_obj)
        head = world_gaussians(actor, world_R, world_t, scale, keep)
        posed_heads.append((world_R, world_t, head))
    dc = None
    if args.color_steps:
        examples = []
        for i, (name, frame, posed) in enumerate(zip(names, frames, posed_heads)):
            R, C = frame[:2]
            head = posed[2]
            original = cv2.imread(str(args.scene / "orig_undist/images" / name))
            if original is None or original.shape[1::-1] != size:
                raise ValueError(f"Wrong native image: {name}")
            examples.append(make_example(make_rasterizer, render, K, R, C, original, observed[i], head, held[i]))
        dc, color_report = refine_dc(bg, examples, render, args.color_steps)
        result["face_color_refinement"] = color_report
        print(color_report, flush=True)
    for i, (name, frame, posed) in enumerate(zip(names, frames, posed_heads)):
        R, C, R_obj, t_obj, local, err = frame
        world_R, world_t, head = posed
        if dc is not None:
            head[1][:, :1] = dc
        roundtrip = (scale * local @ R.T + C - C) @ R
        parity = float(np.linalg.norm(project(roundtrip, K) - project(local, K), axis=1).max())
        fixed = np.linalg.norm(project((anchors - C) @ R, K) - observed[i, SCALE_LANDMARKS], axis=1)
        metric = {"frame": name, "scale_holdout": bool(held[i]),
                  "rigid_fit_median_px": float(np.median(err[FIT_LANDMARKS])),
                  "mouth_holdout_median_px": float(np.median(err[MOUTH_LANDMARKS])),
                  "fixed_world_landmark_median_px": float(np.median(fixed)),
                  "reprojection_parity_px": parity, "object_to_world_rotation": world_R.tolist(),
                  "object_to_world_translation": world_t.tolist()}
        rasterizer = make_rasterizer(K, size, R, C)
        rgb = render(rasterizer, tuple(torch.cat((a, b)) for a, b in zip(bg, head)))
        alpha = render(rasterizer, head, alpha_only=True)
        ref = cv2.imread(str(args.scene / "orig_undist/images" / name))
        if ref is None or ref.shape[1::-1] != size:
            raise ValueError(f"Wrong native image: {name}")
        image = to_bgr(rgb)
        mask = alpha.squeeze().cpu().numpy() > 0.5
        metric["actor_visible_alpha_pixels"] = int(mask.sum())
        world_head = (head[0] - head[0].new_tensor(C)) @ head[0].new_tensor(R)
        metric["head_median_depth"] = float(world_head[:, 2].median())
        for point in project(local, K)[17:68]:
            cv2.circle(ref, tuple(np.rint(point).astype(int)), 2, (40, 230, 255), -1)
        alpha_view = image.copy()
        alpha_view[mask] = (0.62 * alpha_view[mask] + 0.38 * np.array([90, 220, 30])).astype(np.uint8)
        save_jpg(output / (Path(name).stem + "-review.jpg"), np.hstack((ref, image, alpha_view)))
        result["frames"].append(metric)
        print(name, f"fit {metric['rigid_fit_median_px']:.2f} px, mouth {metric['mouth_holdout_median_px']:.2f} px,"
              f" fixed-world {metric['fixed_world_landmark_median_px']:.2f} px, head area {mask.sum()}", flush=True)
    reference_idx = names.index(args.export_frame)
    R, C = frames[reference_idx][:2]
    reference_raster = make_rasterizer(K, size, R, C)
    reference_head = posed_heads[reference_idx][2]
    reference_rgb = render(reference_raster, tuple(torch.cat((a, b)) for a, b in zip(bg, reference_head)))
    temporary_ply = output / "point_cloud.ply.partial"
    result["export_frame"] = args.export_frame
    result["ply_point_counts_scene_and_head"] = write_ply(temporary_ply, bgfile, reference_head)
    delta = (reference_rgb - render(reference_raster, read_gaussians(temporary_ply))).abs()
    result["ply_readback_image_max_rgb_error"] = float(delta.max())
    result["ply_readback_image_mean_rgb_error"] = float(delta.mean())
    if float(delta.max()) > 1e-3:
        raise RuntimeError("The exported PLY did not reproduce the in-memory render.")
    print("PLY read-back max RGB error", float(delta.max()), flush=True)
    # The PLY stays in the background's COLMAP world. Include full intrinsics in the sidecar.
    records = json.loads((args.background / "cameras.json").read_text())
    for camera in records:
        camera["cx"] = float(K[0, 2] + 0.5)
        camera["cy"] = float(K[1, 2] + 0.5)
    (output / "cameras.json").write_text(json.dumps(records, indent=2))
    temporary_ply.replace(output / "point_cloud.ply")
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(output, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for option in ("scene", "background", "bound", "tracker", "landmarks", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--color-steps", default=0, type=int, choices=range(0, 257), metavar="0..256")
    parser.add_argument("--flame-boundary-mask", type=Path, help="Trusted FLAME region asset. Use if the scene contains the torso.")
    parser.add_argument("--export-frame", required=True, help="Undistorted source filename for the static PLY.")
    cv2.setNumThreads(1)
    with torch.no_grad():
        run(parser.parse_args())
