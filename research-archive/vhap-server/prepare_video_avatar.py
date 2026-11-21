"""Fit VHAP and initialize triangle-bound Gaussians for the joint scene fit."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def virtual_crop(points, native_k, size):
    rays = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(native_k).T
    forward = rays.mean(axis=0)
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    virtual_rays = rays @ rotation.T
    if (virtual_rays[:, 2] <= 0).any():
        raise ValueError("Head extends behind its virtual camera")
    normalized = virtual_rays[:, :2] / virtual_rays[:, 2:]
    extent = np.ptp(normalized, axis=0).max()
    if not np.isfinite(extent) or extent <= 1e-6:
        raise ValueError("Invalid face extent")
    focal = size / (2.45 * extent)
    center = (size - 1) / 2
    crop_k = np.array([[focal, 0, center], [0, focal, center], [0, 0, 1.]])
    warp = crop_k @ rotation @ np.linalg.inv(native_k)
    pixels = center + focal * normalized
    return warp, rotation, crop_k, pixels


def write_jpeg(path, image):
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
        raise OSError(f"Could not write {path}")


def make_tracking_data(scene, selection_path, landmarks_path, output, size):
    selection = json.loads(selection_path.read_text("utf-8"))
    with np.load(landmarks_path) as archive:
        names = archive["names"].astype(str).tolist()
        xy = archive["xy"].astype(np.float64)
        hashes = archive["native_sha256"].astype(str).tolist()
        native_k = archive["K"].astype(np.float64)
    train = set(selection["train"])
    held = set(selection["holdout"])
    if len(names) != len(set(names)) or set(names) != train | held or train & held:
        raise ValueError("Selection and landmarks disagree on training and holdout frames")
    if len(train) < 3 or not held or xy.shape != (len(names), 68, 2):
        raise ValueError("At least three training frames, a holdout and STAR landmarks are required")
    if not np.isfinite(xy).all() or native_k.shape != (3, 3) or not np.isfinite(native_k).all():
        raise ValueError("Invalid landmark or camera matrix")
    seq = output / "vhap-input"
    (seq / "images").mkdir(parents=True)
    (seq / "alpha_maps").mkdir()
    (seq / "landmark2d").mkdir()
    crops, normalized_landmarks = [], []
    bounds = []
    for index, (name, point, expected_hash) in enumerate(zip(names, xy, hashes, strict=True)):
        if Path(name).name != name:
            raise ValueError(f"Expected an image basename: {name}")
        source = scene / "orig_undist/images" / name
        if sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Native image changed after selection: {name}")
        image = cv2.imread(str(source))
        person = cv2.imread(str(scene / "mask_undist/images" / name), cv2.IMREAD_GRAYSCALE)
        head = cv2.imread(str(scene / "headmask_undist/images" / Path(name).with_suffix(".png")),
                          cv2.IMREAD_GRAYSCALE)
        if image is None or person is None or head is None or image.shape[:2] != person.shape or head.shape != person.shape:
            raise ValueError(f"Native RGB and masks disagree for {name}")
        warp, rotation, crop_k, pixels = virtual_crop(point, native_k, size)
        crop = cv2.warpPerspective(image, warp, (size, size), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpPerspective(np.minimum(person, head), warp, (size, size), flags=cv2.INTER_LINEAR)
        if (alpha > 128).sum() < 500 or (pixels < -0.5).any() or (pixels > size - 0.5).any():
            raise ValueError(f"Insufficient head support in virtual image for {name}")
        write_jpeg(seq / "images" / f"{index:08d}.jpg", crop)
        write_jpeg(seq / "alpha_maps" / f"{index:08d}.jpg", alpha)
        # VHAP normalizes pixel-boundary coordinates by width and height.
        normalized_landmarks.append(np.column_stack(((pixels + .5) / size, np.ones(68))))
        bounds.append([*pixels.min(axis=0), *pixels.max(axis=0)])
        crops.append({"name": name, "timestep": index, "train": name in train,
                      "warp": warp.tolist(), "virtual_from_native_cv": rotation.tolist(),
                      "K_integer_pixel_centers": crop_k.tolist()})
    np.savez(seq / "landmark2d/STAR.npz", face_landmark_2d=np.asarray(normalized_landmarks, np.float32),
             bounding_box=np.asarray(bounds, np.float32))
    (seq / "cameras.json").write_text(json.dumps(crops, indent=2), "utf-8")
    return seq, names, xy, hashes, native_k, crops


def setup_vhap_cameras(extrinsics=None):
    import torch
    from vhap.data.video_dataset import VideoDataset

    original_load = VideoDataset.load_camera_params
    original_get = VideoDataset.getitem_single_image

    def load(self):
        camera = original_load(self)
        records = json.loads((self.sequence_path / "cameras.json").read_text("utf-8"))
        self.virtual_camera = []
        for record in records:
            k = np.asarray(record["K_integer_pixel_centers"], dtype=np.float32)
            k[:2, 2] += .5
            self.virtual_camera.append(torch.from_numpy(k))
        return camera

    def get(self, index):
        item = original_get(self, index)
        k = self.virtual_camera[item["timestep_index_original"]].clone()
        k[:2] *= item["scale_factor"]
        item["intrinsic"] = k
        if extrinsics is not None:
            item["extrinsic"] = torch.from_numpy(extrinsics[item["timestep_index_original"]].copy())
        return item

    VideoDataset.load_camera_params = load
    VideoDataset.getitem_single_image = get


def fit_vhap(seq, output, records, landmark_epochs, rgb_epochs, seed):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    import tyro
    from vhap.config.base import BaseTrackingConfig
    from vhap.model.tracker import GlobalTracker

    setup_vhap_cameras()
    cfg = tyro.cli(BaseTrackingConfig, args=[
        "--data.root-folder", str(seq.parent), "--data.sequence", seq.name,
        "--data.calibrated", "--data.use-alpha-map", "--exp.output-folder", str(output / "track"),
        "--batch-size", "2",
        "--log.interval-media", "100000", "--log.interval-scalar", "40",
    ])
    torch.manual_seed(seed)
    np.random.seed(seed)
    tracker = GlobalTracker(cfg)
    train_indices = [record["timestep"] for record in records if record["train"]]
    held_indices = [record["timestep"] for record in records if not record["train"]]
    with torch.no_grad():
        _, neutral, landmarks, _ = tracker.forward_flame(torch.zeros(1, device=tracker.device, dtype=torch.long))
        template = landmarks[0, :68].cpu().numpy().astype(np.float64)
        centered_template = neutral[0] - neutral[0].mean(dim=0)
        template_radius = float(centered_template.norm(dim=-1).max())
        if not np.isfinite(template_radius) or template_radius <= 0:
            raise RuntimeError("Neutral FLAME template has an invalid radius")
        root = (tracker.flame.J_regressor[0] @ neutral[0]).cpu().numpy()
        cv_to_gl = np.diag([1., -1., -1.])
        subset = np.arange(17, 68)
        for record in records:
            index = record["timestep"]
            sample = tracker.dataset[index]
            points = np.array(sample["lmk2d"][:, :2], dtype=np.float64, copy=True)
            points -= .5
            k = np.asarray(record["K_integer_pixel_centers"])
            ok, rv, tv = cv2.solvePnP(template[subset], points[subset], k, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or tv[2, 0] <= 0:
                raise RuntimeError(f"VHAP initial pose failed for {record['name']}")
            world_r = cv_to_gl @ cv2.Rodrigues(rv)[0]
            world_t = cv_to_gl @ tv[:, 0] + np.array([0., 0., 1.]) + (world_r - np.eye(3)) @ root
            tracker.rotation[index].copy_(torch.as_tensor(cv2.Rodrigues(world_r)[0][:, 0], device=tracker.device))
            tracker.translation[index].copy_(torch.as_tensor(world_t, device=tracker.device))

    tracker.global_step = 0
    cfg.pipeline.lmk_global_tracking.num_epochs = landmark_epochs
    cfg.pipeline.rgb_global_tracking.num_epochs = rgb_epochs
    cfg.pipeline.lmk_sequential_tracking.num_steps = 40
    cfg.pipeline.lmk_sequential_tracking.disable_jawline_landmarks = True
    cfg.w.always_enable_jawline_landmarks = False
    loader = DataLoader(Subset(tracker.dataset, train_indices), batch_size=cfg.batch_size,
                        shuffle=True, num_workers=2)
    tracker.optimize_stage("lmk_global_tracking", dataloader=loader)
    if rgb_epochs:
        tracker.optimize_stage("rgb_global_tracking", dataloader=loader, lr_scale=.3)
    for index in held_indices:
        tracker.optimize_stage("lmk_sequential_tracking", sample=tracker.dataset.getitem_by_timestep(index))
    tracker.save_result(epoch=1000)
    tracker.tb_writer.close()
    tracked = tracker.out_dir / "tracked_flame_params_1000.npz"
    with np.load(tracked) as state:
        if state["timestep_id"].astype(int).tolist() != list(range(len(records))):
            raise RuntimeError("VHAP did not preserve selected frame order")
        if any(not np.isfinite(state[key]).all() for key in ("shape", "expr", "rotation", "translation", "static_offset")):
            raise RuntimeError("VHAP produced non-finite parameters")

    texture_seed = output / "vhap-face-seed.npz"
    with torch.no_grad():
        flame = tracker.flame
        uv = flame.verts_uvs[flame.textures_idx.long()].mean(dim=1)
        # VHAP converts mesh UVs to image row order for its own atlas sampler.
        grid = torch.stack((2 * uv[:, 0] - 1, 1 - 2 * uv[:, 1]), dim=-1)
        atlas = tracker.get_albedo()
        colors = F.grid_sample(atlas, grid[None, :, None], mode="bilinear",
                               padding_mode="border", align_corners=False)[0, :, :, 0].T
        np.savez(texture_seed, faces=flame.faces.cpu().numpy(),
                 colors=colors.clamp(0, 1).cpu().numpy())
    return tracked, texture_seed, template_radius


HEAD_SEED_OPACITY = 0.55
HEAD_SEED_AXES = (0.70, 0.14, 0.70)


def write_bound_head(tracker, texture_seed, point_cloud, template_radius):
    import torch
    from scene import FlameGaussianModel
    from utils.general_utils import inverse_sigmoid
    from utils.sh_utils import RGB2SH

    keys = ("shape", "static_offset", "expr", "rotation", "translation", "neck_pose", "jaw_pose", "eyes_pose")
    with np.load(tracker) as track, np.load(texture_seed) as seed, torch.no_grad():
        model = FlameGaussianModel(3)
        if not np.array_equal(seed["faces"], model.flame_model.faces.cpu().numpy()):
            raise ValueError("VHAP and the bound head have different FLAME triangle ordering")
        colors = torch.as_tensor(seed["colors"], dtype=torch.float32, device="cuda")
        if colors.shape != (len(model.binding), 3) or not torch.isfinite(colors).all():
            raise ValueError("Invalid VHAP triangle colors")
        model.flame_param = {key: torch.as_tensor(track[key], dtype=torch.float32, device="cuda")
                             for key in keys}
        model.flame_param["dynamic_offset"] = model.flame_param["static_offset"].new_zeros(
            (len(model.flame_param["expr"]), len(model.flame_model.v_template), 3))
        model.create_from_pcd(None, template_radius)
        model._features_dc.copy_(RGB2SH(colors)[:, None, :])
        # Basis 1 is the face normal. Scale is relative to each triangle's metric scale.
        model._scaling.copy_(colors.new_tensor(HEAD_SEED_AXES).log().expand_as(model._scaling))
        model._opacity.copy_(inverse_sigmoid(torch.full_like(model._opacity, HEAD_SEED_OPACITY)))
        point_cloud.parent.mkdir(parents=True)
        model.save_ply(str(point_cloud))
        if not torch.isfinite(model.get_xyz).all():
            raise RuntimeError("VHAP posed seed centers are not finite")
        print(f"Initialized {len(model.binding)} FLAME-bound SH0 Gaussians from VHAP UV colors", flush=True)


def initialize_bound_head(home, tracker, texture_seed, output, template_radius):
    repo = home / "GaussianAvatars"
    env = dict(os.environ)
    env.pop("PYTHONNOUSERSITE", None)
    env["PYTHONPATH"] = os.pathsep.join((str(home / "ga_pkgs"), str(repo)))
    point_cloud = output / "bound-head" / "point_cloud.ply"
    command = [
        str(home / "miniconda3/envs/fastavatar/bin/python"), "-B", str(Path(__file__).resolve()),
        "--initialize-head", str(tracker), str(texture_seed), str(point_cloud), str(template_radius),
    ]
    subprocess.run(command, cwd=repo, env=env, check=True)
    if not point_cloud.is_file() or not point_cloud.with_name("flame_param.npz").is_file():
        raise RuntimeError(f"Bound head initialization did not write {point_cloud}")
    return point_cloud


def main():
    parser = ArgumentParser(description=__doc__)
    for field in ("scene", "selection", "landmarks", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gaussian-steps", type=int, default=0, help="Legacy option. Head seeding uses zero Gaussian steps.")
    parser.add_argument("--vhap-landmark-epochs", type=int, default=8)
    parser.add_argument("--vhap-rgb-epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    if args.size < 128 or args.vhap_landmark_epochs < 1 or args.vhap_rgb_epochs < 0:
        parser.error("Use a crop of at least 128 pixels and valid VHAP stage budgets")
    home = Path.home()
    for field in ("scene", "selection", "landmarks", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty avatar output directory: {args.output}")
    cv2.setNumThreads(1)
    seq, names, xy, hashes, native_k, records = make_tracking_data(
        args.scene, args.selection, args.landmarks, args.output, args.size)
    vhap_repo = home / "VHAP"
    sys.path.insert(0, str(vhap_repo))
    os.chdir(vhap_repo)
    tracked, texture_seed, template_radius = fit_vhap(seq, args.output, records, args.vhap_landmark_epochs,
                                                      args.vhap_rgb_epochs, args.seed)
    bound = initialize_bound_head(home, tracked, texture_seed, args.output, template_radius)
    fit_landmarks = args.output / "landmarks.npz"
    np.savez(fit_landmarks, names=names, xy=xy, native_sha256=hashes, K=native_k,
             tracker_sha256=sha256(tracked.read_bytes()).hexdigest())
    manifest = {"tracker": str(tracked), "bound": str(bound), "landmarks": str(fit_landmarks),
                "frames": records, "gaussian_steps": 0, "head_source": "vhap_flame_initialization",
                "template_radius": template_radius,
                "head_initialization": {"strategy": "VHAP UV colors, one Gaussian per FLAME triangle",
                                        "opacity": HEAD_SEED_OPACITY, "scale_in_triangle_units": HEAD_SEED_AXES},
                "vhap_landmark_epochs": args.vhap_landmark_epochs, "vhap_rgb_epochs": args.vhap_rgb_epochs}
    (args.output / "avatar.json").write_text(json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--initialize-head"]:
        write_bound_head(*map(Path, sys.argv[2:5]), float(sys.argv[5]))
    else:
        main()
