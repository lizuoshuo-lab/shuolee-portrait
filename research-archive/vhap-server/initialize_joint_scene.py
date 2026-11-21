"""Initialize a scene Gaussian node from supported COLMAP tracks and optional body points."""

from argparse import ArgumentParser
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from prepare_video_scene import sha256


def read_colmap(gaussian_repo, sparse):
    # The original SIBR reader retains the observation indices that the 3DGS loader discards.
    reader_path = gaussian_repo / "SIBR_viewers/src/projects/dataset_tools/preprocess/fullColmapProcess/read_write_model.py"
    spec = spec_from_file_location("colmap_sibr_model", reader_path)
    reader = module_from_spec(spec)
    spec.loader.exec_module(reader)
    return reader.read_model(str(sparse), ext=".bin")


def read_scene(scene, gaussian_repo):
    manifest = json.loads((scene / "source-frames.json").read_text(encoding="utf-8"))
    records = manifest["frames"]
    sparse = scene / "orig_undist/sparse"
    digests = {name: sha256(sparse / name) for name in ("cameras.bin", "images.bin", "points3D.bin")}
    for name in ("cameras.bin", "images.bin"):
        if digests[name] != sha256(scene / "mask_undist/sparse" / name):
            raise ValueError(f"RGB and mask COLMAP models disagree: {name}")
    cameras, images, points = read_colmap(gaussian_repo, sparse)
    if len(cameras) != 1:
        raise ValueError("Expected one calibrated COLMAP camera")
    camera = next(iter(cameras.values()))
    if (camera.model != "PINHOLE" or not np.isfinite(camera.params).all() or min(camera.params[:2]) <= 0
            or list(manifest["native_size"]) != [camera.width, camera.height]):
        raise ValueError("Expected the registered native PINHOLE raster")
    names = [record["name"] for record in records]
    by_name = {image.name: image for image in images.values()}
    if (len(set(names)) != len(names) or len(by_name) != len(images) or set(names) != set(by_name)
            or any(Path(name).name != name for name in names)
            or len({Path(name).stem for name in names}) != len(names)):
        raise ValueError("Source manifest and registered COLMAP frame names differ")

    fx, fy, cx, cy = map(float, camera.params)
    camera_rows = []
    for index, name in enumerate(names):
        image = by_name[name]
        rotation = image.qvec2rotmat().T
        center = -rotation @ image.tvec
        if (image.camera_id != camera.id or not np.isfinite(center).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-6)):
            raise ValueError(f"Invalid native COLMAP camera for {name}")
        camera_rows.append(dict(id=index, img_name=Path(name).stem, width=camera.width, height=camera.height,
                                position=center.tolist(), rotation=rotation.tolist(), fx=fx, fy=fy, cx=cx, cy=cy))
    return records, camera, images, by_name, points, camera_rows, digests


def sample_observations(scene, record, image, size):
    name = record["name"]
    paths = (scene / "orig_undist/images" / name, scene / "mask_undist/images" / name,
             scene / "headmask_undist/images" / Path(name).with_suffix(".png"))
    hashes = [sha256(path) for path in paths]
    if hashes[0] != record["native_sha256"]:
        raise ValueError(f"Native source bytes differ from the registration manifest: {name}")
    rgb, person, head = [cv2.imread(str(path), mode) for path, mode in
                         zip(paths, (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_GRAYSCALE), strict=True)]
    if any(array is None or array.shape[:2] != size for array in (rgb, person, head)):
        raise ValueError(f"RGB, person matte and coarse head mask differ in size: {name}")
    if not (head >= 128).any() or not (person >= 224).any() or not (person <= 16).any():
        raise ValueError(f"Incomplete head, body or room mask support: {name}")

    # COLMAP pixel centers start at (0.5, 0.5), so floor is nearest array index.
    xy = image.xys
    height, width = size
    valid = ((image.point3D_ids >= 0) & np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1)
             & (xy < np.array([width, height])).all(axis=1))
    x, y = np.floor(xy[valid]).astype(np.int64).T
    person_values, head_values = person[y, x], head[y, x]
    room = (person_values <= 16) & (head_values <= 40)
    head_hit = (person_values >= 224) & (head_values >= 128)
    labels = np.zeros(len(xy), np.uint8)
    labels[valid] = np.where(room, 1, np.where(head_hit, 2, 3))
    colors = np.zeros((len(xy), 3), np.uint8)
    colors[valid] = rgb[y, x, ::-1]
    return labels, colors, dict(name=name, rgb_sha256=hashes[0], person_sha256=hashes[1], head_sha256=hashes[2])


def sampled_room_points(points, images, samples):
    xyz, rgb = [], []
    counts = Counter()
    for point_id in sorted(points):
        point = points[point_id]
        if not np.isfinite(point.xyz).all() or not np.isfinite(point.error):
            raise ValueError(f"Non-finite COLMAP point {point_id}")
        support = {}
        head_hit = False
        for image_id, pixel_index in zip(point.image_ids, point.point2D_idxs, strict=True):
            image = images.get(image_id)
            if image is None or not 0 <= pixel_index < len(image.point3D_ids) or image.point3D_ids[pixel_index] != point_id:
                raise ValueError(f"Inconsistent COLMAP track backlink for point {point_id}")
            if image_id not in samples:
                continue
            labels, colors = samples[image_id]
            label = labels[pixel_index]
            if label == 1:
                support.setdefault(image_id, colors[pixel_index])
            elif label == 2:
                head_hit = True
        if len(support) < 2:
            counts["dropped_head_overlap" if head_hit else "dropped_other_insufficient_room_support"] += 1
            continue
        xyz.append(point.xyz)
        rgb.append(np.median(list(support.values()), axis=0) / 255.)
        counts["kept_room"] += 1
        if head_hit:
            counts["kept_room_with_head_occlusion"] += 1
    if len(xyz) < 4:
        raise ValueError("Fewer than four COLMAP points have two clean training-room observations")
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.float32), dict(counts)


def read_body(path):
    with np.load(path, allow_pickle=False) as body:
        xyz = np.asarray(body["xyz"], dtype=np.float32)
        colors = body["rgb"]
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.
    elif np.issubdtype(colors.dtype, np.floating):
        colors = colors.astype(np.float32)
    else:
        raise ValueError("Body RGB must be uint8 [0,255] or floating point [0,1]")
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or colors.shape != xyz.shape
            or not np.isfinite(xyz).all() or not np.isfinite(colors).all() or colors.min() < 0 or colors.max() > 1):
        raise ValueError("Body seed must supply finite same-gauge Nx3 xyz and RGB in range")
    return xyz, colors


def write_gaussians(output, gaussian_repo, xyz, rgb, extent):
    # Use the upstream unbound initializer and serialization, with no training setup.
    sys.path.insert(0, str(gaussian_repo))
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    model = GaussianModel(3)
    model.create_from_pcd(BasicPointCloud(xyz, rgb, np.zeros_like(xyz)), extent)
    model.save_ply(str(output))
    vertex = PlyData.read(output)["vertex"].data
    serialized = np.column_stack([vertex[key] for key in ("x", "y", "z")])
    if not np.array_equal(serialized, xyz) or not all(np.isfinite(vertex[key]).all() for key in vertex.dtype.names):
        raise RuntimeError("Scene initialization PLY failed exact XYZ or finite property readback")


def run(args):
    if not args.audit_only and args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    records, camera, images, by_name, points, camera_rows, digests = read_scene(args.scene, args.gaussian_repo)
    holdout = set(args.holdout)
    if len(holdout) != len(args.holdout) or not holdout <= set(by_name) or len(by_name) - len(holdout) < 3:
        raise ValueError("Provide unique registered holdouts and leave at least three training frames")

    samples, evidence = {}, []
    for record in records:
        name = record["name"]
        if name in holdout:
            continue
        image = by_name[name]
        labels, colors, hashes = sample_observations(args.scene, record, image, (camera.height, camera.width))
        samples[image.id] = (labels, colors)
        evidence.append(hashes)
    xyz, rgb, counts = sampled_room_points(points, images, samples)
    centers = np.array([view["position"] for view, record in zip(camera_rows, records, strict=True)
                        if record["name"] not in holdout])
    extent = float(1.1 * np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid scene extent in the registered training views")

    counts["colmap_total"] = len(points)
    body = None
    if args.body_points is not None:
        body_xyz, body_rgb = read_body(args.body_points)
        xyz = np.concatenate((xyz, body_xyz))
        rgb = np.concatenate((rgb, body_rgb))
        body = dict(path=str(args.body_points), sha256=sha256(args.body_points), count=len(body_xyz))
    report = dict(scene=str(args.scene), colmap_binary_sha256=digests,
                  source_manifest_sha256=sha256(args.scene / "source-frames.json"), training_inputs=evidence,
                  holdout=args.holdout, sparse_counts=counts, optional_body_seed=body, scene_extent=extent,
                  gaussian_count=len(xyz), trained_steps=0, camera_count=len(camera_rows),
                  holdout_scope="Holdout images and masks were not sampled for seed selection or color. "
                  "The pre-existing COLMAP calibration and triangulation are reused across the complete sequence.",
                  room_seed_rule="Two distinct training-frame COLMAP observations at person<=16 and head<=40; "
                  "median RGB from their original undistorted pixels. Head overlap can be occlusion and is excluded "
                  "from colors. COLMAP geometry was built after full-person inpainting, so it supplies no body seed.")
    print(json.dumps({key: value for key, value in report.items() if key != "training_inputs"}, indent=2), flush=True)
    if args.audit_only:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".scene-init-", dir=args.output.parent) as scratch:
        staged = Path(scratch) / "output"
        staged.mkdir()
        write_gaussians(staged / "point_cloud.ply", args.gaussian_repo, xyz, rgb, extent)
        (staged / "cameras.json").write_text(json.dumps(camera_rows, indent=2) + "\n", encoding="utf-8")
        (staged / "initialization.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.rename(args.output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout", required=True, nargs="+")
    parser.add_argument("--gaussian-repo", type=Path, default=Path.home() / "gaussian-splatting")
    parser.add_argument("--body-points", type=Path, help="Optional same-world NPZ with xyz and uint8 or normalized rgb.")
    parser.add_argument("--audit-only", action="store_true", help="Verify the inputs without writing files or using CUDA.")
    cv2.setNumThreads(1)
    run(parser.parse_args())
