"""CPU panels for a completed, same-frame replay of the trained bound model."""
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import runtime as rt
from .head_appearance import snapshot_name
from .official_smplifyx import COCO


BODY_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
              (13, 15), (12, 14), (14, 16)]
POSE_KEYS = ("frame", "pose_index", "source_frame_index", "source_sha256", "mask_sha256")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _project(points, K):
    _need(np.isfinite(points).all() and np.all(points[:, 2] > .05), "Invalid camera-space geometry.")
    pixel = points @ K.T
    return pixel[:, :2] / pixel[:, 2:]


def _crop(row):
    wh = np.asarray(row["wh"])
    box = np.asarray(row.get("initial_box_xyxy", [0, 0, *wh]), dtype=float)
    _need(box.shape == (4,) and np.isfinite(box).all(), "Invalid native source crop.")
    low = np.maximum(np.floor(box[:2] - 64).astype(int), 0)
    high = np.minimum(np.ceil(box[2:] + 64).astype(int), wh)
    _need(np.all(high > low), "The native crop is empty.")
    return tuple(np.r_[low, high].tolist())


def _mesh_panel(original, box, vertices, faces, K):
    uv = _project(vertices, K) - box[:2]
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    shade = .3 + .7 * np.abs(normals @ np.array([.3, -.2, -.9327379]))
    layer = np.array(original)
    mask = np.zeros(layer.shape[:2], np.uint8)
    projected = np.rint(uv[faces]).astype(np.int32)
    for index in np.argsort(-triangles[:, :, 2].mean(1), kind="stable"):
        polygon = projected[index]
        if (polygon.max(0) < 0).any() or (polygon.min(0) >= original.size).any():
            continue
        color = tuple(int(channel * shade[index]) for channel in (80, 210, 220))
        cv2.fillConvexPoly(layer, polygon, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, polygon, 255)
    raw = np.array(original)
    keep = mask != 0
    raw[keep] = np.rint(.6 * layer[keep] + .4 * raw[keep]).astype(np.uint8)
    return Image.fromarray(raw)


def _features(original, box, observed, projected):
    panel = original.copy()
    draw = ImageDraw.Draw(panel)
    visible = observed[:, 2] > 0
    green, magenta = "#50ff58", "#ff44dc"
    for xy, color in ((observed[:, :2], green), (projected, magenta)):
        points = xy - box[:2]
        for left, right in BODY_EDGES:
            a, b = COCO[left], COCO[right]
            if visible[a] and visible[b]:
                draw.line([tuple(points[a]), tuple(points[b])], fill=color, width=2)
    for x, y in observed[visible, :2] - box[:2]:
        draw.ellipse((x-3, y-3, x+3, y+3), outline=green, width=1)
    for x, y in projected[visible] - box[:2]:
        draw.ellipse((x-1, y-1, x+1, y+1), fill=magenta)
    return panel


def _panel(parts):
    titles = ("Original native RGB", "Same-frame bound Gaussian",
              "Trained offset mesh", "Input (green) / fit projection (magenta)")
    w, h = parts[0].size
    result = Image.new("RGB", (4*w, h+26), "#181818")
    draw = ImageDraw.Draw(result)
    for col, (part, title) in enumerate(zip(parts, titles, strict=True)):
        result.paste(part, (col*w, 26))
        draw.text((col*w+5, 5), title, fill="white")
    return result

def write_panels(root, replay_receipt, *, output_dir=None):
    """Write one unwarped four-panel PNG per hash-bound training frame."""
    root = Path(root).resolve()
    replay_receipt = Path(replay_receipt)
    if not replay_receipt.is_absolute():
        replay_receipt = root / replay_receipt
    replay_receipt = replay_receipt.resolve()
    audit = (root / "audit").resolve()
    destination = (audit / "bound-panels") if output_dir is None else Path(output_dir)
    if not destination.is_absolute():
        destination = audit / destination
    destination = destination.resolve()
    _need(audit in destination.parents and not destination.exists(), "Use a fresh directory below the run audit folder.")
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = rt.sha(path)
        _need((expected is None or digest == expected) and pins.get(path, digest) == digest,
              f"Replay artifact changed: {path}")
        pins[path] = digest
        return path

    run = root / "gaussian-run"
    train = rt.read(pin(run / "result.json"))
    _need(train["state"] == "completed" and train["gaussian_profile"] == "smpl_sh0"
          and train["sh_degree"] == 0, "Need a completed terminal SH0 snapshot.")
    snapshot = run / snapshot_name(train)
    replay = snapshot / "posed-replay"
    _need(replay_receipt == (replay / "result.json").resolve(), "Use the terminal bound replay receipt.")
    receipt = rt.read(pin(replay_receipt))
    _need(receipt["state"] == "completed" and receipt["optimizer_updates"] == 0
          and receipt["mesh_file"] == "meshes.npz", "Missing completed same-frame bound render and mesh receipt.")
    parity = receipt["reference_reload_parity"]
    _need(0 <= parity["mean_abs"] < 2e-6 and 0 <= parity["max_abs"] < .005
          and parity["archived_max_abs_8bit"] == 0, "The native reference replay failed.")
    manifest = rt.read(pin(snapshot / "bound-replay.json", receipt["bound_replay_sha256"]))
    _need(manifest["schema"] == "shuolee-smplx-bound-v1" and manifest["bound_ply"] == "bound-local.ply"
          and manifest["sh_degree"] == 0 and manifest["points"] == train["points"],
          "Expected a stock trained bound SH0 snapshot.")
    pin(snapshot / "bound-local.ply", manifest["bound_ply_sha256"])
    fit_path = pin(root / "fit/fitted.npz", manifest["inputs_sha256"]["fitted"])
    rows_path = pin(root / "data/training-inputs.json", receipt["training_inputs_sha256"])
    _need(pins[rows_path] == manifest["inputs_sha256"]["training_rows"], "Original training inputs changed.")
    rows = rt.read(rows_path)["records"]
    cameras = rt.read(pin(run / "training-cameras.json", receipt["training_camera_sha256"]))
    records = receipt["records"]
    ids = [row["frame"] for row in rows]
    _need(ids and len(set(ids)) == len(ids) and all(re.fullmatch(r"[A-Za-z0-9_-]+", frame) for frame in ids)
          and ids == [item["frame"] for item in records] and not set(ids) & set(receipt["retained_holdouts"])
          and cameras["source_sha256"] == pins[rows_path],
          "Replay and selected native training identities differ.")
    _need(manifest["frames"] == [{key: item[key] for key in POSE_KEYS} for item in records],
          "Same-frame Gaussian render and terminal bound geometry differ.")
    mesh_path = pin(replay / receipt["mesh_file"], receipt["mesh_sha256"])
    with np.load(mesh_path, allow_pickle=False) as data, np.load(fit_path, allow_pickle=False) as fit:
        mesh = data["vertices_world"]
        faces = data["faces"]
        slots = [item["pose_index"] for item in records]
        _need(all(type(slot) is int for slot in slots) and sorted(slots) == list(range(len(ids)))
              and np.array_equal(data["frame_ids"], ids) and np.array_equal(data["pose_indices"], slots)
              and np.array_equal(data["source_frames"], [row["source_frame_index"] for row in rows])
              and np.array_equal(data["image_sha256"], [row["source_sha256"] for row in rows])
              and np.array_equal(data["mask_sha256"], [row["mask_sha256"] for row in rows]),
              "Bound mesh frame/pose/source identity differs.")
        _need(np.array_equal(fit["frame_ids"][slots], ids)
              and np.array_equal(fit["source_frames"][slots], data["source_frames"])
              and np.array_equal(fit["image_sha256"][slots], data["image_sha256"])
              and np.array_equal(fit["mask_sha256"][slots], data["mask_sha256"])
              and mesh.shape == fit["vertices_world"][slots].shape
              and np.array_equal(faces, fit["faces"]) and len(faces) > 0
              and faces.dtype.kind in "iu" and faces.min() >= 0 and faces.max() < mesh.shape[1],
              "Saved fitted pose and trained mesh identities differ.")
        ref, ref_slot = receipt["reference_frame_id"], receipt["reference_pose_index"]
        _need(ref in ids and ref_slot == slots[ids.index(ref)]
              and ref == manifest["reference_frame_id"] == fit["reference_frame_id"].item()
              and ref_slot == manifest["reference_pose_index"], "Reference pose identity differs.")
        observed = fit["observed_native_keypoints"][slots]
        joints = fit["joints_camera"][slots]
        saved_uv = fit["joints_projected_native"][slots]
        integer_K = fit["K_integer_centers"][slots]
        extrinsics = fit["w2c"][slots]
    _need(observed.shape == joints.shape == (len(ids), 118, 3)
          and saved_uv.shape == (len(ids), 118, 2) and np.isfinite(observed).all()
          and np.all((observed[..., 2] >= 0) & (observed[..., 2] <= 1)), "Invalid fitted observations.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    with TemporaryDirectory(prefix=".bound-panels-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "panels"
        stage.mkdir()
        for i, (row, item, camera) in enumerate(zip(rows, records, cameras["records"], strict=True)):
            frame = row["frame"]
            K, E = integer_K[i], extrinsics[i]
            corner = K.copy()
            corner[:2, 2] += .5
            _need(item["source_frame_index"] == row["source_frame_index"]
                  and item["source_sha256"] == row["source_sha256"] and item["mask_sha256"] == row["mask_sha256"]
                  and item["image"] == frame + ".png" and camera["frame"] == frame
                  and camera["pose_index"] == slots[i] and camera["wh"] == row["wh"]
                  and camera["pixel_convention"] == "pixel_corners"
                  and all(item["camera"][key] == camera[key]
                          for key in ("frame", "pose_index", "wh", "K", "w2c", "pixel_convention"))
                  and np.array_equal(K, row["K_integer_centers"]) and np.array_equal(corner, row["K"])
                  and np.array_equal(corner, camera["K"]) and np.array_equal(E, row["w2c"])
                  and np.array_equal(E, camera["w2c"]), "Same-frame replay camera or source differs: " + frame)
            box = _crop(row)
            source_path = pin(row["image"], row["source_sha256"])
            pin(row["mask"], row["mask_sha256"])
            image_path = pin(replay / item["image"], item["image_sha256"])
            with Image.open(source_path) as source, Image.open(image_path) as rendered:
                _need(source.size == rendered.size == tuple(row["wh"]), "Replay native raster differs: " + frame)
                rgb = source.convert("RGB").crop(box)
                gaussian = rendered.convert("RGB").crop(box)
            uv = _project(joints[i], K)
            projection_error = float(np.abs(uv - saved_uv[i]).max())
            _need(projection_error < .001, "Fitted joint projection disagrees with integer K: " + frame)
            vertices_camera = mesh[i] @ E[:3, :3].T + E[:3, 3]
            four = _panel((rgb, gaussian, _mesh_panel(rgb, box, vertices_camera, faces, K),
                           _features(rgb, box, observed[i], uv)))
            path = stage / (frame + ".png")
            four.save(path)
            panels.append(dict(frame=frame, pose_index=slots[i], source_frame_index=row["source_frame_index"],
                               source_sha256=row["source_sha256"], gaussian_sha256=item["image_sha256"],
                               projection_error_px=projection_error, crop_xyxy=list(box),
                               panel=path.name, panel_sha256=rt.sha(path)))
        _need(all(rt.sha(path) == digest for path, digest in pins.items()), "Replay inputs changed during panel generation.")
        result = dict(state="completed", reference_frame_id=ref, reference_pose_index=ref_slot,
                      replay_receipt_sha256=pins[replay_receipt], mesh_sha256=pins[mesh_path],
                      source_pins_sha256={str(path): digest for path, digest in pins.items()},
                      native_crops=True, same_frame_bound_gaussians=True, optimizer_updates=0, quality_accepted=False,
                      panels=panels)
        rt.save(stage / "result.json", result)
        _need(not destination.exists(), "A prior panel output appeared during review.")
        stage.rename(destination)
    return result
