"""Freeze visible hair silhouette and internal collar evidence from registered VHAP crops."""

from argparse import ArgumentParser
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from detect_vhap_neck import _load_parser
from review_smplx_head_neck import panel


PARSER_WEIGHTS = "18c44e06b77e47da86fe2b48fac78e0ccbd0b5241c829f9487ebf7fd4a6389e6"
MIN_CONFIDENCE = .55
HAIR_BAND = 64


def load_parser(repo, weights, device):
    if "weights_only" in signature(torch.load).parameters:
        return _load_parser(repo, weights, device)
    # Local PyTorch 1.12 lacks weights_only. Only load the already pinned archive in that runtime.
    if sha256(weights.read_bytes()).hexdigest() != PARSER_WEIGHTS:
        raise ValueError("This PyTorch version accepts only the pinned outline parser checkpoint")
    original = torch.load

    def load_pinned(path, *, map_location, weights_only):
        if Path(path).resolve() != weights.resolve() or not weights_only:
            raise ValueError("Expected the pinned outline parser checkpoint")
        return original(path, map_location=map_location)

    torch.load = load_pinned
    try:
        return _load_parser(repo, weights, device)
    finally:
        torch.load = original


@torch.inference_mode()
def consistent_probabilities(model, bgr, device):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = torch.as_tensor(cv2.resize(rgb, (512, 512)), device=device).permute(2, 0, 1)[None].float() / 255.
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    image = (image - mean) / std
    probs = model(torch.cat((image, image.flip(-1)), dim=0))[0].softmax(1).cpu().numpy()
    repeated = probs[1, :, :, ::-1].copy()
    # Mirrored eye, eyebrow and ear classes refer to the opposite side.
    repeated[[2, 3, 4, 5, 7, 8]] = repeated[[3, 2, 5, 4, 8, 7]]
    consistent = np.minimum(probs[0], repeated)
    return cv2.resize(consistent.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)


def outline_targets(alpha, probability):
    """Return hair loss support, observed hair edge, and internal neck/clothing transition."""
    height, width = alpha.shape
    pixel_scale = min(height, width) / 512
    count, component, stats, _ = cv2.connectedComponentsWithStats((alpha >= 128).astype(np.uint8))
    empty = np.zeros_like(alpha, dtype=bool)
    if count < 2:
        return empty, empty, empty
    person = component == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    contours = cv2.findContours(person.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    contour = max(contours, key=cv2.contourArea)[:, 0]
    mask_edge = empty.copy()
    if len(contour) >= 16:
        stride = max(2, round(4 * pixel_scale))
        tangent = np.roll(contour, -stride, axis=0) - np.roll(contour, stride, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.linalg.norm(tangent, axis=1)[:, None].clip(min=1)
        distance = max(3., 7 * pixel_scale)
        bound = np.array([width - 1, height - 1])
        a = np.rint(contour - distance * normal).astype(int)
        b = np.rint(contour + distance * normal).astype(int)
        in_bounds = ((a >= 0) & (a <= bound) & (b >= 0) & (b <= bound)).all(axis=1)
        a, b = a.clip(0, bound), b.clip(0, bound)
        turn = alpha[a[:, 1], a[:, 0]] > alpha[b[:, 1], b[:, 0]]
        x_in, y_in = np.where(turn[:, None], a, b).T
        x_out, y_out = np.where(turn[:, None], b, a).T
        reliable = (in_bounds & (alpha[y_in, x_in] >= 228) & (alpha[y_out, x_out] <= 24)
                    & (probability[17, y_in, x_in] >= MIN_CONFIDENCE)
                    & (probability[0, y_out, x_out] >= MIN_CONFIDENCE))
        edge_x, edge_y = contour[reliable].T
        mask_edge[edge_y, edge_x] = True

    protected = person & (probability[[*range(1, 17), 18]].max(axis=0) >= MIN_CONFIDENCE)
    kernel_size = max(3, round(6 * pixel_scale) | 1)
    protected = cv2.dilate(protected.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)) != 0
    mask_edge &= ~protected
    hair = empty.copy()
    if mask_edge.sum() >= 16:
        distance = cv2.distanceTransform((~mask_edge).astype(np.uint8), cv2.DIST_L2, 5)
        hair = (distance <= HAIR_BAND * pixel_scale) & ~protected
    else:
        mask_edge = empty.copy()

    certain_person = person & (alpha >= 224)
    neck = certain_person & (probability[14] >= MIN_CONFIDENCE)
    cloth = certain_person & (probability[16] >= MIN_CONFIDENCE)
    if not neck.any() or not cloth.any():
        return hair, mask_edge, empty
    distance_neck = cv2.distanceTransform((~neck).astype(np.uint8), cv2.DIST_L2, 5)
    distance_cloth = cv2.distanceTransform((~cloth).astype(np.uint8), cv2.DIST_L2, 5)
    non_collar = probability[[15, 17, 18]].max(axis=0) >= MIN_CONFIDENCE
    collar = (certain_person & ~non_collar & (distance_neck <= 6 * pixel_scale)
              & (distance_cloth <= 6 * pixel_scale)
              & (np.abs(distance_neck - distance_cloth) <= 1.5 * pixel_scale))
    return hair, mask_edge, collar


def run(args):
    model = load_parser(args.parser_repo, args.weights, args.device)
    fields = {key: [] for key in ("hair", "hair_edge", "collar", "names", "crop_sha256", "matte_sha256")}
    previews = []
    statistics = []
    with ZipFile(args.frames) as archive:
        initialization = archive.read("initialization.json")
        records = json.loads(initialization)["frames"]
        views = [record["name"] for record in records if abs(record["baseline_yaw_deg"]) >= 80.][:4]
        for group in ("train", "holdout"):
            candidates = [record for record in records if record.get(group)]
            if candidates:
                views.append(min(candidates, key=lambda r: abs(r["baseline_yaw_deg"]))["name"])
        for i, row in enumerate(records):
            if int(row["timestep"]) != i:
                raise ValueError("VHAP crop archive is out of frame order")
            name = row["name"]
            crop_bytes = archive.read(f"images/{i:08d}.jpg")
            alpha_bytes = archive.read(f"alpha_maps/{i:08d}.jpg")
            bgr = cv2.imdecode(np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR)
            alpha = cv2.imdecode(np.frombuffer(alpha_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if bgr is None or alpha is None or bgr.shape[:2] != alpha.shape:
                raise ValueError(f"Invalid paired crop and alpha: {name}")
            probability = consistent_probabilities(model, bgr, args.device)
            hair, edge, collar = outline_targets(alpha, probability)
            fields["names"].append(name)
            fields["crop_sha256"].append(sha256(crop_bytes).hexdigest())
            fields["matte_sha256"].append(sha256(alpha_bytes).hexdigest())
            for key, target in (("hair", hair), ("hair_edge", edge), ("collar", collar)):
                fields[key].append(target)
            stats = {"name": name, "hair_roi_pixels": int(hair.sum()), "hair_edge_pixels": int(edge.sum()),
                     "collar_pixels": int(collar.sum())}
            statistics.append(stats)
            if name in views:
                hair_preview, collar_preview = bgr.copy(), bgr.copy()
                hair_preview[hair] = (hair_preview[hair] * .72 + np.array([48, 207, 121]) * .28).astype(np.uint8)
                hair_preview[cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (90, 255, 161)
                collar_preview[cv2.dilate(collar.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (25, 166, 255)
                previews.append(np.hstack((panel(bgr, f"{name} | original"),
                                           panel(hair_preview, f"Hair witnesses: {stats['hair_edge_pixels']}"),
                                           panel(collar_preview, f"Skin-cloth seam: {stats['collar_pixels']}"))))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "outline-targets.npz", **fields,
                        initialization_sha256=sha256(initialization).hexdigest(),
                        weight_sha256=sha256(args.weights.read_bytes()).hexdigest())
    if not previews or not cv2.imwrite(str(args.output / "outline-target-review.jpg"), np.vstack(previews)):
        raise OSError("Cannot write the source outline review")
    report = {"frames": statistics, "hair_band_px_at_512": HAIR_BAND, "minimum_parser_confidence": MIN_CONFIDENCE,
              "hair": "two-sided search support near verified exterior alpha edge, excluding protected regions",
              "hair_edge": "verified hair-to-background boundary on the source alpha",
              "collar": "internal, high-confidence neck-skin to clothing band; absent when not visible"}
    (args.output / "outline-targets.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("frames", "parser-repo", "weights", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    run(parser.parse_args())
