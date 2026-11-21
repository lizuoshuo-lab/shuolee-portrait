"""Measure and review the same scene camera with fixed source-image masks."""

from pathlib import Path
import math

import cv2
import numpy as np
import torch


@torch.no_grad()
def assess(rgb, reference, visible_head, masks):
    """Score normalized CHW RGB and the visible head's HW alpha contribution."""
    rgb = rgb.float().clamp(0, 1)
    reference = reference.to(device=rgb.device, dtype=rgb.dtype).clamp(0, 1)
    alpha = visible_head.to(device=rgb.device, dtype=rgb.dtype).clamp(0, 1)
    fixed = {
        name: torch.as_tensor(masks[name], device=rgb.device, dtype=torch.bool)
        for name in ("face", "seam", "scene", "head_allowed")
    }
    residual = rgb - reference
    pixel_error = residual.abs().mean(dim=0) * 255.0
    # A finite floor keeps exact image parity serializable as JSON.
    mse = residual.square().mean().item()
    scores = {"rgb_psnr_db": -10.0 * math.log10(max(mse, 1e-12))}
    for name in ("face", "seam", "scene"):
        mask = fixed[name]
        count = int(mask.sum().item())
        scores[f"{name}_pixels"] = count
        scores[f"{name}_mae_255"] = pixel_error[mask].mean().item() if count else None

    head_mass = alpha.sum().item()
    outside = alpha[~fixed["head_allowed"]].sum().item()
    scores["visible_head_outside_fraction"] = outside / head_mass if head_mass else None
    face_alpha = alpha[fixed["face"]]
    scores["face_uncovered_fraction"] = (
        (face_alpha < 0.5).float().mean().item() if face_alpha.numel() else None
    )
    scores["face_transmission_fraction"] = (
        (1.0 - face_alpha).mean().item() if face_alpha.numel() else None
    )
    return scores


def _bgr(rgb):
    pixels = rgb.detach().float().clamp(0, 1).mul(255).round().byte()
    return cv2.cvtColor(pixels.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)


def _panels(images, labels, width):
    panels = []
    for image, label in zip(images, labels):
        height = round(image.shape[0] * width / image.shape[1])
        interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
        image = cv2.resize(image, (width, height), interpolation=interpolation)
        label_strip = np.full((30, width, 3), 22, dtype=np.uint8)
        cv2.putText(label_strip, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.49, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(np.vstack((label_strip, image)))
    return np.hstack(panels)


def save_review(path, rgb0, rgb1, reference, initial_alpha, final_alpha, masks):
    """Save full frames and identical head-and-collar crops for visual review."""
    source, before, after = map(_bgr, (reference, rgb0, rgb1))
    allowed = torch.as_tensor(masks["head_allowed"]).detach().cpu().numpy().astype(bool)
    seam = torch.as_tensor(masks["seam"]).detach().cpu().numpy().astype(bool)
    bounds = (allowed | seam).astype(np.uint8)
    frame_h, frame_w = bounds.shape
    x, y, w, h = cv2.boundingRect(bounds)
    pad = round(max(w, h) * 0.10)
    crop = np.s_[max(0, y - pad):min(frame_h, y + h + pad),
                 max(0, x - pad):min(frame_w, x + w + pad)] if w and h else np.s_[:, :]
    visible = []
    for alpha in (initial_alpha, final_alpha):
        alpha = alpha.detach().float().clamp(0, 1).cpu().numpy()
        image = cv2.cvtColor(np.rint(255.0 * alpha).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        outside = ~allowed
        image[outside] = 0
        image[:, :, 2][outside] = np.rint(255.0 * alpha[outside]).astype(np.uint8)
        visible.append(image[crop])

    contours, _ = cv2.findContours(seam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seam_overlay = source.copy()
    cv2.drawContours(seam_overlay, contours, -1, (55, 235, 75), 1, cv2.LINE_AA)
    width = 320
    full = _panels((source, before, after), ("Source", "Baseline", "Trained"), width)
    detail = _panels(tuple(image[crop] for image in (source, before, after)),
                     ("Source detail", "Baseline detail", "Trained detail"), width)
    visibility = _panels((seam_overlay[crop], *visible),
                         ("Seam mask", "Baseline alpha | red: outside", "Trained alpha | red: outside"), width)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output.suffix or ".jpg", np.vstack((full, detail, visibility)))
    if not ok:
        raise ValueError(f"Could not encode review image: {output}")
    output.write_bytes(encoded.tobytes())
