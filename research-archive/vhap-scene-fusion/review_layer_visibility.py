"""Review head visibility with the same scene camera and depth sorting as training."""

import json
from pathlib import Path

import cv2
import torch

from joint_review import _bgr, _panels


@torch.no_grad()
def measure_visibility(head_only, visible, head_rgb):
    """Measure occlusion within the fixed source head-RGB support."""
    head_only, visible = head_only.float(), visible.float()
    support = torch.as_tensor(head_rgb, dtype=torch.bool, device=head_only.device)
    if head_only.ndim != 2 or visible.shape != head_only.shape or support.shape != head_only.shape:
        raise ValueError("Layer alpha and image support must have the same HW shape")
    if not torch.isfinite(head_only).all() or not torch.isfinite(visible).all():
        raise ValueError("Layer render returned nonfinite alpha")
    head_only, visible = head_only[support].clamp(0, 1), visible[support].clamp(0, 1)
    mass = head_only.sum().item()
    opaque = head_only >= .8
    return {
        "head_rgb_pixels": head_only.numel(),
        "head_only_alpha_mean": head_only.mean().item() if head_only.numel() else None,
        "head_visible_alpha_mean": visible.mean().item() if visible.numel() else None,
        "head_alpha_mass_blocked_by_scene_fraction": 1.0 - visible.sum().item() / mass if mass else None,
        "head_visible_alpha_mean_on_opaque_head_core": visible[opaque].mean().item() if opaque.any() else None,
        "head_only_opaque_core_pixels": int(opaque.sum().item()),
        "opaque_core_threshold": .8,
    }


@torch.no_grad()
def review_layers(output, frames, scene, head_for, *, render, visible_head, panel_frames=None):
    """Write per-frame layer metrics and aligned RGB crops for requested frames."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels = set(panel_frames) if panel_frames is not None else None
    report = {}
    scene_count = len(scene[0])
    for frame in frames:
        head = head_for(frame)
        joined = tuple(torch.cat((scene_field, head_field)) for scene_field, head_field in zip(scene, head, strict=True))
        only = render(frame.rasterizer, head, alpha_only=True)
        visible = visible_head(frame.rasterizer, joined, scene_count)
        report[frame.name] = measure_visibility(only, visible, frame.masks["head_rgb"])
        if panels is not None and frame.name not in panels:
            continue

        sy, sx = frame.crop
        rgb = (frame.reference, render(frame.rasterizer, scene), render(frame.rasterizer, head),
               render(frame.rasterizer, joined))
        labels = ("Source", "Scene only", "Head only", "Joint render")
        panel = _panels((_bgr(image[:, sy, sx]) for image in rgb), labels, 220)
        path = output / f"{Path(frame.name).stem}-layers.jpg"
        ok, data = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise OSError(f"Could not encode {path}")
        path.write_bytes(data.tobytes())

    (output / "layer-visibility.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
