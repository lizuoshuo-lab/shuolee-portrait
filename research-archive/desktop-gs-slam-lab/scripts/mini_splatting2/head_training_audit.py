"""Read-only usage audit for the isolated training-head L1 intervention."""
import hashlib
import json
import os
from pathlib import Path
import torch


@torch.no_grad()
def audit_head(guidance, camera, image, reference, valid, loss, iteration, model_path):
    region = guidance.region(camera.image_name, image.shape[-2], image.shape[-1], image.device, image.dtype)
    inside = valid * region
    outside = valid * (1 - region)
    absolute = (image - reference).abs()
    row = dict(iteration=iteration, frame=camera.image_name,
               width=image.shape[-1], height=image.shape[-2], coefficient=guidance.coefficient,
               cache_entries=len(guidance.cache), used_frames=len(guidance.used),
               valid_head_pixels=float(inside.sum()), valid_pixels=float(valid.sum()),
               head_l1=float((absolute * inside).sum() / (3 * inside.sum().clamp_min(1))),
               outside_l1=float((absolute * outside).sum() / (3 * outside.sum().clamp_min(1))),
               joint_loss=float(loss), finite=bool(torch.isfinite(loss)))
    assert row['finite'] and row['valid_head_pixels'] > 0
    assert row['cache_entries'] <= guidance.cache_bound
    with (Path(model_path) / 'head-training-audit.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def finish_head(guidance, model_path):
    root = Path(os.environ['MINIGS_HEAD_REGIONS'])
    result = dict(coefficient=guidance.coefficient, training_frames_used=sorted(guidance.used),
                  reference_cache_bound=guidance.cache_bound, training_only=True,
                  region_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                  region_review_sha256=hashlib.sha256((root/'visual-review.json').read_bytes()).hexdigest(),
                  weighting='Normalized RGB L1 only; SSIM and depth unchanged',
                  head_region_is_rectangle_not_segmentation=True)
    (Path(model_path)/'head-guidance-usage.json').write_text(json.dumps(result, indent=2))
