"""Run the official 6DRepNet360 Full Rotation backbone on observed RGB head crops."""

import math

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms


_INPUT = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _backbone():
    # Official test.py uses torchvision Bottleneck and [3,4,6,3]. Its fixed
    # 7x7 average pool agrees with adaptive pooling for 224x224 input.
    return models.resnet50(weights=None, num_classes=6)


def load_profile_pose(weights, device="cpu"):
    """Load an explicit local 6DRepNet360 Full Rotation checkpoint, without network IO."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = dict(checkpoint.get("model_state_dict", checkpoint))
    for field in ("weight", "bias"):
        state[f"fc.{field}"] = state.pop(f"linear_reg.{field}")
    model = _backbone()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def profile_pose_crop(image_bgr, head_mask):
    """Return the RGB square and its native (left, top, right, bottom) pixel bounds."""
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("6DRepNet360 requires an observed uint8 BGR raster")
    mask = np.asarray(head_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("Head support must match the native RGB raster")
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if min(w, h) < 12:
        raise ValueError("Observed head support is missing or too small")
    side = math.ceil(max(w, h) * 1.26)
    left = int(math.floor(x + (w - side) / 2))
    top = int(math.floor(y + (h - side) / 2))
    box = (left, top, left + side, top + side)
    # RVM locates the crop. Keep the RGB scene context the network was trained with.
    native = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return native.crop(box), box


def _rotation(poses):
    x = F.normalize(poses[:, :3], dim=1, eps=1e-8)
    z = F.normalize(torch.cross(x, poses[:, 3:], dim=1), dim=1, eps=1e-8)
    y = torch.cross(z, x, dim=1)
    return torch.stack((x, y, z), dim=2)


@torch.inference_mode()
def predict_profile_rotations(model, images_bgr, head_masks, batch_size=12):
    """Return CPU (N,3,3) rotation matrices in upstream network coordinates."""
    images, masks = list(images_bgr), list(head_masks)
    if len(images) != len(masks) or batch_size < 1:
        raise ValueError("Provide one head mask per image and positive batch size")
    if not images:
        return np.empty((0, 3, 3), dtype=np.float32)
    device = next(model.parameters()).device
    matrices = []
    for start in range(0, len(images), batch_size):
        tensors = [_INPUT(profile_pose_crop(image, mask)[0])
                   for image, mask in zip(images[start:start + batch_size], masks[start:start + batch_size])]
        matrices.append(_rotation(model(torch.stack(tensors).to(device))).cpu())
    matrices = torch.cat(matrices).numpy()
    gram = np.swapaxes(matrices, -1, -2) @ matrices
    if (not np.isfinite(matrices).all() or not np.allclose(gram, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(matrices), 1., atol=1e-4)):
        raise RuntimeError("6DRepNet360 produced a degenerate orientation")
    return matrices
