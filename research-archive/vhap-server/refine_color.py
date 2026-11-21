"""Fit DC color on visible face interiors with a fixed scene and fixed geometry."""

import cv2
import numpy as np
import torch


def refine_dc(background, examples, render, iterations):
    initial = examples[0][1][1][:, :1].detach().clone()
    zeros = torch.zeros_like(examples[0][1][1][:, 1:])

    def rgb(example, dc):
        rasterizer, head, _, _, _ = example
        head = (head[0], torch.cat((dc, zeros), dim=1), *head[2:])
        return render(rasterizer, tuple(torch.cat((a, b)) for a, b in zip(background, head)))

    def loss(example, dc):
        _, _, reference, mask, _ = example
        return (rgb(example, dc)[:, mask] - reference[:, mask]).abs().mean()

    train = [item for item in examples if not item[-1]]
    held = [item for item in examples if item[-1]]
    before = [float(loss(item, initial)) * 255 for item in held]
    with torch.enable_grad():
        dc = initial.clone().requires_grad_()
        optimizer = torch.optim.Adam([dc], lr=0.006)
        for i in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(train[i % len(train)], dc) + 0.025 * (dc - initial).square().mean()
            objective.backward()
            optimizer.step()
    after = [float(loss(item, dc.detach())) * 255 for item in held]
    return dc.detach(), {"iterations": iterations, "held_out_face_rgb_mae_before": before,
                         "held_out_face_rgb_mae_after": after}


def make_example(make_rasterizer, render, K, R, C, original, xy, head, held):
    # Render on the native raster. The CUDA covariance clamp uses the full camera FoV.
    hull = cv2.convexHull(np.rint(xy[:68]).astype(np.int32))
    rasterizer = make_rasterizer(K, original.shape[1::-1], R, C)
    face = np.zeros(original.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(face, hull, 1)
    face = cv2.erode(face, np.ones((7, 7), np.uint8))
    gpu = torch.as_tensor(face, device="cuda", dtype=torch.bool)
    mask = gpu & (render(rasterizer, head, alpha_only=True) > 0.8)
    if int(mask.sum()) < 100:
        raise ValueError("Insufficient visible face pixels for appearance verification.")
    reference = torch.as_tensor(original[..., ::-1].copy(), device="cuda").permute(2, 0, 1).float() / 255
    return rasterizer, head, reference, mask, held
