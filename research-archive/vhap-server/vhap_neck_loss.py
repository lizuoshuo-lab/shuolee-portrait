"""Fit observed neck edges to the visible silhouette of the complete FLAME mesh."""

from collections import defaultdict

import numpy as np
import torch


def _neck_edges(flame):
    """Keep manifold edges whose two endpoints belong to the anatomical neck."""
    faces = flame.faces.detach().cpu().numpy()
    neck = set(flame.mask.get_vid_by_region(("neck",)).cpu().tolist())
    neck.difference_update(flame.mask.get_vid_by_region(("boundary", "face", "left_ear", "right_ear")).cpu().tolist())
    incident = defaultdict(list)
    for index, (a, b, c) in enumerate(faces):
        for start, stop in ((a, b), (b, c), (c, a)):
            if start in neck and stop in neck:
                incident[min(start, stop), max(start, stop)].append(index)
    manifold = [(edge, pair) for edge, pair in incident.items() if len(pair) == 2]
    if not manifold:
        raise ValueError("The FLAME mesh has no interior neck edges")
    edges, neighbors = zip(*manifold, strict=True)
    return (torch.as_tensor(np.asarray(edges), dtype=torch.long, device=flame.faces.device),
            torch.as_tensor(np.asarray(neighbors), dtype=torch.long, device=flame.faces.device))


class NeckContourLoss:
    """Associate one observed left/right boundary point with a live neck edge at the same row."""

    def __init__(self, tracker, observations, weight):
        self.enabled = False
        self.weight = float(weight)
        self.training_counts = defaultdict(int)
        self.training_batches = 0
        self.size = tuple(tracker.image_size)
        self.faces = tracker.flame.faces.long()
        self.edges, neighbors = _neck_edges(tracker.flame)
        self.triangles = self.faces[neighbors]
        self.observations = {}
        for frame, points in observations.items():
            self.observations[int(frame)] = []
            for point in points:
                xy = np.asarray(point["xy"], dtype=float)
                side, split = point["side"], point["split"]
                if (xy.shape != (2,) or not np.isfinite(xy).all()
                        or side not in ("left", "right") or split not in ("fit", "check", "holdout")):
                    raise ValueError(f"Invalid neck contour observation at timestep {frame}")
                self.observations[int(frame)].append((xy.tolist(), side, "check" if split == "holdout" else split))

    def _evaluate(self, sample, rast_dict, split):
        clip = rast_dict["verts_clip"]
        rast = rast_dict["rast_out"].detach()
        height, width = rast.shape[1:3]
        if (height, width) != self.size:
            raise ValueError("Neck observations require the same virtual raster as VHAP")
        times = sample["timestep_index"].tolist()
        rows = [(batch, timestep, xy, side) for batch, timestep in enumerate(times)
                for xy, side, label in self.observations.get(timestep, ()) if label == split]
        if not rows:
            return rows, clip.new_zeros((0,)), clip.new_zeros((0,), dtype=torch.bool)

        batches, _, xy, sides = zip(*rows, strict=True)
        batch = torch.as_tensor(batches, dtype=torch.long, device=clip.device)
        xy = clip.new_tensor(xy)
        right = torch.as_tensor([side == "right" for side in sides], device=clip.device)
        x, y = xy.unbind(-1)
        row_i = y.round().long()
        valid_y = (row_i > 0) & (row_i < height - 1)
        # nvdiffrast raster rows are bottom-up, unlike the virtual JPEGs and observations.
        face_rows = rast[batch, height - 1 - row_i.clamp(0, height - 1), :, 3]
        foreground = face_rows > 0
        leftmost = foreground.long().argmax(dim=-1)
        rightmost = width - 1 - foreground.flip(-1).long().argmax(dim=-1)
        extreme = torch.where(right, rightmost, leftmost)
        face = face_rows.gather(1, extreme[:, None]).long().squeeze(1) - 1

        # Select edges as a discrete visibility decision. Gradients travel only
        # through the continuous 2D intersection of the selected 3D mesh edge.
        ndc = clip[..., :2] / clip[..., 3:].clamp_min(1e-8)
        pixel = ndc * clip.new_tensor([width / 2, -height / 2]) + clip.new_tensor([(width - 1) / 2, (height - 1) / 2])
        endpoints = pixel[:, self.edges][batch]
        start, stop = endpoints.unbind(-2)
        delta = stop - start
        vertical = delta[..., 1]
        denominator = torch.where(vertical.abs() > 1e-5, vertical, torch.ones_like(vertical))
        factor = (y[:, None] - start[..., 1]) / denominator
        predicted = start[..., 0] + factor * delta[..., 0]
        with torch.no_grad():
            triangle = pixel.detach()[:, self.triangles][batch]
            u = triangle[..., 1, :] - triangle[..., 0, :]
            v = triangle[..., 2, :] - triangle[..., 0, :]
            area = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
            opposite_winding = area[..., 0] * area[..., 1] < 0
            row_factor = (row_i[:, None] - start[..., 1]) / denominator
            raster_intersection = start[..., 0] + row_factor * delta[..., 0]
            raster_edge_center = extreme + torch.where(right, .5, -.5)
            distance = (raster_intersection - raster_edge_center[:, None]).abs()
            # An incident triangle can be thinner than a sample pixel. The actual
            # visible triangle must still touch this edge's vertex neighborhood.
            visible = self.faces[face.clamp_min(0)]
            same_neighborhood = (self.edges[None, :, :, None] == visible[:, None, None, :]).any(dim=-1).any(dim=-1)
            depth = clip[..., 3][:, self.edges][batch]
            valid = (opposite_winding & same_neighborhood & (depth > 1e-5).all(dim=-1)
                     & (vertical.abs() > 1e-5) & (factor >= 0) & (factor <= 1)
                     & (row_factor >= 0) & (row_factor <= 1) & (distance <= 0.75)
                     & valid_y[:, None] & (x > 0)[:, None] & (x < width - 1)[:, None]
                     & (extreme > 0)[:, None] & (extreme < width - 1)[:, None] & (face >= 0)[:, None])
            # The observed x plays no role in selecting or accepting a candidate edge.
            selected = distance.masked_fill(~valid, torch.inf).argmin(dim=-1, keepdim=True)
        return rows, predicted.gather(1, selected).squeeze(1), valid.any(dim=-1)

    def term(self, sample, rast_dict):
        rows, predicted, valid = self._evaluate(sample, rast_dict, "fit")
        if not rows:
            return predicted.sum()
        measured = valid.tolist()
        self.training_batches += int(any(measured))
        for (_, timestep, _, _), used in zip(rows, measured, strict=True):
            if used:
                self.training_counts[timestep] += 1
        target = predicted.new_tensor([xy[0] for _, _, xy, _ in rows])
        dx = torch.where(valid, predicted - target, 0.)
        # Pseudo-Huber in pixel units keeps a bounded influence as displacement increases.
        delta = 4. * self.size[1] / 512.
        return (torch.sqrt(dx.square() + delta ** 2) - delta).sum() / valid.sum().clamp_min(1) / self.size[1]

    @torch.no_grad()
    def score(self, sample, rast_dict, split="check"):
        """Return every requested site's validity and horizontal error in virtual pixels."""
        split = "check" if split == "holdout" else split
        rows, predicted, valid = self._evaluate(sample, rast_dict, split)
        return [
            {"timestep": timestep, "side": side, "split": split, "target_xy": xy,
             "predicted_xy": [value, xy[1]] if ok else None,
             "error_px": abs(value - xy[0]) if ok else None,
             "status": "measured" if ok else "no_visible_neck_contour"}
            for (_, timestep, xy, side), value, ok in zip(rows, predicted.tolist(), valid.tolist(), strict=True)
        ]


def configure(tracker, observations, weight=0.5):
    """Install the optional term alongside original VHAP photometric training."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Neck loss weight must be finite and nonnegative")
    loss = NeckContourLoss(tracker, observations, weight)
    original = tracker.compute_photometric_energy

    def compute(sample, verts, faces, albedos, rast_dict, step_i=None, stage=None):
        photo, result = original(sample, verts, faces, albedos, rast_dict, step_i, stage)
        if loss.enabled and stage == "rgb_global_tracking":
            neck = loss.term(sample, rast_dict)
            result["neck_contour_loss"] = neck.detach()
            photo = photo + loss.weight * neck
        return photo, result

    tracker.compute_photometric_energy = compute
    return loss
