"""Fit a shared outward displacement to independently witnessed hair outlines."""

import cv2
from hashlib import sha256
import nvdiffrast.torch as dr
import numpy as np
import torch
from torch.nn import functional as F


def scalp_support(flame):
    faces = flame.faces.long()
    regions = flame.mask.get_vid_by_region
    protected = torch.zeros(len(flame.v_template), dtype=torch.bool, device=faces.device)
    protected[regions(("face", "forehead", "neck", "left_ear", "right_ear", "left_eyeball",
                       "right_eyeball", "boundary", "bottomline"))] = True
    protected[faces[flame.full_lmk_faces_idx.reshape(-1)].flatten()] = True
    touching = faces[protected[faces].any(dim=1)].flatten()
    protected[touching] = True
    allowed = torch.zeros_like(protected)
    allowed[regions(("scalp",))] = True
    ids = torch.where(allowed & ~protected)[0]
    if not len(ids):
        raise ValueError("No verifiable scalp vertices")
    edges = torch.cat((faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]))
    edges = torch.unique(edges.sort(dim=1).values, dim=0)
    allowed = allowed & ~protected
    return ids, edges[allowed[edges].any(dim=1)]


def normals_for(verts, faces, ids):
    tri = verts[:, faces]
    normals = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
    result = torch.zeros_like(verts)
    for corner in range(3):
        result.index_add_(1, faces[:, corner], normals)
    return F.normalize(result[:, ids], dim=-1)


def fit_scalp_outline(tracker, loader, records, sequence, target_path, output):
    names = [r["name"] for r in records]
    with np.load(target_path, allow_pickle=False) as z:
        if z["names"].tolist() != names or z["hair"].shape != (len(names), *tracker.image_size):
            raise ValueError("Scalp observations and timeline differ")
        if str(z["initialization_sha256"].item()) != sha256((sequence / "initialization.json").read_bytes()).hexdigest():
            raise ValueError("Scalp observations and camera manifest differ")
        for field, folder in (("crop_sha256", "images"), ("matte_sha256", "alpha_maps")):
            actual = [sha256((sequence / folder / f"{i:08d}.jpg").read_bytes()).hexdigest() for i in range(len(names))]
            if z[field].tolist() != actual:
                raise ValueError(f"Scalp observations and actual {folder} differ")
        roi = z["hair"].astype(bool)
    target = np.stack([cv2.imread(str(sequence / "alpha_maps" / f"{i:08d}.jpg"), cv2.IMREAD_GRAYSCALE)
                       for i in range(len(names))]) >= 128
    if target.shape != roi.shape or not (roi & target).any() or not (roi & ~target).any():
        raise ValueError("Expected scalp support on both sides of the matte contour")
    sdf = np.stack([np.clip(cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
                            - cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5), -12., 12.)
                    for m in target]) / 12.

    device = tracker.device
    roi = torch.as_tensor(roi, device=device)
    target = torch.as_tensor(target, device=device)
    sdf = torch.as_tensor(sdf, device=device)
    faces = tracker.flame.faces.long().contiguous()
    faces_i = faces.int().contiguous()
    ids, edges = scalp_support(tracker.flame)
    with torch.no_grad():
        timesteps = torch.arange(len(names), device=device)
        posed = torch.cat([tracker.forward_flame(ts)[0] for ts in timesteps.split(8)])
        normal = normals_for(posed, faces, ids)
        # Verify the native normals point outward on the scalp before using a nonnegative field.
        outward = posed[:, ids] - posed[:, ids].mean(dim=1, keepdim=True)
        if float((normal * outward).sum(dim=-1).median()) <= 0:
            raise ValueError("Scalp normals fail the outward orientation check")

    bump = torch.zeros(len(ids), device=device, requires_grad=True)
    opt = torch.optim.Adam([bump], lr=.0015)

    def alpha_for(sample):
        ts = sample["timestep_index"]
        tracker.clear_cache()
        with torch.no_grad():
            tracker.fill_cam_params_into_sample(sample)
        modified = posed[ts].index_add(1, ids, normal[ts] * bump[None, :, None])
        raster = tracker.rasterize_flame(sample, modified, faces, train_mode=True)
        rast = raster["rast_out"]
        hard = (rast[..., -1:] > 0).to(modified)
        return dr.antialias(hard.contiguous(), rast, raster["verts_clip"], faces_i)[..., 0].flip(1)

    @torch.no_grad()
    def scores():
        result = {}
        for sample in loader:
            pred = alpha_for(sample) >= .5
            for j, i in enumerate(sample["timestep_index"].tolist()):
                if roi[i].any():
                    truth, valid = target[i], roi[i]
                    ninter = int((pred[j] & truth & valid).sum())
                    result[names[i]] = dict(intersection=ninter, union=int(((pred[j] | truth) & valid).sum()),
                                             missing=int((~pred[j] & truth & valid).sum()),
                                             outside=int((pred[j] & ~truth & valid).sum()))
        return result

    before = scores()
    first_gradient = None
    steps = 0
    max_offset = .035
    for epoch in range(16):
        for sample in loader:
            ts = sample["timestep_index"]
            if not roi[ts].any():
                continue
            alpha = alpha_for(sample)
            data = ((alpha - target[ts].float()) * sdf[ts] * roi[ts]).sum() / roi[ts].sum()
            full = bump.new_zeros(posed.shape[1]).index_copy(0, ids, bump)
            smooth = ((full[edges[:, 0]] - full[edges[:, 1]]) / max_offset).square().mean()
            loss = data + .04 * smooth + .006 * (bump / max_offset).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(bump.grad).all():
                raise ValueError("Nonfinite scalp gradient")
            if first_gradient is None and bump.grad.any():
                first_gradient = float(bump.grad.abs().max())
            opt.step()
            with torch.no_grad():
                bump.clamp_(0., max_offset)
            steps += 1
        if epoch % 4 == 3:
            print(f"Scalp: pass {epoch + 1}/16, {steps} updates, largest offset {float(bump.detach().max()):.4f}", flush=True)

    after = scores()
    displacement = bump.detach()
    if not displacement.any() or not first_gradient or steps == 0:
        raise ValueError("No actual silhouette-driven scalp update")
    np.savez_compressed(output / "shared-scalp-offset.npz", native_vertex_ids=ids.cpu().numpy(),
                        outward_offset=displacement.cpu().numpy())
    result = dict(steps=steps, frame_count=len(after), vertex_count=len(ids),
                  first_gradient_max=first_gradient, offset_max=float(displacement.max()),
                  offset_median=float(displacement.median()), initial=before, fitted=after)
    return (ids, displacement), result
