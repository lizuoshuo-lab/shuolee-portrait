"""A shared clothing guide from fixed-pose masks and verified native RGB tracks.

This only moves the existing garment-capable offset layer. Two-dimensional
silhouettes and appearance tracks constrain the observed part of the surface;
they do not measure hidden cloth depth.
"""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from .coverage_refinement import _clip

POLICY = dict(long_edge=896, cycles=12, offset_lr_m=.001, max_change_m=.018,
              silhouette_window_px=14, mask_edge_rgb_min=9, source_bins_min=2,
              track_fb_px=.65, track_lk_error=18., track_baseline_max_px=24.,
              minimum_tracks=24, minimum_track_views=4, silhouette_gain_px=.10,
              silhouette_view_regression_px=.65, track_mean_regression_px=.20,
              track_view_regression_px=.65, minimum_normal_dot=.70,
              minimum_area_ratio=.58, maximum_area_ratio=1.70,
              feasible_line_fractions=[1., .5, .25, .125])


def _native_images(row, dims):
    import cv2
    from scipy.ndimage import distance_transform_edt, maximum_filter

    w, h = dims
    with Image.open(row['image']) as im:
        gray = np.array(im.convert('L'))
        rgb = np.asarray(im.convert('RGB').resize((w, h), Image.Resampling.BOX), np.float32)
    with Image.open(row['mask']) as im:
        mask = np.array(im.convert('L'))
        target = np.asarray(im.convert('L').resize((w, h), Image.Resampling.BOX), np.float32)/255
    binary = target >= .5
    if gray.shape != mask.shape or gray.shape[::-1] != tuple(row['wh']) or not binary.any() or binary.all():
        raise ValueError('Clothing guide image and foreground mask must match the actual native frame.')
    distance = distance_transform_edt(binary)-distance_transform_edt(~binary)
    gx = cv2.Sobel(rgb, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    gy = cv2.Sobel(rgb, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    strength = (gx*gx+gy*gy).max(axis=2)**.5
    edge = (np.abs(distance) <= 1.45) & (maximum_filter(strength, size=3) >= POLICY['mask_edge_rgb_min'])
    return gray, mask, target, edge


def _project(world, row):
    E, K = (world.new_tensor(row[name]) for name in ('w2c', 'K'))
    return (world @ E[:3, :3].T + E[:3, 3]) @ K.T


def _raster(world, row, faces, dimensions, scale, dr, context, *, aa):
    projected = _project(world, row)
    m = projected.new_tensor([*scale, 1.])
    clip = _clip(projected*m, *dimensions)[None].contiguous()
    raster, _ = dr.rasterize(context, clip, faces, (dimensions[1], dimensions[0]), grad_db=False)
    alpha = (raster[..., 3:] > 0).to(world.dtype)
    if aa:
        alpha = dr.antialias(alpha, raster, clip, faces)
    return alpha[0, :, :, 0].flip(0), raster[0, :, :, 3].flip(0).long()-1, projected


def _lookup(owner, xy, scale):
    xy = (xy+.5)*np.asarray(scale)
    ij = np.floor(xy).astype(np.int64)
    x, y = ij.T
    valid = (x >= 0) & (y >= 0) & (x < owner.shape[1]) & (y < owner.shape[0])
    return np.where(valid, owner[y.clip(0, owner.shape[0]-1), x.clip(0, owner.shape[1]-1)], -1)


def _barycentric(projected, triangles, xy):
    # OpenCV LK gives integer-centered coordinates; K uses pixel corners.
    depths = projected[triangles, 2]
    corners = projected[triangles, :2]/depths[..., None]
    edge = np.stack((corners[:, 0]-corners[:, 2], corners[:, 1]-corners[:, 2]), axis=-1)
    det = np.linalg.det(edge)
    good = (np.abs(det) > 1e-5) & (depths.min(axis=1) > .2)
    edge[~good] = np.eye(2)
    weights = np.linalg.solve(edge, (xy+.5-corners[:, 2])[..., None])[..., 0]
    weights = np.column_stack((weights, 1-weights.sum(axis=1)))/np.maximum(depths, .2)
    weights /= np.maximum(weights.sum(axis=1)[:, None], 1e-8)
    good &= np.isfinite(weights).all(axis=1) & (weights.min(axis=1) >= -.05) & (weights.max(axis=1) <= 1.05)
    return weights, good


def _track(from_gray, to_gray, points):
    import cv2

    opts = dict(winSize=(19, 19), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
    forward, a, error = cv2.calcOpticalFlowPyrLK(from_gray, to_gray, points, None, **opts)
    backward, b, _ = cv2.calcOpticalFlowPyrLK(to_gray, from_gray, forward, None, **opts)
    moved = forward.reshape(-1, 2)
    ok = (a[:, 0].astype(bool) & b[:, 0].astype(bool) & (error[:, 0] < POLICY['track_lk_error'])
          & (np.linalg.norm(backward.reshape(-1, 2)-points.reshape(-1, 2), axis=1) < POLICY['track_fb_px'])
          & np.isfinite(moved).all(1))
    return np.where(np.isfinite(moved), moved, -10000.), ok


def _inside(mask, xy):
    x, y = np.rint(xy).astype(np.int64).T
    return ((x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
            & (mask[y.clip(0, mask.shape[0]-1), x.clip(0, mask.shape[1]-1)] >= 252))


def _rgb_tracks(views, rows, faces, garment, scale):
    """Track real textures across a source triple and reject ambiguous reprojections."""
    import cv2

    tracks = defaultdict(list)
    anchors = defaultdict(list)
    ordered = sorted(range(len(views)), key=lambda i: rows[i]['source_frame_index'])
    for prev, mid, nxt in zip(ordered, ordered[1:], ordered[2:]):
        v = views[mid]
        selected = (v.owner >= 0) & garment[np.maximum(v.owner, 0)]
        roi = cv2.resize(selected.astype(np.uint8), tuple(rows[mid]['wh']), interpolation=cv2.INTER_NEAREST)
        roi = cv2.erode(roi & (v.mask >= 252).astype(np.uint8), np.ones((7, 7), np.uint8))
        p = cv2.goodFeaturesToTrack(v.gray, maxCorners=200, qualityLevel=.02, minDistance=9, mask=roi)
        if p is None:
            continue
        xy = p.reshape(-1, 2)
        owner = _lookup(v.owner, xy, scale)
        weights, ok = _barycentric(v.projected, faces[np.maximum(owner, 0)], xy)
        ok &= (owner >= 0) & garment[np.maximum(owner, 0)]
        triangles = faces[np.maximum(owner, 0)]
        moved = {}
        for i in (prev, nxt):
            u = views[i]
            xy_i, valid = _track(v.gray, u.gray, p)
            moved[i] = xy_i
            target_owner = _lookup(u.owner, xy_i, scale)
            target_tri = faces[np.maximum(target_owner, 0)]
            adjacent = (triangles[:, :, None] == target_tri[:, None, :]).any((1, 2))
            predicted = (u.projected[triangles]*weights[..., None]).sum(axis=1)
            reproj = predicted[:, :2]/np.maximum(predicted[:, 2:3], .2)-.5
            ok &= (valid & _inside(u.mask, xy_i) & (target_owner >= 0) & adjacent
                   & (np.linalg.norm(reproj-xy_i, axis=1) <= POLICY['track_baseline_max_px']))
        if not ok.any():
            continue
        # The anchor establishes the barycentric identity. Only independently
        # tracked coordinates in the two other views enter the RGB constraint.
        anchors[mid].extend(owner[ok].tolist())
        for i in (prev, nxt):
            tracks[i].append((triangles[ok], weights[ok], moved[i][ok]+.5))
    return tracks, anchors


def fit_shared_clothing(model, rows, slots, row_bins, reference):
    """Return a single rest-space offset, supported faces, and review evidence."""
    import cv2
    import nvdiffrast.torch as dr
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from utils.graphics_utils import compute_vertex_normals

    cv2.setNumThreads(4)
    dev, dtype = model.base.device, model.base.dtype
    zeros = torch.zeros(len(model.faces), dtype=torch.bool, device=dev)
    base_offset = model.static_offset.detach().clone()
    width, height = rows[0]['wh']
    if any(tuple(row['wh']) != (width, height) for row in rows):
        raise ValueError('The clothing guide requires the same native image domain in the selected roster.')
    factor = min(1., POLICY['long_edge']/max(width, height))
    dims = tuple(max(8, 8*round(d*factor/8)) for d in (width, height))
    scale = tuple(d/o for d, o in zip(dims, (width, height), strict=True))
    context = dr.RasterizeCudaContext(device=dev)
    faces = model.faces.long()
    raster_faces = faces.int().contiguous()
    cloth = model.clothing[faces].all(1).cpu().numpy()
    vertices = faces.cpu().numpy()
    with torch.no_grad():
        original = model.posed_vertices(slice(None)).detach()
        normal = compute_vertex_normals(original[reference:reference+1], faces)[0]
        direct = torch.linalg.solve_ex(model.linear[reference], normal[..., None])
        raw_direction = direct.result[..., 0]
        good = ((direct.info == 0) & torch.isfinite(raw_direction).all(dim=1)
                & (raw_direction.norm(dim=1) > .5))
        direction = torch.nn.functional.normalize(torch.where(good[:, None], raw_direction, 0.), dim=1)
        views, face_bins = [], defaultdict(set)
        for row, slot, source_bin in zip(rows, slots, row_bins, strict=True):
            gray, mask, target, edge = _native_images(row, dims)
            alpha, owner, projected = _raster(original[slot], row, raster_faces, dims, scale, dr, context, aa=False)
            owned = owner.cpu().numpy()
            inside = owned >= 0
            visible_boundary = inside & ~binary_erosion(inside)
            clothing_boundary = visible_boundary & cloth[np.maximum(owned, 0)]
            near_model = distance_transform_edt(~clothing_boundary) if clothing_boundary.any() else np.full(inside.shape, np.inf)
            near_observed = distance_transform_edt(~edge) if edge.any() else np.full(inside.shape, np.inf)
            window = POLICY['silhouette_window_px']
            region = (near_model <= window) & (near_observed <= window)
            normalizer = max(1, int((region & edge).sum()))
            weight = np.where(region, np.clip(1.-near_observed/(window+1), 0., 1.), 0.).astype(np.float32)
            ids = np.unique(owned[clothing_boundary & region])
            for face in ids:
                face_bins[int(face)].add(int(source_bin))
            views.append(SimpleNamespace(gray=gray, mask=mask, owner=owned, projected=projected.cpu().numpy(),
                target=torch.as_tensor(target, device=dev), weight=torch.as_tensor(weight, device=dev),
                normalizer=normalizer, candidate_faces=len(ids)))
        tracks, anchors = _rgb_tracks(views, rows, vertices, cloth, scale)
        for i, ids in anchors.items():
            for face in set(ids):
                face_bins[int(face)].add(int(row_bins[i]))
        vote = np.array([len(face_bins[i]) for i in range(len(faces))], dtype=np.int16)
        supported = torch.as_tensor(cloth & (vote >= POLICY['source_bins_min']), device=dev)
        eligible = torch.zeros(len(base_offset), dtype=torch.bool, device=dev)
        eligible[faces[supported].reshape(-1)] = True
        eligible &= model.clothing & good
        support_face = supported & eligible[faces].all(1)
        targets = {}
        for i, parts in tracks.items():
            triangle, bary, coords = [np.concatenate(x) for x in zip(*parts)]
            keep = eligible[torch.as_tensor(triangle, device=dev)].any(1).cpu().numpy()
            if keep.any():
                targets[i] = (torch.as_tensor(triangle[keep], device=dev),
                              torch.as_tensor(bary[keep], device=dev, dtype=dtype),
                              torch.as_tensor(coords[keep], device=dev, dtype=dtype))
        track_count = sum(len(x[0]) for x in targets.values())
        support_counts = dict(supported_faces=int(support_face.sum()), eligible_vertices=int(eligible.sum()),
                              native_rgb_track_anchors=sum(len(x) for x in anchors.values()),
                              native_rgb_tracked_observations=track_count, rgb_tracked_views=len(targets),
                              distinct_source_bins=len(set(row_bins)))
    detail = dict(schema='shuolee-shared-clothing-guide-v1', policy=POLICY,
        inputs=dict(frames=[row['frame'] for row in rows], image_sha256=[row['source_sha256'] for row in rows],
                    mask_sha256=[row['mask_sha256'] for row in rows], raster_wh=list(dims)),
        support=support_counts, posture_and_identity_frozen=True, protected_vertices_bit_exact=True,
        meaning='Mask and consistent RGB tracks constrain visible clothing; hidden depth is not measured.')
    if (not support_face.any() or track_count < POLICY['minimum_tracks']
            or len(targets) < POLICY['minimum_track_views']):
        return base_offset, zeros, dict(detail, state='insufficient_rgb_or_multiview_support',
                                        selected_cycle=0, maximum_change_m=0., adam_updates=0)

    def evaluate(offset, i):
        slot = slots[i]
        verts = model.base[slot] + (model.linear[slot] @ offset[..., None]).squeeze(-1)
        v = views[i]
        alpha, _, projected = _raster(verts, rows[i], raster_faces, dims, scale, dr, context, aa=True)
        sil = ((alpha-v.target).abs()*v.weight).sum()/(v.normalizer*scale[0])
        if i not in targets:
            return sil, None
        tri, bary, pixels = targets[i]
        xyz = (projected[tri]*bary[..., None]).sum(1)
        predicted = xyz[:, :2]/xyz[:, 2:3].clamp_min(.2)
        rgb_track = ((predicted-pixels).square().sum(1)+.25).sqrt().mean()-.5
        return sil, rgb_track

    @torch.no_grad()
    def metrics(offset):
        results = []
        for i, row in enumerate(rows):
            sil, phot = evaluate(offset, i)
            results.append(dict(frame=row['frame'], edge_error_native_px_equivalent=float(sil),
                                rgb_track_robust_mean_px=float(phot) if phot is not None else None,
                                tracked_observations=len(targets[i][0]) if i in targets else 0,
                                boundary_faces=views[i].candidate_faces))
        return results

    @torch.no_grad()
    def mesh_gate(candidate):
        updated = model.base + (model.linear @ candidate[..., None]).squeeze(-1)
        scope = eligible[faces].any(1)
        a, b = original[:, faces[scope]], updated[:, faces[scope]]
        norm = lambda v: torch.linalg.cross(v[:, :, 1]-v[:, :, 0], v[:, :, 2]-v[:, :, 0])
        na, nb = norm(a), norm(b)
        length_a, length_b = na.norm(dim=-1), nb.norm(dim=-1)
        ratio = length_b/length_a.clamp_min(1e-10)
        dot = (na*nb).sum(dim=-1)/(length_a*length_b).clamp_min(1e-20)
        return dict(normal_dot_min=float(dot.min()), area_ratio_min=float(ratio.min()),
                    area_ratio_max=float(ratio.max()), finite=bool(torch.isfinite(updated).all()))

    limits = POLICY
    initial = metrics(base_offset)
    scalar = torch.nn.Parameter(torch.zeros(len(base_offset), device=dev, dtype=dtype))
    adam = torch.optim.Adam([scalar], lr=limits['offset_lr_m'])
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = torch.unique(edges.sort(dim=1).values, dim=0)
    edges = edges[eligible[edges].any(1)]
    chosen, best, attempts, updates = base_offset, None, [], 0
    indices = sorted(range(len(rows)), key=lambda i: (row_bins[i], rows[i]['source_frame_index']))
    for cycle in range(1, limits['cycles']+1):
        order = indices[cycle-1:]+indices[:cycle-1]
        for i in order:
            d = scalar*eligible
            offset = base_offset+direction*d[:, None]
            sil, rgb_track = evaluate(offset, i)
            smooth = ((d[edges[:, 0]]-d[edges[:, 1]])/.005).square().mean()
            trust = (d[eligible]/limits['max_change_m']).square().mean()
            loss = sil + (0.16*rgb_track if rgb_track is not None else 0.) + .15*smooth + .08*trust
            adam.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(scalar.grad).all():
                raise RuntimeError('Nonfinite shared clothing geometry gradient.')
            adam.step()
            updates += 1
            with torch.no_grad():
                scalar.clamp_(-limits['max_change_m'], limits['max_change_m'])
        with torch.no_grad():
            delta_offset = direction*(scalar*eligible)[:, None]
            for fraction in limits['feasible_line_fractions']:
                candidate = base_offset+fraction*delta_offset
                topology = mesh_gate(candidate)
                if (topology['finite'] and topology['normal_dot_min'] >= limits['minimum_normal_dot']
                        and topology['area_ratio_min'] >= limits['minimum_area_ratio']
                        and topology['area_ratio_max'] <= limits['maximum_area_ratio']):
                    break
            after = metrics(candidate)
            delta = [r['edge_error_native_px_equivalent']-b['edge_error_native_px_equivalent']
                     for r, b in zip(after, initial, strict=True) if b['boundary_faces']]
            track_delta = [after[i]['rgb_track_robust_mean_px']-initial[i]['rgb_track_robust_mean_px'] for i in targets]
            record = dict(cycle=cycle, topology_line_fraction=fraction, mesh=topology, edge_gain_px=-float(np.mean(delta)) if delta else 0.,
                          max_edge_regression_px=max(delta, default=0.),
                          track_mean_regression_px=float(np.mean(track_delta)),
                          max_track_regression_px=max(track_delta))
            record['accepted'] = (topology['finite'] and topology['normal_dot_min'] >= limits['minimum_normal_dot']
                and topology['area_ratio_min'] >= limits['minimum_area_ratio']
                and topology['area_ratio_max'] <= limits['maximum_area_ratio']
                and record['edge_gain_px'] >= limits['silhouette_gain_px']
                and record['max_edge_regression_px'] <= limits['silhouette_view_regression_px']
                and record['track_mean_regression_px'] <= limits['track_mean_regression_px']
                and record['max_track_regression_px'] <= limits['track_view_regression_px'])
            attempts.append(record)
            if record['accepted'] and (best is None or record['edge_gain_px'] > best[0]['edge_gain_px']):
                chosen, best = candidate.clone(), (record, after)
    if not torch.equal(chosen[~eligible], base_offset[~eligible]) or not torch.equal(chosen[~model.clothing], base_offset[~model.clothing]):
        raise RuntimeError('Shared clothing guide moved protected geometry.')
    return chosen, support_face if best else zeros, dict(detail,
        state='accepted_visible_surface' if best else 'rejected_keep_original_surface',
        selected_cycle=best[0]['cycle'] if best else 0,
        selected_topology_line_fraction=best[0]['topology_line_fraction'] if best else 0., before=initial, after=best[1] if best else initial,
        optimizer_trials=attempts, adam_updates=updates,
        maximum_change_m=float((chosen-base_offset).norm(dim=1).max()),
        changed_vertices=int(((chosen-base_offset).abs().sum(1) > 0).sum()))


def clothing_attachment_loss(model, supported_faces, visible):
    """Favor supported cloth tangency, with no penalty for tangential spread."""
    from utils.general_utils import build_rotation

    rows = torch.where(supported_faces[model.binding] & visible & (model.get_opacity[:, 0] >= .02))[0]
    owner = model.binding[rows]
    scale = model.face_scaling[owner, 0]
    xyz = model._xyz[rows, 1]*scale
    rotations = build_rotation(model._rotation[rows])
    sigma = model._scaling[rows].exp()*scale[:, None]
    normal_sigma = ((rotations[:, 1, :]*sigma).square().sum(1)+1e-12).sqrt()
    near = torch.relu(xyz.abs()-.004)/.012
    thin = torch.relu(normal_sigma-.004)/.012
    return (.018*torch.nn.functional.smooth_l1_loss(near, torch.zeros_like(near), reduction='sum')
            + .012*torch.nn.functional.smooth_l1_loss(thin, torch.zeros_like(thin), reduction='sum'))/max(1, len(rows))



@torch.no_grad()
def measure_clothing_attachment(model, supported_faces):
    from utils.general_utils import build_rotation

    rows = torch.where(supported_faces[model.binding] & (model.get_opacity[:, 0] >= .02))[0]
    result = dict(state='measured' if len(rows) else 'no_supported_active_rows',
                  supported_faces=int(supported_faces.count_nonzero()), active_rows=len(rows),
                  scope='Active Gaussian rows on the supported garment guide in the reference pose.')
    if not len(rows):
        return result
    scale = model.face_scaling[model.binding[rows], 0]
    center = model._xyz[rows, 1]*scale
    rotation = build_rotation(model._rotation[rows])
    axes = model._scaling[rows].exp()*scale[:, None]
    normal = (rotation[:, 1, :]*axes).square().sum(1).sqrt()
    ordered = axes.sort(dim=1)
    flat = ordered.values[:, 1]/ordered.values[:, 0].clamp_min(1e-9) >= 1.5
    align = rotation[:, 1, :].gather(1, ordered.indices[:, :1])[:, 0].abs().clamp(0., 1.)
    angle = torch.rad2deg(align.acos())

    def q(x):
        return dict(zip(('p50','p95'), x.quantile(x.new_tensor([.5,.95])).tolist())) if len(x) else None

    return dict(result, signed_normal_center_mm=q(1000*center), abs_normal_center_mm=q(1000*center.abs()),
        normal_sigma_mm=q(1000*normal), flattened_rows=int(flat.sum()),
        flattened_thin_axis_angle_deg=q(angle[flat]))
