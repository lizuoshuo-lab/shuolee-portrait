"""Measure static-background consistency in the same registered views as the neck test."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np

from colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
from fit_native_skin_profile import camera


def measure(root, split, output):
    plan = json.loads(split.read_bytes())
    if Path(plan["input"]).resolve() != root.resolve():
        raise ValueError("Expected the same source registration as the neck experiment")
    scene = root / "scene"
    sparse = scene / "orig_undist/sparse"
    intrinsics = read_intrinsics_binary(sparse / "cameras.bin")
    images = {im.name: im for im in read_extrinsics_binary(sparse / "images.bin").values()}
    frames = json.loads((root / "head-neck/initialization.json").read_bytes())["frames"]
    held = set(plan["splits"]["held"])
    if set(images) != {frame["name"] for frame in frames}:
        raise ValueError("Registered frames disagree")
    tracks, coords, views, frame_ids = [], [], [], []
    for i, frame in enumerate(frames):
        name = frame["name"]
        k, r, c = camera(frame)
        im = images[name]
        cam = intrinsics[im.camera_id]
        rotation = qvec2rotmat(im.qvec)
        np.testing.assert_allclose(r.T, rotation, atol=1e-7)
        np.testing.assert_allclose(c, -rotation.T @ im.tvec, atol=1e-7)
        np.testing.assert_allclose(k[[0,1,0,1],[0,1,2,2]], cam.params - [0,0,.5,.5], atol=1e-7)
        mask = cv2.imread(str(scene/"mask_undist/images"/name), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (cam.height, cam.width):
            raise ValueError(name)
        xy = im.xys - .5
        px = np.rint(xy).astype(np.int64)
        ok = (im.point3D_ids >= 0) & (px >= 0).all(axis=1) & (px < [cam.width,cam.height]).all(axis=1)
        ii = np.flatnonzero(ok)
        ii = ii[mask[px[ii,1],px[ii,0]] < 10]
        tracks.extend(im.point3D_ids[ii])
        coords.extend(xy[ii])
        frame_ids.extend([i]*len(ii))
        views.append((k,r,c))

    train_names = set(plan["splits"]["nonheld"])
    frame_ids, xy = np.asarray(frame_ids), np.asarray(coords)
    _, group = np.unique(tracks,return_inverse=True)
    train = np.array([f["name"] in train_names for f in frames])[frame_ids]
    count = np.bincount(group[train], minlength=group.max()+1)
    keep = count >= 3
    # A linear intersection of training rays per track. Do not read the saved 3D point positions.
    lhs, rhs = np.zeros((len(keep),3,3)), np.zeros((len(keep),3))
    for i,(k,r,c) in enumerate(views):
        sel = train & (frame_ids==i) & keep[group]
        u = np.column_stack((xy[sel],np.ones(sel.sum()))) @ np.linalg.inv(k).T @ r.T
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        perpendicular = np.eye(3) - u[:,:,None] * u[:,None,:]
        np.add.at(lhs, group[sel], perpendicular)
        np.add.at(rhs, group[sel], perpendicular @ c)
    conditioned = np.zeros_like(keep)
    conditioned[keep] = np.linalg.cond(lhs[keep]) < 1e10
    points = np.zeros_like(rhs)
    points[conditioned] = np.linalg.solve(lhs[conditioned], rhs[conditioned,:,None])[:,:,0]
    all_errors, per_view = [], {}
    for i,(frame,(k,r,c)) in enumerate(zip(frames,views)):
        if frame["name"] not in held:
            continue
        sel = (frame_ids==i) & conditioned[group]
        local = (points[group[sel]] - c) @ r
        proj = local @ k.T
        err = np.linalg.norm(xy[sel] - proj[:,:2]/proj[:,2:], axis=1)
        err[local[:,2] <= 0] = np.inf
        per_view[frame["name"]] = summarize(err)
        all_errors.extend(err)
    report = {"split":str(split), "policy":"Outside dilated person mask; at least 3 nonheld observations; condition number below 1e10; no residual filtering.",
              "caveat":"Held 2D observations excluded only from triangulation. Camera calibration and track associations used the complete video.",
              "triangulated_track_count":int(conditioned.sum()), "unconditioned_tracks":int((keep & ~conditioned).sum()),
              "held":summarize(all_errors),"by_view":per_view}
    with output.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)


def summarize(errors):
    errors = np.asarray(errors,dtype=float)
    return {"n":len(errors),"over_3":int((errors>3).sum()),"behind_or_nonfinite":int((~np.isfinite(errors)).sum()),
            "median_px":float(np.median(errors)), "p95_px":float(np.percentile(errors,95))}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("run","split","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    measure(args.run,args.split,args.output)
