"""Attach an observation-guided upper shoulder patch and review it on the source views."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

from build_vhap_shoulder_patch import build_patch
from fit_vhap_lower_seam import shoulder_axes, unit
from review_smplx_head_neck import mesh_overlay, panel, project


def shoulder_anchors(body, obs, records):
    names = body["names"].tolist()
    if names != obs["names"].tolist():
        raise ValueError("Shoulder detections do not match the registered body")
    ids = [obs["joint_names"].tolist().index(key) for key in ("left_shoulder", "right_shoulder")]
    points = obs["keypoints"][:, ids].astype(np.float64)
    scores = obs["scores"][:, ids]
    flips = obs["flip_disagreement"][:, ids]
    box_width = obs["boxes"][:, 2] - obs["boxes"][:, 0]
    valid = (scores > .5) & (flips < np.maximum(7, .06*box_width[:, None]))
    joints = body["joints_world"]
    axes = shoulder_axes(joints)
    base = joints[:, [16, 17]].astype(np.float64)
    center = body["native_camera_center"]
    cameras = body["native_camera_to_world_rotation"]
    initial_depth = ((base - center[:, None]) @ cameras)[..., 2]
    profile = np.asarray([abs(r["baseline_yaw_deg"]) > 58 for r in records])
    valid[np.where(profile)[0], initial_depth[profile].argmax(axis=1)] = False
    check = np.arange(len(base)) % 5 == 1
    fit = valid & ~check[:, None]

    def prediction(p):
        target = base + p.reshape(1, 2, 3) @ axes.transpose(0, 2, 1)
        homo = ((target - center[:, None]) @ cameras) @ body["native_K"].T
        return homo[..., :2]/homo[..., 2:]

    def error(p):
        return np.r_[((prediction(p) - points)[fit] * scores[fit, None]).ravel(), 100*p]

    original = np.zeros(6)
    p = least_squares(error, original, loss="soft_l1", f_scale=4., bounds=(-.045, .045)).x
    def score(fraction):
        stats = {}
        for key, select in (("fit", fit), ("check", valid & check[:, None])):
            if not select.any():
                raise ValueError("No eligible shoulder anchor observations")
            stats[key] = {"measurements": int(select.sum()), **{n: float(np.median(np.linalg.norm(
                (prediction(x) - points)[select], axis=-1))) for n, x in (("before_px", original), ("after_px", fraction*p))}}
        return stats

    return base, axes, p.reshape(2, 3) @ axes.transpose(0, 2, 1), valid, score


def refine_patch(patch, anchors, axes, delta):
    before = patch["vertices_world"]
    center = anchors.mean(axis=1, keepdims=True)
    lateral = ((before - center) * axes[:, None, :, 0]).sum(axis=-1)
    width = np.linalg.norm(anchors[:, 0] - anchors[:, 1], axis=1, keepdims=True)
    t = np.clip((np.abs(lateral)/(width*.5) - .25)/.75, 0, 1)
    t = t*t*(3 - 2*t)
    canonical = patch["canonical_vertices"]
    distance = cdist(canonical, canonical[patch["patch_boundary_vertex_ids"]]).min(axis=1)
    seam_fade = np.clip(distance / .095, 0, 1)
    t *= (seam_fade**2 * (3 - 2*seam_fade))[None]
    movement = np.where(lateral[..., None] >= 0, delta[:, None, 0], delta[:, None, 1]) * t[..., None]
    p = before[:, patch["faces"]]
    n = np.cross(p[:, :, 1] - p[:, :, 0], p[:, :, 2] - p[:, :, 0])
    displacement = 1.
    for _ in range(7):
        after = (before + displacement*movement).astype(before.dtype)
        q = after[:, patch["faces"]]
        m = np.cross(q[:, :, 1] - q[:, :, 0], q[:, :, 2] - q[:, :, 0])
        if (unit(n)*unit(m)).sum(axis=-1).min() >= .25 and (np.linalg.norm(m, axis=-1)/np.linalg.norm(n, axis=-1)).min() >= .5:
            return after, displacement
        displacement *= .5
    return before, 0.


def run(args):
    cv2.setNumThreads(1)
    before = np.load(args.before, allow_pickle=False)
    fitted = np.load(args.fitted, allow_pickle=False)
    body = np.load(args.body, allow_pickle=False)
    obs = np.load(args.observations, allow_pickle=False)
    for key in ("names", "faces", "native_K", "native_camera_center", "native_camera_to_world_rotation",
                "virtual_crop_homographies", "neck_boundary_vertex_ids", "scene_scale"):
        if not np.array_equal(before[key], fitted[key]):
            raise ValueError(f"Input changed during lower-seam fit: {key}")
    names = fitted["names"].tolist()
    vp = fitted["vertices_world"]
    old = before["vertices_world"]
    patch = build_patch(args.body, args.reference, args.model, vp)
    faces = fitted["faces"]
    patch_faces = patch["faces"]
    neck_ids = patch["head_boundary_vertex_ids"]
    patch_ids = patch["patch_boundary_vertex_ids"]
    nonseam = np.setdiff1d(np.arange(patch["vertices_world"].shape[1]), patch_ids)
    remap = np.zeros(patch["vertices_world"].shape[1], dtype=int)
    remap[patch_ids] = neck_ids
    remap[nonseam] = len(vp[0]) + np.arange(len(nonseam))
    args.output.mkdir(parents=True, exist_ok=False)

    with ZipFile(args.frames) as source:
        records = json.loads(source.read("initialization.json"))["frames"]
        if [r["name"] for r in records] != names:
            raise ValueError("Registered crops do not match the attachment mesh")
        anchors, axes, delta, valid, score = shoulder_anchors(body, obs, records)
        shoulder, step = refine_patch(patch, anchors, axes, delta)
        if not np.array_equal(shoulder[:, patch_ids], vp[:, neck_ids]):
            raise RuntimeError("Patch is no longer welded to the existing neck ring")
        stitched = np.concatenate((vp, shoulder[:, nonseam]), axis=1)
        stitched_faces = np.vstack((faces, remap[patch_faces])).astype(np.int32)
        tints = np.repeat(np.array([[246, 210, 87], [60, 174, 241]]), [len(faces), len(patch_faces)], axis=0)
        # Preserve provenance and label which faces are shirt/shoulder proxy versus skin.
        metadata = {k: fitted[k] for k in fitted.files if k not in ("vertices_world", "faces")}
        np.savez_compressed(args.output / "head-neck-shoulders.npz", vertices_world=stitched, faces=stitched_faces,
                            cloth_faces=np.arange(len(faces), len(stitched_faces)), **metadata)
        ffmpeg_args = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
                       "1152x384", "-framerate", "3", "-i", "-", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(args.output / "all-frames.mp4")]
        sheets = {}
        with subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE) as video:
            for i, r in enumerate(records):
                original = cv2.imdecode(np.frombuffer(source.read(f"images/{i:08d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
                c = fitted["native_camera_to_world_rotation"][i]
                o = fitted["native_camera_center"][i]
                h = fitted["virtual_crop_homographies"][i] @ fitted["native_K"]
                start = mesh_overlay(original, (old[i] - o) @ c, h, faces)
                end = mesh_overlay(original, (stitched[i] - o) @ c, h, stitched_faces, tint=tints)
                xy, _ = project((anchors[i] + step*delta[i] - o) @ c, h)
                for side in np.flatnonzero(valid[i]):
                    x, y = np.rint(xy[side]).astype(int)
                    cv2.circle(end, (x, y), 4, (0, 250, 255), -1, cv2.LINE_AA)
                row = np.hstack([panel(original, f"{r['name']} | original"), panel(start, "Before | head/neck only"),
                                 panel(end, "After | lower seam + shoulder patch")])
                video.stdin.write(row.tobytes())
                if r["name"] in ("00010.jpg", "00510.jpg", "00190.jpg", "00700.jpg", "00830.jpg", "00840.jpg", "00850.jpg", "00860.jpg"):
                    sheets[r["name"]] = row
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("Could not write review video")
    for label, items in (("frontal", ["00010.jpg", "00510.jpg"]), ("oblique", ["00190.jpg", "00700.jpg"]),
                         ("profile", ["00830.jpg", "00840.jpg", "00850.jpg", "00860.jpg"])):
        if all(name in sheets for name in items):
            cv2.imencode(".jpg", np.vstack([sheets[name] for name in items]), [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(
                args.output / f"{label}-comparison.jpg")
    result = dict(views=len(names), shoulder_anchor_step=step, shoulder_anchor_stats=score(step),
                  welded_vertices=len(neck_ids), head_vertices=len(vp[0]), shoulder_faces=len(patch_faces))
    (args.output / "shoulder-fit.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("before", "fitted", "body", "observations", "reference", "model", "frames", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
