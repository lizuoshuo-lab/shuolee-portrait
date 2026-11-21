"""Set FLAME's initial cervical inclination relative to the subject's upper torso."""

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from fit_native_skin_profile import camera, project


def torso_axes(joints):
    def unit(v):
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        if np.any(norm < 1e-6):
            raise ValueError("Degenerate upper-torso anatomical frame")
        return v / norm
    left = unit(joints[..., 16, :] - joints[..., 17, :])
    vertical = joints[..., [16,17], :].mean(axis=-2) - joints[..., [1,2], :].mean(axis=-2)
    up = unit(vertical - (vertical * left).sum(-1, keepdims=True) * left)
    return left, up, np.cross(left, up)


def inclination(vertices, rim, ears, up, forward):
    line = vertices[..., ears, :].mean(-2) - vertices[..., rim, :].mean(-2)
    return np.rad2deg(np.arctan2((line * forward).sum(-1), (line * up).sum(-1)))


@torch.no_grad()
def neutralize(tracker, source, settings, gauge, origin, anatomy, template, correspondence, masks):
    records = settings["frames"]
    names = [r["name"] for r in records]
    with anatomy.open("rb") as stream, np.load(stream) as body:
        if names != body["names"].tolist() or not np.isclose(float(body["scene_scale"]),settings["scene_scale"]):
            raise ValueError("A same-sequence, same-scale upper-torso frame is required")
        left, up, forward = torso_axes(body["joints_world"])
    with masks.open("rb") as stream:
        regions = pickle.load(stream, encoding="latin1")
    rim = np.intersect1d(regions["neck"], regions["boundary"])
    ears = np.r_[regions["left_ear"], regions["right_ear"]]
    mapping = np.load(correspondence, allow_pickle=False)
    with np.load(template, allow_pickle=False) as neutral_body:
        vertices = neutral_body["v_template"]
        _, neutral_up, neutral_forward = torso_axes(neutral_body["J_regressor"] @ vertices)
        neutral = vertices[mapping]
        if np.dot(neutral[regions["face"]].mean(0)-neutral[ears].mean(0), neutral_forward) <= 0:
            raise ValueError("The anatomical forward axis opposes the face")
        target = float(inclination(neutral, rim, ears, neutral_up, neutral_forward))
    flame = tracker.flame
    old = Rotation.from_rotvec(source["rotation"])
    t0 = source["translation"]
    joint = (flame.J_regressor[:2] @ flame.v_template).cpu().numpy()
    arm = joint[1]-joint[0]
    left_tracker = left @ gauge.T
    device = tracker.device

    def poses(delta):
        root = Rotation.from_rotvec(np.deg2rad(delta)[:,None] * left_tracker) * old
        return {"rotation": root.as_rotvec(), "neck_pose": (root.inv()*old).as_rotvec(),
                "translation": t0 + old.apply(arm) - root.apply(arm)}

    def neutral_mesh(delta):
        seed = poses(delta)
        v, l = [], []
        for lo in range(0,len(records),12):
            sl = slice(lo,lo+12)
            size = len(seed["rotation"][sl])
            def value(key):
                return torch.as_tensor(seed[key][sl],dtype=torch.float32,device=device)
            def zeros(n):
                return torch.zeros((size,n),dtype=torch.float32,device=device)
            vs, landmarks = flame(zeros(tracker.shape.numel()),zeros(tracker.expr.shape[-1]),
                                  value("rotation"),value("neck_pose"),zeros(3),zeros(6),value("translation"))
            v.append(vs.cpu().numpy())
            l.append(landmarks.cpu().numpy())
        return np.concatenate(v),np.concatenate(l)

    delta = np.zeros(len(records))
    before_mesh, before_lm = neutral_mesh(delta)
    before = inclination(before_mesh @ gauge,rim,ears,up,forward)
    after = before
    for _ in range(6):
        if np.max(np.abs(target-after)) < .02:
            break
        probe = inclination(neutral_mesh(delta+.25)[0] @ gauge,rim,ears,up,forward)
        slope = (probe-after)/.25
        valid = np.abs(slope)>.05
        step = np.divide(target-after,slope,out=np.zeros_like(delta),where=valid)
        delta = np.clip(delta + np.clip(step,-15.,15.),-40.,40.)
        after = inclination(neutral_mesh(delta)[0] @ gauge,rim,ears,up,forward)
    seed = poses(delta)
    after_mesh, after_lm = neutral_mesh(delta)
    root = Rotation.from_rotvec(seed["rotation"])
    composed = root * Rotation.from_rotvec(seed["neck_pose"])
    np.testing.assert_allclose((old.inv()*composed).magnitude(),0,atol=1e-9)
    np.testing.assert_allclose(root.apply(arm)+seed["translation"],old.apply(arm)+t0,atol=1e-8)
    scale = settings["scene_scale"]
    face_shift = np.concatenate([
        np.linalg.norm(project(a[17:68]@gauge*scale+origin,*camera(r))[0] -
                       project(b[17:68]@gauge*scale+origin,*camera(r))[0],axis=1)
        for a,b,r in zip(before_lm,after_lm,records)])
    report = {"reference":"Neutral SMPL-X template, lower neck rim to mean ears, sagittal angle relative to shoulder-hip up",
              "neutral_template_deg":target,"old_deg":before.tolist(),"new_deg":after.tolist(),
              "root_correction_deg":delta.tolist(),"near_target_frames":int((np.abs(after-target)<=.1).sum()),
              "frames":names,"inner_face_motion_registered_max_px":float(face_shift.max()),
              "stage":"initialization only","input_anatomy":str(anatomy),"template":str(template)}
    return seed, report, (before_mesh, after_mesh)
