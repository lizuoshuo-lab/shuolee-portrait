"""Fit two shared, post-skinning FLAME contour offsets with camera and pose fixed."""

from argparse import ArgumentParser
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import minimize
import torch

from audit_side_geometry import draw_boundary, make_panel, mesh_mask
from fit_side_frames import alpha_maps, mask_scores
from fuse_scene import local_head, make_rasterizer, prepare_scene, render, to_bgr, world_gaussians
from profile_shape import PROFILE_LIMITS, profile_basis, profile_weights
from train_joint import JointParameters, combine


def run(args):
    result = json.loads((args.run / "model/metrics.json").read_text())
    side = json.loads((args.run / "side/side-poses.json").read_text())
    settings = {key: Path(path) for key, path in result["input_paths"].items()}
    settings["export_frame"] = result["reference"]
    K, size, cameras, _, _, track, actor, keep, _, scene = prepare_scene(SimpleNamespace(**settings))
    time = {row["name"]: row["timestamp_s"]
            for row in json.loads((settings["scene"] / "source-frames.json").read_text())["frames"]}
    hold = side["review_frames"]
    records = side["frames"]
    fit = [n for n, r in records.items() if r.get("angular_training_candidate")
           and abs(r["camera_relative_yaw_deg"]) >= 28
           and all(abs(time[n] - time[h]) > .6 for h in hold)]
    fit = sorted(fit, key=lambda n: records[n]["camera_relative_yaw_deg"])
    if len(fit) > 10:
        fit = [fit[i] for i in np.linspace(0, len(fit) - 1, 10).round().astype(int)]
    if (sum(records[n]["camera_relative_yaw_deg"] > 0 for n in fit) < 2
            or sum(records[n]["camera_relative_yaw_deg"] < 0 for n in fit) < 2):
        raise RuntimeError("Need reliable fit views on both sides of the head, outside review time blocks.")
    names = fit + hold
    evidence, _, _, _, _ = alpha_maps(settings["scene"], size, K, names)
    poses = {n: tuple(np.asarray(records[n][k]) for k in ("rotation", "translation")) for n in names}
    used_cameras = {name: cameras[Path(name).stem] for name in names}
    rasters = {name: make_rasterizer(K, size, *used_cameras[name]) for name in names}
    scale = side["scale"]
    binding = actor.binding[keep].long()
    local_head(actor, track, 0)
    reference_face_size = actor.face_scaling[binding].reshape(-1, 1, 1).clone()
    articulation = {k: torch.as_tensor([v], device="cuda", dtype=torch.float32)
                    for k, v in side["propagated_articulation"].items()}
    local_head(actor, articulation, 0)
    vertices = actor.verts.clone()
    faces = actor.flame_model.faces[binding].cpu().numpy()
    weights, protected = profile_weights(actor, settings["flame_boundary_mask"])
    basis = profile_basis(actor, weights)
    assert basis.shape == (vertices.shape[1], 2, 3) and protected.shape == (vertices.shape[1],)
    if not torch.isfinite(basis).all() or torch.count_nonzero(basis[protected]):
        raise RuntimeError("Profile offsets must be finite and zero at protected points.")
    canonical = actor.verts_cano.clone()
    params = JointParameters(scene, world_gaussians(actor, np.eye(3), np.zeros(3), 1., keep))
    checkpoint = torch.load(args.run / "model/training_state.pt", weights_only=True, map_location="cpu")
    params.load_state_dict(checkpoint["parameters"])
    fitted_scene = params.scene(scene)

    def set_shape(coefficients):
        delta = (basis * basis.new_tensor(coefficients)[None, :, None]).sum(dim=1)
        updated = vertices + delta[None, :, :]
        if not torch.equal(updated[:, protected], vertices[:, protected]):
            raise RuntimeError("Protected face or landmark vertices moved.")
        actor.update_mesh_properties(updated, canonical)

    def get_head(name):
        A, t = poses[name]
        head = world_gaussians(actor, A, t, scale, keep)
        axes = head[0].new_tensor(A) @ actor.face_orien_mat[binding]
        axes *= scale * actor.face_scaling[binding].reshape(-1, 1, 1) / reference_face_size
        return params.head(SimpleNamespace(head=head, axes=axes), geometry=True)

    limits = np.array([PROFILE_LIMITS[0][1], abs(PROFILE_LIMITS[1][0])])

    def objective(coefficients):
        set_shape(coefficients)
        losses = []
        for n in fit:
            obs = evidence[n]
            crop = obs["crop"]
            valid = obs["valid"][crop]
            target = obs["target"][crop]
            alpha = render(rasters[n], get_head(n), alpha_only=True)[crop].cpu().numpy()
            losses.append(float(np.abs(alpha - target)[valid].sum() / (target & valid).sum()))
        return mean(losses) + .010 * float(np.square(coefficients / limits).sum())

    initial = np.zeros(2)
    base_loss = objective(initial)
    optimized = minimize(objective, initial, method="Powell", bounds=[(0, limits[0]), (-limits[1], 0)],
                         options={"maxfev": 48, "xtol": .0005, "ftol": 1e-4})
    coefficients = optimized.x if optimized.fun < base_loss else initial
    print(f"Fit {fit}; coefficients {coefficients.tolist()}; loss {base_loss:.5f} -> {optimized.fun:.5f}", flush=True)
    rows, scores = [], {}
    for n in hold:
        obs = evidence[n]
        source = cv2.imread(str(settings["scene"] / "orig_undist/images" / n))
        crop, valid = obs["crop"], obs["boundary_valid"]
        overlay = source.copy()
        draw_boundary(overlay, obs["target"], valid, (72, 242, 74))
        panels = [(source, f"{n} original")]
        outline = overlay.copy()
        case = {}
        for label, values, color in (("Before", initial, (255, 169, 37)),
                                     ("Refined", coefficients, (40, 140, 255))):
            set_shape(values)
            head = get_head(n)
            alpha = render(rasters[n], head, alpha_only=True).cpu().numpy()
            draw_boundary(outline, alpha >= .5, valid, color)
            mesh = mesh_mask(actor.verts[0].cpu().numpy(), faces, K, used_cameras[n], poses[n], scale, size)
            case[label] = {"mesh": mask_scores(mesh, obs), "gaussian": mask_scores(alpha, obs)}
            rgb = to_bgr(render(rasters[n], combine(fitted_scene, head)))
            panels.append((rgb, label))
        panels.append((outline, "Green photo / blue old / orange new"))
        rows.append(np.hstack([make_panel(image[crop], text) for image, text in panels]))
        scores[n] = case
    if not cv2.imwrite(str(args.output / "profile-refinement.jpg"), np.vstack(rows)):
        raise OSError("Could not write profile-refinement.jpg")
    iou = {label: mean(v[label]["gaussian"]["iou"] for v in scores.values()) for label in ("Before", "Refined")}
    if all(v[label]["gaussian"]["edge_p90_px"] is not None
           for v in scores.values() for label in ("Before", "Refined")):
        edge = {label: mean(v[label]["gaussian"]["edge_p90_px"] for v in scores.values())
                for label in ("Before", "Refined")}
    else:
        edge = None
    accepted = iou["Refined"] > iou["Before"] + .003 and edge is not None and edge["Refined"] <= edge["Before"] + .5
    report = {"scalp_and_posterior_neck_flame_units": coefficients.tolist(), "fit_frames": fit,
              "heldout_frames": hold, "fit_loss_before": base_loss, "fit_loss_after": float(optimized.fun),
              "heldout_gaussian_iou": iou, "heldout_gaussian_edge_p90_mean_px": edge,
              "protected_vertices": int(protected.sum()), "protected_vertices_max_displacement": 0.0,
              "shape_candidate_passed_contour_comparison": bool(accepted), "frames": scores}
    (args.output / "profile-refinement.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    with torch.no_grad():
        run(args)
