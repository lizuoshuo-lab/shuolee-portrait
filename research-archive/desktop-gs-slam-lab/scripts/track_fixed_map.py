"""Six-DoF photometric pose recovery against a frozen gsplat Gaussian map.

Default example uses a held-out Tiny NeRF view. Ground truth sets a deliberately
perturbed initialization and is used only for evaluation, never in the objective.
This is not online SLAM. A generic NPZ bundle supports later RGB-D map exercises.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]


def hat(vector):
    """SO(3) hat operator, differentiable even at the zero rotation vector."""
    x, y, z = vector.unbind()
    zero = x * 0.0
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)


def se3_exp(twist):
    """Exact matrix exponential of [translation tangent, rotation tangent].

    Its translation block is V(omega) @ v, rather than simply v. This avoids
    interpreting an unconstrained 4x4 matrix as a camera rotation.
    """
    upper = torch.cat((hat(twist[3:]), twist[:3, None]), dim=1)
    algebra = torch.cat((upper, twist.new_zeros(1, 4)), dim=0)
    return torch.matrix_exp(algebra)


def pose_errors(w2c, ground_truth):
    estimate = torch.linalg.inv(w2c.detach().double()).cpu()
    truth = torch.linalg.inv(ground_truth.detach().double()).cpu()
    translation = torch.linalg.vector_norm(estimate[:3, 3] - truth[:3, 3])
    relative_rotation = estimate[:3, :3] @ truth[:3, :3].T
    # atan2 is better conditioned than acos near the correct pose.
    skew = torch.stack((relative_rotation[2, 1] - relative_rotation[1, 2],
                        relative_rotation[0, 2] - relative_rotation[2, 0],
                        relative_rotation[1, 0] - relative_rotation[0, 1]))
    sine = torch.linalg.vector_norm(skew) / 2.0
    cosine = (torch.trace(relative_rotation) - 1.0) / 2.0
    rotation = torch.atan2(sine, cosine).abs() * 180.0 / math.pi
    orthogonality = torch.linalg.matrix_norm(w2c[:3, :3].T @ w2c[:3, :3]
                                           - torch.eye(3, device=w2c.device, dtype=w2c.dtype))
    return {"translation_error_dataset_units": float(translation),
            "rotation_error_degrees": float(rotation),
            "rotation_orthogonality_frobenius": float(orthogonality),
            "rotation_determinant": float(torch.linalg.det(w2c[:3, :3]))}


def cpu_math_check():
    vector = torch.tensor([.05, -.03, .02, .07, -.04, .03], dtype=torch.float64,
                          requires_grad=True)
    zero_vector = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    transform = se3_exp(vector)
    identity = torch.eye(4, dtype=torch.float64)
    errors = {
        "zero_twist_identity_max_error": float((se3_exp(vector * 0) - identity).abs().max()),
        "inverse_max_error": float((se3_exp(-vector) @ transform - identity).abs().max()),
        "rotation_orthogonality_max_error": float((transform[:3, :3].T @ transform[:3, :3]
                                                   - identity[:3, :3]).abs().max()),
        "rotation_determinant_error": float((torch.linalg.det(transform[:3, :3]) - 1).abs()),
        "gradcheck_nonzero": bool(torch.autograd.gradcheck(se3_exp, (vector,), eps=1e-6,
                                                          atol=1e-5, rtol=1e-4)),
        "gradcheck_zero": bool(torch.autograd.gradcheck(se3_exp, (zero_vector,),
                                                       eps=1e-6, atol=1e-5, rtol=1e-4))}
    if max(errors[key] for key in errors if key.endswith("error")) > 1e-10:
        raise RuntimeError(f"SE3 rigid transform validation failed: {errors}")
    return errors


def save_image(tensor, path):
    array = tensor.detach().cpu().numpy()
    Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion_covariance(quats, scales):
    # gsplat uses scalar-first (w,x,y,z) quaternions.
    w, x, y, z = torch.nn.functional.normalize(quats, dim=-1).unbind(-1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    factor = rotation * scales[:, None, :]
    return factor @ factor.transpose(-1, -2)


def pose_gradient_probe(loss_from_twist, device="cuda", dtype=torch.float32, epsilon=1e-4,
                        min_cosine=.8, max_relative_l2=.5):
    """Independent six-DoF gradient check for any scalar loss callable.

    loss_from_twist receives a length-six perturbation about zero; it must build
    a differentiable SE3 camera and compute the intended data loss, without GT.
    This helper makes no assumptions about map storage or image/depth objective.
    """
    probe = torch.zeros(6, device=device, dtype=dtype, requires_grad=True)
    analytic = torch.autograd.grad(loss_from_twist(probe), probe)[0]
    numeric = []
    with torch.no_grad():
        for dimension in range(6):
            offset = torch.zeros(6, device=device, dtype=dtype)
            offset[dimension] = epsilon
            plus = loss_from_twist(offset)
            minus = loss_from_twist(-offset)
            numeric.append(float((plus - minus) / (2 * epsilon)))
    numeric = torch.tensor(numeric, device=device, dtype=dtype)
    cosine = float(torch.nn.functional.cosine_similarity(analytic[None], numeric[None]))
    relative_l2 = float(torch.linalg.vector_norm(analytic - numeric)
                        / torch.linalg.vector_norm(numeric).clamp_min(1e-12))
    if not torch.isfinite(analytic).all() or not torch.isfinite(numeric).all():
        raise RuntimeError("Pose gradient is non-finite")
    if torch.linalg.vector_norm(analytic) < 1e-10:
        raise RuntimeError("Pose gradient vanished")
    if cosine < min_cosine or relative_l2 > max_relative_l2:
        raise RuntimeError(f"Pose finite difference check failed: cosine={cosine}, relative_l2={relative_l2}")
    return {"analytic_six_dof": analytic.detach().cpu().tolist(),
            "finite_difference_six_dof": numeric.cpu().tolist(),
            "finite_difference_epsilon": epsilon, "cosine_similarity": cosine,
            "relative_l2_error": relative_l2, "all_finite": True,
            "nonzero_analytic_components": int((analytic.abs() > 1e-10).sum())}


def load_inputs(args):
    map_path = Path(args.map)
    if not map_path.is_absolute():
        map_path = ROOT / map_path
    checkpoint = torch.load(map_path, map_location="cpu", weights_only=True)
    required = {"means", "log_scales", "quats", "color_logits", "opacity_logits"}
    if not required.issubset(checkpoint["parameters"]):
        raise ValueError(f"Map parameters require {required}")
    frozen = {key: value.detach().to(device="cuda", dtype=torch.float32).contiguous()
              for key, value in checkpoint["parameters"].items() if key in required}
    if any(value.requires_grad or not torch.isfinite(value).all() for value in frozen.values()):
        raise ValueError("Map must be frozen and finite")
    bundle_path = Path(args.bundle) if args.bundle else ROOT / "data/tiny-nerf/tiny_nerf_data.npz"
    if not bundle_path.is_absolute():
        bundle_path = ROOT / bundle_path
    data = np.load(bundle_path, allow_pickle=False)
    images = data["images"]
    target = torch.from_numpy(images[args.view]).to(device="cuda", dtype=torch.float32)
    if target.ndim != 3 or target.shape[-1] != 3 or target.min() < 0 or target.max() > 1:
        raise ValueError("Bundle images must be float RGB [V,H,W,3] in [0,1]")
    height, width = target.shape[:2]
    if "w2c" in data:
        truth = torch.from_numpy(data["w2c"][args.view]).to(device="cuda", dtype=torch.float32)
        intrinsic = data["intrinsics"]
        intrinsic = intrinsic[args.view] if intrinsic.ndim == 3 else intrinsic
        intrinsic = torch.as_tensor(intrinsic, device="cuda", dtype=torch.float32)[None]
        unit = args.translation_unit or "dataset coordinate units"
        convention = "provided OpenCV world-to-camera"
    else:
        pose = torch.from_numpy(data["poses"][args.view]).to(device="cuda", dtype=torch.float32)
        conversion = torch.diag(torch.tensor([1., -1., -1., 1.], device="cuda"))
        truth = torch.linalg.inv(pose @ conversion)
        focal = float(data["focal"])
        intrinsic = torch.tensor([[focal, 0., width/2.], [0., focal, height/2.],
                                   [0., 0., 1.]], device="cuda")[None]
        unit = "synthetic scene coordinate units; not calibrated meters"
        convention = "OpenGL c2w @ diag(1,-1,-1,1), inverted to OpenCV w2c"
    if "heldout_local_indices" in data:
        heldout = [int(value) for value in data["heldout_local_indices"]]
        training = [int(value) for value in data["train_local_indices"]]
    else:
        heldout = checkpoint.get("test_views", [])
        training = checkpoint.get("training_views", [])
    split_metadata = {"target_local_view_index": args.view,
                      "target_dataset_frame_id": int(data["frame_ids"][args.view]) if "frame_ids" in data else args.view,
                      "target_in_map_training_views": args.view in training,
                      "target_in_map_heldout_views": args.view in heldout,
                      "map_training_view_indices": training, "map_heldout_view_indices": heldout}
    return map_path, frozen, target, truth.contiguous(), intrinsic, width, height, unit, convention, split_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="outputs/tiny-nerf-gsplat-001/gaussians.pt")
    parser.add_argument("--bundle", help="NPZ images/w2c/intrinsics, or official Tiny NeRF NPZ")
    parser.add_argument("--view", type=int, default=101)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--translation", type=float, nargs=3, default=[.1, .05, -.03])
    parser.add_argument("--rotation-deg", type=float, nargs=3, default=[3., -4., 0.])
    parser.add_argument("--translation-unit", default=None)
    parser.add_argument("--lr-translation", type=float, default=.003)
    parser.add_argument("--lr-rotation", type=float, default=.0015)
    parser.add_argument("--lr-decay", type=float, default=.995)
    parser.add_argument("--render-mode", choices=["viewmat", "camera-covariance"], default="viewmat")
    parser.add_argument("--target-source", choices=["dataset", "rendered"], default="dataset",
                        help="Rendered target is a synthetic gradient sanity test only")
    parser.add_argument("--run", default="tiny-fixed-map-pose-001")
    parser.add_argument("--finite-difference-epsilon", type=float, default=1e-4)
    parser.add_argument("--check-only", action="store_true", help="CPU SE3 checks without CUDA")
    args = parser.parse_args()
    math_checks = cpu_math_check()
    if args.check_only:
        print(json.dumps({"status": "passed", "cpu_se3_checks": math_checks}, indent=2))
        return
    if args.steps < 1 or args.finite_difference_epsilon <= 0 or not 0 < args.lr_decay <= 1:
        raise ValueError("Positive steps/finite difference epsilon and lr decay in (0,1] required")
    if not args.run or not args.run.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run name must contain only letters, numbers, dashes and underscores")
    output = ROOT / "outputs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    from gsplat import rasterization
    map_path, frozen, target, truth, intrinsics, width, height, unit, convention, split_metadata = load_inputs(args)
    hash_before = checkpoint_hash(map_path)
    perturbation = torch.tensor(args.translation + [math.radians(v) for v in args.rotation_deg], device="cuda")
    initial = (se3_exp(perturbation) @ truth).detach()
    translation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    rotation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    normalized_quats = torch.nn.functional.normalize(frozen["quats"], dim=-1)
    scales = frozen["log_scales"].exp()
    colors = frozen["color_logits"].sigmoid()
    opacities = frozen["opacity_logits"].sigmoid()
    covariances = quaternion_covariance(normalized_quats, scales)
    identity = torch.eye(4, device="cuda")[None]

    def current_pose(twist=None):
        if twist is None:
            twist = torch.cat((translation, rotation))
        return se3_exp(twist) @ initial

    def render(pose, mode=None):
        mode = mode or args.render_mode
        if mode == "viewmat":
            return rasterization(frozen["means"], normalized_quats, scales, opacities, colors,
                                 pose[None].contiguous(), intrinsics, width, height,
                                 packed=False)[0][0]
        # Equivalent parameterization that transfers pose gradients to means/covars.
        camera_means = frozen["means"] @ pose[:3, :3].T + pose[:3, 3]
        camera_covars = pose[:3, :3] @ covariances @ pose[:3, :3].T
        return rasterization(camera_means, None, None, opacities, colors,
                             identity, intrinsics, width, height, covars=camera_covars,
                             packed=False)[0][0]

    def objective(prediction):
        return ((prediction - target) ** 2).mean()

    with torch.no_grad():
        ground_truth_render = render(truth)
        if args.target_source == "rendered":
            target = ground_truth_render.detach()
        first_prediction = render(initial)
        initial_loss = float(objective(first_prediction))
        gt_pose_rgb_loss = float(objective(ground_truth_render))
        equivalence_error = float((render(initial, "viewmat")
                                  - render(initial, "camera-covariance")).abs().max())
    save_image(target, output / "target.png")
    save_image(first_prediction, output / "initial.png")
    save_image(ground_truth_render, output / "gt_pose_render.png")

    # Centered finite difference check is independent of CUDA's analytic backward.
    gradient_checks = pose_gradient_probe(lambda twist: objective(render(current_pose(twist))),
                                         epsilon=args.finite_difference_epsilon)
    gradient_checks["camera_covariance_render_max_abs_difference"] = equivalence_error
    optimizer = torch.optim.Adam([{"params": [translation], "lr": args.lr_translation},
                                  {"params": [rotation], "lr": args.lr_rotation}], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history = []
    best_loss, best_pose, best_step = initial_loss, initial.clone(), 0
    best_twist = torch.zeros(6, device="cuda")
    for step in range(args.steps):
        pose = current_pose()
        prediction = render(pose)
        loss = objective(prediction)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite objective at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
               for parameter in (translation, rotation)):
            raise RuntimeError(f"Non-finite or missing pose gradient at {step}")
        if float(loss.detach()) < best_loss:
            best_loss, best_pose, best_step = float(loss.detach()), pose.detach().clone(), step
            best_twist = torch.cat((translation.detach(), rotation.detach())).clone()
        if step % 10 == 0 or step == args.steps - 1:
            errors = pose_errors(pose, truth)
            history.append({"step": step, "rgb_mse": float(loss.detach()), **errors,
                            "translation_gradient_l2": float(translation.grad.norm()),
                            "rotation_gradient_l2": float(rotation.grad.norm())})
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step={step+1}/{args.steps} rgb_mse={float(loss.detach()):.7f}", flush=True)
    torch.cuda.synchronize()
    optimization_seconds = time.perf_counter() - started
    with torch.no_grad():
        last_pose = current_pose()
        last_loss = float(objective(render(last_pose)))
        if last_loss < best_loss:
            best_loss, best_pose, best_step = last_loss, last_pose.detach().clone(), args.steps
            best_twist = torch.cat((translation.detach(), rotation.detach())).clone()
        final_prediction = render(best_pose)
    map_unchanged = hash_before == checkpoint_hash(map_path)
    if not map_unchanged or any(value.requires_grad for value in frozen.values()):
        raise RuntimeError("The frozen map was modified")
    save_image(final_prediction, output / "final.png")
    panel = Image.new("RGB", (3*max(width, 200), max(height, 200)+40), "white")
    draw = ImageDraw.Draw(panel)
    for column, (filename, label) in enumerate((("target.png", "Target RGB"),
                                               ("initial.png", "Perturbed camera"),
                                               ("final.png", "Pose optimized / fixed map"))):
        with Image.open(output / filename) as frame:
            panel.paste(frame.resize((max(width, 200), max(height, 200))),
                        (column*max(width, 200), 35))
        draw.text((column*max(width, 200)+4, 8), label, fill="black")
    panel.save(output / "comparison.png")
    with (output / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save({"initial_w2c": initial.cpu(), "estimated_w2c": best_pose.cpu(),
                "last_w2c": last_pose.detach().cpu(), "evaluation_gt_w2c": truth.cpu(),
                "best_translation_delta": best_twist[:3].cpu(),
                "best_rotation_delta": best_twist[3:].cpu(),
                "last_translation_delta": translation.detach().cpu(),
                "last_rotation_delta": rotation.detach().cpu(), "best_step": best_step,
                "intrinsics": intrinsics.cpu(), "config": vars(args)}, output / "pose.pt")
    result = {"status": "completed", "scope": "fixed-map single-view six-DoF photometric pose recovery; not online SLAM",
              "initialization": "ground-truth w2c with explicit left SE3 perturbation",
              "gt_usage": ("synthetic control: GT camera creates rendered target; initialization and evaluation"
                           if args.target_source == "rendered" else
                           "initialization and evaluation only; no GT pose in photometric objective"),
              "map_checkpoint": str(map_path.relative_to(ROOT)), "map_sha256": hash_before,
              "map_unchanged": map_unchanged, "optimized_parameter_count": 6,
              "map_trainable_parameter_count": 0, "target_view": args.view,
              "target_source": args.target_source, "resolution": [width, height],
              "data_split": split_metadata,
              "camera_convention": convention, "translation_unit": unit,
              "config": vars(args), "initial_rgb_mse": initial_loss, "final_rgb_mse": best_loss,
              "last_rgb_mse": last_loss, "gt_pose_rgb_mse_for_evaluation": gt_pose_rgb_loss,
              "initial_psnr_db": -10*math.log10(max(initial_loss, 1e-12)),
              "final_psnr_db": -10*math.log10(max(best_loss, 1e-12)),
              "initial_pose_error": pose_errors(initial, truth),
              "final_pose_error": pose_errors(best_pose, truth),
              "last_pose_error": pose_errors(last_pose, truth),
              "best_step_selected_by_rgb_loss_only": best_step,
              "all_iteration_gradients_finite": True, "gradient_checks": gradient_checks,
              "cpu_se3_checks": math_checks, "optimization_seconds_excludes_compile_and_checks": optimization_seconds,
              "peak_torch_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "limitations": ["single target frame", "GT perturbed initialization", "frozen learned map has reconstruction error",
                               "best photometric camera can differ from true camera", "no online trajectory, mapping, loop closure, or ATE"]}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
