"""Refine profile yaw and image-plane translation from the observed silhouette."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation


PROFILE_MIN_YAW = 55.
YAW_GRID = np.arange(-12., 12.1, 4.)


def fit_profile_pose(baseline, camera, K, scale, center, evidence, render_alpha):
    """Use disjoint contour bands for optimization and acceptance at fixed depth."""
    crop = evidence["crop"]
    target = evidence["target"][crop]
    valid = evidence["valid"][crop]
    rows = np.flatnonzero((target & valid).any(axis=1))
    if len(rows) < 36 or (target & valid).sum() < 400:
        return baseline, {"accepted": False, "reason": "insufficient_silhouette"}
    # Leave out bands of observable surface, including their exterior background.
    bands = np.clip((np.arange(len(target)) - rows[0]) * 6 // len(rows), 0, 5)
    training = valid & (bands[:, None] % 2 == 0)
    check = valid & ~training
    if min((target & training).sum(), (target & check).sum()) < 120:
        return baseline, {"accepted": False, "reason": "insufficient_split_support"}

    R, C = camera
    A, t = baseline
    center_world = scale * A @ center + t
    depth = float((R.T @ (center_world - C))[2])
    if not np.isfinite(depth) or depth <= 0:
        return baseline, {"accepted": False, "reason": "invalid_depth"}
    shift_limit = min(14., max(5., .10 * np.count_nonzero(target.any(axis=0))))
    image_axes = R[:, :2] * (depth / np.array([K[0, 0], K[1, 1]]))

    def pose_at(yaw, shift):
        rotation = A @ Rotation.from_rotvec([0., np.deg2rad(yaw), 0.]).as_matrix()
        translation = center_world - scale * rotation @ center + image_axes @ shift
        return rotation, translation

    def alpha_at(yaw, shift):
        return render_alpha(pose_at(yaw, shift))

    def error(alpha, region):
        return float(np.abs(alpha - target)[region].sum() / (target & region).sum())

    def fit_position(yaw):
        def cost(shift):
            return error(alpha_at(yaw, shift), training) + .004 * float(np.square(shift / shift_limit).sum())

        optimum = minimize(cost, np.zeros(2), method="Powell", bounds=[(-shift_limit, shift_limit)] * 2,
                           options={"maxfev": 35, "xtol": .15, "ftol": 1e-4})
        return float(optimum.fun), optimum.x

    fixed_yaw_cost, fixed_yaw_xy = fit_position(0.)
    grid_scores = [fit_position(float(yaw))[0] if yaw else fixed_yaw_cost for yaw in YAW_GRID]
    best = int(np.argmin(grid_scores))
    if best in (0, len(YAW_GRID) - 1):
        return baseline, {"accepted": False, "reason": "yaw_optimum_at_search_limit",
                          "grid_yaw_deg": float(YAW_GRID[best]), "grid_training_errors": grid_scores}
    search = minimize_scalar(lambda yaw: fit_position(yaw)[0], method="bounded",
                             bounds=(YAW_GRID[best - 1], YAW_GRID[best + 1]),
                             options={"maxiter": 8, "xatol": .35})
    yaw = float(search.x)
    best_cost, shift = fit_position(yaw)
    candidate = alpha_at(yaw, shift)
    fixed_yaw = alpha_at(0., fixed_yaw_xy)
    original = render_alpha(baseline)
    # Compare against both the delivered baseline and translation optimized on the same data.
    old_error = error(original, check)
    fixed_error = error(fixed_yaw, check)
    new_error = error(candidate, check)
    spread = min(fit_position(yaw + delta)[0] for delta in (-2., 2.)) - best_cost
    accepted = (2. <= abs(yaw) <= 10. and spread > .0007 and fixed_yaw_cost - best_cost > .002
                and new_error < min(old_error, fixed_error) - .002
                and error(candidate, valid) < error(original, valid) - .002)
    return (pose_at(yaw, shift) if accepted else baseline,
            {"accepted": bool(accepted), "reason": "accepted" if accepted else "yaw_not_supported",
             "tested_yaw_delta_deg": yaw, "tested_shift_px_at_pivot": shift.tolist(),
             "train_before": fixed_yaw_cost, "train_after": best_cost, "yaw_cost_rise_at_two_deg": spread,
             "validation_baseline": old_error, "validation_translation_only": fixed_error,
             "validation_profile_pose": new_error, "fit_pixels": int(training.sum()), "check_pixels": int(check.sum())})
