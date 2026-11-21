"""Controlled gsplat feature/depth/alpha and six-DoF gradient experiments.

All data are synthetic. Features are unbounded camera-z, one, and camera-z^2;
they are not RGB intensities and must not be clipped/sigmoided. This checks the
pinned gsplat backend, not numerical equivalence with SplaTAM's original backend.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from track_fixed_map import se3_exp, quaternion_covariance

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 5e-6


def front_to_back_moments(depths, effective_alphas):
    transmittance = 1.0
    weights = []
    for alpha in effective_alphas:
        weights.append(transmittance * alpha)
        transmittance *= 1-alpha
    weights = np.asarray(weights, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    coverage = float(weights.sum())
    accumulated = float(weights @ depths)
    second = float(weights @ (depths**2))
    expected = accumulated/coverage
    variance = second/coverage-expected**2
    return {"weights": weights.tolist(), "coverage": coverage,
            "accumulated_depth": accumulated, "expected_depth": expected,
            "accumulated_second_moment": second, "conditional_depth_variance": variance}


def cpu_check():
    one = front_to_back_moments([2.], [.5])
    two = front_to_back_moments([2., 4.], [.5, .7])
    if abs(one["coverage"]-.5) > 1e-12 or abs(one["accumulated_depth"]-1.) > 1e-12:
        raise AssertionError("Hand-computed single-Gaussian control failed")
    if abs(one["expected_depth"]-2.) > 1e-12 or abs(one["conditional_depth_variance"]) > 1e-12:
        raise AssertionError("Single-depth conditional moment control failed")
    if abs(two["coverage"]-.85) > 1e-12 or abs(two["accumulated_depth"]-2.4) > 1e-12:
        raise AssertionError("Hand-computed overlapping compositing control failed")
    if abs(two["accumulated_second_moment"]-7.6) > 1e-12 or two["conditional_depth_variance"] <= .9:
        raise AssertionError("Two-depth variance control failed")
    return {"status": "passed", "one_center_hand_computation": one,
            "two_center_hand_computation": two,
            "features_are_unbounded_not_rgb": True}


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_gradient_and_difference(loss_from_twist, epsilon=1e-4):
    twist = torch.zeros(6, device="cuda", requires_grad=True)
    loss = loss_from_twist(twist)
    gradient = torch.autograd.grad(loss, twist)[0]
    differences = []
    with torch.no_grad():
        for axis in range(6):
            perturbation = torch.zeros(6, device="cuda")
            perturbation[axis] = epsilon
            plus = loss_from_twist(perturbation)
            minus = loss_from_twist(-perturbation)
            differences.append(float((plus-minus)/(2*epsilon)))
    numeric = torch.tensor(differences, device="cuda")
    finite = bool(torch.isfinite(loss) and torch.isfinite(gradient).all() and torch.isfinite(numeric).all())
    cosine = float(torch.nn.functional.cosine_similarity(gradient[None], numeric[None]))
    relative = float((gradient-numeric).norm()/numeric.norm().clamp_min(1e-12))
    result = {"scalar_loss": float(loss.detach()), "analytic_six_dof": gradient.detach().cpu().tolist(),
              "finite_difference_six_dof": numeric.cpu().tolist(), "finite_difference_epsilon": epsilon,
              "all_finite": finite, "nonzero_analytic_components": int((gradient.abs()>1e-9).sum()),
              "gradient_direction_cosine": cosine, "finite_difference_relative_l2_error": relative}
    if not finite or result["nonzero_analytic_components"] != 6 or cosine < .98 or relative > .2:
        raise AssertionError(f"Actual scalar-loss camera gradient control failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="renderer-feature-semantics-001")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    checks = cpu_check()
    if args.check_only:
        print(json.dumps(checks, indent=2))
        return
    if not args.run or not args.run.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run name must contain only letters, numbers, dashes and underscores")
    output = ROOT/"outputs"/args.run
    output.mkdir(parents=True, exist_ok=False)
    source = {"gsplat_version": "1.5.3", "gsplat_commit": "937e29912570c372bed6747a5c9bf85fed877bae",
              "script_sha256": hash_file(Path(__file__)),
              "rendering_source_sha256": hash_file(ROOT/"repos/gsplat/gsplat/rendering.py"),
              "projection_wrapper_sha256": hash_file(ROOT/"repos/gsplat/gsplat/cuda/_wrapper.py"),
              "pose_helper_sha256": hash_file(ROOT/"scripts/track_fixed_map.py"),
              "official_api": "https://docs.gsplat.studio/main/apis/rasterization.html",
              "backend_scope": "gsplat CUDA rasterization only; original SplaTAM backend not run"}
    config = {"width": 32, "height": 32, "focal": 24., "cx_cy": [15.5, 15.5],
              "sample_pixel_y_x": [15, 15], "features": ["camera_z", "one", "camera_z_squared"],
              "background": "zero", "colors_are_raw_features": True,
              "packed": False, "rasterize_mode": "classic", "tolerance": TOLERANCE,
              "one_depths": [2.], "one_opacities": [.5],
              "two_depths": [2., 4.], "two_opacities": [.5, .7],
              "finite_difference_epsilon": 1e-4}
    (output/"source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__), output/"source_script.py")
    (output/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    from gsplat import rasterization
    device = "cuda"
    width = height = 32
    K = torch.tensor([[24.,0.,15.5],[0.,24.,15.5],[0.,0.,1.]], device=device)[None]
    identity = torch.eye(4, device=device)[None]
    scene_results = {}
    all_arrays = {}
    started = time.perf_counter()
    for name, depths, opacities in (("one", [2.], [.5]), ("two", [2.,4.], [.5,.7])):
        means = torch.tensor([[0.,0.,z] for z in depths], device=device)
        # Equal projected footprints at the principal point; center alpha is opacity.
        scales = torch.tensor([[.09*z,.09*z,.09*z] for z in depths], device=device)
        quats = torch.tensor([[1.,0.,0.,0.]], device=device).repeat(len(depths),1)
        opacity = torch.tensor(opacities, device=device)
        z = means[:,2]
        features = torch.stack((z,torch.ones_like(z),z*z),-1)
        renders = {}
        for mode in ("RGB", "RGB+D", "RGB+ED"):
            render, alpha, _ = rasterization(means,quats,scales,opacity,features,identity,K,width,height,
                                             packed=False,render_mode=mode)
            renders[mode] = (render[0],alpha[0,...,0])
            key = name+"_"+mode.replace("+","_")
            all_arrays[key] = render[0].detach().cpu().numpy()
            all_arrays[key+"_alpha"] = alpha[0,...,0].detach().cpu().numpy()
        feature, alpha = renders["RGB"]
        builtin_D = renders["RGB+D"][0][...,3]
        builtin_ED = renders["RGB+ED"][0][...,3]
        D, S, M2 = feature.unbind(-1)
        expected_depth = D/alpha.clamp_min(1e-10)
        check_errors = {"coverage_feature_vs_alpha_max_error": float((S-alpha).abs().max()),
                        "depth_feature_vs_builtin_D_max_error": float((D-builtin_D).abs().max()),
                        "builtin_ED_vs_D_over_alpha_max_error": float((builtin_ED-expected_depth).abs().max()),
                        "RGB_vs_RGBD_feature_channels_max_error": float((feature-renders["RGB+D"][0][...,:3]).abs().max()),
                        "RGB_vs_RGBED_feature_channels_max_error": float((feature-renders["RGB+ED"][0][...,:3]).abs().max())}
        center = feature[15,15].detach().cpu().tolist()
        center_alpha = float(alpha[15,15])
        hand = front_to_back_moments(depths,opacities)
        hand_errors = {"center_coverage_abs_error": abs(center[1]-hand["coverage"]),
                       "center_accumulated_depth_abs_error": abs(center[0]-hand["accumulated_depth"]),
                       "center_second_moment_abs_error": abs(center[2]-hand["accumulated_second_moment"])}
        conditional_variance = center[2]/center_alpha-(center[0]/center_alpha)**2
        result = {"center_raw_features_D_S_M2": center,"center_alpha": center_alpha,
                  "center_builtin_D": float(builtin_D[15,15]), "center_builtin_ED": float(builtin_ED[15,15]),
                  "center_second_moment_minus_accumulated_depth_squared": center[2]-center[0]**2,
                  "center_conditional_depth_variance": conditional_variance,
                  "center_coverage_strict_greater_than_0_5": bool(S[15,15]>.5),
                  "center_coverage_greater_or_equal_0_5": bool(S[15,15]>=.5),
                  "raw_feature_image_max": float(feature.max()), "formula_comparison_errors": check_errors,
                  "hand_computation": hand, "hand_computation_errors": hand_errors}
        if max(check_errors.values()) > TOLERANCE or max(hand_errors.values()) > TOLERANCE:
            raise AssertionError(f"Feature values contradict independent compositing control: {name}: {result}")
        if result["raw_feature_image_max"] <= 1 or abs(result["center_second_moment_minus_accumulated_depth_squared"])<.1:
            raise AssertionError("Unbounded feature / distinct second moment control did not exercise the contract")
        if name=="two" and conditional_variance<.9:
            raise AssertionError("Two overlapping depths did not produce expected positive conditional variance")
        scene_results[name]=result
    (output/"feature_semantics.json").write_text(json.dumps(scene_results,indent=2),encoding="utf-8")
    np.savez_compressed(output/"raw_outputs.npz",**all_arrays)
    (output/"actual_outputs.json").write_text(json.dumps({key:value.tolist() for key,value in all_arrays.items()},indent=2),encoding="utf-8")

    # Deliberately asymmetric anisotropic map avoids an accidental zero roll gradient.
    world_means=torch.tensor([[-.13,.07,2.],[.09,-.04,2.8],[.04,.11,3.5]],device=device)
    world_scales=torch.tensor([[.12,.20,.09],[.23,.13,.10],[.18,.25,.14]],device=device)
    world_quats=torch.nn.functional.normalize(torch.tensor([[1.,.1,.05,.08],[1.,-.06,.12,.1],[1.,.1,-.08,.04]],device=device),dim=-1)
    world_covariances=quaternion_covariance(world_quats,world_scales)
    world_opacity=torch.tensor([.55,.45,.7],device=device)
    def render_camera(twist,route,detach_features=False):
        pose=se3_exp(twist)
        camera_means=world_means@pose[:3,:3].T+pose[:3,3]
        z=camera_means[:,2]
        features=torch.stack((z,torch.ones_like(z),z*z),-1)
        if detach_features:
            features=features.detach()
        if route=="direct_viewmat":
            return rasterization(world_means,world_quats,world_scales,world_opacity,features,
                                 pose[None].contiguous(),K,width,height,packed=False)[0][0]
        camera_covariances=pose[:3,:3]@world_covariances@pose[:3,:3].T
        return rasterization(camera_means,None,None,world_opacity,features,identity,K,width,height,
                             covars=camera_covariances,packed=False)[0][0]
    target_twist=torch.tensor([.03,-.02,.04,math.radians(1),math.radians(-2),math.radians(1.5)],device=device)
    with torch.no_grad():
        target=render_camera(target_twist,"direct_viewmat")
        zero=torch.zeros(6,device=device)
        direct=render_camera(zero,"direct_viewmat")
        transformed=render_camera(zero,"camera_means_covariances")
    channel_scales=torch.tensor([4.,1.,16.],device=device)
    def loss(twist,route,detach_features=False):
        prediction=render_camera(twist,route,detach_features)
        return (((prediction-target)/channel_scales)**2).mean()
    gradient_results={route:scalar_gradient_and_difference(lambda x:loss(x,route))
                      for route in ("direct_viewmat","camera_means_covariances")}
    analytic_direct=np.array(gradient_results["direct_viewmat"]["analytic_six_dof"])
    analytic_transformed=np.array(gradient_results["camera_means_covariances"]["analytic_six_dof"])
    comparison={"render_max_absolute_difference":float((direct-transformed).abs().max()),
                "scalar_loss_absolute_difference":abs(gradient_results["direct_viewmat"]["scalar_loss"]-gradient_results["camera_means_covariances"]["scalar_loss"]),
                "analytic_gradient_relative_l2_difference":float(np.linalg.norm(analytic_direct-analytic_transformed)/max(np.linalg.norm(analytic_direct),1e-12))}
    if comparison["render_max_absolute_difference"]>2e-5 or comparison["scalar_loss_absolute_difference"]>1e-6 or comparison["analytic_gradient_relative_l2_difference"]>1e-3:
        raise AssertionError(f"Equivalent camera routes disagree: {comparison}")
    # A negative control: detaching depth features must change the camera gradient.
    detached_twist=torch.zeros(6,device=device,requires_grad=True)
    detached_gradient=torch.autograd.grad(loss(detached_twist,"direct_viewmat",True),detached_twist)[0].detach().cpu().numpy()
    detach_difference=float(np.linalg.norm(detached_gradient-analytic_direct)/max(np.linalg.norm(analytic_direct),1e-12))
    comparison["detaching_z_feature_gradient_relative_change"]=detach_difference
    comparison["detached_feature_analytic_six_dof"]=detached_gradient.tolist()
    # Isolate the feature-z graph: with one Gaussian, D/S equals its camera z.
    # At the principal pixel, depth loss has dL/d(t_z)=2*(2-2.2)=-0.4.
    # Detaching feature z kills this derivative; geometric alpha terms cancel.
    def isolated_depth_loss(twist,detach_features=False):
        pose=se3_exp(twist)
        one_world=torch.tensor([[0.,0.,2.]],device=device)
        camera=one_world@pose[:3,:3].T+pose[:3,3]
        z=camera[:,2]
        one_features=torch.stack((z,torch.ones_like(z),z*z),-1)
        if detach_features:
            one_features=one_features.detach()
        rendered=rasterization(one_world,torch.tensor([[1.,0.,0.,0.]],device=device),
                               torch.full((1,3),.18,device=device),torch.tensor([.5],device=device),
                               one_features,pose[None].contiguous(),K,width,height,packed=False)[0][0]
        expected_depth=rendered[15,15,0]/rendered[15,15,1].clamp_min(1e-10)
        return (expected_depth-2.2)**2
    connected=torch.zeros(6,device=device,requires_grad=True)
    connected_loss=isolated_depth_loss(connected)
    connected_gradient=torch.autograd.grad(connected_loss,connected)[0]
    detached=torch.zeros(6,device=device,requires_grad=True)
    detached_loss=isolated_depth_loss(detached,True)
    isolated_detached_gradient=torch.autograd.grad(detached_loss,detached)[0]
    with torch.no_grad():
        plus=torch.zeros(6,device=device); plus[2]=1e-4
        minus=-plus
        isolated_numeric_z=float((isolated_depth_loss(plus)-isolated_depth_loss(minus))/(2e-4))
    negative_control={"definition":"one Gaussian ED=D/S at principal pixel; L=(ED-2.2)^2",
                      "expected_translation_z_derivative":-.4,
                      "analytic_six_dof_with_z_graph":connected_gradient.detach().cpu().tolist(),
                      "analytic_six_dof_with_z_detached":isolated_detached_gradient.detach().cpu().tolist(),
                      "finite_difference_translation_z_derivative":isolated_numeric_z,
                      "connected_scalar_loss":float(connected_loss.detach()),
                      "detached_scalar_loss":float(detached_loss.detach())}
    if abs(float(connected_gradient[2])+.4)>2e-5 or abs(isolated_numeric_z+.4)>.002 or abs(float(isolated_detached_gradient[2]))>2e-5:
        raise AssertionError(f"Feature-z graph negative control failed: {negative_control}")
    comparison["isolated_feature_z_graph_negative_control"]=negative_control
    all_arrays["gradient_target_features"]=target.cpu().numpy()
    all_arrays["gradient_direct_features"]=direct.cpu().numpy()
    all_arrays["gradient_camera_covariance_features"]=transformed.cpu().numpy()
    np.savez_compressed(output/"raw_outputs.npz",**all_arrays)
    # Portable actual values allow viewing formulas without decoding a model/checkpoint.
    (output/"actual_outputs.json").write_text(json.dumps({key:value.tolist() for key,value in all_arrays.items()},indent=2),encoding="utf-8")
    torch.cuda.synchronize()
    result={"status":"passed","source":source,"config":config,"cpu_controls":checks,
            "synthetic_scene_results":scene_results,"camera_gradient_controls":gradient_results,
            "equivalent_camera_route_comparison":comparison,
            "target_pose_twist_for_synthetic_gradient_control":target_twist.cpu().tolist(),
            "gradient_control_geometry":{"means":world_means.cpu().tolist(),"scales":world_scales.cpu().tolist(),
                                         "quats_wxyz":world_quats.cpu().tolist(),"opacities":world_opacity.cpu().tolist()},
            "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
            "elapsed_seconds_includes_first_render_and_checks":time.perf_counter()-started,
            "limitations":["controlled tiny synthetic scenes, not dataset metrics",
                           "z is camera-axis depth, not Euclidean ray range",
                           "second moment describes alpha-weighted Gaussian-center z, not intra-Gaussian depth variance",
                           "zero background required for the feature=alpha identities tested here",
                           "ED uses alpha denominator clamped at 1e-10; compare handling of empty/near-empty pixels explicitly",
                           "classic packed=False renderer only; sorting/visibility can affect finite differences",
                           "original SplaTAM backend not executed: no cross-backend numerical equivalence claim"]}
    (output/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
