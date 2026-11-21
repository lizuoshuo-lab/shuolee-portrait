"""Run one eye-balanced, strict-head appearance control on the exact 006 stock model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import FunctionType

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_head_appearance_pilot_006 as pilot


OUT = pilot.LAB/'shuolee-smpl-006-eye-balanced-quality'
ACCEPTANCE = pilot.ACCEPTANCE.with_name('smpl-eye-balanced-quality-006-preregistration.json')
ACCEPTANCE_SHA = '3b13d2da185d95c393c79714d125995e90e4d648d11510ed460392d1954e4472'
PILOT_CODE_SHA = '5d4a1431f1198728420f1707824f8e7e9015039c75622b845cce1a9079e6503b'
WHOLE_FRACTION, EYE_FRACTION = .75, .25
EYES = (('eye_36_41', 86, 92), ('eye_42_47', 92, 98))
read, pin, require, sha, save = pilot.read, pilot.pin, pilot.require, pilot.sha, pilot.save


def rebound(function, **overrides):
    return FunctionType(function.__code__, dict(function.__globals__, **overrides), function.__name__)


def roi_sha(pixels):
    return hashlib.sha256(np.asarray(pixels, dtype='<i8', order='C').tobytes()).hexdigest()


def eye_region(observed, projected, alpha, box):
    if (observed.shape != (6,3) or not np.isfinite(observed).all() or not np.isfinite(projected).all()
            or (observed[:,2] < .5).any()):
        return None, dict(eligible=False, reason='unsupported_eye_landmarks')
    residual = float(np.linalg.norm(observed[:,:2]-projected, axis=1).max())
    summary = dict(observed_fit_max_native_px=residual)
    if residual > 6.:
        return None, dict(summary, eligible=False, reason='native_observed_fit_over_6px')
    x0,y0,x1,y1 = box
    xy = observed[:,:2] - [x0,y0]
    width,height = x1-x0,y1-y0
    if not ((xy >= 0) & (xy < [width,height])).all():
        return None, dict(summary, eligible=False, reason='whole_eye_outside_target_crop')
    raster = Image.new('1', (width,height))
    ImageDraw.Draw(raster).polygon([tuple(point) for point in xy], fill=1)
    rows,columns = np.nonzero(np.asarray(raster))
    summary.update(pixels=len(rows))
    if len(rows) < 32:
        return None, dict(summary, eligible=False, reason='whole_eye_under_32_pixels')
    values = alpha[0,y0+rows,x0+columns]
    if not (values > .99).all():
        return None, dict(summary, eligible=False, reason='whole_eye_not_confident_foreground')
    pixels = np.stack((rows,columns)).astype('<i8', copy=False)
    return pixels, dict(summary, eligible=True, crop_relative_rc_sha256=roi_sha(pixels))


def balanced_loss(stock_loss, eyes, rgb, alpha, target_rgb, target_alpha, box, *, alpha_floor, face_pixels, face_weight):
    if not eyes:
        return stock_loss(rgb,alpha,target_rgb,target_alpha,box,alpha_floor=alpha_floor,
                          face_pixels=face_pixels,face_weight=face_weight)
    import torch

    photo,l1,ssim,coverage = stock_loss(rgb,alpha,target_rgb,target_alpha,box,alpha_floor=alpha_floor,
                                        face_pixels=face_pixels,face_weight=face_weight*WHOLE_FRACTION)
    x0,y0,_,_ = box
    eye_losses = []
    for rows,columns in eyes:
        black_error = rgb[:,y0+rows,x0+columns]-target_rgb[:,rows,columns]
        white_error = black_error+(1-alpha[:,y0+rows,x0+columns])
        eye_losses.append(.5*(black_error.abs()+white_error.abs()).mean())
    return photo+face_weight*EYE_FRACTION*torch.stack(eye_losses).mean(),l1,ssim,coverage


def differential_control(stock_loss):
    import torch

    box = (8,8,56,56)
    mask = torch.zeros((1,64,64))
    mask[:,8:56,8:56] = 1.
    rgb = (torch.full((3,64,64),.2)*mask).requires_grad_()
    alpha = (.7*mask).requires_grad_()
    target = torch.full((3,48,48),.1)
    whole = torch.tensor([[3,7,13,18],[4,8,14,19]])
    eye0 = torch.tensor([[3],[4]])
    eye1 = torch.tensor([[7,13],[8,14]])
    args = (rgb,alpha,target,mask,box)
    opts = dict(alpha_floor=1e-4,face_pixels=whole,face_weight=.05)
    reference = stock_loss(*args,**opts)
    fallback = balanced_loss(stock_loss,(),*args,**opts)
    require(all(torch.equal(a,b) for a,b in zip(reference,fallback,strict=True)), 'No-eye loss is not exact stock.')
    grads = lambda out: torch.autograd.grad(out[0]+.1*out[3],(rgb,alpha),retain_graph=True)
    require(all(torch.equal(a,b) for a,b in zip(grads(reference),grads(fallback),strict=True)), 'No-eye gradients differ.')
    base = stock_loss(*args,**dict(opts,face_weight=0.))
    changed = balanced_loss(stock_loss,(eye0,eye1),*args,**opts)
    independent = [stock_loss(*args,**dict(opts,face_pixels=p))[0]-base[0] for p in (whole,eye0,eye1)]
    expected = base[0]+WHOLE_FRACTION*independent[0]+EYE_FRACTION*(independent[1]+independent[2])/2
    torch.testing.assert_close(changed[0],expected,rtol=1e-6,atol=1e-7)
    require(all(torch.equal(a,b) for a,b in zip(reference[1:],changed[1:],strict=True)), 'Base loss or alpha weighting changed.')
    torch.testing.assert_close(grads(changed),grads((expected,*base[1:])),rtol=1e-6,atol=1e-7)
    require(torch.as_tensor(whole,device=whole.device).data_ptr() == whole.data_ptr(), 'Torch target alias changed.')
    return dict(no_eye_forward_backward_bitwise_stock=True, two_eye_unequal_pixel_count_equal_group_mean=True,
                weighted_photometric_forward_backward_verified=True, alpha_and_base_components_bitwise_unchanged=True)


def preflight():
    pin(Path(pilot.__file__),PILOT_CODE_SHA)
    pin(ACCEPTANCE,ACCEPTANCE_SHA)
    policy = read(ACCEPTANCE)
    optimizer, settings = policy['optimizer'], policy['single_change']
    eligibility = policy['eye_training_eligibility']
    require(policy['state'] == 'registered_before_006_eye_balance_trial' and policy['output'] == str(OUT)
            and policy['start'] == {'path':str(pilot.STOCK/'result.json'),'sha256':pilot.STOCK_RESULT}
            and policy['initial_stock_commit'] == pilot.face_trial.BASE_COMMIT
            and policy['prior_pilot_preregistration_sha256'] == pilot.ACCEPTANCE_SHA
            and optimizer == dict(iterations=600,fresh=True,seed=0,feature_dc_lr=.0025,opacity_logit_lr=.05,
                 same_prior_slots=True,geometry_and_nonhead_bytewise_frozen=True,no_refit_or_density_edits=True,no_parameter_sweep=True)
            and settings['total_paired_face_weight'] == .05 and settings['alpha_weight_unchanged'] == .1
            and settings['whole_face_fraction_when_eye_eligible'] == WHOLE_FRACTION
            and settings['eligible_eyes_mean_fraction'] == EYE_FRACTION
            and eligibility == dict(adapted_slots=[[a,b] for _,a,b in EYES],contiguous_intervals_end_exclusive=True,
                all_six_confidence_min=.5,max_native_observed_fit_euclidean_px=6.,polygon_min_pixels=32,
                all_polygon_foreground_alpha_strictly_over=.99,use_training_observations_only=True,physical_visibility_claim=False),
            'The only registered loss change or eye eligibility changed.')
    controls = policy['accepted_control']
    input_pins = {str(Path(pilot.__file__)):PILOT_CODE_SHA,str(ACCEPTANCE):ACCEPTANCE_SHA}
    for proof in controls.values():
        input_pins[proof['path']] = pin(Path(proof['path']),proof['sha256'])
    stock, state = pilot.preflight()
    require(stock == read(Path(controls['preregistration']['path'])) and stock['editable_points'] == 28407
            and stock['appearance_steps'] == 600 and stock['face_weight'] == .05 and stock['alpha_weight'] == .1,
            'The exact stock input, sampler, or original uniform face objective changed.')
    eyes_by_frame, reports = [], {}
    fit = state.saved.fit
    for row, identity, (_,alpha,box,pixels) in zip(state.rows,state.saved.facts['frames'],state.targets,strict=True):
        frame,slot = row['frame'],identity['pose_index']
        require(frame == identity['frame'] == fit['frame_ids'][slot]
                and frame not in stock['protected_holdouts'] and frame not in policy['protected_holdouts'],
                'A protected observation entered eye training.')
        observed, projected = (fit[key][slot] for key in ('observed_native_keypoints','joints_projected_native'))
        xyz = fit['joints_camera'][slot]
        ray = xyz@np.asarray(row['K']).T
        projected_native = ray[:,:2]/ray[:,2:3]-.5
        require(np.allclose(projected,projected_native,rtol=0,atol=.001),'Stored eye projection is outside the native K convention.')
        regions, metrics = [], {}
        for name,a,b in EYES:
            region,metrics[name] = eye_region(observed[a:b],projected[a:b],alpha,box)
            if region is not None:
                regions.append(region)
        reports[frame] = dict(pose_index=slot,crop_xyxy=list(box),source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256'],
            original_face_pixels=int(pixels.shape[1]),original_face_crop_relative_rc_sha256=roi_sha(pixels),eyes=metrics)
        eyes_by_frame.append(tuple(regions))
    require(any(eyes_by_frame) and policy['reference']['frame'] == state.rows[0]['frame'] == 'frame_000033'
            and len(eyes_by_frame[0]) == 2,'Reference or selected eye support is unavailable.')
    parity = differential_control(state.loss)
    plan = dict(stock,acceptance_plan_sha256=ACCEPTANCE_SHA,prior_uniform_face_acceptance_sha256=pilot.ACCEPTANCE_SHA,
        wrapper_code_sha256=sha(Path(__file__)),eye_training_rois=reports,eye_training_roi_coordinate_order='crop row then column, int64 LE',
        eye_balance=settings,eye_training_eligibility=eligibility,cpu_differential_control=parity,
        eligible_eye_count=sum(map(len,eyes_by_frame)),eligible_eye_frames=sum(map(bool,eyes_by_frame)),
        input_sha256=stock['input_sha256']|input_pins,registered_ready=True)
    return plan,state,eyes_by_frame


def launch(plan,state,eyes_by_frame):
    stock_polish = rebound(pilot.polish,OUT=OUT)
    stock_delivery = rebound(pilot.delivery,OUT=OUT)

    def polish(inputs, exact_plan):
        import torch

        original_loss = inputs.loss
        cuda_regions = {}
        for index,((rgb,mask,box,pixels),regions) in enumerate(zip(inputs.targets,eyes_by_frame,strict=True)):
            face_pixels = torch.as_tensor(pixels,device='cuda')
            cuda_regions[face_pixels.data_ptr()] = tuple(torch.as_tensor(region,device='cuda') for region in regions)
            inputs.targets[index] = rgb,mask,box,face_pixels
        require(len(cuda_regions) == len(inputs.targets),'The stock face ROI identifiers overlap.')

        def loss(rgb,alpha,target_rgb,target_alpha,box,*,alpha_floor,face_pixels,face_weight):
            key = face_pixels.data_ptr()
            require(key in cuda_regions,'The unchanged optimizer presented an unknown registered face ROI.')
            return balanced_loss(original_loss,cuda_regions[key],rgb,alpha,target_rgb,target_alpha,box,
                                 alpha_floor=alpha_floor,face_pixels=face_pixels,face_weight=face_weight)

        inputs.loss = loss
        return stock_polish(inputs,exact_plan)

    def delivery(inputs,model,exact_plan):
        return dict(stock_delivery(inputs,model,exact_plan),eye_balanced_quality=dict(
            wrapper_code_sha256=exact_plan['wrapper_code_sha256'],prior_uniform_face_acceptance_sha256=pilot.ACCEPTANCE_SHA,
            eligible_eye_count=exact_plan['eligible_eye_count'],eligible_eye_frames=exact_plan['eligible_eye_frames'],
            loss_blend=exact_plan['eye_balance'],registered_eye_rois=True,cpu_differential_control=exact_plan['cpu_differential_control']))

    run = rebound(pilot.run,OUT=OUT,ACCEPTANCE_SHA=ACCEPTANCE_SHA,polish=polish,delivery=delivery)
    run(plan,state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','run'))
    args = parser.parse_args()
    if args.action != 'run':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan,state,regions = preflight()
    if args.action == 'run':
        launch(plan,state,regions)
        return
    path = OUT/'preregistration.json'
    if args.action == 'stage':
        OUT.mkdir(exist_ok=False)
        save(path,plan)
    print(json.dumps(dict(state='eye_quality_inputs_verified',staged=args.action=='stage',gpu_calls=0,
        preregistration=str(path),preregistration_sha256=sha(path) if path.is_file() else None,
        eye_frames=plan['eligible_eye_frames'],eligible_eyes=plan['eligible_eye_count'],
        reference_eyes=plan['eye_training_rois']['frame_000033']['eyes'],cpu_controls=plan['cpu_differential_control'])),flush=True)


if __name__ == '__main__':
    main()
