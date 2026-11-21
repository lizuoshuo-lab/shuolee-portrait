"""Zero-update, full-resolution memory diagnostics; never initialize 4x training.

Run inside the existing WSL Python environment. Existing diagnostic outputs are
refused rather than overwritten. The only fitted model read is the completed
1x world PLY; all full-size targets and cameras come from the current 4x inputs.
"""

from pathlib import Path
import gc
import importlib.util
import json
import os
import resource
import sys
import time
import traceback

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).with_name('standard_msv2_demo.py')
MEMORY_LIMIT_GIB = 13.5
LIVE_PADDING_GIB = 0.5
FULL_WH = [10816, 6080]
GIB = 1024 ** 3


def experiment():
    spec = importlib.util.spec_from_file_location('full4_resource_experiment', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def descriptor_count():
    return len(os.listdir('/proc/self/fd'))


def memory(torch):
    """Call after synchronization; preserve both current and peak counters."""
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved(),
        allocated_gib=torch.cuda.memory_allocated() / GIB,
        reserved_gib=torch.cuda.memory_reserved() / GIB,
        max_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
        max_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
        open_file_descriptors=descriptor_count(),
    )


def verify_native(ex, plan):
    assert ex.sha(ex.ROOT / 'fast.config') == plan['fast_sha256']
    for item in plan['source_checks']:
        path = ex.ROOT / 'source' / item['path']
        assert ex.sha(path) == item['sha256'], str(path)


def clear_target_cache(ex):
    ex.LazyCamera.cached_id = None
    ex.LazyCamera.cached_tensor = None
    ex.LazyCamera.last = None


def gradient_difference(torch, reference, candidate):
    """Compare CPU gradients in bounded chunks, including exact equality."""
    assert reference.device.type == candidate.device.type == 'cpu'
    assert reference.shape == candidate.shape
    a, b = reference.reshape(-1), candidate.reshape(-1)
    largest, total, reference_maximum = 0.0, 0.0, 0.0
    exact = True
    chunk = 4 * 1024 ** 2
    for offset in range(0, a.numel(), chunk):
        x, y = a[offset:offset + chunk], b[offset:offset + chunk]
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        exact = exact and torch.equal(x, y)
        delta = (x - y).abs()
        largest = max(largest, float(delta.max()))
        total += float(delta.sum(dtype=torch.float64))
        reference_maximum = max(reference_maximum, float(x.abs().max()))
    return dict(
        torch_equal=exact,
        max_abs=largest,
        mean_abs=total / a.numel(),
        reference_max_abs=reference_maximum,
        max_abs_relative_to_reference_max=largest / max(reference_maximum, 1e-30),
    )


def layout_check(ex, torch, model, rows, pipe, background):
    from gaussian_renderer import render_imp
    from fused_ssim import fused_ssim
    from utils.loss_utils import l1_loss

    row = rows[0]
    camera = ex.LazyCamera(row, 0, 1)
    assert [camera.width, camera.height] == FULL_WH
    assert ex.sha(row['image']) == row['source_sha256']
    old_uint8 = camera.rgb
    new_uint8 = camera.rgb
    assert np.array_equal(old_uint8, new_uint8)
    with Image.open(row['image']) as source:
        assert source.mode == 'RGB' and list(source.size) == FULL_WH
        assert np.array_equal(old_uint8, np.asarray(source))

    old_target = torch.from_numpy(old_uint8).to(
        device='cuda', dtype=torch.float32).div_(255.).permute(2, 0, 1)
    new_target = camera.original_image
    assert not old_target.is_contiguous()
    assert new_target.is_contiguous(), 'The active adapter must return contiguous CHW'
    targets_equal = torch.equal(old_target, new_target)
    assert targets_equal
    strides = dict(old=list(old_target.stride()), new=list(new_target.stride()))

    # The actual image is rendered once without a Gaussian autograd graph.
    with torch.no_grad():
        prediction = render_imp(camera, model, pipe, background)['render'].detach()
    assert torch.isfinite(prediction).all()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Do not keep both SSIM autograd graphs, or both image gradients, on GPU.
    torch.cuda.reset_peak_memory_stats()
    image = prediction.clone().requires_grad_(True)
    l1 = l1_loss(image, old_target)
    ssim_value = fused_ssim(image.unsqueeze(0), old_target.unsqueeze(0))
    loss = .8 * l1 + .2 * (1. - ssim_value)
    loss.backward()
    torch.cuda.synchronize()
    old_values = dict(loss=float(loss), l1=float(l1), ssim=float(ssim_value))
    old_gradient = image.grad.detach().cpu()
    old_memory = memory(torch)
    del image, l1, ssim_value, loss, old_target
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    image = prediction.clone().requires_grad_(True)
    l1 = l1_loss(image, new_target)
    ssim_value = fused_ssim(image.unsqueeze(0), new_target.unsqueeze(0))
    loss = .8 * l1 + .2 * (1. - ssim_value)
    loss.backward()
    torch.cuda.synchronize()
    new_values = dict(loss=float(loss), l1=float(l1), ssim=float(ssim_value))
    new_gradient = image.grad.detach().cpu()
    new_memory = memory(torch)
    difference = gradient_difference(torch, old_gradient, new_gradient)
    loss_error = abs(old_values['loss'] - new_values['loss'])
    assert all(np.isfinite(v) for v in (*old_values.values(), *new_values.values()))
    # Record exact equality separately; these bounds allow float32 roundoff only.
    gradient_bound = 1e-12 + 1e-5 * difference['reference_max_abs']
    passed = loss_error <= 1e-7 and difference['max_abs'] <= gradient_bound
    result = dict(
        passed=passed, frame=row['frame'], wh=FULL_WH,
        uint8_bitwise_equal=True, source_png_equal=True,
        target_float32_bitwise_equal=targets_equal, target_strides=strides,
        old=old_values, contiguous=new_values,
        loss_torch_equal=old_values['loss'] == new_values['loss'],
        loss_abs_error=loss_error,
        loss_relative_error=loss_error / max(abs(old_values['loss']), 1e-30),
        image_gradient=difference,
        tolerances=dict(loss_abs=1e-7, gradient_abs=1e-12, gradient_relative=1e-5),
        old_memory=old_memory, contiguous_memory=new_memory,
    )
    return result


def capacity_check(ex, torch, model, rows, opt, pipe, background, state, persist):
    from gaussian_renderer import render_imp
    from fused_ssim import fused_ssim
    from utils.loss_utils import l1_loss

    model.training_setup(opt)
    moments = [
        torch.zeros_like(parameter)
        for group in model.optimizer.param_groups
        for parameter in group['params']
        for _ in range(2)
    ]
    padding = torch.zeros(int(LIVE_PADDING_GIB * GIB), dtype=torch.uint8, device='cuda')
    cameras = [ex.LazyCamera(row, uid, 1) for uid, row in enumerate(rows)]
    assert all([camera.width, camera.height] == FULL_WH for camera in cameras)
    torch.cuda.synchronize()
    state['capacity_setup'] = dict(
        points=len(model.get_xyz), cameras=len(cameras),
        manual_adam_moment_bytes=sum(t.numel() * t.element_size() for t in moments),
        live_padding_bytes=padding.numel() * padding.element_size(),
        memory=memory(torch), optimizer_updates=0,
        native_adjacent_iteration_lifetimes_preserved=True,
        trim_order='backward, synchronize, empty_cache, synchronize, zero_grad',
    )
    state['views'] = []
    persist()
    start = time.monotonic()

    # These same-named locals deliberately survive each iteration, matching
    # native train.py: the previous image/GT still exist during new assignment.
    render_pkg = image = gt_image = l1 = ssim_value = loss = None
    for index, viewpoint_cam in enumerate(cameras):
        state.update(current_view_index=index, current_frame=viewpoint_cam.image_name)
        torch.cuda.reset_peak_memory_stats()
        iteration_start = time.monotonic()
        render_pkg = render_imp(viewpoint_cam, model, pipe, background)
        image = render_pkg['render']
        viewspace_point_tensor = render_pkg['viewspace_points']
        visibility_filter = render_pkg['visibility_filter']
        radii = render_pkg['radii']
        gt_image = viewpoint_cam.original_image.cuda()
        assert gt_image.is_contiguous()
        l1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = .8 * l1 + .2 * (1. - ssim_value)
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all()
        before_trim = memory(torch)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_trim = memory(torch)
        model.optimizer.zero_grad(set_to_none=True)
        record = dict(
            index=index, frame=viewpoint_cam.image_name, wh=FULL_WH,
            loss=float(loss), l1=float(l1), ssim=float(ssim_value),
            seconds=time.monotonic() - iteration_start,
            memory_before_trim=before_trim, memory_after_trim=after_trim,
            memory_after_zero_grad=memory(torch),
            optimizer_updates=0,
        )
        state['views'].append(record)
        state.update(processed_views=index + 1, updated=ex.now())
        state['max_allocated_bytes'] = max(
            state.get('max_allocated_bytes', 0), before_trim['max_allocated_bytes'])
        state['max_reserved_bytes'] = max(
            state.get('max_reserved_bytes', 0), before_trim['max_reserved_bytes'])
        persist()
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(cameras):
            print('FULL4_ZERO_UPDATE_PROFILE', json.dumps(dict(
                views=index + 1, total=len(cameras), frame=viewpoint_cam.image_name,
                allocated_gib=after_trim['allocated_gib'],
                reserved_gib=after_trim['reserved_gib'],
                peak_allocated_gib=before_trim['max_allocated_gib'],
                peak_reserved_gib=before_trim['max_reserved_gib'],
                fd=after_trim['open_file_descriptors'],
            )), flush=True)

    # The extra buffers remain live throughout all views; none are model state.
    assert len(moments) == 2 * sum(len(g['params']) for g in model.optimizer.param_groups)
    assert padding.numel() == int(LIVE_PADDING_GIB * GIB)
    state['capacity_seconds'] = time.monotonic() - start
    state['max_allocated_gib'] = state['max_allocated_bytes'] / GIB
    state['max_reserved_gib'] = state['max_reserved_bytes'] / GIB


def main():
    ex = experiment()
    output = ex.ROOT / 'full4-memory-layout-validation.json'
    record = ex.RECORD / '四倍显存布局核验.json'
    assert not output.exists() and not record.exists(), 'Existing diagnostics must be preserved'
    native = ex.ROOT / 'original'
    execution = ex.read(native / 'execution.json')
    assert execution['state'] == 'training_completed' and execution['points'] > 0
    expected_points=execution['points']
    expected_views=ex.EXPECTED_FRAMES or (execution['train_frames']+execution['test_frames'])
    ply = native / 'model/point_cloud/iteration_18000/point_cloud.ply'
    original_sha = ex.sha(ply)
    assert original_sha == execution['ply_sha256']
    plan = ex.read(ex.ROOT / 'plan.json')
    verify_native(ex, plan)
    rows = ex.read(ex.ROOT / 'swinir4x/training-inputs.json')['records']
    assert len(rows) == expected_views
    assert all(row['wh'] == FULL_WH for row in rows)
    assert len({row['frame'] for row in rows}) == expected_views
    state = dict(
        state='running', phase='initializing', started=ex.now(),
        scope='Resource/layout diagnostics only; frozen 1x model, real 4x inputs, zero updates',
        model=str(ply), model_sha256_before=original_sha, source_points=expected_points,
        optimizer_updates=0, used_as_fourx_training_initialization=False,
        full4_training_completed=False, quality_accepted=False,
        memory_limit_gib=MEMORY_LIMIT_GIB, live_padding_gib=LIVE_PADDING_GIB,
        source_views=len(rows), native_sources_unchanged=True,
        adapter_sha256=ex.sha(SCRIPT), profile_sha256=ex.sha(__file__),
        limitation='Frozen 1x geometry plus live padding does not bound every fresh-4x rasterization working set',
    )
    def persist():
        ex.save(output, state)
        ex.save(record, state)
    persist()
    caught = None
    torch = None
    try:
        torch = ex.runtime()
        total = torch.cuda.get_device_properties(0).total_memory
        assert MEMORY_LIMIT_GIB * GIB < total
        fraction = float(MEMORY_LIMIT_GIB * GIB / total)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        state.update(device_total_memory_bytes=total, allocator_fraction=fraction,
                     nofile_limits=list(resource.getrlimit(resource.RLIMIT_NOFILE)))
        from scene.gaussian_model import GaussianModel
        model = GaussianModel(3)
        model.load_ply(str(ply))
        model.active_sh_degree = 3
        assert len(model.get_xyz) == expected_points
        args, data, opt, pipe = ex.options(4)
        assert args.imp_metric == 'outdoor' and args.resolution == 1
        background = torch.zeros(3, device='cuda')
        state['phase'] = 'layout_parity'
        persist()
        state['layout'] = layout_check(ex, torch, model, rows, pipe, background)
        print('FULL4_LAYOUT_PARITY', json.dumps(state['layout']), flush=True)
        persist()
        assert state['layout']['passed'], state['layout']
        clear_target_cache(ex)
        gc.collect()
        torch.cuda.empty_cache()
        state['phase'] = 'all_views_capacity'
        persist()
        capacity_check(ex, torch, model, rows, opt, pipe, background, state, persist)
        verify_native(ex, plan)
        assert state['processed_views'] == expected_views
        state.update(state='passed', phase='completed', passed=True, finished=ex.now())
    except BaseException as error:
        caught = error
        state.update(state='failed', passed=False, finished=ex.now(),
                     error=repr(error), traceback=traceback.format_exc())
        if torch is not None:
            try:
                state['memory_at_failure'] = memory(torch)
            except BaseException as snapshot_error:
                state['memory_snapshot_error'] = repr(snapshot_error)
    finally:
        state['model_sha256_after'] = ex.sha(ply)
        state['model_file_unchanged'] = state['model_sha256_after'] == original_sha
        if not state['model_file_unchanged']:
            state.update(state='failed', passed=False, error='Source model SHA changed')
            caught = caught or RuntimeError('Source model SHA changed')
        persist()
    if caught is not None:
        raise RuntimeError('Full4 resource diagnostic failed; see ' + str(output)) from caught
    print('FULL4_RESOURCE_DIAGNOSTIC_PASSED', str(output), flush=True)


if __name__ == '__main__':
    main()
