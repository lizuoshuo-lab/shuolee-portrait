"""Isolated, upstream-derived validity-mask interface. No real-data training.

The original utility is copied and patched only in a new experiment directory.
Existing projects, masks, snapshots and their dependency hashes are preserved.
"""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import os
import time
import traceback
from datetime import datetime, timezone

H = Path(__file__).resolve().parent
WIN_BASE = 'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit'
WIN_UPSTREAM = 'D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core/utils/loss_utils.py'


def path(value):
    if os.name == 'nt':
        return Path('\\\\?\\' + str(Path(value).resolve()))
    return Path('/mnt/' + value[0].lower() + value[2:])


OUT = path(WIN_BASE + '/loss-validity-contract')
UPSTREAM = path(WIN_UPSTREAM)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


HELPERS = '''
# Local optional validity adapter; original all-pixel path below is unchanged.
# A true bit means a KNOWN supervision pixel (foreground OR background).
# This must never be supplied with a binary person/foreground mask by mistake.
def _check_validity(img1, img2, valid_mask):
    if img1.shape != img2.shape or img1.ndim not in (3, 4):
        raise ValueError("Expected matching CHW or NCHW RGB tensors")
    if img1.dtype != img2.dtype or img1.device != img2.device:
        raise ValueError("Image dtype/device mismatch")
    if valid_mask.dtype != torch.bool or valid_mask.device != img1.device:
        raise ValueError("Validity must be a boolean tensor on the image device")
    expected = (*img1.shape[:-3], 1, *img1.shape[-2:])
    if tuple(valid_mask.shape) != expected:
        raise ValueError("Validity must have one channel and the same batch/spatial shape")
    if not bool(valid_mask.any()):
        raise ValueError("No known RGB supervision pixels")
    return valid_mask


def _ssim_support(valid_mask, window_size):
    # Outside-image padding stays known zero, exactly as in the original SSIM.
    unknown = (~valid_mask).to(dtype=torch.float32)
    return F.max_pool2d(unknown, window_size, stride=1, padding=window_size // 2) == 0


def _known_only(image, valid_mask):
    # Prevent NaN/Inf in unknown targets from entering neighborhood arithmetic.
    return torch.where(valid_mask, image, torch.zeros_like(image))

'''


def patched_source(source):
    s = source

    def replace(before, after):
        nonlocal s
        assert s.count(before) == 1, before
        s = s.replace(before, after, 1)

    replace('def l1_loss(network_output, gt):\n    return torch.abs((network_output - gt)).mean()',
        HELPERS + '''def l1_loss(network_output, gt, valid_mask=None):
    if valid_mask is not None:
        valid_mask = _check_validity(network_output, gt, valid_mask)
        difference = _known_only(network_output, valid_mask) - _known_only(gt, valid_mask)
        return torch.masked_select(torch.abs(difference), valid_mask).mean()
    return torch.abs((network_output - gt)).mean()''')
    replace('def ssim(img1, img2, window_size=11, size_average=True):\n    channel',
        '''def ssim(img1, img2, window_size=11, size_average=True, valid_mask=None):
    if valid_mask is not None:
        if not size_average:
            raise ValueError("Masked SSIM currently supports the training scalar reduction only")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("SSIM support requires a positive odd window size")
        valid_mask = _check_validity(img1, img2, valid_mask)
        img1 = _known_only(img1, valid_mask)
        img2 = _known_only(img2, valid_mask)
    channel''')
    replace('return _ssim(img1, img2, window, window_size, channel, size_average)',
        'return _ssim(img1, img2, window, window_size, channel, size_average, valid_mask)')
    replace('def _ssim(img1, img2, window, window_size, channel, size_average=True):',
        'def _ssim(img1, img2, window, window_size, channel, size_average=True, valid_mask=None):')
    replace('    if size_average:\n        return ssim_map.mean()',
        '''    if valid_mask is not None:
        support = _ssim_support(valid_mask, window_size)
        if not bool(support.any()):
            raise ValueError("No complete known SSIM neighborhoods")
        return torch.masked_select(ssim_map, support).mean()
    if size_average:
        return ssim_map.mean()''')
    ast.parse(s)
    return s


def prepare():
    assert not OUT.exists(), 'Do not overwrite the completed/registered contract.'
    original = UPSTREAM.read_text(encoding='utf-8')
    candidate = patched_source(original)
    OUT.mkdir()
    (OUT / 'loss_utils_original.py').write_text(original, encoding='utf-8')
    (OUT / 'loss_utils_validity.py').write_text(candidate, encoding='utf-8')
    (OUT / 'validity-only.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True), fromfile='upstream/loss_utils.py',
        tofile='isolated/loss_utils_validity.py')), encoding='utf-8')
    plan = dict(updated=now(), state='registered_before_CPU_execution',
        hypothesis='Unknown occlusion must neither become black background nor train device color as person appearance. L1 and11x11 SSIM need validity support, while the original fully supervised path remains equivalent.',
        scope='An isolated copy of upstream L1/SSIM gains an optional boolean validity argument. No original source, training runner, masks or Gaussian parameters are changed.',
        validity_semantics='true=known visible-person OR known background; false=unknown occlusion. A person segmentation mask is not a validity mask.',
        budget=dict(device='CPU', seconds=120, optimization_updates=0, real_training=False, seeds=1),
        controls=['Original/default and all-valid loss+gradient parity in float32/float64, CHW/NCHW',
            'Unknown finite/NaN/Inf value invariance and zero gradients for prediction AND target',
            'Independent SSIM support enumeration at center and image boundaries',
            'Known pixels retain nonzero gradients and L1 normalization uses valid channel count',
            'Empty valid set and no full SSIM windows reject explicitly; malformed masks reject',
            'Original unmasked loss is a negative control for unknown-pixel leakage'],
        stop='120seconds, nonfinite result or failed control stops; preserve failure, no parameter sweep. A confirmed implementation bug may be fixed with its failure retained.',
        source=dict(path=WIN_UPSTREAM, sha256=sha(UPSTREAM)),
        hashes={p.name: sha(p) for p in [OUT / 'loss_utils_original.py', OUT / 'loss_utils_validity.py', OUT / 'validity-only.diff']},
        test_script_sha256=sha(Path(__file__)), quality_accepted=False,
        real_occluder_masks_available=False, model_promoted=False)
    save('plan.json', plan)
    print(json.dumps(dict(state=plan['state'], path=str(OUT)), ensure_ascii=False))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(OUT / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run():
    assert not (OUT / 'result.json').exists(), 'Already completed; do not rerun.'
    import signal
    import torch
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU120s budget')))
    signal.alarm(120)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    assert plan['source']['sha256'] == sha(UPSTREAM)
    assert plan['test_script_sha256'] == sha(Path(__file__))
    for name, digest in plan['hashes'].items():
        assert sha(OUT / name) == digest
    old = module('original_photometric', 'loss_utils_original.py')
    new = module('validity_photometric', 'loss_utils_validity.py')
    checks = []

    def evaluate(fn, a, b, **kw):
        x, y = a.clone().requires_grad_(), b.clone().requires_grad_()
        value = fn(x, y, **kw)
        assert torch.isfinite(value)
        gx, gy = torch.autograd.grad(value, (x, y))
        assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
        return value.detach(), gx, gy

    def compare(ref, got, dtype, label):
        atol = 2e-6 if dtype == torch.float32 else 2e-12
        errors = [float((x-y).abs().max()) for x, y in zip(ref, got)]
        for x, y in zip(ref, got):
            torch.testing.assert_close(x, y, atol=atol, rtol=0)
        checks.append(dict(name=label, passed=True, max_value_grad_errors=errors, atol=atol))

    for dtype in (torch.float32, torch.float64):
        for shape in ((3, 31, 37), (2, 3, 31, 37)):
            x, y = torch.rand(shape, dtype=dtype), torch.rand(shape, dtype=dtype)
            valid = torch.ones((*shape[:-3], 1, *shape[-2:]), dtype=torch.bool)
            for name in ('l1_loss', 'ssim'):
                ref = evaluate(getattr(old, name), x, y)
                compare(ref, evaluate(getattr(new, name), x, y), dtype, f'{name}_default_{dtype}_{len(shape)}D')
                compare(ref, evaluate(getattr(new, name), x, y, valid_mask=valid), dtype, f'{name}_allvalid_{dtype}_{len(shape)}D')

    x, y = torch.rand(3, 31, 37, dtype=torch.float64), torch.rand(3, 31, 37, dtype=torch.float64)
    valid = torch.ones(1, 31, 37, dtype=torch.bool)
    valid[:, 10:14, 13:17] = False
    unknown = (~valid).expand_as(x)
    for name in ('l1_loss', 'ssim'):
        ref = evaluate(getattr(new, name), x, y, valid_mask=valid)
        assert ref[1][unknown].count_nonzero() == ref[2][unknown].count_nonzero() == 0
        assert ref[1][~unknown].abs().sum() > 0
        checks.append(dict(name=name+'_unknown_grad_zero_known_grad_nonzero', passed=True))
        for mode in ('finite', 'nonfinite'):
            xx, yy = x.clone(), y.clone()
            if mode == 'finite':
                xx[unknown], yy[unknown] = 1e8, -1e6
            else:
                xx[unknown], yy[unknown] = float('nan'), float('inf')
            compare(ref, evaluate(getattr(new, name), xx, yy, valid_mask=valid), torch.float64, name+'_unknown_'+mode+'_invariance')
        negative = evaluate(getattr(old, name), x, y)
        leaked = float(negative[1][unknown].abs().sum())
        assert leaked > 0
        checks.append(dict(name=name+'_original_unmasked_negative_control', passed=True, unknown_prediction_grad_L1=leaked))

    known_difference = (x-y).abs()[valid.expand_as(x)].mean()
    assert torch.equal(new.l1_loss(x, y, valid_mask=valid), known_difference)
    checks.append(dict(name='L1_known_channel_count_normalization', passed=True))
    for pos, excluded in (((0, 0), 36), ((15, 18), 121), ((30, 36), 36)):
        mask = torch.ones(1, 31, 37, dtype=torch.bool)
        mask[0, pos[0], pos[1]] = False
        actual = new._ssim_support(mask, 11)
        manual = torch.ones_like(mask)
        for r in range(31):
            for c in range(37):
                manual[0, r, c] = bool(mask[:, max(0, r-5):min(31, r+6), max(0, c-5):min(37, c+6)].all())
        assert torch.equal(actual, manual) and int((~actual).sum()) == excluded
        checks.append(dict(name='SSIM_support_'+str(pos), passed=True, excluded_windows=excluded))

    def rejects(label, fn):
        try:
            fn()
        except ValueError as error:
            checks.append(dict(name=label, passed=True, error=str(error)))
        else:
            raise AssertionError(label + ' did not reject')

    empty = torch.zeros_like(valid)
    isolated = empty.clone(); isolated[0, 15, 18] = True
    rejects('empty_L1', lambda: new.l1_loss(x, y, valid_mask=empty))
    rejects('empty_SSIM', lambda: new.ssim(x, y, valid_mask=empty))
    rejects('SSIM_no_complete_window', lambda: new.ssim(x, y, valid_mask=isolated))
    rejects('nonboolean_validity', lambda: new.l1_loss(x, y, valid_mask=valid.float()))
    rejects('wrong_mask_shape', lambda: new.ssim(x, y, valid_mask=valid[0]))
    rejects('unsupported_masked_reduction', lambda: new.ssim(x, y, valid_mask=valid, size_average=False))
    rejects('even_SSIM_window', lambda: new.ssim(x, y, valid_mask=valid, window_size=10))
    assert new.l1_loss(x, x, valid_mask=valid) == 0
    torch.testing.assert_close(new.ssim(x, x, valid_mask=valid), torch.tensor(1., dtype=x.dtype), atol=1e-12, rtol=0)
    checks.append(dict(name='equal_known_images_zero_photometric_error', passed=True))
    assert plan['source']['sha256'] == sha(UPSTREAM)
    assert not torch.cuda.is_initialized()
    result = dict(updated=now(), state='isolated_CPU_contract_passed', seconds=time.monotonic()-start,
        checks=checks, controls_passed=len(checks), GPU_used=False, optimization_updates=0,
        original_source_unchanged=True, training_input_unchanged=True, candidate_integrated=False,
        real_occlusion_semantics_validated=False, quality_accepted=False,
        end_to_end_15min_accepted=False, model_promoted=False,
        limit='Synthetic correctness only; does not create or verify real occluder labels or improve a reconstruction by itself.')
    save('result.json', result)
    save('progress.json', dict(updated=now(), state=result['state'], pid=None, optimization_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}, ensure_ascii=False))
    signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            prepare()
        else:
            save('progress.json', dict(updated=now(), state='CPU_contract_running', pid=os.getpid(), optimization_updates=0))
            run()
    except BaseException as error:
        if OUT.exists():
            save(args.mode+'-failure.json', dict(updated=now(), error=repr(error), traceback=traceback.format_exc(), retry=False))
            save('progress.json', dict(updated=now(), state='failed', pid=None))
        raise
