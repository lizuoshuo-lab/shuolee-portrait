"""Twelve new-person views; unchanged reviewed SH0/ROI2x/paired-background profile.

This is a narrow input/reference/count adapter over the existing trainer.
The optional snapshot alpha export is read-only; no opacity or loss change.
"""
import ast
import difflib
import inspect
import os
import re
import signal
import sys
import traceback
from pathlib import Path

import numpy as np
import train_winterhat_half8_sh0124 as previous
from train_indoor_circle16_sh0124 import reused
import cross_subject_sh0124 as gs

SUBJECT = 'football-half12-paired18k126'
OUT = gs.ROOT/SUBJECT
FIT = gs.ROOT/'football-half12-pose126'
HANDOFF = FIT/'gaussian-native-handoff'
N = 12


# Scene.save is called inside the original no_grad checkpoint branch. Keep an
# explicit no_grad block as well, use the unpaired evaluation renderer, and do
# not consume the training sampler or paired-background generator.
ALPHA_EXPORT = '''                with torch.no_grad():
                    from PIL import ImageFilter
                    cuda_rng_before = torch.cuda.get_rng_state().clone()
                    cpu_rng_before = torch.get_rng_state().clone()
                    transmission_rgb = render(cams[i], model, pipe,
                        torch.ones_like(background),
                        override_color=torch.zeros_like(model.get_xyz))['render']
                    channel_delta = float((transmission_rgb-transmission_rgb[:1]).abs().max())
                    assert channel_delta < 2e-6, channel_delta
                    cumulative_alpha = 1.-transmission_rgb.mean(dim=0)
                    assert torch.isfinite(cumulative_alpha).all()
                    assert float(cumulative_alpha.min()) >= -2e-5 and float(cumulative_alpha.max()) <= 1.+2e-5
                    assert torch.equal(cuda_rng_before, torch.cuda.get_rng_state())
                    assert torch.equal(cpu_rng_before, torch.get_rng_state())
                    alpha_array = cumulative_alpha.detach().cpu().numpy().astype(np.float32)
                    alpha_path = folder/(row['frame']+'_cumulative-alpha.npy')
                    np.save(alpha_path, alpha_array, allow_pickle=False)
                    alpha_display = Image.fromarray(np.rint(np.clip(alpha_array,0,1)*255).astype(np.uint8))
                    display_path = folder/(row['frame']+'_cumulative-alpha.png')
                    alpha_display.save(display_path)
                    source_mask = Image.open(row['mask']).convert('L')
                    interior_mask = source_mask.filter(ImageFilter.MinFilter(11))
                    quantiles = [0.,.01,.05,.1,.25,.5,.75,.9,.95,.99,1.]
                    regions = {}
                    for region_name, region_mask in [('mask',source_mask),('mask_eroded_5_derived_pixels',interior_mask)]:
                        region_values = alpha_array[np.asarray(region_mask)>0]
                        regions[region_name] = dict(pixels=int(region_values.size),
                            quantiles=[float(x) for x in np.quantile(region_values,quantiles)] if region_values.size else None,
                            fraction_below_0_5=float((region_values<.5).mean()) if region_values.size else None,
                            fraction_below_0_95=float((region_values<.95).mean()) if region_values.size else None)
                    save(folder/(row['frame']+'_cumulative-alpha.json'),dict(
                        frame=row['frame'], alpha_npy_sha256=sha(alpha_path), display_png_sha256=sha(display_path),
                        mask_sha256=row['mask_sha256'], channel_max_abs_difference=channel_delta,
                        original_min=float(alpha_array.min()),original_max=float(alpha_array.max()),
                        quantile_probabilities=quantiles,regions=regions,
                        measurement='1-T from zero override RGB on white background through the unchanged native rasterizer; includes its sorting and truncation.',
                        limitations='Cumulative alpha along this camera ray, not per-point opacity, front-surface-only opacity or full physical geometry truth.',
                        eroded_region_is_diagnostic_only=True,training_mask_unchanged=True,
                        optimizer_updates_added=0, RNG_unchanged=True))
'''


def adapted_prepare(reference, frozen_frames):
    """Compile the existing successful prepare function without executing it."""
    before = inspect.getsource(previous.prepare)
    code = before
    begin = code.index('    expected=[')
    end = code.index("    assert [r['frame']", begin)
    code = code[:begin] + ("    source_plan=gs.read(FIT/'plan.json');fit_result=gs.read(FIT/'result.json')\n"
        "    expected=list(frozen_frames)\n") + code[end:]
    code = re.sub(r'\b8\b', str(N), code)
    code = code.replace('all8', 'all12')
    # Change standalone count labels, never the substring in e.g. weight.
    code = re.sub(r'\beight\b', 'twelve', code)
    code = re.sub(r'\bEight\b', 'Twelve', code)
    assert 'shared_laplacian_weight' in code and 'shared_laplacian_wtwelve' not in code
    code = code.replace('All eight', 'All twelve').replace('all eight', 'all twelve')
    for old, new, occurrences in [
        ("row['wh']==[1520,2704]", "row['wh']==[2704,1520]", 1),
        ("REFERENCE='frame_000071'", 'REFERENCE=reference', 1),
        ('code.replace("\'frame_000105\'","\'frame_000071\'")', 'code.replace("\'frame_000105\'",repr(reference))', 1),
        ("reference_frame='frame_000071'", 'reference_frame=reference', 1),
        ("record_id='001'", "record_id='011'", 2),
    ]:
        assert code.count(old) == occurrences, (old, code.count(old))
        code = code.replace(old, new)
    code = code.replace('winterhat', 'football').replace('Winterhat', 'Football').replace('冬帽', '足球场新')
    ast.parse(code)
    scope = dict(previous.prepare.__globals__, OUT=OUT, FIT=FIT, HANDOFF=HANDOFF,
        SUBJECT=SUBJECT, reference=reference, frozen_frames=frozen_frames, __file__=__file__)
    exec(compile(code, str(Path(__file__))+':new-person-twelve-inputs', 'exec'), scope)
    return scope['prepare'], before, code


def add_alpha_export(code):
    anchor = "                image(rgb).save(folder/(row['frame']+'_gaussian.png'))\n"
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor+ALPHA_EXPORT)
    ast.parse(code)
    return code


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists() and not HANDOFF.exists()
    plan = gs.read(FIT/'plan.json')
    rows = plan['rows']
    frames = [r['frame'] for r in rows]
    reference = plan['reference_frame']
    assert len(set(frames)) == len(rows) == N and reference in frames
    assert reference != frames[0], 'Source selection must explicitly choose a good non-first reference.'
    with np.load(FIT/'selected-fit.npz', allow_pickle=False) as z:
        assert z['betas'].shape == (1,10)
        assert z['body_pose_axis_angle'].shape == (1,63)
        assert z['pose_embedding'].shape == (1,32)
        assert z['frame_ids'].astype(str).tolist() == frames
        # The fitter stores the single shared full K as3x3, not an N-camera stack.
        assert z['K_integer_centers'].shape == (3,3)
        assert all(np.array_equal(z['K_integer_centers'],np.asarray(r['K_integer_centers'])) for r in rows)
    fn, before, changed = adapted_prepare(reference, frames)
    fn()
    (OUT/'twelve-prepare-interface.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),changed.splitlines(True),fromfile='existing-winterhat-input-interface',
        tofile='new-football-twelve-input-interface')))
    adapter = OUT/'adapter.py'
    original = adapter.read_text()
    adapter.write_text(add_alpha_export(original))
    (OUT/'read-only-alpha-export.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),adapter.read_text().splitlines(True),
        fromfile='unchanged-training-profile',tofile='snapshot-read-only-cumulative-alpha')))
    current = gs.read(OUT/'plan.json')
    current.update(updated=gs.now(), label='足球场新人物 · 半圈12帧 SH0',record_id='011',
        user_authorization='做新的人物；半圈12，一圈24；沿已跑通SH0表面绑定路线。',
        reference_frame=reference, initial_admission_max_step=1000, automatic_18k_continuation=False,
        hypothesis='Apply the same frozen ROI2x/paired-background/SH0/18k profile to a new person with12 reviewed views.',
        only_parameter_change='New video, twelve frozen source views and a fresh official SMPLify-X subject fit; no opacity/loss/LR change.',
        causal_scope='New-person pipeline trial; there is no same-person old baseline or isolated view-count ablation.',
        fixed='SH0;person ROI2x;original loss and LR amplitudes;18k clock;shared clothing offset1e-6;protected face/head/hands/feet;source-selected reference '+reference,
        criteria='Actual all12 raw/mask/current-mesh/Gaussian/alpha review at1000, then a hash-bound decision; at most18000, no automatic retry.',
        known_limitation='One-side through front to opposite-side coverage; no rear observations or calibrated180-degree claim. Fingers near/in pockets unknown, hand2D uniformly disabled; fullK remains conditional.',
        alpha_diagnostic=dict(enabled=True, snapshot_only=True, loss_or_optimizer_changed=False,
            raw_float32_saved=True, display_only_clamp=True, erosion_radius_derived_pixels=5,
            renderer='Native rasterizer sorting/truncation, zero override RGB with white background',
            surface_opacity_acceptance_claim=False), quality_accepted=False)
    for path in list(current['source_hashes']):
        if Path(path).is_relative_to(OUT):
            current['source_hashes'][path] = gs.sha(path)
    for path in [Path(__file__),Path(previous.__file__),gs.HERE/'body_root_sh_colors123.py']:
        current['source_hashes'][str(path)] = gs.sha(path)
    gs.save(OUT/'plan.json',current)
    assert 'frame_000071' not in adapter.read_text() and 'frame_000105' not in adapter.read_text()
    assert f"== {N}" in adapter.read_text() and f"self.num_timesteps = {N}" in adapter.read_text()
    print('FOOTBALL_TWELVE_PROFILE_PREPARED', reference, flush=True)


def preflight():
    edits = [('paired.calls==8','paired.calls==12'),("fn('eight-preflight')","fn('twelve-preflight')")]
    fn, diff = reused(previous.preflight,edits,dict(OUT=OUT),'football-twelve-paired-preflight')
    (OUT/'twelve-preflight-interface.diff').write_text(diff)
    fn()


def train():
    assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
    assert gs.read(OUT/'validity-cuda-preflight.json')['passed']
    assert gs.read(OUT/'validity-review.json')['training_admitted']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Registered2400seconds')))
    signal.alarm(2400)
    try:
        gs.train(SUBJECT)
    finally:
        signal.alarm(0)


def audit():
    _, _, code = adapted_prepare('frame_000033',[f'frame_{i:06d}' for i in range(12)])
    # Evaluate only the read-only code-string transformation within prepare.
    # Do not execute its mesh preparation, mkdir, image creation or GPU path.
    function = ast.parse(code).body[0]
    start = next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='before' for t in node.targets)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and node.value.func.attr=='read_text')
    end = next(i for i,node in enumerate(function.body[start:],start) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='ast'
        and node.value.func.attr=='parse')
    scope = dict(previous.prepare.__globals__,reference='frame_000033')
    exec(compile(ast.Module(body=function.body[start:end],type_ignores=[]),'<read-only-profile-string-audit>','exec'),scope)
    template = scope['code']
    changed = add_alpha_export(template)
    for marker in ["receipt['frames'] == 12",'self.num_timesteps = 12',
                   'assert len(training_slots)==12','counts = np.zeros(12, dtype=int)',
                   'frames_rendered=12','training_frames=12, review_frames=12']:
        assert changed.count(marker)==1,marker
    assert 'frame_000071' not in changed and 'frame_000105' not in changed
    assert "'frame_000033'" in changed
    assert 'KnownPixelLoss' not in changed and 'photometric.use_camera' not in changed
    assert "deadline=time.monotonic()+600" in changed
    assert "'lr': 1.e-6" in changed
    assert "override_color=torch.zeros_like(model.get_xyz)" in changed
    review_path = Path(__file__).with_name('review_football_half12_gaussian126.py')
    ast.parse(review_path.read_text())
    assert not OUT.exists() and not HANDOFF.exists()
    result = dict(updated=gs.now(),state='static_interface_audit_passed',GPU_started=False,
        experiment_created=False,helper_sha256=gs.sha(__file__),frames=N,
        review_helper_sha256=gs.sha(review_path),source_count_transform_checked=True,
        reference_from_fit_plan=True, prepare_ast_compiled=True, snapshot_adapter_ast_compiled=True,
        review_timeout_seconds=600,total_budget_seconds=2400,
        original_training_parameters_unchanged=True,alpha_diagnostic_only=True,
        quality_accepted=False)
    gs.save(Path(__file__).with_suffix('.audit.json'),result)
    print(__import__('json').dumps(result),flush=True)


if __name__ == '__main__':
    try:
        mode = sys.argv[1]
        {'audit':audit,'prepare':prepare,'preflight':preflight,'train':train}[mode]()
        if mode == 'train':
            gs.save(OUT/'process-exit.json',dict(updated=gs.now(),exit_code=0))
    except BaseException as error:
        if OUT.exists():
            gs.save(OUT/(sys.argv[1]+'-failure.json'),dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise
