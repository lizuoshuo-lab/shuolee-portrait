"""Thin CPU adapter: equal-clock129/131 final metrics, plus131 progression.

static builds/compiles only; run is for root after actual terminal exit.
The frozen1000 helper/artifacts and all training sources remain untouched.
"""
import argparse
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

import summarize_leafwoman_source_comparison131 as core

CORE_SHA = '5a84c74f1019d87b81d32aae8ae38002d36d4dd2ed3f2b049b883b0c082f1de7'
EARLY_SHA = 'e4bdfda30ac9fae773b9ce31c1c24ea2de30584d2571bdc7b197248337869425'
STEP = 18000
LABELS = ('baseline12918000', 'candidate13118000')
OUTPUT = core.CANDIDATE / 'gaussian-run/step-018000/summary131-final.json'


def final_snapshot_contract(snapshot_result, root, read):
    assert snapshot_result['step'] == STEP
    assert snapshot_result['frames_rendered'] == 12 and snapshot_result['active_sh_degree'] == 0
    run_result = read(root / 'gaussian-run/result.json')
    assert run_result['snapshot_step'] == STEP and run_result['optimizer_updates'] == STEP - 1
    assert run_result['configured_schedule_steps'] == STEP
    assert run_result['points'] == snapshot_result['points']
    for key in ('optimizer_updates', 'global_iteration'):
        if key in snapshot_result:
            assert snapshot_result[key] == (STEP - 1 if key == 'optimizer_updates' else STEP)


def extend_result(result, baseline, candidate, read, bind, generated_sha):
    early_path = candidate / 'gaussian-run/step-001000/summary131.json'
    assert bind(early_path) == EARLY_SHA
    early = read(early_path)
    assert early['source_roots']['candidate1311000'] == str(candidate)
    earlier = {row['frame']: row for row in early['records']}
    assert set(earlier) == {row['frame'] for row in result['records']}
    for row in result['records']:
        old = earlier[row['frame']]
        for key in ('derived_wh', 'identical_source_RGB_sha256', 'identical_mask_sha256', 'identical_validity_sha256'):
            assert old[key] == row[key]
        for region in core.REGIONS:
            assert old['models']['candidate1311000']['regions'][region]['pixels'] == row['models'][LABELS[1]]['regions'][region]['pixels']
    curve = {}
    for region in core.REGIONS:
        first = early['area_weighted_aggregate'][region]['candidate1311000']
        last = result['area_weighted_aggregate'][region][LABELS[1]]
        assert first['pixels'] == last['pixels']
        metrics = ['quantized_RGB_L1'] + ([] if region == core.REGIONS[0] else ['alpha_fraction_below_0_5', 'alpha_fraction_below_0_95'])
        curve[region] = dict(step1000=first, step18000=last,
            delta_18000_minus_1000={key: last[key] - first[key] for key in metrics})
    geometry = {}
    completions = {}
    for label, root in ((LABELS[0], baseline), (LABELS[1], candidate)):
        path = root / 'gaussian-run/geometry-step-018000.json'
        geometry[label] = (dict(path=str(path), sha256=bind(path), measurements=read(path)) if path.exists()
            else dict(available=False, reason='No saved geometry receipt; no geometry recomputation or substitution.'))
        exit_path = root / ('final-process-exit.json' if label == LABELS[0] else 'resume-process-exit.json')
        exit_receipt = read(exit_path)
        assert exit_receipt['exit_code'] == 0
        completions[label] = dict(path=str(exit_path), sha256=bind(exit_path), receipt=exit_receipt)
    code = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX/train.py')
    bind(code)
    code_lines = code.read_text().splitlines()
    assert 'scene.save(iteration)' in code_lines[191]
    assert 'if iteration < opt.iterations:' in code_lines[207] and 'gaussians.optimizer.step()' in code_lines[208]
    result.update(frozen_1000_metrics_helper_sha256=CORE_SHA, generated_metrics_AST_sha256=generated_sha,
        snapshot_phase=dict(global_iteration=18000, actual_optimizer_updates=17999,
            rule='OriginalScene.save precedes current density/Adam; terminal18000 skips optimizer.step. Both candidates use the same phase.',
            source=dict(path=str(code), sha256=bind(code), lines='190-209')),
        progression_within131=dict(earlier_summary=str(early_path), earlier_summary_sha256=EARLY_SHA,
            earlier_optimizer_updates=1000, final_optimizer_updates=17999,
            earlier_points=early['point_SH0_color_ranges']['candidate1311000']['points'],
            final_points=result['point_SH0_color_ranges'][LABELS[1]]['points'], regions=curve,
            scope='Descriptive progression of the same131 candidate, not a scientificcontrol. Optimization, geometry/colors/scales/opacity, density/pruning and resetclock co-occur; no unique-cause/reset attribution.'),
        saved_mesh_geometry=geometry, observed_completion_receipts=completions,
        actual_image_review_claim=False, quality_accepted=False)
    return result


def build_summary():
    assert core.sha(core.__file__) == CORE_SHA
    source = textwrap.dedent(inspect.getsource(core.summarize))
    edits = [
        ("root / 'gaussian-run/step-001000'", "root / 'gaussian-run/step-018000'"),
        ("        assert result['step'] == result['optimizer_updates'] == result['global_iteration'] == 1000\n        assert result['frames_rendered'] == 12 and result['active_sh_degree'] == 0\n        assert result['checkpoint_phase'] == 'after_density_adam_zero_grad'",
         "        final_snapshot_contract(result, runs[label], read)"),
        ("candidate / 'dc-projection-1000-receipt.json'", "candidate / 'dc-projection-resume-receipt.json'"),
        ("    assert projection['original_Adam_steps'] == projection['DC_projection_steps'] == 1000\n    assert projection['complete_checkpoint_after_projection'] and projection['Adam_moments_preserved']",
         "    assert projection['actual_total_optimizer_updates'] == projection['DC_projection_steps'] == STEP - 1\n    assert projection['same_fit_cameras_masks_loss_rates_density_reset_and_original_clock']\n    assert projection['final18000_snapshot_is_pre_terminal_optimizer_branch']"),
        ("    for path, digest in reads.items():\n        assert sha(path) == digest, path\n", ""),
        ("schema='leafwoman-frozen-source-comparison131-v1'", "schema='leafwoman-frozen-final-source-comparison131-v1'"),
        ("scope='Only1291000 versus1311000, each1000 actualoptimizerupdates; no18k/ceiling control.'",
         "scope='Equalclock12918000 versus13118000, each17999 actualoptimizerupdates; separate1311000 progression is not a scientificcontrol. Noceiling intervention is a control.'"),
        ("    with output.open('x', encoding='utf-8') as handle:",
         "    result = extend_result(result, baseline, candidate, read, bind, GENERATED_SHA)\n    for path, digest in reads.items():\n        assert sha(path) == digest, path\n    with output.open('x', encoding='utf-8') as handle:"),
    ]
    for old, new in edits:
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new)
    tree = ast.parse(source)
    generated_sha = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    namespace = dict(core.__dict__, LABELS=LABELS, STEP=STEP, __file__=__file__,
        final_snapshot_contract=final_snapshot_contract, extend_result=extend_result, GENERATED_SHA=generated_sha)
    exec(compile(tree, str(__file__) + '::frozen_metrics_adapter', 'exec'), namespace)
    return namespace['summarize'], generated_sha, len(edits)


def static():
    import json
    _, generated_sha, edits = build_summary()
    assert core.sha(core.__file__) == CORE_SHA
    print(json.dumps(dict(state='CPU_AST_only_prepared', helper_sha256=core.sha(__file__),
        frozen_1000_helper_sha256=CORE_SHA, generated_metrics_AST_sha256=generated_sha,
        exact_bounded_adapter_edits=edits, reads_future_snapshot=False, GPU_started=False,
        run_command='python -B summarize_leafwoman_final_comparison131.py run',
        output=str(OUTPUT), quality_accepted=False), indent=2))


def run():
    # Explicit completion gates are read only on the later root-authorized run.
    import json
    for path in (core.BASELINE / 'final-process-exit.json', core.CANDIDATE / 'resume-process-exit.json'):
        receipt = json.loads(path.read_text())
        assert receipt['exit_code'] == 0, path
    function, _, _ = build_summary()
    return function(core.BASELINE, core.CANDIDATE, OUTPUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['static', 'run'])
    globals()[parser.parse_args().mode]()
