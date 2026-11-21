"""Freeze stage-specific copies of existing, validated read-only diagnostics.

The previous nine-model audit and videos remain unchanged. This adapter only
changes candidate lists, output locations, a shared display scale, and records
opacity coverage so a candidate cannot hide failures through the common mask.
"""
import hashlib
import json
from pathlib import Path
import time

SCRIPTS = Path(__file__).parent
ROOT = SCRIPTS.parents[1] / 'outputs/mini-splatting2-mip-depth-001'


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new, 1)


def main():
    records = {}
    source = (SCRIPTS / 'evaluate_depth_spread.py').read_text()
    old = """names={f'{key}{seed}':f'{prefix}-18000-s{seed}' for key,prefix in
        [('combined','native-motion-mip-depthmask'),('depth003','native-depthguide003'),('depth010','native-depthguide010')]
        for seed in range(3)}"""
    new = """names={'reference0':'native-depthguide003-18000-s0',
        'control0':'native-depthspread000-18000-s0',
        'spread0030':'native-depthspread003-18000-s0',
        'spread0100':'native-depthspread010-18000-s0'}"""
    source = replace_once(source, old, new)
    source = source.replace("'depth-spread-evaluation-001'", "'depth-spread-candidates-002'")
    source = source.replace("all9 models", "all4 models")
    source = source.replace("cache['combined0']", "cache['control0']")
    source = replace_once(source, "cache[key]['rgb_error']=", "cache[key]['opaque']=(moment['alpha'].cpu().numpy()>.95)\n            cache[key]['rgb_error']=")
    source = replace_once(source, "baseline_subject_quartile_limits=limits.tolist())",
        "baseline_subject_quartile_limits=limits.tolist(),\n                opacity_coverage=dict(full=float(data['opaque'].mean()),\n                    subject=float(data['opaque'][subject].mean()),head=float(data['opaque'][head].mean())),\n                unmasked_rgb_mae=dict(full=float(data['rgb_error'].mean()),\n                    subject=float(data['rgb_error'][subject].mean()),head=float(data['rgb_error'][head].mean())))")
    source = replace_once(source, "heatmap(data['relative_std'],common).save", "heatmap(data['relative_std'],common,maximum=.50).save")
    source = replace_once(source, "bins='four subject-ROI quartiles of combined seed0 dispersion, shared across all models in each view',",
        "bins='four subject-ROI quartiles of zero-variance control dispersion, shared across all models in each view',\n        display_range_relative_std=[0,.50],\n        intervention='fixed mean-depth weight .003; variance weights 0, .003, .01; seed0 only',")
    output = SCRIPTS / 'evaluate_depth_spread_candidates.py'
    assert not output.exists()
    output.write_text(source, encoding='utf-8')
    records[output.name] = dict(adapted_from='evaluate_depth_spread.py', sha256=hashlib.sha256(output.read_bytes()).hexdigest())

    source = (SCRIPTS / 'render_native_depth_paths.py').read_text()
    source = replace_once(source,
        "NAMES={'Baseline':'native-motion-mip-depthmask-18000-s0',\n       'Depth 0.003':'native-depthguide003-18000-s0','Depth 0.010':'native-depthguide010-18000-s0'}",
        "NAMES={'Spread 0':'native-depthspread000-18000-s0',\n       'Spread 0.003':'native-depthspread003-18000-s0','Spread 0.010':'native-depthspread010-18000-s0'}")
    source = replace_once(source, "out=ROOT/'native-depth-paths-001'", "out=ROOT/'native-spread-paths-001'")
    output = SCRIPTS / 'render_native_spread_paths.py'
    assert not output.exists()
    output.write_text(source, encoding='utf-8')
    records[output.name] = dict(adapted_from='render_native_depth_paths.py', sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    target = ROOT / 'native-spread-acceptance-adapters.json'
    assert not target.exists()
    target.write_text(json.dumps(dict(created=time.time(), files=records,
        modifications=['candidate and output names', 'same depth-spread heatmap range 0..0.5 for all models',
            'per-model opacity coverage and unmasked RGB error'],
        metrics_unchanged=True, previous_sources_unchanged=True), indent=2))
    print(json.dumps(records))


if __name__ == '__main__':
    main()
