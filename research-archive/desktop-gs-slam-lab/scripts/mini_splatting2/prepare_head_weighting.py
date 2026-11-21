"""Copy the validated cache runtime and change only the RGB L1 weighting."""
import ast
import difflib
import hashlib
import json
import shutil
import time
from pathlib import Path

S = Path(__file__).parent
ROOT = S.parents[1]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME = Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')


def main():
    check = json.loads((ROOT/'native-head-interface-validation-003/summary.json').read_text())
    assert check['status'] == 'interface_validated' and len(check['rows']) == 4
    assert all(row['interface_passed'] for row in check['rows'])
    helper = S/'head_region_l1.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == '8d490be4b711aa31f42bb3a71ee6ffdce96d8a0d508a4188bbb52de779ee2301'
    source = RUNTIME/'repo-depth-cache'; destination = RUNTIME/'repo-depth-head'
    stage = ROOT/'native-head-runtime-001'
    if destination.exists():
        # Resume only an untouched copy left by a preparation assertion.
        assert stage.exists() and not (stage/'manifest.json').exists()
        assert (source/'msv2/train_depth_guided.py').read_bytes() == (destination/'msv2/train_depth_guided.py').read_bytes()
        (stage/'preparation-recovery.json').write_text(json.dumps(dict(
            original_failure='Whitespace-sensitive SSIM source anchor; no trainer write or training occurred',
            copied_trainer_verified_unchanged=True, recovered=time.time()), indent=2))
    else:
        stage.mkdir(exist_ok=False)
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'build'))
    path = destination/'msv2/train_depth_guided.py'
    before = path.read_text(); current = before; edits = []
    def replace(old, new):
        nonlocal current
        assert current.count(old) == 1, old
        current = current.replace(old, new, 1); edits.append((old, new))
    replace('from native_depth_guidance import DepthGuidance',
            'from native_depth_guidance import DepthGuidance\nfrom head_region_l1 import TrainingHeadL1\nfrom head_training_audit import audit_head, finish_head')
    replace('    first_iter = 0',
            '    head_guidance = TrainingHeadL1(os.environ["MINIGS_HEAD_REGIONS"], float(os.environ["MINIGS_HEAD_WEIGHT"]))\n    first_iter = 0')
    replace('        Ll1 = ((image-gt_image).abs()*valid).sum()/(3*valid.sum().clamp_min(1))',
            '        Ll1 = head_guidance(viewpoint_cam.image_name, image, gt_image, valid)')
    replace('        iter_end.record()',
            '        if iteration % 100 == 0:\n            audit_head(head_guidance, viewpoint_cam, image, gt_image, valid, loss, iteration, dataset.model_path)\n        iter_end.record()')
    replace('    depth_guidance.finish()',
            '    depth_guidance.finish()\n    finish_head(head_guidance, dataset.model_path)')
    ast.parse(current)
    neutral = current
    for old, new in reversed(edits): neutral = neutral.replace(new, old, 1)
    assert ast.dump(ast.parse(neutral), include_attributes=False) == ast.dump(ast.parse(before), include_attributes=False)
    # The expression replacement does not change SSIM/depth construction or joint backward.
    anchor = '        ssim_value ='
    assert before[before.index(anchor):before.index('        iter_end.record()')] == current[current.index(anchor):current.index('        if iteration % 100 == 0:\n            audit_head')]
    path.write_text(current)
    for name in ['head_region_l1.py', 'head_training_audit.py']:
        shutil.copy2(S/name, destination/name)
    unchanged = ['native_depth_guidance.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','native_mip_filter.py','config/fast']
    changed = ['msv2/train_depth_guided.py','head_region_l1.py','head_training_audit.py']
    hashes = lambda names: {name: hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in names}
    manifest = dict(status='prepared_pending_smoke', created=time.time(), repository=str(destination),
                    neutral_ast_equivalent=True, ssim_depth_joint_backward_byte_identical=True,
                    unchanged_files=hashes(unchanged), changed_files=hashes(changed),
                    baseline='native-resolution-cache-full-18000-s0', original_topology_schedule=True,
                    intervention='Training-only reviewed head rectangle coefficient2 in normalized RGB L1',
                    interface_validation='native-head-interface-validation-003/summary.json',
                    exact_gaussian_gradient_equivalence=False, native_repeatability_unresolved=True,
                    rejected_gates_preserved=['native-head-gradient-validation-001','native-head-repeatability-validation-002'])
    for name, digest in manifest['unchanged_files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    (stage/'training.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), current.splitlines(True), fromfile='repo-depth-cache/train', tofile='repo-depth-head/train')))
    (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest), flush=True)


if __name__ == '__main__': main()
