"""One fresh 18k duration control; unchanged original shared-offset trainer."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import os
import shutil
import sys
import run_official_gaussians123 as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    assert os.name == 'posix'
    plan_path = Path(args.plan)
    plan = runner.read(plan_path)
    assert plan['actual_step_budget'] == 18000
    assert plan['changed_variable'] == 'executed Gaussian steps: 1000 -> 18000'
    source, target = Path(plan['source_root']), Path(plan['target_root'])
    for name, digest in plan['source_hashes'].items():
        assert runner.sha(runner.PROJECT/name) == digest, name
    for name, digest in plan['input_hashes'].items():
        assert runner.sha(source/name) == digest, name
    assert runner.sha(runner.__file__) == plan['existing_launcher_sha256']
    assert not target.exists(), 'Fresh run only; no overwrite or partial optimizer resume.'
    target.mkdir()
    for name in plan['input_hashes']:
        dest = target/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, dest)
    runner.save(target/'control-input.json', dict(plan=plan, plan_sha256=runner.sha(plan_path),
        fit_reused_not_refitted=True, Gaussian_from_scratch=True, end_to_end_benchmark=False,
        reason='User requested a longer trial of the same original shared vertexoffset mechanism.'))
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    runner.SOURCES = {name: plan['source_hashes']['src/shuolee_portrait/'+name]
                      for name in ('training.py', 'export.py')}
    runner.CFG = plan['config']
    runner.run(SimpleNamespace(root=str(target), project_root=str(runner.PROJECT), assets_root=str(runner.ASSETS)))


if __name__ == '__main__':
    main()
