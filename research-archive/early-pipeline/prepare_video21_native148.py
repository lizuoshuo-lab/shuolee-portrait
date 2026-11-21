"""Run the existing native intake on the user-selected video21.mp4."""
import argparse
from pathlib import Path
import types

import prepare_core_subjects124 as core
from run_official_gaussians123 import PROJECT, now, save, sha


SUBJECT = 'seated021'
VIDEO = Path('/mnt/d/00/工作/数据集/video21.mp4')
DIGEST = '3cc2d32acb63c70668bf5d98bad00165b50fad0229e99bdbf1c7c3f3d294fd91'
DATA = Path('/opt/gs-slam-lab/shuolee-cross-subject-148')/SUBJECT
REPORT = core.REC/'seated021-native148'
PINS = {
    Path(core.__file__): 'e388e65e945f56c2fad0ee355fdb31f05ff85878bc5fc8280cf17b49360b14a8',
    Path(__file__).with_name('preprocess_demo_fps3_123.py'): '9d99f91032cd82ad36eac1b13ddab7db3e7b84cd4fa2ca35cd2328eee4566ac6',
    PROJECT/'src/shuolee_portrait/observations.py': 'ea7e6998d92583bd44e0e447ee26f91c806350e58df3d56ff4c9449f5caa35b7',
    core.CALIB: '7cded3ee3a21d6c6c0e496a61295db726fa964140b59db6eb5082788f7fe1f50',
}


def paths(subject):
    assert subject == SUBJECT
    return VIDEO, DATA, REPORT, '021', DIGEST


def run(mode):
    assert sha(VIDEO) == DIGEST
    for path, digest in PINS.items():
        assert sha(path) == digest, path
    original = getattr(core, mode)
    scope = dict(core.__dict__, paths=paths, __file__=__file__)
    entry = types.FunctionType(original.__code__, scope, original.__name__)
    assert entry.__code__ is original.__code__
    entry(SUBJECT)
    save(REPORT/(mode+'-receipt.json'), dict(updated=now(), exit_code=0, mode=mode, original_function_reused=True,
        result_sha256=sha(REPORT/'result.json'), video_sha256=DIGEST, training_roster_frozen=False,
        conditional_K=True, geometry_fitted=False, gaussian_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('decode', 'observe'))
    run(parser.parse_args().mode)
