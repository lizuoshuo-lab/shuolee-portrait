"""Use the existing native CPU review for the chronological seed control."""
import argparse
import inspect
import review_smplifyx_temporal_root123 as original
import run_smplifyx_forward_guard123 as forward


def main(max_new):
    forward.bind()
    source = inspect.getsource(original.main)
    for before, after in [
        ('SMPL-X_连续朝向_', 'SMPL-X_正向目标保护_'),
        ("'temporal_body_root'", "'forward_guarded_body_root'"),
        ("            s=load(run.OUT/'frames'/seed/'winner-camera-space.npz')",
         "            assert run.sha(fit_receipt['seed_path']) == fit_receipt['seed_sha256']\n            s=load(fit_receipt['seed_path'])")]:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    scope = dict(original.main.__globals__, run=forward.temporal, __file__=__file__)
    exec(compile(source, '<same-native-review-forward-seed-records>', 'exec'), scope)
    scope['main'](max_new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new',type=int,default=5)
    args = parser.parse_args()
    assert 1 <= args.max_new <= 59
    main(args.max_new)
