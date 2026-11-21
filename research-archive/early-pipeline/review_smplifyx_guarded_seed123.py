"""Reuse the native CPU review with separate guarded-control record names."""
import argparse
import inspect
import review_smplifyx_temporal_root123 as original
import run_smplifyx_guarded_seed123 as guarded


def main(max_new):
    guarded.bind()
    source = inspect.getsource(original.main)
    before = "SMPL-X_连续朝向_"
    assert source.count(before) == 1
    source = source.replace(before, "SMPL-X_原版目标保护_")
    source = source.replace("'temporal_body_root'", "'guarded_body_root'")
    scope = dict(original.main.__globals__, run=guarded.temporal, __file__=__file__)
    exec(compile(source, '<same-native-review-separate-guarded-records>', 'exec'), scope)
    scope['main'](max_new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new', type=int, default=5)
    args = parser.parse_args()
    assert 1 <= args.max_new <= 59
    main(args.max_new)
