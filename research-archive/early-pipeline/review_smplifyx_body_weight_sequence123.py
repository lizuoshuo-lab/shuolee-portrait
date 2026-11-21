"""Reuse the native per-frame CPU review for the uniform body-data control."""
import argparse
import inspect
import review_smplifyx_temporal_root123 as original
import run_smplifyx_body_weight_sequence123 as run


def main(max_new):
    run.bind()
    source=inspect.getsource(original.main)
    replacements=[('SMPL-X_连续朝向_','SMPL-X_身体数据权重全59_'),
        ("'temporal_body_root'","'uniform_body_observation_amplitude2'"),
        ("            s=load(run.OUT/'frames'/seed/'winner-camera-space.npz')",
         "            assert run.sha(fit_receipt['seed_path']) == fit_receipt['seed_sha256']\n            s=load(fit_receipt['seed_path'])")]
    for before,after in replacements:
        assert source.count(before)==1,before
        source=source.replace(before,after)
    scope=dict(original.main.__globals__,run=run.warm.forward.temporal,__file__=__file__)
    exec(compile(source,'<same-native-review-uniform-body-weight>','exec'),scope)
    scope['main'](max_new)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-new',type=int,default=5);args=parser.parse_args()
    assert 1<=args.max_new<=59
    main(args.max_new)
