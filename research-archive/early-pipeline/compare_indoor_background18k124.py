"""Reuse frozen four-camera RGB/gray/alpha review at18000 steps."""
import inspect
import compare_indoor_background_orbit124 as original

if __name__ == '__main__':
    source=inspect.getsource(original.main)
    for old,new in {
        "new=ROOT/'indoor-paired-background124'":"new=ROOT/'indoor-paired-background18k124'",
        "'gaussian-run/step-001000/world.ply'":"'gaussian-run/step-018000/world.ply'",
        'black1000':'black18000', 'paired1000':'paired18000',
        'source_step=1000,candidate_step=1000':'source_step=18000,candidate_step=18000',
        'BG1000':'BG18000', 'same1000 steps':'same18000 steps',
        '分析_室内配对背景124_1000_':'分析_室内配对背景长程124_18000_',
    }.items():
        assert old in source,old
        source=source.replace(old,new)
    namespace=dict(original.main.__globals__)
    exec(compile(source,'<fixed-four-camera-background18k-review>','exec'),namespace)
    namespace['main']()
