"""Same6 source-pixel review, fixed-black versus paired-background at equal step."""
import inspect
import sys
import review_indoor_clock124 as original


def main(step):
    source = inspect.getsource(original.main)
    changes = {
        "z=original.ROOT/'indoor-zoom-clock18k124'":
            "z=original.ROOT/'indoor-paired-background18k124'",
        'baseline=original.Z if step==1000 else original.B':
            "baseline=original.ROOT/'indoor-zoom-clock18k124'",
        'baseline_step=1000 if step==1000 else 18000': 'baseline_step=step',
        '高斯_室内放大时序18k124_': '高斯_室内配对背景长程124_',
        "f'zoom_clock{STEP}'": "f'paired_background{STEP}'",
        "    scope=dict(original.main.__globals__,Z=z,B=baseline,STEP=step,BASE_STEP=baseline_step)":
            "    source=source.replace('_原图_当前mesh_原图训练_放大训练.png','_原图_当前mesh_固定黑底_配对随机底.png')\n"
            "    source=source.replace('source_and_baseline_resampled=False','source_resampled=False,baseline_area_downsampled=True')\n"
            "    scope=dict(original.main.__globals__,Z=z,B=baseline,STEP=step,BASE_STEP=baseline_step)",
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    scope=dict(original.main.__globals__)
    exec(compile(source,'<same-six-view-long-background-review>','exec'),scope)
    scope['main'](step)


if __name__ == '__main__':
    main(int(sys.argv[1]))
