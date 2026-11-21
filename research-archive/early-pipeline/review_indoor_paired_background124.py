"""Reuse exact all-six source-pixel comparison, against frozen clock1000."""
import inspect
import review_indoor_clock124 as original


def main():
    source=inspect.getsource(original.main)
    replacements={
        "z=original.ROOT/'indoor-zoom-clock18k124'":"z=original.ROOT/'indoor-paired-background124'",
        'baseline=original.Z if step==1000 else original.B':"baseline=original.ROOT/'indoor-zoom-clock18k124'",
        '高斯_室内放大时序18k124_':'高斯_室内配对背景124_',
        "f'zoom_clock{STEP}'":"f'paired_background{STEP}'",
    }
    for old,new in replacements.items():assert source.count(old)==1,old;source=source.replace(old,new)
    namespace=dict(original.main.__globals__)
    exec(compile(source,'<reuse-six-view-review-paired-background>','exec'),namespace)
    namespace['main'](1000)


if __name__=='__main__':main()
