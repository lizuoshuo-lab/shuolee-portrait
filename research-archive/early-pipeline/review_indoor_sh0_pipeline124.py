"""Existing saved-render review with a distinct indoor record prefix."""
import inspect
import sys
import review_cross_subject_sh0124 as existing

source = inspect.getsource(existing.main)
assert source.count('_跨人物SH0_') == 1
source = source.replace('_跨人物SH0_', '_跨人物SH0_室内六帧124_')
scope = dict(existing.__dict__)
exec(compile(source, __file__+':record-prefix', 'exec'), scope)

if __name__ == '__main__':
    scope['main']('indoor-selected-sh0-pipeline124', int(sys.argv[1]))
