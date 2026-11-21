"""Preserve prior winterhat figures; reuse actual saved native render review."""
import inspect
import sys
import review_cross_subject_sh0124 as existing
source=inspect.getsource(existing.main)
assert source.count('_跨人物SH0_')==1
source=source.replace('_跨人物SH0_','_冬帽时序换算124_')
scope=dict(existing.__dict__)
exec(compile(source,__file__+':record-prefix','exec'),scope)
if __name__=='__main__':scope['main']('winterhat-clock18k-cache124',int(sys.argv[1]))
