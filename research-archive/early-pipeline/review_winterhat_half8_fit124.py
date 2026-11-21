"""Reuse the native-pixel and four-direction fit reviewer for winterhat."""
import inspect
import review_sofa_half8_fit124 as existing
import run_winterhat_half8_fit124 as fit

source=inspect.getsource(existing.main)
for old,new in [("Path('/mnt/d/00/记录/006')","Path('/mnt/d/00/记录/001')"),
                ('沙发半圈8帧124','冬帽半圈8帧124'),
                ('SOFA_HALF8_NATIVE_REVIEW_READY','WINTERHAT_HALF8_NATIVE_REVIEW_READY')]:
    assert old in source
    source=source.replace(old,new)
namespace=dict(existing.main.__globals__,fit=fit)
exec(compile(source,'<same-native-winterhat-fit-review>','exec'),namespace)
if __name__=='__main__':namespace['main']()
