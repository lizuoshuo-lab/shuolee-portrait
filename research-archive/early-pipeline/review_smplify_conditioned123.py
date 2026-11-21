"""Reuse exactly the frozen integer crop/overlay renderer for the numerical repair."""
from pathlib import Path
source=(Path(__file__).parent/'review_smplify_corrected123.py').read_text()
needle="O0=O; O=O/'corrected-reference'"
assert source.count(needle)==1
exec(compile(source.replace(needle,"O0=O; O=O/'conditioned-reference'"),str(__file__),'exec'))
