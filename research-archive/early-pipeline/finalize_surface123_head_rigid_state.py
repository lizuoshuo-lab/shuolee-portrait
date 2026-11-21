"""Complete state write after the ledger dictionary/list mismatch; no experiment repeats."""
from pathlib import Path
import shutil,json
from grayT123_head_rigid_common import *
H=Path(__file__).parent;C=H.parent/'shuolee-coordination';R=REC.parent/'robust-portrait-122'
L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
con=read(REC/'head-rigid-conclusion.json');stamp=con['updated'];summary=con['summary']
assert sha(OUT/'result.json')==read(REC/'grayT-head-rigid-completion.json')['result_sha256']
source=(H/'record_surface123_head_rigid.py').read_text(encoding='utf-8');tail=source[source.index("ledger=read(REC/'evidence-ledger.json')"):]
exec(compile(tail,str(H/'record_surface123_head_rigid.py')+':state-only-recovery','exec'))
