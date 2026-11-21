"""Recover only the verified renamed input, before any decoding was performed."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'pattern-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
mapping=next(x for x in read(REC/'source-name-reconciliation.json')['subjects'] if x['subject']=='pattern')
assert len(mapping['exact_matches'])==1 and read(OUT/'initial-failure.json')['decoder_started'] is False
match=mapping['exact_matches'][0]
source=Path('/mnt/d')/match['path'].replace('\\','/')[3:]
assert hashlib.sha256(source.read_bytes()).hexdigest()==mapping['expected_sha256']
resolution=dict(updated=datetime.now(timezone.utc).isoformat(),source=str(source),sha256=match['sha256'],change='Source basename only. Exact full-file content matches the original registered input. No previous decoder, camera or fit ran.',original_plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest())
(OUT/'input-resolution.json').write_text(json.dumps(resolution,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
script=(H/'prepare_pattern_native123.py').read_text(encoding='utf-8')
a=script.index("video=Path('/mnt/d/00/工作/数据集')");b=script.index('start=time.monotonic()',a)
script=script[:a]+f"video=Path({str(source)!r})\nassert not RUN.exists() and not Path(subject['run_root']).exists()\n"+script[b:]
exec(compile(script,str(H/'prepare_pattern_native123.py'),'exec'))
