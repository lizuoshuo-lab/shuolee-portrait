"""Correct hand-rounded prose to exact result rounding before acceptance; no solver rerun."""
from pathlib import Path
import json
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/robust-portrait-122/residual-distortion-direction'
V=T/'viewer';S=Path('D:/00/工作/查看器');L=Path('D:/00/工作日志/2026-10-27/内参文件解析')
r=json.loads((O/'result.json').read_text(encoding='utf-8'));x=next(x for x in r['per_line'] if x['id']=='B1')
assert f'{x["before_rms_px"]:.3f}'=='0.493' and f'{x["after_rms_px"]:.3f}'=='0.787'
targets=[O/'review.json',H/'publish_residual_lines122.py']
parents=[V,S,L,V/'new-capture-118/calibration-note',S/'new-capture-118/calibration-note']
targets.extend(p/'122-residual-lines/review.json' for p in parents)
reports=[V/'body-smplx.html',S/'progress.html',L/'内参文件解析与待核实项.html',V/'new-capture-118/calibration-note/01_内参文件解析与待核实项.html',S/'new-capture-118/calibration-note/01_内参文件解析与待核实项.html']
for p in targets+reports:
 s=p.read_text(encoding='utf-8-sig').replace('0.494','0.493').replace('0.790','0.787')
 if p in reports:
  a=s.index('<!-- ROBUST122_D1_BEGIN -->');b=s.index('<!-- ROBUST122_D1_END -->',a)
  sec=s[a:b].replace('<table>','<table style="width:100%;border-collapse:collapse;text-align:left">').replace('<th>','<th style="padding:8px;border-bottom:1px solid #536050">').replace('<td>','<td style="padding:6px 8px;border-bottom:1px solid #28372d">')
  s=s[:a]+sec+s[b:]
 p.write_text(s,encoding='utf-8')
(O/'display-correction.json').write_text(json.dumps(dict(reason='Prose rounded from console table rather than exact JSON; corrected B1 prose to 0.493/0.787 before acceptance.',result_unchanged=True,solver_rerun=False),indent=2),encoding='utf-8')
print('B1_DISPLAY_ROUNDING_CORRECTED')
