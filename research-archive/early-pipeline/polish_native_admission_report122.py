from pathlib import Path
import shutil
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
V=T/'viewer';S=Path('D:/00/工作/查看器');L=Path('D:/00/工作日志/2026-10-27/内参文件解析')
for p in [V/'body-smplx.html',S/'progress.html',L/'内参文件解析与待核实项.html']:
    s=p.read_text(encoding='utf-8-sig');start=s.index('<!-- ROBUST122_T3_BEGIN -->');end=s.index('<!-- ROBUST122_T3_END -->')+len('<!-- ROBUST122_T3_END -->')
    part=s[start:end];assert 't3-admission-table' not in part
    part=part.replace('<table>','<style>.t3-admission-table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:15px;line-height:1.6}.t3-admission-table td,.t3-admission-table th{border:1px solid #3e5147;padding:8px 10px;vertical-align:top;text-align:left}.t3-admission-table th{background:#1c3026}</style><table class="t3-admission-table"><colgroup><col style="width:18%"><col style="width:42%"><col style="width:40%"></colgroup>')
    p.write_text(s[:start]+part+s[end:],encoding='utf-8')
for root in [V/'new-capture-118/calibration-note',S/'new-capture-118/calibration-note']:
    shutil.copy2(L/'内参文件解析与待核实项.html',root/'01_内参文件解析与待核实项.html')
print('T3_REPORT_TABLE_LAYOUT_UPDATED')
