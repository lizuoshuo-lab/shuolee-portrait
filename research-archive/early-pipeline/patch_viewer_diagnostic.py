from pathlib import Path
roots=[Path('D:/00/工作/查看器'),Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer')]
for root in roots:
 p=root/'index.html';text=p.read_text(encoding='utf-8-sig')
 if 'id="latest-diagnostic"' not in text:
  marker='<p id="latest-status" class="micro" role="status">历史对照模式</p>'
  assert marker in text
  text=text.replace(marker,marker+'\n<div id="latest-diagnostic" class="micro" hidden><strong id="diagnostic-title"></strong><p id="diagnostic-summary"></p><a id="diagnostic-report" target="_blank" rel="noopener">查看本轮诊断与图片 ↗</a></div>')
 p.write_text(text,encoding='utf-8')
 p=root/'viewer.js';text=p.read_text(encoding='utf-8-sig')
 if 'function showLatestDiagnostic' not in text:
  text=text.replace('async function refreshLatest(){', '''function showLatestDiagnostic(d){
  if(!d||!$('latest-diagnostic'))return;
  $('latest-diagnostic').hidden=false;
  text('diagnostic-title','最新诊断 · '+d.round+' · '+d.title);
  text('diagnostic-summary',d.summary);
  const url=new URL(d.reportUrl||'progress.html',location.href);
  $('diagnostic-report').href=url.origin===location.origin?url.href:'progress.html';
}
async function refreshLatest(){''')
  text=text.replace('if(!followingLatest||busy||checkingLatest)return false;','if(busy||checkingLatest)return false;')
  text=text.replace('const latest=await response.json();','const latest=await response.json();\n    showLatestDiagnostic(latest.latestDiagnostic);\n    if(!followingLatest)return true;')
 p.write_text(text,encoding='utf-8')
 print(root)
