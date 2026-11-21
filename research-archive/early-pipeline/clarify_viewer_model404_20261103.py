"""Keep model loading unchanged; make a missing synced model actionable."""
import datetime, hashlib, json, shutil
from pathlib import Path

BASE=Path(__file__).parent
V=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer')
S=Path('D:/00/工作/查看器')
OUT=BASE/'remy00-delivery-20261103'/'model-selection-404'
OUT.mkdir(exist_ok=True)

helper='''async function modelReadError(record,status){
  const requested=new URL(record.url,location.href);
  let message='无法读取模型文件（HTTP '+status+'）。\\n文件：'+record.url;
  if(status===404){
    message+='\\n此模型条目存在，但当前查看器未找到对应文件。';
    if(requested.origin===location.origin){
      try{
        const response=await fetch('viewer-info',{cache:'no-store',signal:AbortSignal.timeout(2000)});
        if(response.ok){
          const info=await response.json();
          if(typeof info.root==='string')message+='\\n当前查看器目录：'+info.root;
        }
      }catch{}
    }
    message+='\\n请将对应文件在 WPS 下载到本机，或点“返回查看器”后用“打开文件”导入点云＋相机 ZIP（不用解压）。';
    if(record.id==='remy-20261103-112806')message+='\\nRemy 配套包：工作/查看器/exports/Remy_261103_112806_点云与相机.zip';
  }
  const error=new Error(message);error.httpStatus=status;error.requestedUrl=requested.href;
  return error;
}
'''
js=(V/'viewer.js').read_text(encoding='utf-8')
assert js==(S/'viewer.js').read_text(encoding='utf-8')
assert 'async function modelReadError' not in js
js=js.replace('async function readBytes(record){',helper+'async function readBytes(record){',1)
needle="if(!response.ok)throw Error('无法读取模型文件（HTTP '+response.status+'）。');"
assert js.count(needle)==1
js=js.replace(needle,'if(!response.ok)throw await modelReadError(record,response.status);',1)
js=js.replace("version:'shuolee-viewer-20261102-sh-visible'","version:'shuolee-viewer-20261103-model-path'")
js=js.replace("window.viewerState={...window.viewerState,status:'error',error:String(error)};",
              "window.viewerState={...window.viewerState,status:'error',error:String(error),httpStatus:error.httpStatus||null,requestedUrl:error.requestedUrl||null};")
receipts=[]
for root,label in [(V,'research'),(S,'sync')]:
    backup=OUT/label;backup.mkdir()
    for name in ['viewer.js','index.html','viewer.css']:
        shutil.copy2(root/name,backup/name)
    (root/'viewer.js').write_text(js,encoding='utf-8',newline='\n')
    html=(root/'index.html').read_text(encoding='utf-8')
    html=html.replace('20261102-sh-visible','20261103-model-path')
    html=html.replace('页面版本 2026.11.02.2 · SH0 直接查看','页面版本 2026.11.03.3 · 模型路径检查')
    (root/'index.html').write_text(html,encoding='utf-8',newline='\n')
    css=(root/'viewer.css').read_text(encoding='utf-8')
    css+='\n#loading.error #loading-detail{white-space:pre-line;overflow-wrap:anywhere;max-width:min(42rem,85vw)}\n'
    (root/'viewer.css').write_text(css,encoding='utf-8',newline='\n')
    receipts.append(dict(root=str(root),viewer_js_sha256=hashlib.sha256((root/'viewer.js').read_bytes()).hexdigest()))

guide=S/'exports/Remy打开步骤.txt'
original=guide.read_text(encoding='utf-8')
shutil.copy2(guide,OUT/'Remy打开步骤.before.txt')
note='\n2026-11-03 补充：已确认 00 是“选择 Remy 模型后”404。上面的 ZIP 服务修复不能单独解释这个现象；本机相同模型文件已实际加载。新页面在404时会显示请求的文件路径及当前查看器目录，便于确认同步缺文件或旧入口。直接打开完整 ZIP 仍是现在最短的操作。00 实际加载结果待验证。\n'
guide.write_text(original+note,encoding='utf-8')
shutil.copy2(guide,V/'exports'/guide.name)
receipt=dict(updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    symptom='Human confirmed 404 after selecting Remy, not downloading ZIP.',
    root_cause_on_company00='unconfirmed; local requested SOG is present and renders',
    changes='Actionable 404 path/root and paired ZIP instructions only; no loading fallback or model changes',
    roots=receipts,company00_verified=False)
(OUT/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
