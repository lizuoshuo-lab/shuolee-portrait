from pathlib import Path
import re,json,hashlib,shutil
V=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\viewer')
S=Path(r'D:\00\工作\查看器')
B=Path(__file__).parent/'viewer-camera-sync-before-20261027'
B.mkdir(exist_ok=True)
for n in ['viewer.js','index.html','viewer.css','使用说明.md']:
    if not (B/n).exists():shutil.copy2(S/n,B/n)
js=(V/'viewer.js').read_text(encoding='utf-8')
if "fetch('viewer-info'" not in js:
    js=js.replace('async function init(){', '''async function init(){
  try{
    const response=await fetch('viewer-info',{cache:'no-store'});
    if(response.ok){const host=await response.json();if(typeof host.root==='string'&&host.identity?.startsWith('shuolee-viewer:')){text('app-source','当前文件夹：'+host.root);$('app-source').hidden=false;}}
  }catch{}
''')
    (V/'viewer.js').write_text(js,encoding='utf-8',newline='\n')
h=(V/'index.html').read_text(encoding='utf-8')
h=h.replace('2026.10.27.2 · Camera / WASD','2026.10.27.3 · 通用 Camera / WASD')
if 'id="app-version"' not in h:
    h=h.replace('<aside>','<aside>\n<p class="micro" id="app-version">页面版本 2026.10.27.2 · Camera / WASD</p><p class="micro" id="app-source" hidden></p>')
    h=h.replace('href="viewer.css"','href="viewer.css?v=20261027-3"')
(V/'index.html').write_text(h,encoding='utf-8',newline='\n')
syncjs=(S/'viewer.js').read_text(encoding='utf-8-sig')
builtin=re.search(r'const BUILTINS = .*?;',syncjs).group(0)
js=re.sub(r'const BUILTINS = .*?;',lambda m:builtin,js,count=1)
(S/'viewer.js').write_text(js,encoding='utf-8',newline='\n')
h=h.replace('body-smplx.html#boundary-analysis','progress.html#diagnosis').replace('href="report.html"','href="progress.html"')
(S/'index.html').write_text(h,encoding='utf-8',newline='\n')
for n in ['viewer.css','camera-io.js','camera-formats.js']:shutil.copy2(V/n,S/n)
readme=(S/'使用说明.md').read_text(encoding='utf-8-sig')
readme=readme.replace('页面功能版本为2026.10.27.2','页面功能版本为2026.10.27.3')
readme=readme.replace('- “下载全部相机”保存当前相机列表与模型SHA256（有记录时）；“下载当前视角”保存当前K和位姿，手动视角标记synthetic，不自动当训练相机。刷新网页需重新加载手动导入的文件。','- 相机下载默认输出标准3DGS cameras.json数组。可选COLMAP相机包（PINHOLE完整fx/fy/cx/cy，Hamilton wxyz，世界到相机外参）；部分3DGS查看器会忽略额外cx/cy，偏心主点请选COLMAP或核对目标软件。完整来源JSON保留模型SHA256与像素约定。手动视角标记synthetic，不作为实拍相机；下载不改变相机或点云。')
readme=readme.replace('当前版本为2026.10.26.1','启动器版本为2026.10.26.1；页面功能版本为2026.10.27.2')
readme=readme.replace('- 实拍相机仅对预设模型提供；直接导入文件先适应整体。不要把坐标轴改变当成人体拟合修复。', '''- 相机：先“打开文件”选择PLY，再“加载相机 JSON”选择同一轮配套文件。支持3DGS标准cameras数组、本站模型JSON和含K/wh/w2c的cameras.json。文件声明integer_centers时主点只转换一次+0.5；未声明的旧3DGS数组按COLMAP边缘坐标读取并提示。不会自动旋转、缩放或配准点云；格式通过不证明相机物理标定准确。
- “下载全部相机”保存当前相机列表与模型SHA256（有记录时）；“下载当前视角”保存当前K和位姿，手动视角标记synthetic，不自动当训练相机。刷新网页需重新加载手动导入的文件。
- WASD：先点击三维场景，W/S前后、A/D左右，Q/E沿当前向上轴升降，Shift三倍速度；速度菜单可选0.1×。Esc退出键盘控制。点击菜单/其他窗口后停止移动；不会修改点云位置。''')
(S/'使用说明.md').write_text(readme,encoding='utf-8',newline='\n')
receipt={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['viewer.js','index.html','viewer.css','camera-io.js','camera-formats.js','使用说明.md']}
(B/'update-receipt.json').write_text(json.dumps(dict(files=receipt,host_executable_changed=False,cloud_upload_verified=False,remote00_verified=False),indent=2)+'\n')
print(json.dumps(receipt,indent=2))
