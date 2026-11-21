from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

H = Path(__file__).parent
W = H / 'remy00-delivery-20261103'
S = Path('D:/00/工作/查看器')
V = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer')
W.mkdir(exist_ok=True)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

source = S / 'ViewerHost.cs'
old = source.read_text(encoding='utf-8-sig')
assert 'const string Build = "2026.10.26.1";' in old
assert 'case ".zip":' not in old
for label, root in [('sync', S), ('research', V)]:
    backup = W / 'before' / label
    backup.mkdir(parents=True, exist_ok=True)
    for name in ['ViewerHost.cs', '高斯与点云查看器.exe']:
        src = root / name
        if src.exists():
            dst = backup / name
            if dst.exists():
                raise FileExistsError(dst)
            shutil.copy2(src, dst)

new = old.replace('const string Build = "2026.10.26.1";',
                  'const string Build = "2026.11.03.1";')
anchor = 'case ".json":mime="application/json; charset=utf-8";break;'
assert new.count(anchor) == 1
new = new.replace(anchor, anchor + 'case ".zip":mime="application/zip";break;')
staged = W / 'ViewerHost.cs'
staged.write_text(new, encoding='utf-8', newline='\n')
exe = W / '高斯与点云查看器.exe'
subprocess.run([
    'C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe',
    '/nologo', '/target:winexe', '/optimize+', '/out:' + str(exe),
    '/r:System.Windows.Forms.dll', '/r:System.Drawing.dll',
    '/r:System.Web.Extensions.dll', str(staged),
], check=True)

pack = H / 'remy-import-20261103/scene.zip'
assert digest(pack) == '0d1ac215954d1b8d3b23271fa3f89cd01cc2a7efaff86975cc99973d177c9a93'
package_name = 'Remy_261103_112806_点云与相机.zip'
for root in [S, V]:
    shutil.copy2(staged, root / staged.name)
    shutil.copy2(exe, root / exe.name)
    (root / 'exports').mkdir(exist_ok=True)
    shutil.copy2(pack, root / 'exports' / package_name)
    assert digest(root / 'exports' / package_name) == digest(pack)

instructions = '''Remy 在 00 上打开（2026-11-03）

1. 等 WPS 将本文件旁边的 Remy_261103_112806_点云与相机.zip 下载到本机。
   完整大小：11,898,161 字节（约 11.35 MiB）。不用解压。
2. 打开“工作/查看器/高斯与点云查看器.exe”。用这个文件夹的程序，避免旧桌面入口指向另一份目录。
3. 在查看器点击“打开文件”，选择上面的 ZIP。
   应显示 789,005 个高斯、174 个相机。模型和相机已经配套，不需要重新导入相机。

这个操作直接读取本地包，不依赖之前的 Remy 分享地址，也不依赖旧模型列表中的文件地址。
本机已完成 Gaussian / 点云两种模式的实际导入核验；00 的云同步及实际打开尚未核验。

已修复 Windows 启动器把存在的 ZIP 返回 404 的问题，新启动器为 2026.11.03.1。
若使用浏览器地址打开，请从本文件夹重新启动后使用新标签，不要照搬另一台电脑的 127.0.0.1 端口。
关闭旧浏览器标签不等于退出旧查看器服务；旧服务可在系统托盘“退出查看器服务”。
若仍是 404，保留完整地址或错误截图，以便区分旧目录、文件未下载和分享网页失效。
'''
(S / 'exports/Remy打开步骤.txt').write_text(instructions, encoding='utf-8-sig', newline='\r\n')
readme = S / '使用说明.md'
text = readme.read_text(encoding='utf-8-sig')
text = text.replace('启动器版本为2026.10.26.1；页面功能版本为2026.10.27.4',
                    '启动器版本为2026.11.03.1；页面功能版本为2026.11.02.2')
text += '\n\n## Remy 在 00 上出现 404\n\n直接在“打开文件”中选择 `exports/Remy_261103_112806_点云与相机.zip`，无需解压。应显示789,005个高斯和174个相机。完整包约11.35 MiB；先等WPS下载完成。详细步骤见 `exports/Remy打开步骤.txt`。新Windows启动器已支持ZIP下载，旧启动器会对存在的ZIP返回404；旧桌面入口或旧端口仍可能指向另一份目录。00同步与运行尚未核验。\n'
readme.write_text(text, encoding='utf-8', newline='\n')

receipt = {
    'updated': datetime.now(timezone.utc).isoformat(),
    'state': 'prepared_pending_runtime_verification',
    'launcher_build': '2026.11.03.1',
    'scope': 'ZIP MIME whitelist plus matching launcher version; no model or camera transformation',
    'old_launcher_test': {'build': '2026.10.26.1', 'metadata_http': 200, 'sog_http': 200, 'existing_zip_http': 404},
    'package': str(S / 'exports' / package_name),
    'package_bytes': pack.stat().st_size,
    'package_sha256': digest(pack),
    'launcher_sha256': digest(exe),
    'company00_verified': False,
    'company00_failure_url_known': False,
    'note': 'ZIP 404 is reproduced locally. It is not yet confirmed as the exact failing URL on company00.',
}
(W / 'delivery.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False))
