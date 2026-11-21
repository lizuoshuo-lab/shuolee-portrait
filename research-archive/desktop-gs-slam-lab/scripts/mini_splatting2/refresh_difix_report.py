import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
S=Path(__file__).parent
V=S.parents[1]/'outputs/mini-splatting2-mip-depth-001/viewer'
data=json.loads((V/'novel-view-assets/summary.json').read_text())
data['distillation']=json.loads((V/'distill-view-assets/summary.json').read_text())
for directory in ['novel-view-assets','distill-view-assets']:
    for kind in ['elevation','distance']:
        dest=V/directory/f'{kind}-ab.jpg'
        if not dest.exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-i',str(V/directory/f'{kind}-ab.mp4'),'-frames:v','1','-q:v','2',str(dest)],check=True)
(V/'novel-views.html').write_text((S/'novel_view_report.html').read_text(encoding='utf-8').replace('__DATA__',json.dumps(data).replace('</','<\\/')),encoding='utf-8')
print('refreshed report and paired posters')
