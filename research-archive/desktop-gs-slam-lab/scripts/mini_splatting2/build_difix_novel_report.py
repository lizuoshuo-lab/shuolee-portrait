"""Publish actual no-GT path videos with an explicit, locally stored human rubric."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
import numpy as np

S=Path(__file__).parent;ROOT=S.parents[1]/'outputs/mini-splatting2-mip-depth-001';OUT=ROOT/'difix-novel-paths-001'


def encode(paths,destination,labels=None):
    font_path='C:/Windows/Fonts/arial.ttf' if Path('C:/Windows/Fonts/arial.ttf').exists() else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,20)
    width=576*len(paths[0]);height=1024+(32 if labels else 0)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for frames in paths:
            canvas=Image.new('RGB',(width,height),'#19373e');draw=ImageDraw.Draw(canvas)
            for j,file in enumerate(frames):
                picture=Image.open(file).convert('RGB');assert picture.size==(576,1024)
                canvas.paste(picture,(576*j,32 if labels else 0))
                if labels:draw.text((576*j+12,4),labels[j],fill='white',font=font)
            process.stdin.write(np.asarray(canvas).tobytes())
    finally:process.stdin.close()
    error=process.stderr.read().decode();code=process.wait();assert code==0,error


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert len(summary['rows'])==120
    cams=json.loads((OUT/'cameras.json').read_text())
    data=dict(protocol=summary['protocol'],cameras=cams,
        assignments={'elevation':{'A':'input','B':'output'},'distance':{'A':'output','B':'input'}},
        human_ratings=[],human_ratings_count=0,accepted=False,
        baseline='MiniGS2 + Mip + depth',candidate='Difix reference-conditioned image diffusion',
        continuity=json.loads((OUT/'continuity.json').read_text()) if (OUT/'continuity.json').exists() else None)
    viewer=ROOT/'viewer';assets=viewer/'novel-view-assets';assets.mkdir(exist_ok=False)
    for folder in ['input','output','reference']:shutil.copytree(OUT/folder,assets/folder)
    for kind,assignment in data['assignments'].items():
        selected=[c for c in cams if c['kind']==kind]
        for arm in ['input','output']:
            encode([[OUT/arm/(c['name']+'.png')] for c in selected],assets/f'{kind}-{arm}.mp4')
        frames=[[OUT/assignment[label]/(c['name']+'.png') for label in ['A','B']] for c in selected]
        encode(frames,assets/f'{kind}-ab.mp4',['A','B'])
    (assets/'summary.json').write_text(json.dumps(data,indent=2))
    template=(S/'novel_view_report.html').read_text(encoding='utf-8')
    target=viewer/'novel-views.html';assert not target.exists()
    target.write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')),encoding='utf-8')
    index=viewer/'index.html';old=index.read_text(encoding='utf-8')
    shutil.copy2(index,OUT/'index-before-novel.html')
    if 'href="novel-views.html"' not in old:
        index.write_text(old.replace('<a class="report-link"','<a class="report-link" href="novel-views.html">新视角：扩散补全与主观 A/B 对比 ↗</a><a class="report-link"',1),encoding='utf-8')
    result=dict(status='published_pending_browser_review',created=time.time(),url='http://127.0.0.1:8820/novel-views.html',
        human_scores_not_fabricated=True,subjective_scores_initially_empty=True,geometry_updated=False,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.glob('*.mp4')})
    (OUT/'publication.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
