"""Identical qualitative camera paths for native depth candidates. No photo GT."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import imageio_ffmpeg
from local_refinement import ROOT,cameras,read_model,render

NAMES={'Original L1':'native-resolution-cache-full-18000-s0','Head L1 x2':'native-head-full-18000-s0'}


@torch.no_grad()
def main():
    out=ROOT/'native-head-paths-001';out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    paths_path=ROOT/'geometry-evaluation/paths.json';paths=json.loads(paths_path.read_text())['frames']
    cams=cameras();reference=cams[24];w=reference['width'];h=reference['height']
    width=(2*w+1)//2*2;height=(h+32+1)//2*2
    models={};frozen={}
    for key,name in NAMES.items():
        m=json.loads((ROOT/name/'manifest.json').read_text());assert m['status']=='complete'
        p=Path(m['baked_ply']);models[key]=read_model(p)
        frozen[key]=dict(run=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    protocol=dict(status='rendering',started=time.time(),models=frozen,paths_sha256=hashlib.sha256(paths_path.read_bytes()).hexdigest(),
        fps=10,frames_per_path=60,resolution_per_panel=[w,h],rendering='gsplat with baked3D smoothing and2D Mip compensation',
        independent_gt=False,scope='qualitative stress test only; orbit elevation +/-25degrees and radius0.5..1.5 relative to the reference')
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for kind in ['elevation','distance']:
        video=out/(kind+'.mp4')
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
             '-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        with (out/f'{kind}-encode.log').open('w') as log:
            process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            try:
                for path in [x for x in paths if x['kind']==kind]:
                    c=dict(reference);R=np.array(path['rotation']);P=np.array(path['position']);V=np.eye(4);V[:3,:3]=R.T;V[:3,3]=-R.T@P
                    c['viewmat']=torch.tensor(V,device='cuda',dtype=torch.float32)[None]
                    panel=Image.new('RGB',(width,height),'#172d3a');draw=ImageDraw.Draw(panel)
                    for j,(key,model) in enumerate(models.items()):
                        image,_,_=render(model,c,divisor=1,mode='mip')
                        rgb=(image.clamp(0,1)*255).round().byte().cpu().numpy()
                        panel.paste(Image.fromarray(rgb),(j*w,32));draw.text((j*w+10,6),key,fill='white',font=font)
                    process.stdin.write(np.asarray(panel).tobytes())
                    if path['index'] in [0,15,45]:panel.save(out/f'{kind}-{path["index"]:02d}.png')
            finally:process.stdin.close()
            if process.wait()!=0:raise RuntimeError(f'Video encoding failed: {kind}')
        print(json.dumps(dict(path=kind,frames=60,bytes=video.stat().st_size)),flush=True)
    for item in frozen.values():assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    protocol.update(status='complete',finished=time.time(),videos={k:hashlib.sha256((out/f'{k}.mp4').read_bytes()).hexdigest() for k in ['elevation','distance']})
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
