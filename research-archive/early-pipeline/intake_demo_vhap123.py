"""Read the newly authorized demo; no optimization or change to source pixels."""
from pathlib import Path
import json, hashlib, subprocess
import cv2
import numpy as np
from PIL import Image
from run_official_gaussians123 import save, now, sha

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
VIDEO=Path('/mnt/d/00/工作/数据集/demo.mp4')

if __name__=='__main__':
    ROOT.mkdir(exist_ok=False)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(VIDEO)]))
    save(ROOT/'ffprobe.json',probe)
    stream=next(x for x in probe['streams'] if x['codec_type']=='video')
    cap=cv2.VideoCapture(str(VIDEO)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids=np.linspace(0,count-1,6).round().astype(int).tolist()
    frames=[]; panel=Image.new('RGB',(1200,800),(240,240,240))
    for slot,i in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));p=ROOT/f'input-frame-{i:06d}.png';im.save(p)
        frames.append(dict(source_frame_index=i,path=str(p),sha256=sha(p),wh=im.size))
        im.thumbnail((400,400));panel.paste(im,((slot%3)*400,(slot//3)*400))
    cap.release();panel.save(ROOT/'intake-contact.png')
    save(ROOT/'intake.json',dict(updated=now(),video=str(VIDEO),sha256=sha(VIDEO),bytes=VIDEO.stat().st_size,
        native_size=[stream['width'],stream['height']],frames=count,fps=stream['avg_frame_rate'],duration=probe['format']['duration'],
        samples=frames,user_direction='Use dataset demo, high quality; fit both hands and face; VHAP initialized mesh before bound Gaussians',
        previous_grass_trial='stopped after actual1000-step review; preserved',training_started=False,quality_accepted=False))
    print(json.dumps({k:v for k,v in json.loads((ROOT/'intake.json').read_text()).items() if k!='samples'},ensure_ascii=False),flush=True)
