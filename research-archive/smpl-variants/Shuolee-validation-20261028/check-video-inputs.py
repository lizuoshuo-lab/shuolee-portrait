"""Probe actual 00 inputs and decode one frame with the production resize filter."""
from pathlib import Path
import hashlib, json, subprocess, sys
from PIL import Image

repo = Path('D:/Projects/Shuolee-validation-20261028/smpl-x')
sys.path.insert(0, str(repo/'src'))
from shuolee_portrait.video119 import validate_video_stream

art = Path('D:/Projects/Shuolee-validation-20261028')
out = art/'input-frames'
out.mkdir(exist_ok=True)
rows = []
for video in sorted(Path('D:/00/工作/数据集').glob('*.mp4')):
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(video)],text=True))
    stream = info['streams'][0]
    validate_video_stream(stream)
    target = out/(video.stem+'.png')
    assert not target.exists()
    result = subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(video),'-vf',"scale=w='min(iw,1600)':h='min(ih,1600)':force_original_aspect_ratio=decrease",'-frames:v','1','-pix_fmt','rgb24',str(target)],capture_output=True,text=True,check=True)
    with Image.open(target) as im:
        wh = list(im.size)
        assert im.mode == 'RGB' and max(wh) == 1600
        assert (wh[0] > wh[1]) == (stream['width'] > stream['height'])
    row = dict(file=video.name,sha256=hashlib.sha256(video.read_bytes()).hexdigest(),source_wh=[stream['width'],stream['height']],decoded_wh=wh,orientation='landscape' if wh[0]>wh[1] else 'portrait',codec=stream['codec_name'],pixel_format=stream['pix_fmt'],frames=int(stream['nb_frames']),sampled_frames=(int(stream['nb_frames'])+4)//5,first_frame_decoded=True,full_reconstruction_tested=False,decoder_stderr=result.stderr)
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
(art/'video-inputs.json').write_text(json.dumps(dict(state='passed',scope='stream validation and one real decoded frame per input; not full reconstruction',source_git=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),videos=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
