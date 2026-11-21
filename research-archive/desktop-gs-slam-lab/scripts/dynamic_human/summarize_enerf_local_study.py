"""Paired results and actual decoded video counts; no perceptual acceptance."""
from pathlib import Path
import json,subprocess
import numpy as np

LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029'
r=json.loads((out/'evaluation/metrics.json').read_text());rows=r['views']
lookup={(v['camera'],v['common_frame'],v['model']):v for v in rows}
models=r['models'];comparisons=[]
for baseline in ['static','same-index']:
    values=[]
    for cam in r['cameras']:
        for t in r['frames']:
            a=lookup[cam,t,baseline];b=lookup[cam,t,'offset-index']
            values.append(dict(camera=cam,common_frame=t,foreground_psnr_delta=b['foreground_psnr_db']-a['foreground_psnr_db'],upper_body_psnr_delta=float(np.mean([p['upper_body_psnr_db'] for p in b['people']])-np.mean([p['upper_body_psnr_db'] for p in a['people']])),outside_pixels_delta=b['outside_foreground_color_pixels']-a['outside_foreground_color_pixels']))
    comparisons.append(dict(baseline=baseline,candidate='offset-index',pairs=75,foreground_improved=sum(v['foreground_psnr_delta']>0 for v in values),foreground_mean_delta=float(np.mean([v['foreground_psnr_delta'] for v in values])),worst_pair=min(values,key=lambda v:v['foreground_psnr_delta']),best_pair=max(values,key=lambda v:v['foreground_psnr_delta']),by_camera={c:float(np.mean([v['foreground_psnr_delta'] for v in values if v['camera']==c])) for c in r['cameras']},values=values))
bin=Path('C:/Users/Administrator/ffmpeg/bin');videos=[]
for cam in r['cameras']:
    path=out/'comparison'/f'camera{cam}-continuous-comparison.mp4'
    spec=json.loads(subprocess.check_output([str(bin/'ffprobe.exe'),'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]
    assert int(spec['nb_read_frames'])==25
    subprocess.run([str(bin/'ffmpeg.exe'),'-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True)
    videos.append(dict(file=path.relative_to(out).as_posix(),**spec,full_decode_exit_code=0))
result=dict(status='paired_statistics_and_all3videos_verified_visual_review_separate',models=r['summary'],comparisons=comparisons,videos=videos,limits=r['scope'])
(out/'paired-statistics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(models=r['summary'],comparisons=[{k:v for k,v in c.items() if k!='values'} for c in comparisons],videos=videos),indent=2))
