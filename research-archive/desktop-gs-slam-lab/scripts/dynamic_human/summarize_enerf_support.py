"""Paired native metrics, baseline repeat sensitivity and decoded videos."""
from pathlib import Path
import json,subprocess
import numpy as np
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-support-030';r=json.loads((out/'evaluation/metrics.json').read_text());rows={(v['camera'],v['common_frame'],v['model']):v for v in r['views']};old=json.loads((out.parent/'dynamic-human-enerf-local-time-029/evaluation/metrics.json').read_text());oldrows={(v['camera'],v['common_frame']):v for v in old['views'] if v['model']=='offset-index'};comparisons=[];repeat=[]
for g in r['models'][1:]:
    values=[]
    for c in r['cameras']:
        for t in r['frames']:
            a=rows[c,t,'frozen'];b=rows[c,t,g];values.append(dict(camera=c,frame=t,foreground_psnr_delta=b['foreground_psnr_db']-a['foreground_psnr_db'],upper_body_psnr_delta=float(np.mean([p['upper_body_psnr_db'] for p in b['people']])-np.mean([p['upper_body_psnr_db'] for p in a['people']])),foreground_gradient_mae_delta=b['foreground_gradient_mae']-a['foreground_gradient_mae'],silhouette_iou_delta=b['silhouette_iou']-a['silhouette_iou'],outside_alpha_mass_delta=b['outside_alpha_mass_pixels']-a['outside_alpha_mass_pixels']))
    comparisons.append(dict(candidate=g,baseline='frozen',pairs=75,psnr_improved=sum(v['foreground_psnr_delta']>1e-8 for v in values),psnr_unchanged=sum(abs(v['foreground_psnr_delta'])<=1e-8 for v in values),gradient_error_reduced=sum(v['foreground_gradient_mae_delta']<-1e-8 for v in values),foreground_mean_delta=float(np.mean([v['foreground_psnr_delta'] for v in values])),silhouette_mean_delta=float(np.mean([v['silhouette_iou_delta'] for v in values])),gradient_mean_delta=float(np.mean([v['foreground_gradient_mae_delta'] for v in values])),outside_alpha_mass_mean_delta=float(np.mean([v['outside_alpha_mass_delta'] for v in values])),worst_psnr_pair=min(values,key=lambda v:v['foreground_psnr_delta']),by_camera={c:float(np.mean([v['foreground_psnr_delta'] for v in values if v['camera']==c])) for c in r['cameras']},values=values))
for c in r['cameras']:
    for t in r['frames']:
        a=oldrows[c,t];b=rows[c,t,'frozen'];assert a['source_sha256']==b['source_sha256'];repeat.append(dict(camera=c,frame=t,foreground_psnr_delta=b['foreground_psnr_db']-a['foreground_psnr_db']))
binary=Path('C:/Users/Administrator/ffmpeg/bin');videos=[]
for c in r['cameras']:
    path=out/'comparison'/f'camera{c}-continuous-comparison.mp4';s=json.loads(subprocess.check_output([str(binary/'ffprobe.exe'),'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0];assert int(s['nb_read_frames'])==25;subprocess.run([str(binary/'ffmpeg.exe'),'-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True);videos.append(dict(file=path.relative_to(out).as_posix(),**s,full_decode_exit_code=0))
result=dict(status='four_condition_descriptive_statistics_and3videos_verified_visual_review_separate',models=r['summary'],comparisons=comparisons,baseline_repeat=dict(scope='Same-seed frozen run compared with029, descriptive numerical sensitivity, not multi-seed robustness',mean_delta=float(np.mean([v['foreground_psnr_delta'] for v in repeat])),mean_absolute_delta=float(np.mean([abs(v['foreground_psnr_delta']) for v in repeat])),max_absolute_delta=float(max(abs(v['foreground_psnr_delta']) for v in repeat)),pairs=repeat),videos=videos,limits=r['scope'])
(out/'paired-statistics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(models=result['models'],comparisons=[{k:v for k,v in c.items() if k!='values'} for c in comparisons],baseline_repeat={k:v for k,v in result['baseline_repeat'].items() if k!='pairs'}),indent=2))
