from pathlib import Path
import json,subprocess,shutil,argparse
import numpy as np
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,default=ROOT);ap.add_argument('--before-root',type=Path,default=ROOT/'fixed_offsets');ap.add_argument('--after-root',type=Path,default=ROOT/'estimated_offsets');ap.add_argument('--before-label',default='Same frame-index');ap.add_argument('--after-label',default='Capture-time hypothesis');args=ap.parse_args();branches={'before':args.before_root,'after':args.after_root};OUT=args.output_root/'temporal-comparison';OUT.mkdir(exist_ok=True);read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert (args.output_root/'paired-comparison/summary.json').exists();records=read(ROOT/'input-audit.json')['records'];cameras=sorted({r['camera'] for r in records});transitions=[];videos=[];ff=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe');fp=ff.with_name('ffprobe.exe');frames=[745,747,749,751,753,755]
for c in cameras:
 seq=[]
 for t in frames:
  prefix=f'camera{c}-frame{t:06d}';z=np.load(ROOT/'native-input'/f'{prefix}.npz');ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);rr={}
  for branch,path in branches.items():
   p=np.load(path/'native-evaluation'/f'{prefix}-prediction.npz');rr[branch]=np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1)
  seq.append(dict(frame=t,rgb=z['rgb'],ref=ref,matte=z['matte'],valid=z['valid'],bbox=z['bbox'],**rr))
 for a,b in zip(seq,seq[1:]):
  valid=a['valid']&b['valid']&((a['matte']>.5)|(b['matte']>.5));target=(b['ref']-a['ref']).astype(np.float64);scores={}
  for branch in branches:scores[branch]=float(np.abs(((b[branch]-a[branch]).astype(np.float64)-target)[valid]).mean())
  transitions.append(dict(camera=c,frames=[a['frame'],b['frame']],foreground_union_pixels=int(valid.sum()),mean_absolute_temporal_difference_residual=scores))
 if c in ['01','04','08','17']:
  boxes=np.array([r['bbox'] for r in seq]);x0,y0=boxes[:,:2].min(0);x1,y1=boxes[:,2:].max(0);w=int(x1-x0);h=int(y1-y0);cw=w+w%2;hh=h+40+(h%2);frameout=OUT/f'camera{c}-frames';frameout.mkdir(exist_ok=True)
  for i,r in enumerate(seq):
   im=Image.new('RGB',(3*cw,hh),'white');d=ImageDraw.Draw(im)
   for j,(label,key) in enumerate([('Original photograph','rgb'),(args.before_label,'before'),(args.after_label,'after')]):
    arr=r[key] if key=='rgb' else u8(r[key]);im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(j*cw,40));d.text((j*cw+4,10),f"{label} | camera {c} | frame {r['frame']} | reserved time",fill='black')
   path=frameout/f'frame{i:02d}.png';im.save(path);saved=np.array(Image.open(path))
   for j,key in enumerate(['rgb','before','after']):
    arr=r[key] if key=='rgb' else u8(r[key]);assert np.array_equal(saved[40:40+h,j*cw:j*cw+w],arr[y0:y1,x0:x1])
  video=OUT/f'camera{c}-reserved-time-comparison.mp4';subprocess.run([str(ff),'-v','error','-y','-framerate','4','-i',str(frameout/'frame%02d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
  probe=json.loads(subprocess.check_output([str(fp),'-v','error','-count_frames','-show_streams','-of','json',str(video)]));stream=next(s for s in probe['streams'] if s['codec_type']=='video');assert int(stream['nb_read_frames'])==6;subprocess.run([str(ff),'-v','error','-i',str(video),'-f','null','-'],check=True);videos.append(dict(camera=c,file=video.name,frames=6,width=int(stream['width']),height=int(stream['height']),display_fps=4,real_capture_fps_certified=False,PNG_comparison_frames='Nativepixelcropsverified;videoH264lossyencodingforplayback'))
 save(OUT/'progress.json',dict(cameras=len({r['camera'] for r in transitions}),transitions=len(transitions),videos=len(videos)))
save(OUT/'summary.json',dict(status='75native_reserved_time_transition_residuals_and4decoded_comparison_videos_verified',transitions=transitions,videos=videos,
 mean_temporal_residual={branch:float(np.mean([r['mean_absolute_temporal_difference_residual'][branch] for r in transitions])) for branch in branches},branch_paths={k:str(v) for k,v in branches.items()},
 qualification='Pixeltemporal-difference residual includesmotion/occlusion andmatteerror; notopticalflowcompensatedflicker or independenttemporalGT. Playback4fps isorderedinspection, notcertifiedcapturespeed.',goal_complete=False));shutil.copy2(__file__,args.output_root/Path(__file__).name);print('75 temporal differences and4decoded6frame comparisonvideos verified')
