"""Independently check native outputs and complete25-frame sequences."""
from pathlib import Path
import json,hashlib,subprocess,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';T=O/'continuity';T.mkdir(exist_ok=True);P=T/'panels';P.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (T/'summary.json').exists();plan=read(I/'continuity-protocol.json');B=['dynamic_clock','path_rigidity','path_rigidity_binding'];roots={b:(I/b if b=='dynamic_clock' else O/b) for b in B};labels=['Original','Euclidean graph','Surface-path edges','Surface-path edges + binding'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
reports={b:{(r['camera'],r['frame']):r for folder in ['native-evaluation','continuity-extra'] for r in read(roots[b]/folder/'metrics.json')['records']} for b in B};before={b:sha(roots[b]/'resume.pt') for b in B};records=[];transitions=[];panels=[];videos=[];checks=0
for c,sequence in plan['sequences'].items():
    x0,y0,x1,y1=sequence['full_person_union_bbox'];w=x1-x0;h=y1-y0;previous=None
    for r in sequence['records']:
        f=r['frame'];stem=f'camera{c}-frame{f:06d}';assert sha(I/r['input'])==r['input_sha256'];z=np.load(I/r['input']);rgb=z['rgb'];matte=z['matte'];valid=z['valid'];target=rgb.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]);preds={};arrays=[rgb];m=valid&(matte>.5);vals={}
        for b in B:
            er=roots[b]/r['prediction_directory'];mr=reports[b][c,f];assert mr['input_sha256']==r['input_sha256']
            for file,digest in mr['output_sha256'].items():assert sha(er/file)==digest
            q=np.load(er/f'{stem}-prediction.npz');p=np.clip(q['rgb']+.5*(1-q['alpha'][...,None]),0,1);preds[b]=p;arrays.append(u8(p))
            for suffix,arr in [('trained',u8(p)),('alpha',u8(q['alpha']))]:assert np.array_equal(np.array(Image.open(er/f'{stem}-{suffix}.png')),arr);checks+=1
            e=p.astype(np.float64)[m]-target.astype(np.float64)[m];vals[b]=dict(psnr_db=float(-10*np.log10(np.mean(e*e))),mae=float(abs(e).mean()));assert abs(vals[b]['psnr_db']-mr['metrics']['foreground']['psnr_db'])<1e-9
        records.append(dict(camera=c,frame=f,split=r['split'],metrics=vals))
        if previous is not None:
            assert f==previous['frame']+1;mask=valid&previous['valid']&((matte>.5)|(previous['matte']>.5));dt=(target-previous['target']).astype(np.float64)
            transitions.append(dict(camera=c,frames=[f-1,f],pixels=int(mask.sum()),mean_absolute_difference_residual={b:float(abs((preds[b]-previous['preds'][b]).astype(np.float64)[mask]-dt[mask]).mean()) for b in B}))
        previous=dict(frame=f,valid=valid,matte=matte,target=target,preds=preds)
        im=Image.new('RGB',(w*4,h+32),'white');draw=ImageDraw.Draw(im)
        for k,(arr,label) in enumerate(zip(arrays,labels)):
            im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(k*w,32));draw.text((k*w+4,7),f'{label} | {f}',font=font,fill='black')
        path=P/f'{stem}.png';im.save(path);actual=np.array(Image.open(path))
        for k,arr in enumerate(arrays):assert np.array_equal(actual[32:,k*w:(k+1)*w],arr[y0:y1,x0:x1])
        panels.append(dict(camera=c,frame=f,file=path.relative_to(T).as_posix(),sha256=sha(path),bbox=sequence['full_person_union_bbox']))
        save(T/'progress.json',dict(images=len(records),PNG_checks=checks,consecutive_transitions=len(transitions)));print('ACTOR23_CONTINUITY',c,f,flush=True)
    ff=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe');fp=ff.with_name('ffprobe.exe');out=T/f'camera{c}-native-continuity.mp4'
    assert not out.exists()
    subprocess.run([str(ff),'-v','error','-n','-framerate','5','-start_number','600','-i',str(P/f'camera{c}-frame%06d.png'),'-frames:v','25','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True,timeout=120)
    info=json.loads(subprocess.check_output([str(fp),'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(out)]))['streams'][0]
    assert int(info['nb_read_frames'])==25 and info['r_frame_rate']=='5/1'
    subprocess.run([str(ff),'-v','error','-i',str(out),'-f','null','-'],check=True,timeout=90)
    videos.append(dict(file=out.name,sha256=sha(out),**info,scope='Native pixel panels, only even-dimension padding;H264lossy viewingcopy;5fpsslowreview,notcaptureframe-rate.'))
assert checks==300 and len(records)==50 and len(transitions)==48 and all(sha(roots[b]/'resume.pt')==before[b] for b in B)
save(T/'records.json',dict(records=records));save(T/'transition-records.json',dict(records=transitions));save(T/'summary.json',dict(status='50native_panels_300PNGs_48consecutive_transitions_2videos_verified_visual_review_required',metrics={c:{b:float(np.mean([r['metrics'][b]['psnr_db'] for r in records if r['camera']==c])) for b in B} for c in plan['sequences']},temporal_difference_residual={c:{b:float(np.mean([r['mean_absolute_difference_residual'][b] for r in transitions if r['camera']==c])) for b in B} for c in plan['sequences']},panels=panels,videos=videos,PNG_checks=checks,checkpoints=before,scope=plan['scope'],goal_complete=False));shutil.copy2(__file__,T/Path(__file__).name);print('ACTOR23_CONTINUITY_AUDIT_COMPLETE',flush=True)
