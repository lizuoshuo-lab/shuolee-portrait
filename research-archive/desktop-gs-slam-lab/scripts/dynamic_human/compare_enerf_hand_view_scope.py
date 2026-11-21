from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';O=L/'outputs/dynamic-human-enerf-hand-view-scope-089';R=O/'comparison';R.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
branches=['parent','single_time','single_camera'];roots=dict(parent=U/'parent',single_time=U/'single_time',single_camera=O/'single_camera');labels=['Original','Parent','15-view fit','1-view fit'];reports={b:{r['camera']:r for r in read(root/'native-evaluation/metrics.json')['records'] if r['frame']==732} for b,root in roots.items()};inputs={r['camera']:r for r in read(U/'evaluation-protocol.json')['records'] if r['frame']==732};crops={r['camera']:r['crop'] for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records'] if r['frame']==732};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);rows=[];panels=[];pngchecks=0;maxerr=0.
def panel(c,name,bb,arrays):
 x0,y0,x1,y1=map(int,bb);w=x1-x0;h=y1-y0;im=Image.new('RGB',(w*4,h+32),'white');d=ImageDraw.Draw(im)
 for k,(a,label) in enumerate(zip(arrays,labels)):im.paste(Image.fromarray(a[y0:y1,x0:x1]),(k*w,32));d.text((k*w+4,8),label,font=font,fill='black')
 p=R/f'camera{c}-frame000732-{name}.png';im.save(p);actual=np.array(Image.open(p))
 for k,a in enumerate(arrays):assert np.array_equal(actual[32:,k*w:(k+1)*w],a[y0:y1,x0:x1])
 panels.append(dict(file=p.name,sha256=sha(p),camera=c,bbox=list(map(int,bb))))
for c,r in sorted(inputs.items()):
 stem=f'camera{c}-frame000732';ip=(U if r.get('input_location')=='088' else S)/'native-input'/f'{stem}.npz';assert sha(ip)==r['input_sha256'];z=np.load(ip);matte=z['matte'];valid=z['valid'];target=z['rgb'].astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]);bb=np.zeros_like(valid);x0,y0,x1,y1=z['bbox'];bb[y0:y1,x0:x1]=True;masks=dict(full_valid=valid,foreground=valid&(matte>.5),foreground_bbox=valid&bb)
 if c in crops:
  xx0,yy0,xx1,yy1=crops[c];roi=np.zeros_like(valid);roi[yy0:yy1,xx0:xx1]=True;masks['hand_context']=roi&masks['foreground']
 metrics={};arrays=[z['rgb']]
 for b in branches:
  rr=reports[b][c];er=S/'local_bspline/native-evaluation' if rr['reused079parent_prediction'] else roots[b]/'native-evaluation';p=np.load(er/f'{stem}-prediction.npz');pred=np.clip(p['rgb']+.5*(1-p['alpha'][...,None]),0,1);metrics[b]={}
  for name,m in masks.items():
   dif=pred.astype(np.float64)[m]-target.astype(np.float64)[m];v=dict(psnr_db=float(-10*np.log10(np.mean(dif*dif))),mae=float(abs(dif).mean()),pixels=int(m.sum()));metrics[b][name]=v
   if name!='hand_context':
    delta=max(abs(v[k]-rr['metrics'][name][k]) for k in ['psnr_db','mae']);maxerr=max(maxerr,delta);assert delta<1e-9
  for suffix,a in [('trained',u8(pred)),('alpha',u8(p['alpha']))]:assert np.array_equal(np.array(Image.open(er/f'{stem}-{suffix}.png')),a);pngchecks+=1
  arrays.append(u8(pred))
 if c in crops:panel(c,'hand',crops[c],arrays)
 if c in ['00','04','06','12']:panel(c,'full-person',z['bbox'],arrays)
 rows.append(dict(camera=c,frame=732,fitted_in089=c=='04',unseen_in079_088=c in ['00','06','12'],metrics=metrics));print('VIEW_SCOPE_COMPARE',c,flush=True)
assert pngchecks==108 and len(rows)==18
a,b=[dict(np.load(p/'schedule.npz')) for p in [U/'single_time',O/'single_camera']];assert np.array_equal(a['background'],b['background']) and np.array_equal(a['frame'],b['frame']) and set(b['camera'])=={'04'}
audit=read(O/'single_camera/frozen-state-audit.json');assert audit['all384motion_nodes_exact'] and audit['allbindings_exact'] and audit['nonhand_Gaussians']==246654 and audit['all_optimizer_steps']==3000;assert sha(O/'single_camera/resume.pt')==audit['checkpoint_sha256']
summary={label:{b:{metric:float(np.mean([r['metrics'][b][metric]['psnr_db'] for r in selected if metric in r['metrics'][b]])) for metric in ['foreground','hand_context'] if any(metric in r['metrics'][b] for r in selected)} for b in branches} for label,selected in [('fitted_camera',[r for r in rows if r['camera']=='04']),('other14_known_cameras',[r for r in rows if r['camera']!='04' and not r['unseen_in079_088']]),('three_previously_heldout_cameras',[r for r in rows if r['unseen_in079_088']])]}
handfiles=sorted(R.glob('*-hand.png'));sheets=[]
for i in range(0,len(handfiles),3):
 ims=[Image.open(p) for p in handfiles[i:i+3]];im=Image.new('RGB',(640,sum(x.height for x in ims)+16),'white');y=0
 for x in ims:im.paste(x,(0,y));y+=x.height+8
 p=R/f'hand-review-{i//3+1:02d}.png';im.save(p);sheets.append(dict(file=p.name,sources=[f.name for f in handfiles[i:i+3]]))
save(R/'records.json',dict(records=rows));save(R/'summary.json',dict(status='18nativeviews108PNGs_audited_visualreview_pending',metrics=summary,PNG_checks=pngchecks,max_recomputed_metric_difference=maxerr,matched_background_LR_steps_parent_to088=True,only_camera_sampling_changes=True,checkpoint_sha256=audit['checkpoint_sha256'],panels=panels,review_sheets=sheets,goal_complete=False));print(json.dumps(summary),flush=True)
