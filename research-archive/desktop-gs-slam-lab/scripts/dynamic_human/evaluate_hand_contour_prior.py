"""Native source-photo qualification of paired contour-refined MANO candidates."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from fit_hand_contour_prior import coverage
from fit_hamer_consensus import SharedHand
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-hand-contour-prior-102';E=O/'evaluation';E.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tt=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
assert not (E/'summary.json').exists();torch.set_num_threads(4);protocol=read(O/'protocol.json');z=np.load(H/'all_views/initialization.npz');B=tt(z['B']);W=tt(z['w2c']);K=tt(z['K']);font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
sources=dict(initial=H/'all_views/parameters.npz',keypoint_control=O/'keypoint_control/parameters.npz',contour_candidate=O/'contour_candidate/parameters.npz');before={k:sha(p) for k,p in sources.items()};initial=dict(np.load(sources['initial']));pred={};records=[];panels=[]
for branch in ['keypoint_control','contour_candidate']:
 assert read(O/branch/'summary.json')['steps']==1500
assert np.array_equal(np.load(O/'keypoint_control/schedule.npy'),np.load(O/'contour_candidate/schedule.npy'))
with torch.no_grad():
 for branch,path in sources.items():
  state=dict(np.load(path));model=SharedHand(state);model.requires_grad_(False);J,V,_,_=model(B);q=J@W[:,:3,:3].transpose(-1,-2)+W[:,None,:3,3];vv=q@K.transpose(-1,-2);uv=(vv[:,:,:2]/vv[:,:,2:]).cpu().numpy();faces=torch.tensor(model.mano.faces,dtype=torch.long,device='cuda');arrays={}
  assert np.array_equal(state['betas'],initial['betas']) and np.array_equal(state['log_scale'],initial['log_scale'])
  for r in protocol['records']:
   p=O/r['input'].replace('\\','/');assert sha(p)==r['input_sha256'];a=np.load(p);i=r['source_index'];alpha,_=coverage(V[i],faces,tt(a['w2c']),tt(a['K']));alpha=alpha.cpu().numpy();assert np.isfinite(alpha).all() and alpha.min()>=0 and alpha.max()<=1.00001
   out=E/f"{branch}-camera{r['camera']}-frame{r['frame']:06d}.npz";np.savez_compressed(out,alpha=alpha,joints_projected_xy=uv[i],bbox=a['bbox']);arrays[r['camera'],r['frame']]=alpha
  pred[branch]=dict(alpha=arrays,uv=uv,parameters_sha256=before[branch]);print('PRIOR_EVALUATION',branch,flush=True)
 for c in sorted({r['camera'] for r in protocol['records']}):
  im=Image.new('RGB',(640,404),'white');draw=ImageDraw.Draw(im)
  for row,f in enumerate([730,732]):
   r=next(r for r in protocol['records'] if r['camera']==c and r['frame']==f);a=np.load(O/r['input'].replace('\\','/'));rgb=a['rgb'];pos=a['positive'];neg=a['negative'];known=pos|neg;metrics={};images=[rgb]
   for b in sources:
    alpha=pred[b]['alpha'][c,f];v=np.clip(alpha,1e-5,1-1e-5);mask=alpha>=.5;metrics[b]=dict(positive_BCE=float(-np.log(v[pos]).mean()),negative_BCE=float(-np.log1p(-v[neg]).mean()),positive_coverage_fraction=float(mask[pos].mean()),negative_leak_fraction=float(mask[neg].mean()),known_domain_IoU=float((mask&pos).sum()/max(int(((mask&known)|pos).sum()),1)))
    over=rgb.copy();contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,contours,-1,(0,190,255),1);images.append(over)
   records.append(dict(camera=c,frame=f,mask_fit=r['fit'],metrics=metrics))
   for col,(arr,label) in enumerate(zip(images,['Source','Initial MANO contour','Keypoint control','Contour candidate'])):
    y=row*202;draw.text((col*160+3,y+3),label,font=font,fill='black');draw.text((col*160+3,y+19),f"{c}/{f} | {'Mask fit' if r['fit'] else 'Mask diagnostic'}",font=font,fill='black');im.paste(Image.fromarray(arr),(col*160,y+42))
  path=E/f'camera{c}-contours.png';im.save(path);panels.append(dict(camera=c,file=path.name,sha256=sha(path)))
errors={b:np.linalg.norm(pred[b]['uv']-z['xy'],axis=-1) for b in sources}
aggregate={}
for split in ['mask_fit','mask_diagnostic']:
 rr=[r for r in records if r['mask_fit']==(split=='mask_fit')];aggregate[split]={b:{k:float(np.mean([r['metrics'][b][k] for r in rr])) for k in rr[0]['metrics'][b]} for b in sources}
summary=dict(status='22native_contour_contexts_and105keypoint_predictions_evaluated_visual_review_pending',sources=before,records=records,panels=panels,aggregate=aggregate,
 candidate_keypoint_residual={b:dict(median_px=float(np.median(a)),p90_px=float(np.quantile(a,.9))) for b,a in errors.items()},
 source_parameters_unchanged=all(sha(path)==before[b] for b,path in sources.items()),matched1500steps_and_schedule=True,source_photo_RGB_unchanged=True,
 scope='04/13diagnostic masks excludedfromnewmaskloss, buttheirHaMeRpredictions alreadyusedinbothbranches. ThisisnotindependentcameraorRGBreconstructionvalidation. ContoursarethresholdedGaussiancoverage oftheMANOmesh, notrenderedskintexture. Original079andallGaussianappearanceparametersunchanged.',goal_complete=False)
assert summary['source_parameters_unchanged'];save(E/'summary.json',summary);shutil.copy2(__file__,E/Path(__file__).name);print(json.dumps(dict(aggregate=aggregate,candidate_keypoint_residual=summary['candidate_keypoint_residual'])),flush=True)
