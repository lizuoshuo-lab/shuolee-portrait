"""Prepare conservative photograph-derived hand constraints, not finger GT."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';M=L/'outputs/dynamic-human-enerf-open-hand-flow-083/hand-masks';O=L/'outputs/dynamic-human-hand-contour-prior-102';O.mkdir(exist_ok=True);(O/'inputs').mkdir(exist_ok=True);(O/'review').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'protocol.json').exists()
cameras=['01','03','04','05','08','09','10','11','13','15','17'];diagnostic=['04','13'];frames=[730,732]
source=np.load(H/'all_views/predictions.npz');lookup={(str(c),int(f)):i for i,(c,f) in enumerate(zip(source['camera'],source['frame']))};prior=np.load(H/'all_views/initialization.npz');assert np.allclose(source['capture_time'],prior['t'])
meta={(r['camera'],r['frame']):r for r in read(M/'completion.json')['records']};out=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
for c in cameras:
 im=Image.new('RGB',(640,404),'white');draw=ImageDraw.Draw(im)
 for k,f in enumerate(frames):
  r=meta[c,f];mp=M/r['filename'];assert sha(mp)==r['sha256'];m=np.load(mp);mask=m['mask'];box=m['crop'];x0,y0,x1,y1=box;p=S/'native-input'/f'camera{c}-frame{f:06d}.npz';assert sha(p)==r['input_sha256'];z=np.load(p);rgb=z['rgb'][y0:y1,x0:x1];i=lookup[c,f];uv=source['projected_xy'][i]-[x0,y0];wrist=uv[0];palm=uv[[5,9,13,17]].mean(0);direction=palm-wrist;direction/=np.linalg.norm(direction)
  yy,xx=np.mgrid[:160,:160];xy=np.stack([xx+.5,yy+.5],-1);distal=((xy-wrist)*direction).sum(-1)>5
  near=cv2.dilate(mask.astype(np.uint8),np.ones((41,41),np.uint8)).astype(bool)
  positive=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&distal
  negative=~cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)&distal&near
  positive&=z['valid'][y0:y1,x0:x1];negative&=z['valid'][y0:y1,x0:x1];assert positive.sum()>50 and negative.sum()>50 and not (positive&negative).any()
  K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0
  stem=f'camera{c}-frame{f:06d}';dest=O/'inputs'/f'{stem}.npz';np.savez_compressed(dest,rgb=rgb,positive=positive,negative=negative,mask=mask,distal=distal,bbox=box,K=K,w2c=z['w2c'],capture_time=source['capture_time'][i],source_index=i)
  row=dict(camera=c,frame=f,source_index=i,capture_time=float(source['capture_time'][i]),fit=c not in diagnostic,input=str(dest.relative_to(O)),input_sha256=sha(dest),mask_sha256=sha(mp),source_sha256=sha(p),positive_pixels=int(positive.sum()),negative_pixels=int(negative.sum()));out.append(row)
  arr=[rgb,np.rint(rgb*mask[...,None]+128*(1-mask[...,None])).astype(np.uint8),rgb.copy(),rgb.copy()];arr[2][positive]=(0.5*arr[2][positive]+[0,100,0]).astype(np.uint8);arr[3][negative]=(0.5*arr[3][negative]+[80,0,100]).astype(np.uint8)
  for j,(a,label) in enumerate(zip(arr,['Source','SAM candidate','Positive interior','Negative outside'])):
   y=k*202;draw.text((j*160+3,y+3),f'{c}/{f} | {label}',font=font,fill='black');draw.text((j*160+3,y+19),'MASK DIAGNOSTIC' if c in diagnostic else 'MASK FIT',font=font,fill='black');im.paste(Image.fromarray(a),(j*160,y+42))
 im.save(O/'review'/f'camera{c}-constraints.png')
assert len(out)==22 and sum(r['fit'] for r in out)==18
save(O/'protocol.json',dict(status='photo_constraints_prepared_visual_check_required_before_training',records=out,
    initial_parameters_sha256=sha(H/'all_views/parameters.npz'),initial_keypoint_inputs_sha256=sha(H/'all_views/initialization.npz'),
    paired_branches=['keypoint_control','contour_candidate'],steps_each=1500,seed=10201,
    fitting_cameras=[c for c in cameras if c not in diagnostic],mask_diagnostic_cameras=diagnostic,frames=frames,
    input_scope='Use already-reviewed083SAM proposals only at730/732. Neither masks nor HaMeR joints are GT. Exclude728missingfingers,14/16toycontamination,andunreviewed02/07. All105originalsoftkeypointobservations remain identical inbothbranches.04/13heldoutfrommaskloss only,notfromprior2Dkeypoints.',
    domains='1pxerodedpositiveinterior;negativeoutside3pxdilationwithin20pxmaskneighborhood. Both restricted todistalhalfplane5pxpastfrozen097wrist alongwrist-toMCPdirection, avoiding uncertain artificialwristcut. Missingfingerswithin3pxboundary band areunknown,not negatives.',
    model='Same9continuouscubicB-splinecontrols for16rotationvectors andwristtranslation. FixedMANOshape/scale from097;posecoefficients boundedwithin0.35radandtranslation0.04*scene_scale ofstart. No body/camera/time/GS changes.',
    contour_surrogate='1538trianglecentroidGaussians, fixedalpha0.8, tangentstd0.55edgeRMS,normalstd0.0825edgeRMS; no learnablecolors/alpha/offsets. Approximate differentiable handcoverage, not photorealistic reconstruction.',
    loss='OriginalrobustHaMeRreprojection andsamecontinuoustemporalregularizers plus8*(mean-positive negative-log-alpha +0.25mean-negative negative-log-one-minus-alpha) in candidateonly. Persteponeof18maskfitcontexts fromfixedbalancedpermutation. OriginalimagesandheldoutRGBassessmentunchanged.',
    verification='RequirefiniteCUDAgradientsandnativeconstraintreview; compareposeprojectionandsurrogatealpha onall22contexts,including4maskdiagnostics, and105keypointsbeforeanyGaussianreconstructionclaim.',goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name);print('Prepared22photo constraint contexts;18fit,4maskdiagnostics; no training.')
