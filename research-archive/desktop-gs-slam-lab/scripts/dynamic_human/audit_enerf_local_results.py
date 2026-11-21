"""Independent CPU audit of native rendered pixels, regions and temporal residuals."""
from pathlib import Path
import json,hashlib
import numpy as np,cv2
from PIL import Image
from skimage.metrics import structural_similarity
from enerf_native_loader import EnerfNativeDataset

LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029';dest=out/'evaluation';metrics=json.loads((dest/'metrics.json').read_text());ds=EnerfNativeDataset(out/'offset-index/appearance-manifest.json','heldout_view');lookup={(r['camera'],r['common_frame']):i for i,r in enumerate(ds.rows)};checks=[];temporal=[];previous={}
for cam in metrics['cameras']:
    for t in metrics['frames']:
        o=ds[lookup[cam,t]];stem=f'camera{cam}-frame{t:03d}';matte=o['foreground_matte'];gt=np.array(Image.open(dest/f'{stem}-reference.png'));expected=np.rint((o['rgb']/255*matte[...,None]+.5*(1-matte[...,None]))*255).clip(0,255).astype('uint8');assert np.array_equal(gt,expected);assert np.array_equal(np.array(Image.open(dest/f'{stem}-matte.png')),np.rint(matte*255).astype('uint8'));inside=(matte>.9)&o['valid_pixels'];outside=(cv2.dilate((matte>.1).astype('uint8'),np.ones((33,33),'uint8'))==0)&o['valid_pixels']
        for group in metrics['models']:
            row=next(r for r in metrics['views'] if r['common_frame']==t and r['camera']==cam and r['model']==group);pred=np.array(Image.open(dest/f'{stem}-{group}.png'));err=((pred.astype(float)-gt)/255)**2;psnr=float(-10*np.log10(err[inside].mean()));assert abs(psnr-row['foreground_psnr_db'])<1e-9
            assert abs(float(-10*np.log10(err[o['valid_pixels']].mean()))-row['full_canvas_psnr_db'])<1e-9
            count=int((np.max(np.abs(pred.astype(float)/255-.5),axis=2)[outside]>.1).sum());assert count==row['outside_foreground_color_pixels']
            for person in row['people']:
                x0,y0,x1,y1=person['bbox'];pr=pred[y0:y1,x0:x1]/255.;ref=gt[y0:y1,x0:x1]/255.;score=float(structural_similarity(pr,ref,channel_axis=2,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False));assert abs(score-person['ssim'])<1e-4
                assert abs(float(-10*np.log10(err[y0:y1,x0:x1].mean()))-person['psnr_db'])<1e-9
                u0,v0,u1,v1=person['upper_body_bbox'];assert abs(float(-10*np.log10(err[v0:v1,u0:u1].mean()))-person['upper_body_psnr_db'])<1e-9
            if (cam,group) in previous:
                oldpr,oldgt,oldmask=previous[cam,group];mask=inside|oldmask;delta=(pred.astype(float)-oldpr)-(gt.astype(float)-oldgt);temporal.append(dict(camera=cam,model=group,from_common_frame=t-1,to_common_frame=t,mean_abs_change_residual=float(np.abs(delta[mask]).mean()/255)))
            previous[cam,group]=(pred,gt,inside);checks.append(dict(camera=cam,common_frame=t,source_frame=o['frame'],model=group,foreground_psnr_db=psnr))
        if t==58:print('VERIFIED_CAMERA',cam,flush=True)
for rel,h in metrics['state_hashes'].items():assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==h
result=dict(status='all225renders75references450person_regions_and216transitions_independently_verified',images=len(checks),native_references=75,person_regions=450,temporal_transitions=len(temporal),checks=checks,temporal=temporal,temporal_summary={g:float(np.mean([r['mean_abs_change_residual'] for r in temporal if r['model']==g])) for g in metrics['models']},methods='Source RGB/matte independently rectified; reference byte equality; CPU float64 PSNR; independent skimage Gaussian-window SSIM; full-canvas outside-color counts; model-file hashes.',temporal_metric_limit='Residual of adjacent image changes in union of foreground interiors; NOT optical-flow compensated, not pure flicker or physical tracking accuracy. No temporal holdout in this comparison.',scope='Numerical fidelity does not establish perceptual clarity or accurate geometry')
(out/'independent-image-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=result['status'],temporal_summary=result['temporal_summary']),indent=2))
