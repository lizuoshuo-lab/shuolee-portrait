"""CPU/source readback of all saved RGB/alpha outputs, including alternate gradients."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,cv2
from PIL import Image
from skimage.metrics import structural_similarity
from enerf_native_loader import EnerfNativeDataset

ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['heldout_view','train'],default='heldout_view');ap.add_argument('--camera');args=ap.parse_args()
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-support-030';dest=out/('evaluation' if args.split=='heldout_view' else 'training-evaluation');m=json.loads((dest/'metrics.json').read_text());source=out.parent/'dynamic-human-enerf-local-time-029';ds=EnerfNativeDataset(source/'offset-index/appearance-manifest.json',args.split);lookup={(r['camera'],r['common_frame']):i for i,r in enumerate(ds.rows)};rows={(r['camera'],r['common_frame'],r['model']):r for r in m['views']};checks=[];temporal=[];previous={};max_ssim_difference=0.
def gradient(rgb):
    x=rgb.astype('float64')/255;gray=x@np.array([.299,.587,.114]);v=np.pad(gray,1,mode='reflect')
    dx=(-v[:-2,:-2]+v[:-2,2:]-2*v[1:-1,:-2]+2*v[1:-1,2:]-v[2:,:-2]+v[2:,2:])/8
    dy=(-v[:-2,:-2]-2*v[:-2,1:-1]-v[:-2,2:]+v[2:,:-2]+2*v[2:,1:-1]+v[2:,2:])/8
    return np.stack([dx,dy],-1)
selected_cameras=m['cameras'] if args.camera is None else [args.camera]
assert all(c in m['cameras'] for c in selected_cameras)
for cam in selected_cameras:
    for t in m['frames']:
        o=ds[lookup[cam,t]];stem=f'camera{cam}-frame{t:03d}';matte=o['foreground_matte'];valid=o['valid_pixels'];gt=np.array(Image.open(dest/f'{stem}-reference.png'));expected=np.rint((o['rgb']/255*matte[...,None]+.5*(1-matte[...,None]))*255).clip(0,255).astype('uint8');assert np.array_equal(gt,expected)
        if args.split=='heldout_view':assert np.array_equal(gt,np.array(Image.open(source/'evaluation'/f'{stem}-reference.png')))
        assert np.array_equal(np.array(Image.open(dest/f'{stem}-matte.png')),np.rint(matte*255).astype('uint8'));inside=(matte>.9)&valid;sil=(matte>.5)&valid;outside=(cv2.dilate((matte>.1).astype('uint8'),np.ones((33,33),'uint8'))==0)&valid;gg=gradient(gt)
        for g in m['models']:
            row=rows[cam,t,g];pr=np.array(Image.open(dest/f'{stem}-{g}.png'));alpha=np.array(Image.open(dest/f'{stem}-{g}-alpha.png'));err=((pr.astype(float)-gt)/255)**2;ge=np.abs(gradient(pr)-gg)
            assert abs(-10*np.log10(err[inside].mean())-row['foreground_psnr_db'])<1e-9
            assert abs(-10*np.log10(err[valid].mean())-row['full_canvas_psnr_db'])<1e-9
            assert abs(ge[inside].mean()-row['foreground_gradient_mae'])<1e-7
            predsil=(alpha>=128)&valid;assert abs((sil&predsil).sum()/(sil|predsil).sum()-row['silhouette_iou'])<1e-12
            assert abs(np.abs(alpha/255-matte)[sil].mean()-row['alpha_foreground_mae'])<1e-12
            assert abs((alpha[outside].astype(float)/255).sum()-row['outside_alpha_mass_pixels'])<1e-9
            assert int((np.max(np.abs(pr.astype(float)/255-.5),axis=2)[outside]>.1).sum())==row['outside_foreground_color_pixels']
            for p in row['people']:
                x0,y0,x1,y1=p['bbox'];score=structural_similarity(pr[y0:y1,x0:x1]/255,gt[y0:y1,x0:x1]/255,channel_axis=2,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False);delta=abs(score-p['ssim']);max_ssim_difference=max(max_ssim_difference,delta);assert delta<1e-4
                assert abs(-10*np.log10(err[y0:y1,x0:x1].mean())-p['psnr_db'])<1e-9
                u0,v0,u1,v1=p['upper_body_bbox'];assert abs(-10*np.log10(err[v0:v1,u0:u1].mean())-p['upper_body_psnr_db'])<1e-9;assert abs(ge[v0:v1,u0:u1].mean()-p['upper_body_gradient_mae'])<1e-7
            if args.split=='heldout_view' and (cam,g) in previous:
                oldpr,oldgt,oldmask=previous[cam,g];mask=inside|oldmask;delta=(pr.astype(float)-oldpr)-(gt.astype(float)-oldgt);temporal.append(dict(camera=cam,model=g,from_frame=t-1,to_frame=t,mean_abs_change_residual=float(np.abs(delta[mask]).mean()/255)))
            previous[cam,g]=(pr,gt,inside);checks.append(dict(camera=cam,frame=t,model=g,foreground_psnr_db=row['foreground_psnr_db']))
    print('VERIFIED_CAMERA',cam,flush=True)
for rel,h in m['state_hashes'].items():assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==h
assert len(checks)==len(selected_cameras)*len(m['frames'])*4 and len(temporal)==(len(selected_cameras)*(len(m['frames'])-1)*4 if args.split=='heldout_view' else 0)
r=dict(status=f'{len(checks)}RGB_and_alpha_outputs_independently_verified',split=args.split,cameras=selected_cameras,images=len(checks),alpha_images=len(checks),references=len(selected_cameras)*len(m['frames']),person_regions=2*len(checks),temporal_transitions=len(temporal),max_ssim_implementation_difference=max_ssim_difference,checks=checks,temporal=temporal,temporal_summary={g:float(np.mean([r['mean_abs_change_residual'] for r in temporal if r['model']==g])) for g in m['models']} if temporal else {},methods='Fresh source RGB/matte; heldout references additionally byte-matched to029. CPU float64 RGB, separate NumPy Sobel stencil, skimage SSIM, threshold silhouette and alpha residuals, all model hashes.',limits='Author mattes include held toys and boundary errors; gradient difference is fidelity not sharpness score. Adjacent image change residual has no flow compensation and is not pure flicker or true3Dmotion accuracy. Train probes are supervised fit only.')
name='independent-image-audit' if args.split=='heldout_view' else 'independent-training-image-audit'
if args.camera:name+=f'-camera{args.camera}'
(out/f'{name}.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=r['status'],temporal_summary=r['temporal_summary']),indent=2))
