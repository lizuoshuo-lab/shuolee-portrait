"""Independent saved-image/source checks for the native ENeRF pilot."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np,cv2
from PIL import Image
from skimage.metrics import structural_similarity
from enerf_native_loader import EnerfNativeDataset
from audit_enerf_input import digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--frame',type=int,default=0);ap.add_argument('--source-frames');a=ap.parse_args();out=Path(a.out);first=out/'first-frame/evaluation';report=json.loads((first/'metrics.json').read_text());checks=[];cache_checks=[]
    source_frames=json.loads(Path(a.source_frames).read_text())['source_frames'] if a.source_frames else {}
    for split in ('train','heldout_view'):
        ds=EnerfNativeDataset(a.manifest,split)
        for i,r in enumerate(ds.rows):
            wanted=source_frames.get(r['camera'],a.frame) if split=='train' else a.frame
            matches=r['common_frame']==a.frame if 'common_frame' in r else r['frame']==wanted
            if not matches:continue
            if split=='train':
                o=ds[i];cache=np.load(out/f"train-camera-{r['camera']}.npz")
                for key,other in [('rgb','rgb'),('matte','foreground_matte'),('valid','valid_pixels'),('K','K'),('w2c','w2c')]:assert np.array_equal(cache[key],o[other])
                cache_checks.append(dict(camera=r['camera'],file_sha256=digest(out/f"train-camera-{r['camera']}.npz")))
            row=next((x for x in report['views'] if x['split']==split and x['camera']==r['camera']),None)
            if row is None:continue
            o=ds[i];name=f"{split}-camera{r['camera']}";pred=np.array(Image.open(first/f'{name}-render.png'));gt=np.array(Image.open(first/f'{name}-reference.png'))
            expected=np.clip((o['rgb']/255*o['foreground_matte'][...,None]+.5*(1-o['foreground_matte'][...,None]))*255+.5,0,255).astype('uint8');assert np.array_equal(gt,expected)
            e=((pred.astype(float)-gt)/255)**2;mask=(o['foreground_matte']>.9)&o['valid_pixels'];score=float(-10*np.log10(e[mask].mean()))
            # A fixed dB tolerance is inappropriate near high-PSNR predictions:
            # verify the reported float MSE lies inside exact 8-bit rounding
            # bounds, using the independently recomputed FLOAT source target.
            ref_float=o['rgb']/255*o['foreground_matte'][...,None]+.5*(1-o['foreground_matte'][...,None])
            lo=np.maximum(pred.astype(float)/255-.5/255,0);hi=np.minimum(pred.astype(float)/255+.5/255,1)
            min_error=np.maximum(np.maximum(lo-ref_float,ref_float-hi),0)**2
            max_error=np.maximum((lo-ref_float)**2,(hi-ref_float)**2)
            mse_bounds=[float(min_error[mask].mean()),float(max_error[mask].mean())];float_mse=10**(-row['foreground_psnr_db']/10)
            assert mse_bounds[0]-1e-10<=float_mse<=mse_bounds[1]+1e-10
            for person in row['people']:
                x0,y0,x1,y1=person['bbox'];pr=pred[y0:y1,x0:x1]/255.;ref=gt[y0:y1,x0:x1]/255.;cpu_ss=structural_similarity(pr,ref,channel_axis=2,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False);assert abs(cpu_ss-person['bbox_ssim'])<1e-4
                assert abs(float(-10*np.log10(((pr-ref)**2).mean()))-person['bbox_psnr_db'])<1e-6
            checks.append(dict(camera=r['camera'],split=split,psnr_db=score,saved_vs_float_psnr_difference=score-row['foreground_psnr_db'],float_mse_within_8bit_quantization_bounds=True,mse_rounding_bounds=mse_bounds,reference_exact_native_rectified_matte_composite=True,ssim_cpu_agrees=True))
    result=dict(status='saved_first_frame_metrics_sources_and_training_cache_independently_verified',native_first_frame_train_cache_count=len(cache_checks),training_cache=cache_checks,views=checks,methods='CPU float64 saved-PNG PSNR, independently implemented skimage SSIM, source RGB/matte remap recomputation',scope='Source fidelity and image metrics do not establish accurate 3D geometry or continuous motion')
    (out/'first-frame-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],train_cache=len(cache_checks),views=len(checks))))

if __name__=='__main__':main()
