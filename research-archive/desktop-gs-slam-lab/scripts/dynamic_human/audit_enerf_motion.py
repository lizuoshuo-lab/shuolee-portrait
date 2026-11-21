"""Recompute saved native temporal comparisons on CPU and encode previews."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np,cv2
from PIL import Image
from enerf_native_loader import EnerfNativeDataset
from audit_enerf_input import digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--manifest',required=True);a=ap.parse_args();out=Path(a.out);dest=out/'motion/evaluation';report=json.loads((dest/'metrics.json').read_text());byid={(r['camera'],r['frame']):r for r in report['views']};rows=[]
    for split in ('heldout_view','heldout_time'):
        ds=EnerfNativeDataset(a.manifest,split)
        for i,r in enumerate(ds.rows):
            o=ds[i];name=f"camera{o['camera']}-frame{o['frame']:03d}";gt=np.array(Image.open(dest/f'{name}-reference.png'));pr=np.array(Image.open(dest/f'{name}-render.png'));expected=(o['rgb']/255*o['foreground_matte'][...,None]+.5*(1-o['foreground_matte'][...,None]));expected=(expected*255+.5).astype('uint8');assert np.array_equal(gt,expected)
            mask=(o['foreground_matte']>.9)&o['valid_pixels'];psnr=float(-10*np.log10((((pr.astype(float)-gt)/255)**2)[mask].mean()));orig=byid[(r['camera'],r['frame'])];assert abs(psnr-orig['foreground_psnr_db'])<1e-9
            if 'static_foreground_psnr_db' in orig:
                base=np.array(Image.open(dest/f'{name}-static.png'));base_psnr=float(-10*np.log10((((base.astype(float)-gt)/255)**2)[mask].mean()));assert abs(base_psnr-orig['static_foreground_psnr_db'])<1e-9
            outside=(cv2.dilate((o['foreground_matte']>.1).astype('uint8'),np.ones((33,33),np.uint8))==0)&o['valid_pixels']
            # Native full canvas exposes floaters that tight human crops exclude.
            leak=outside&(np.max(np.abs(pr.astype(float)-127.5),axis=2)>25.5)
            rows.append(dict(camera=r['camera'],frame=r['frame'],psnr_db=psnr,full_canvas_psnr_db=float(-10*np.log10((((pr.astype(float)-gt)/255)**2)[o['valid_pixels']].mean())),far_background_color_residual_pixels=int(leak.sum()),far_background_pixel_count=int(outside.sum()),render_sha256=digest(dest/f'{name}-render.png'),reference_matches_native_source=True))
        print('AUDIT',split,len(rows),flush=True)
    ff=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe');fp=ff.with_name('ffprobe.exe');videos=[]
    for c in ['00','06','12']:
        video=dest/f'camera{c}-motion-comparison.mp4'
        subprocess.run([str(ff),'-hide_banner','-loglevel','error','-y','-framerate','30','-i',str(dest/f'video-camera{c}/%03d.png'),'-vf','scale=1600:-2:flags=lanczos','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
        probe=json.loads(subprocess.check_output([str(fp),'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(video)]));assert int(probe['streams'][0]['nb_read_frames'])==60
        subprocess.run([str(ff),'-v','error','-i',str(video),'-f','null','-'],check=True);videos.append(dict(file=video.name,sha256=digest(video),stream=probe['streams'][0]))
    final=dict(status='270_native_pairs_and180_static_controls_independently_recomputed_three_videos_decoded',images=rows,videos=videos,leakage_definition='Any rendered channel differs from neutral gray by >0.1 outside the author matte dilated by16 native pixels. A color residual diagnostic, not an alpha/3D correctness metric.',source_fps='Unverified. Video encoded at30fps for inspection only.',visual_review='Video decoding is not all-frame perceptual approval')
    (out/'motion-independent-audit.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(dict(status=final['status'],images=len(rows),videos=len(videos))))

if __name__=='__main__':main()
