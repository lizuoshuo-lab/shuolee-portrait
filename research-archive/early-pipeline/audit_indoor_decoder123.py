"""One bounded, read-only comparison of four already saved decoder pairs."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, time, signal
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
import cv2
from PIL import Image
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
PRE=T/'shuolee/current/local-clothed-surface-123/indoor-preflight'
OUT=PRE/'decoder-comparison'
RUN=Path('/opt/gs-slam-lab/shuolee-local-surface-123/preflight/indoor')
OLD=Path('/opt/gs-slam-lab/shuolee-raw-domain-121/indoor/observations.json')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):return dict(mean=float(x.mean()),p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),maximum=float(x.max()))
def main():
    assert not OUT.exists(), 'Once-only comparison already exists'
    source=read(PRE/'result.json'); old=read(OLD)['records']; new=read(RUN/'decoded.json')['records']
    old={r['raw_source_frame_index']:r for r in old}; new={r['source_frame_index']:r for r in new}
    indices=[r['source_frame'] for r in source['shared_historical_frame_checks']]
    assert indices==[0,115,230,845]
    OUT.mkdir()
    plan=dict(registered=now(),script_sha256=sha(Path(__file__)),input_result_sha256=sha(PRE/'result.json'),
        hypothesis='Separate saved RGB-decoder differences from a possible image-grid displacement. No physical camera inference is made.',
        fixed_pairs=indices,operations=['Absolute RGB differences on full raster, 16px interior and bottom16rows',
          'Descriptive affine RGB conversion fitted on a fixed stride16 interior grid; no image is modified',
          'Hann-windowed grayscale phase correlation on nine fixed512x512 native interior tiles; report every tile, without filtering by response'],
        tile_origins_xy=[[x,y] for y in [32,504,976] for x in [32,1096,2160]],
        maximum_seconds=120,CPU_threads=1,GPU=0,new_decodes=0,features=0,calibration=0,training=0,
        stop='Stop on source SHA, WH or timestamp mismatch. Differences in RGB units are not reprojection errors. Translation diagnostics cannot certify physical calibration or subpixel landmark accuracy.')
    save(OUT/'plan.json',plan); started=time.monotonic()
    def timeout(*_):raise TimeoutError('120second decoder audit budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);cv2.setNumThreads(1)
    save(OUT/'progress.json',dict(state='running',pid=os.getpid(),updated=now()))
    result=[];hann=cv2.createHanningWindow((512,512),cv2.CV_32F)
    for index in indices:
        o,n=old[index],new[index];op=Path(o['image']);npth=RUN/'raw'/n['name']
        assert sha(op)==o['source_sha256'] and sha(npth)==n['source_sha256']
        assert abs(o['pts_seconds']-n['time_seconds'])<1e-5
        a=np.array(Image.open(op).convert('RGB'));b=np.array(Image.open(npth).convert('RGB'))
        assert a.shape==b.shape==(1520,2704,3)
        diff=np.abs(a.astype(np.int16)-b.astype(np.int16))
        rows=diff.mean(axis=(1,2)); top=np.argsort(-rows)[:20]
        # Matrix only describes a photometric relationship; it is never applied to training images.
        X=a[16:-16:16,16:-16:16].reshape(-1,3).astype(np.float64)
        Y=b[16:-16:16,16:-16:16].reshape(-1,3).astype(np.float64)
        XX=np.column_stack([X,np.ones(len(X))]);coef=np.linalg.lstsq(XX,Y,rcond=None)[0]
        residual=np.abs(XX@coef-Y)
        ag=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32);bg=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(np.float32)
        tiles=[]
        for x,y in plan['tile_origins_xy']:
            shift,response=cv2.phaseCorrelate(ag[y:y+512,x:x+512].copy(),bg[y:y+512,x:x+512].copy(),hann)
            tiles.append(dict(origin_xy=[x,y],old_to_new_shift_xy=list(shift),shift_norm_px=float(np.linalg.norm(shift)),response=float(response)))
        result.append(dict(source_frame=index,old_sha256=o['source_sha256'],new_sha256=n['source_sha256'],
            PTS_equal=True,WH_equal=True,full_RGB_difference=stats(diff),interior16_RGB_difference=stats(diff[16:-16,16:-16]),
            bottom16_RGB_difference=stats(diff[-16:]),highest_difference_rows=[dict(row=int(j),mean_RGB_difference=float(rows[j])) for j in top],
            affine_RGB_matrix_with_bias=coef.tolist(),fixed_sample_RGB_residual_after_affine=stats(residual),
            fixed_tiles=tiles,all9_max_shift_norm_px=max(t['shift_norm_px'] for t in tiles)))
    signal.alarm(0)
    r=dict(updated=now(),state='comparison_completed_interpretation_pending',pairs=result,seconds=time.monotonic()-started,
        source_color_metadata=read(PRE/'ffprobe.json')['streams'][0],
        pixel_warp_applied=False,old_features_reused=False,physical_calibration_verified=False,training_started=False,
        limits=['Phase correlation describes these decoded image pairs only; it does not measure camera projection or landmark accuracy.',
                'An affine color fit does not establish the decoder implementation cause. No global frame warp or K update is made.'])
    save(OUT/'result.json',r);save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None))
    print(json.dumps(dict(seconds=r['seconds'],pairs=[dict(source_frame=x['source_frame'],interior=x['interior16_RGB_difference'],bottom=x['bottom16_RGB_difference'],max_tile_shift=x['all9_max_shift_norm_px'],affine_residual=x['fixed_sample_RGB_residual_after_affine']) for x in result]),ensure_ascii=False))
if __name__=='__main__':main()
