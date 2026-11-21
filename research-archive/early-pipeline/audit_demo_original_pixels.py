"""Read-only video-to-training pixel and camera lineage verification."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/mnt/d/00/记录/2026-11-04_MSV2_mask原始_插值2倍_4倍对比')
BASE = Path('/opt/gs-slam-lab/static-person-upsampling-20261102/demo')
RUNS = {
    1: Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-native-20261104'),
    2: Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-interpolate2x-20261104'),
    4: Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-20261104'),
}
SR = Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    plan = read(BASE / 'plan.json')
    video = Path(plan['source_video'])
    assert sha(video) == plan['source_video_sha256']
    parent = read(Path(plan['source_run']) / 'data/training-inputs.json')['records']
    native = read(BASE / 'original/data/training-inputs.json')['records']
    arms = {s: read(r / 'training-inputs.json')['records'] for s, r in RUNS.items()}
    sr_records = {r['frame']: r for r in read(SR / 'superresolution.json')['records']}
    assert len(parent) == len(native) == 59
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height','-of','json',str(video)]))
    wh = [probe['streams'][0][key] for key in ['width','height']]
    assert wh == [2704, 1520]
    ids = [p['source_frame_index'] for p in parent]
    select = '+'.join(f'eq(n\\,{index})' for index in ids)
    command = ['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(video),
        '-vf','select='+select,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    process = subprocess.Popen(command,stdout=subprocess.PIPE)
    checks = []
    for i, (p, n) in enumerate(zip(parent, native)):
        index = p['source_frame_index']
        payload = process.stdout.read(wh[0]*wh[1]*3)
        assert len(payload) == wh[0]*wh[1]*3
        decoded = np.frombuffer(payload,dtype=np.uint8).reshape(wh[1],wh[0],3)
        raw = np.asarray(Image.open(p['image']).convert('RGB'))
        assert p['wh'] == wh and raw.shape == decoded.shape
        assert sha(p['image']) == p['source_sha256']
        error = np.abs(raw.astype(np.int16) - decoded.astype(np.int16))
        assert error.max() == 0, (n['frame'], 'decoded pixels changed', int(error.max()))
        x0,y0,x1,y1 = n['crop_xyxy']
        crop = decoded[y0:y1, x0:x1]
        observed = np.asarray(Image.open(n['image']).convert('RGB'))
        assert np.array_equal(crop, observed), (n['frame'], 'crop changed')
        assert list(observed.shape[1::-1]) == n['wh'] == [476, 960]
        assert sha(n['image']) == n['source_sha256']
        C = np.array([[1.,0,-x0],[0,1.,-y0],[0,0,1.]])
        assert np.array_equal(C @ np.asarray(p['K']), np.asarray(n['K']))
        armchecks = {}
        for scale, rows in arms.items():
            r = rows[i]
            assert r['frame'] == n['frame'] and r['source_frame_index'] == n['source_frame_index']
            assert r['wh'] == [476 * scale, 960 * scale]
            assert r['w2c'] == n['w2c'] == p['w2c']
            assert np.array_equal(np.diag([scale,scale,1]) @ np.asarray(n['K']), np.asarray(r['K']))
            k = np.asarray(r['K']).copy(); k[:2,2] -= .5
            assert np.array_equal(k, r['K_integer_centers'])
            assert sha(r['image']) == r['source_sha256']
            with Image.open(r['image']) as im:
                assert list(im.size) == r['wh']
                if scale <= 2:
                    expected = Image.fromarray(crop)
                    if scale == 2: expected = expected.resize(tuple(r['wh']), Image.Resampling.LANCZOS)
                    assert np.array_equal(np.asarray(expected), np.asarray(im.convert('RGB')))
            if scale == 4:
                s = sr_records[n['frame']]
                assert s['native_sha256'] == n['source_sha256']
                assert s['output_sha256'] == r['source_sha256']
            armchecks[str(scale)] = dict(wh=r['wh'], image_sha256=r['source_sha256'], K_scaled=True,
                source_exact_crop=scale == 1, direct_crop_lanczos=scale == 2,
                direct_crop_swinir_manifest_verified=scale == 4)
        checks.append(dict(frame=n['frame'], source_frame_index=index, video_wh=wh,
            raw_png_sha256=p['source_sha256'], native_crop_sha256=n['source_sha256'],
            video_to_raw_max_pixel_error=int(error.max()), crop_xyxy=n['crop_xyxy'],
            crop_pixel_exact=True, arms=armchecks))
        if i % 10 == 0 or i == 58: print('PIXELS_VERIFIED', i+1, '/59', flush=True)
    assert process.stdout.read() == b''
    assert process.wait() == 0
    adapter_checks = {}
    for scale, run in RUNS.items():
        code = (run / 'experiment.py').read_text(encoding='utf-8')
        tree = ast.parse(code)
        classes = {n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
        camera = ast.get_source_segment(code,classes['Camera'])
        scene = ast.get_source_segment(code,classes['Scene'])
        assert '.resize(' not in camera and 'interpolate(' not in camera
        assert 'self.image_width,self.image_height=row[\'wh\']' in camera
        assert "def getTrainCameras_warn_up(self,*args,**kwargs):return self.cameras" in scene
        adapter_checks[str(scale)] = dict(adapter_sha256=sha(run/'experiment.py'),
            camera_preserves_input_pixels=True, warmup_uses_full_resolution=True,
            actual_preflight=read(run/'preflight.json'))
    result = dict(created=datetime.now(timezone.utc).isoformat(), passed=True, frames=59,
        source_video=str(video), source_video_sha256=plan['source_video_sha256'],
        video_decoded_wh=wh, native_crop_wh=[476,960], crop_xyxy=plan['crop_xyxy'],
        redecoding_command=command, decoder='FFmpeg native output with no spatial scale filter',
        rgb_downsample_before_enlargement=False, no_training_half_size_warmup=True,
        maximum_video_to_png_pixel_error=0, original_pixels_exact=True,
        training_wh={'original':[476,960],'Lanczos2x':[952,1920],'SwinIR4x':[1904,3840]},
        records=checks, adapter_checks=adapter_checks,
        scope='Decoded pixels from the provided demo.mp4; no claim about compression before this video was recorded.')
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'原视频到训练像素核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ALL59_ORIGINAL_VIDEO_PIXELS_EXACT_NO_PRE_DOWNSAMPLING',flush=True)

if __name__ == '__main__': main()
