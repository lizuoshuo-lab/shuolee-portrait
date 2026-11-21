"""Recover missing native diagnostic images from their immutable video/data."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

base = Path('/mnt/d/Projects/Shuolee-validation-20261029/one-shot-reconstruction')
out = base/'gray-image-recovery'
out.mkdir(exist_ok=False)
delivery = Path('/mnt/d/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图')
manifest = json.loads((delivery/'清单.json').read_text(encoding='utf-8-sig'))
video = Path('/mnt/d/00/工作/数据集/video3.mp4')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(video) == manifest['source_video_sha256']
subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-noautorotate',
    '-i', str(video), '-vf', r'select=eq(n\,0)+eq(n\,375)+eq(n\,470)',
    '-vsync', '0', '-frames:v', '3', '-pix_fmt', 'rgb24', str(out/'%03d.png')], check=True)
root = Path(manifest['run'])
with np.load(root/'fit/observation-weights.npz', allow_pickle=False) as f:
    print('weights', {k:dict(shape=list(f[k].shape), nonzero_first=np.flatnonzero(f[k][0]).tolist()) for k in f.files})
for frame, number in zip([manifest['frames'][0], manifest['frames'][4], manifest['frames'][5]], [1,2,3]):
    p=out/f'{number:03d}.png'
    print(frame['frame'], sha(p), 'exact_original_hash', sha(p) == frame['files'][0]['sha256'])
original = np.asarray(Image.open(delivery/'frame_000001/01_完整原图.png').convert('RGB'))
for suffix in ['02_检测点完整图.png','03_SMPL-X投影完整图.png']:
    annotated=np.asarray(Image.open(delivery/'frame_000001'/suffix).convert('RGB'))
    changed=(original!=annotated).any(-1)
    colors, counts=np.unique(annotated[changed],axis=0,return_counts=True)
    order=np.argsort(counts)[-8:]
    print(suffix, 'changed_pixels',int(changed.sum()), 'colors',list(zip(colors[order].tolist(),counts[order].tolist())))
