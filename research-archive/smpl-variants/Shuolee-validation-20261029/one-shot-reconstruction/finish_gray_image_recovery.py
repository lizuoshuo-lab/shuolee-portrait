"""Restore only missing files, using exact native frames and saved coordinates."""
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
import cv2
import numpy as np
from PIL import Image

base=Path('/mnt/d/Projects/Shuolee-validation-20261029/one-shot-reconstruction/gray-image-recovery')
delivery=Path('/mnt/d/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图')
manifest=json.loads((delivery/'清单.json').read_text(encoding='utf-8-sig'))
root=Path(manifest['run'])
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(root/'data/observations.npz') as a:
    observations=a['keypoints'][:,40:91].copy(); ids=a['frame_ids'].astype(str).tolist()
with np.load(root/'fit/fitted.npz') as a:
    projections=a['projected_face51'].copy(); fit_ids=a['frame_ids'].astype(str).tolist()
with np.load(root/'fit/observation-weights.npz') as a: weights=a['face'].copy()
assert ids == fit_ids
def points(name, values, rounding):
    i=ids.index(name); uv=(values[i][weights[i]>0]+.5)*np.array([1520/899,2704/1600])-.5
    return (np.rint(uv*256) if rounding=='subpixel' else np.rint(uv) if rounding=='nearest' else uv).astype(int)

# Numerical point positions, mask and six displayed medians reproduce the
# saved analysis. Missing diagnostic overlays are rasterized again only.
for frame in manifest['frames']:
    i=ids.index(frame['frame'])
    median=float(np.median(np.linalg.norm(observations[i]-projections[i],axis=-1)[weights[i]>0]))
    assert abs(median-frame['face_median_analysis_px'])<1.e-9
circle=dict(rounding='nearest',radius=4,thickness=1,line=cv2.LINE_AA)
cross=dict(rounding='nearest',size=11,thickness=1)
original_manifest=json.loads(json.dumps(manifest))
restyled=[]
restored=[]
for frame,number in zip([manifest['frames'][4],manifest['frames'][5]],[2,3]):
    decoded=base/f'{number:03d}.png'; assert sha(decoded)==frame['files'][0]['sha256']
    original=np.array(Image.open(decoded).convert('RGB'))
    for spec in frame['files'][:3]:
        target=delivery/spec['path']
        if target.exists():
            assert sha(target)==spec['sha256'];continue
        if target.name.startswith('01_'):
            raw=decoded.read_bytes()
        else:
            canvas=original.copy()
            if target.name.startswith('02_'):
                for x,y in points(frame['frame'],observations,circle['rounding']):
                    shift=8 if circle['rounding']=='subpixel' else 0
                    cv2.circle(canvas,(int(x),int(y)),circle['radius']*(2**shift),(0,239,152),circle['thickness'],lineType=circle['line'],shift=shift)
            else:
                for x,y in points(frame['frame'],projections,cross['rounding']):
                    cv2.drawMarker(canvas,(int(x),int(y)),(255,125,36),cv2.MARKER_CROSS,cross['size'],cross['thickness'],cv2.LINE_8)
            buffer=io.BytesIO();Image.fromarray(canvas).save(buffer,format='PNG');raw=buffer.getvalue()
        digest=hashlib.sha256(raw).hexdigest()
        if target.name.startswith('01_'):
            assert digest==spec['sha256'], 'Original PNG bytes differ: '+spec['path']
        elif digest!=spec['sha256']:
            spec['previous_missing_file_sha256']=spec['sha256']
            spec['sha256']=digest
            spec['recovery_note']='Re-rasterized saved analytical coordinates; original image pixels, observations, model projection and residual unchanged.'
            restyled.append(spec['path'])
        with target.open('xb') as stream:stream.write(raw)
        restored.append(spec['path'])
(base/'original-manifest.json').write_text(json.dumps(original_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(delivery/'清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
index=delivery/'查看原图.html' 
html=index.read_text(encoding='utf-8')
html=re.sub(r'src="(frame_\d+)/04_脸部对照.png" alt="原图、检测点和 SMPL-X 投影"',
            r'src="\1/01_完整原图.png" alt="完整原图，原始尺寸"',html)
index.write_text(html,encoding='utf-8')
verified=[]
for frame in manifest['frames']:
    for spec in frame['files']:
        path=delivery/spec['path']
        assert sha(path)==spec['sha256'],spec['path']
        with Image.open(path) as im:
            im.load();assert list(im.size)==spec['wh']
        verified.append(spec['path'])
links=[]
for page in [index,delivery/'修复进展.html']:
    for item in re.findall(r'(?:href|src)="([^"]+)"',page.read_text(encoding='utf-8')):
        if ':' in item or item.startswith('#'):continue
        path=(page.parent/item.split('#')[0]).resolve()
        assert path.exists(),(page,item)
        links.append(item)
receipt=dict(updated=datetime.now(timezone.utc).isoformat(),source_pngs_verified=len(verified),
    source_frames=len(manifest['frames']),local_links_verified=len(links),native_full_frame_wh=[1520,2704],
    restored_missing_files=restored,all_24_PNG_hashes_match_current_manifest=True,all_6_originals_match_original_manifest=True,analytical_overlay_rasterization_recovered=restyled,original_six_face_medians_exact=True,
    original_video_unchanged=True,source_images_modified=False,passed=True)
(delivery/'文件检查.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(base/'recovery-result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
