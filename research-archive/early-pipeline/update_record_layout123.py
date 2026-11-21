"""Compose saved scientific evidence for the user's revised image-only rules.

Original native pixels are cropped uniformly within each comparison, never
rescaled or synthesized. Source files are archived separately before removal
from the delivery directory; this script does not delete or move them.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import sys
import numpy as np
from PIL import Image

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REF = T / 'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
OUT = REF / 'record-layout-v2'
RECORD = Path('D:/00/记录/005')
RULES = Path('D:/00/记录/规则.md')
THOUGHT = Path('D:/00/思路.md')
COORD = H.parent / 'shuolee-coordination'
RULE_HASH = '215e173880652fc1a873474a27abb5727a9c6b11e7542a1f18dfcc64113f070c'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def prepare():
    assert sha(RULES) == RULE_HASH, 'Rules changed; read them before composing.'
    assert not OUT.exists(), 'Preparation already exists; inspect it, do not duplicate.'
    old = read(REF / 'image-record-delivery.json')
    assert old['image_count'] == 81
    expected = {x['path']: x for x in old['files']}
    OUT.mkdir()
    outputs, archives = [], []
    sources_seen = set()
    for frame in [0, 155, 310]:
        name = f'20261029_SMPLifyX数值修复_视频帧{frame:06d}.png'
        paths = {key: RECORD / stage / sub / name for key, stage, sub in [
            ('raw', 'mask', '原图'), ('mask', 'mask', 'mask'),
            ('k_raw', '特征点', '原图'), ('2d', '特征点', '2d特征点'),
            ('3d', '特征点', '空间特征点'), ('s_raw', 'smpl-x', '原图'),
            ('smplx', 'smpl-x', 'smpl-x')]}
        for path in paths.values():
            relative = path.relative_to(RECORD).as_posix()
            assert sha(path) == expected[relative]['sha256']
            if relative not in sources_seen:
                archives.append(dict(source=str(path), destination=str(OUT / 'source-images' / relative),
                                     sha256=sha(path), record_relative=relative))
                sources_seen.add(relative)
        arrays = {k: np.asarray(Image.open(p).convert('RGB')) for k, p in paths.items()}
        assert all(v.shape == (1520, 2704, 3) for v in arrays.values())
        assert np.array_equal(arrays['raw'], arrays['k_raw'])
        assert np.array_equal(arrays['raw'], arrays['s_raw'])
        mask = np.asarray(Image.open(paths['mask']).convert('L'))
        yy, xx = np.where(mask > 0)
        assert len(xx)
        box = (max(0, int(xx.min()) - 64), max(0, int(yy.min()) - 64),
               min(2704, int(xx.max()) + 65), min(1520, int(yy.max()) + 65))
        l, t, r, b = box
        # Preserve the existing soft mask instead of inventing a hard boundary.
        foreground = np.rint(arrays['raw'].astype(np.float32) * (mask[..., None] / 255.0)).astype(np.uint8)
        stages = [
            ('mask', ['原图', 'mask', 'mask内部人像'], [arrays['raw'], arrays['mask'], foreground]),
            ('特征点', ['原图', '2D特征点', '3D特征点投影'], [arrays['raw'], arrays['2d'], arrays['3d']]),
            ('smpl-x', ['原图', 'SMPLX叠加'], [arrays['raw'], arrays['smplx']])]
        for stage, labels, values in stages:
            cropped = [v[t:b, l:r].copy() for v in values]
            joined = np.concatenate(cropped, axis=1)
            output = RECORD / stage / (f'20261029_视频帧{frame:06d}_从左到右_' + '_'.join(labels) + '.png')
            assert not output.exists()
            Image.fromarray(joined).save(output)
            reopened = np.asarray(Image.open(output).convert('RGB'))
            assert np.array_equal(joined, reopened)
            width = r - l
            assert all(np.array_equal(reopened[:, i*width:(i+1)*width], v) for i, v in enumerate(cropped))
            outputs.append(dict(path=str(output), relative=output.relative_to(RECORD).as_posix(),
                frame=frame, stage=stage, left_to_right=labels, crop_xyxy=list(box),
                pixels=[joined.shape[1], joined.shape[0]], sha256=sha(output),
                native_pixel_size_preserved=True, same_crop_all_panels=True, words_drawn=False))
    save(OUT / 'prepared.json', dict(created=now(), rule=str(RULES), rule_sha256=sha(RULES),
        thought=str(THOUGHT), thought_sha256=sha(THOUGHT), outputs=outputs, archives=archives,
        archive_complete=False, gaussian_generated=False, new_work_log_images=False))
    print(json.dumps(dict(prepared_composites=len(outputs), source_images_to_archive=len(archives)), ensure_ascii=False))

def verify():
    p = read(OUT / 'prepared.json')
    assert sha(RULES) == p['rule_sha256']
    for f in p['archives']:
        assert not Path(f['source']).exists()
        assert sha(Path(f['destination'])) == f['sha256']
    for f in p['outputs']:
        assert sha(Path(f['path'])) == f['sha256']
    old = read(REF / 'image-record-delivery.json')
    current = {x['relative']: x['sha256'] for x in p['outputs']}
    current.update({x['path']: x['sha256'] for x in old['files'] if x['path'].startswith('分析/')})
    files = sorted(x for x in RECORD.rglob('*') if x.is_file())
    assert len(files) == len(current) == 69
    entries = []
    for f in files:
        relative = f.relative_to(RECORD).as_posix()
        assert f.suffix.lower() == '.png' and sha(f) == current[relative]
        with Image.open(f) as im:
            pixels = list(im.size)
            im.verify()
        entries.append(dict(path=relative, sha256=sha(f), pixels=pixels))
    receipt = dict(updated=now(), root=str(RECORD), image_count=len(files),
        stage_counts=dict(Counter(f.relative_to(RECORD).parts[0] for f in files)),
        current_rules_sha256=sha(RULES), source_receipt=str(REF / 'image-record-delivery.json'),
        one_comparison_image_per_frame_per_stage=True, native_pixels_preserved=True,
        no_words_drawn=True, source_archive_verified=True, images_only=True,
        actual_visual_review_pending=True, gaussian_steps=0, WPS_cloud_sync_verified=False,
        new_work_log_images=False, composites=p['outputs'], files=entries)
    save(OUT / 'result.json', receipt)
    policy = read(COORD / 'image-record-policy.json')
    policy.update(updated=now(), rules_sha256=sha(RULES), latest_image_count=len(files),
        latest_delivery_receipt=str(OUT / 'result.json'),
        comparison_layout='One image per source frame and stage; left-to-right panel identities in filename.',
        mask_panels=['原图', 'mask', 'mask内部人像'], common_native_crop_allowed=True,
        image_geometric_warp_allowed=False, text_in_image_allowed=False, new_work_log_image_writes=False)
    save(COORD / 'image-record-policy.json', policy)
    print(json.dumps({k: receipt[k] for k in ['image_count', 'stage_counts', 'source_archive_verified',
                                            'images_only', 'actual_visual_review_pending']}, ensure_ascii=False))

if __name__ == '__main__':
    {'--prepare': prepare, '--verify': verify}[sys.argv[1]]()
