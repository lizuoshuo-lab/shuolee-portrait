"""One CPU-only six-source intake; no masks, detectors, fit or training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
from PIL import Image

DATASET = Path('/mnt/d/00/工作/数据集')
ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT = ROOT / 'cross-subject-sh0-124/new-person-intake129/video22-arc'
VIDEO = DATASET / 'video22.mp4'
EXPECTED_SHA = 'c5aac80e574dba892137d59186ff17713ab0403ec1a969d788b054a364f68cd0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists(), 'One six-frame intake only; inspect existing outputs instead.'
    assert shutil.disk_usage(ROOT).free > 2 * 2**30
    before = sha(VIDEO)
    assert before == EXPECTED_SHA
    registry = ROOT / 'source-name-reconciliation.json'
    old = ROOT / 'cross-subject-sh0-124/new-person-intake126/intake.json'
    registered = json.loads(registry.read_text())
    prior = json.loads(old.read_text())
    known = {s['expected_sha256']: s['subject'] for s in registered['subjects']}
    known.update(prior['known_source_hashes'])
    known.update({'4e1c1e52818362dcab22bb3ff9548a2d1a8c8b3f6affb8bd13dacfcc12a706a0':'football126',
                  '71d993f2f1e2c1e2ad1f5e3b58c4f351cd375e172f92c3526818dfb4ab380edb':'officewoman127',
                  '965befa346fdb0b5e5140745ab5651ef6ed7f8a87ae32777e1d88017bc45e547':'graywoman128'})
    assert before not in known
    inventory = [dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(DATASET.glob('*.mp4'), key=lambda p:p.name)]
    duplicates = [r['name'] for r in inventory if r['sha256'] == before]
    assert duplicates == ['video22.mp4']
    probe_command = ['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_streams',
                     '-show_entries','frame=best_effort_timestamp_time:stream','-of','json',str(VIDEO)]
    probe = json.loads(subprocess.check_output(probe_command))
    stream = probe['streams'][0]
    assert (stream['width'], stream['height']) == (2704, 1520)
    assert not stream.get('tags', {}).get('rotate')
    assert all(float(x.get('rotation', 0)) == 0 for x in stream.get('side_data_list', []))
    assert stream.get('color_transfer') not in ('smpte2084', 'arib-std-b67')
    times = np.asarray([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    assert len(times) >= 6 and np.isfinite(times).all() and np.all(np.diff(times) > 0)
    targets = np.linspace(times[0], times[-1], 6)
    indices = [int(np.argmin(np.abs(times - t))) for t in targets]
    assert len(set(indices)) == 6
    OUT.mkdir(parents=True)
    expression = '+'.join(r'eq(n\,%d)' % i for i in indices)
    command = ['ffmpeg','-nostdin','-v','error','-hwaccel','none','-threads','4','-noautorotate',
               '-i',str(VIDEO),'-vf','select='+expression,'-vsync','0','-frames:v','6',
               '-pix_fmt','rgb24',str(OUT/'source-%02d.png')]
    subprocess.run(command, check=True)
    records = []
    contact = Image.new('RGB', (676 * 3, 380 * 2))
    for i, index in enumerate(indices):
        path = OUT / ('source-%02d.png' % (i + 1))
        image = Image.open(path).convert('RGB')
        assert image.size == (2704, 1520)
        contact.paste(image.resize((676, 380), Image.Resampling.LANCZOS), ((i % 3)*676, (i // 3)*380))
        records.append(dict(order=i+1, source_index=index, source_PTS_seconds=float(times[index]),
            target_PTS_seconds=float(targets[i]), path=str(path), sha256=sha(path), native_wh=[2704,1520], resized=False))
    contact_path = OUT/'source-contact-six.png'
    contact.save(contact_path)
    assert sha(VIDEO) == before
    result = dict(updated=datetime.now(timezone.utc).isoformat(), state='six_CPU_sources_pending_actual_review',
        source_video=str(VIDEO), source_sha256_before=before, source_sha256_after=before,
        source_bytes=VIDEO.stat().st_size, source_wh=[2704,1520], source_stream=stream,
        decoded_source_frame_count=len(times), sampling='Six fixed equally spaced true-PTS targets; nearest existing frame, lower index tie. No fit/image-quality selection.',
        records=records, source_contact=str(contact_path), source_contact_sha256=sha(contact_path),
        contact_only_resized_for_display=True, native_sources_no_resize_or_geometric_warp=True,
        source_pixel_format=stream.get('pix_fmt'), decoded_display_pixel_format='rgb24_8bit',
        source_color_transfer=stream.get('color_transfer'), probe_command=probe_command, CPU_decode_command=command,
        dataset_mp4_inventory=inventory, exact_same_content_filenames=duplicates,
        known_source_sha256=known, known_exact_source_match=False,
        registry_paths_sha256={str(p):sha(p) for p in (registry,old)},
        actual_source_visual_review=False, new_person_admitted=False, physical_identity_verified=False,
        source_arm_face_leg_visibility_review_pending=True, training_view_target=None,
        full3fps_preprocessing=False, masks_or_detectors=False, GPU_executed=False,
        fitting_or_training=False, frozen_sources_modified=False, helper_sha256=sha(__file__), quality_accepted=False)
    (OUT/'arc-intake.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(path=str(OUT/'arc-intake.json'), contact=str(contact_path), source_indices=indices,
                          source_PTS_seconds=times[indices].tolist(), source_sha256=before)))


if __name__ == '__main__':
    main()
