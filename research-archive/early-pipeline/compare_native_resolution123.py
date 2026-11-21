"""Read two saved step1000 models at one identical native camera; no updates."""
import argparse
import fcntl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def main(root, baseline):
    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    sys.path.insert(0, '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import camera, image
    from prepare_native_resolution_review123 import join
    state = rt.read(root / 'gaussian-launch/progress.json')
    assert state['state'] == 'stopped_after_review_exported_quality_pending'
    assert state['step'] == 1000
    assert rt.sha(root / 'fit/fitted.npz') == rt.sha(baseline / 'fit/fitted.npz')
    assert rt.sha(root / 'data/training-inputs.json') == rt.sha(baseline / 'data/training-inputs.json')
    rows = rt.read(root / 'data/training-inputs.json')['records']
    ids = [r['frame'] for r in rows]
    slot = ids.index('frame_000032')
    row = rows[slot]
    out = root / 'export/native-resolution-comparison'
    if out.exists():
        raise RuntimeError('Preserve existing comparison.')
    out.mkdir()
    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        bg = torch.zeros(3, device='cuda')
        cam, cam_record = camera(row, slot, 1520, pose_index=slot)
        arrays, hashes = [], []
        with torch.no_grad():
            for label, directory in [('old899', baseline), ('native1520', root)]:
                snap = directory / 'gaussian-run/step-001000'
                parity = rt.read(snap / 'native-ply-parity.json')
                assert parity['passed'] and parity['reference_frame_id'] == 'frame_000032'
                assert rt.sha(snap / 'world.ply') == parity['sha256']
                model = GaussianModel(3)
                model.load_ply(str(snap / 'world.ply'))
                rendered = render(cam, model, pipe, bg)['render']
                arrays.append(rendered.permute(1, 2, 0).cpu().numpy())
                image(rendered).save(out / f'{label}.png')
                hashes.append(dict(label=label, sha256=parity['sha256'], points=len(model.get_xyz)))
            portable = GaussianModel(3)
            portable.load_ply(str(root / 'export/point_cloud.ply'))
            row_export = rt.read(root / 'export/cameras.json')['records'][slot]
            portable_cam, _ = camera(row_export, slot, 1520, pose_index=slot)
            exported = render(portable_cam, portable, pipe, bg)['render'].permute(1, 2, 0).cpu().numpy()
            diff = abs(exported - arrays[1])
            assert float(diff.mean()) < 2e-6 and float(diff.max()) < .005
    assert rt.sha(row['image']) == row['source_sha256']
    raw = Image.open(row['image']).convert('RGB')
    mask = np.asarray(Image.open(row['mask']).convert('L')) > 254
    assert raw.size == (2704, 1520) and mask.shape == (1520, 2704)
    target = np.asarray(raw).astype(np.float32) / 255
    crops = rt.read(baseline / 'export/final-review-images/reference-index.json')['images']
    receipt = dict(created=rt.now(), reference_frame_id=ids[slot], source_frame=row['source_frame_index'],
                   source_sha256=rt.sha(root / 'export/point_cloud.ply'), input_sha256=rt.sha(row['image']),
                   native_camera=cam_record, model_hashes=hashes, model_updates=0,
                   same_camera=True, same_pose=True, no_image_resize=True, no_image_text=True,
                   actual_visual_review=False, parity=dict(mean_abs=float(diff.mean()), max_abs=float(diff.max())),
                   images=[], metric_scope='Fitted reference frame masked RGB MAE; not heldout geometry or independent subpixel accuracy.')
    names = {'全身':'body','面部':'face','左手':'left-hand','右手':'right-hand','双脚':'feet'}
    records = Path('/mnt/d/00/记录/005')
    for crop in crops:
        box = crop['crop_integer_xyxy']; label = crop['region']; alias = names[label]
        panels = [raw.crop(box)] + [Image.open(out / f'{name}.png').crop(box) for name in ('old899', 'native1520')]
        joined = join(panels)
        joined.save(out / f'comparison-{alias}.png')
        join([panels[0], panels[2]]).save(out / f'current-{alias}.png')
        record_dir = records / ('3DGS' if label == '全身' else '分析')
        record_dir.mkdir(exist_ok=True)
        dest = record_dir / f'20261030_分辨率对照_s001000_frame_000032_{label}_左原图_中899训练_右1520训练_同原生相机.png'
        joined.save(dest)
        x0,y0,x1,y1 = box
        roi = mask[y0:y1, x0:x1]
        errors = [float(abs(a[y0:y1,x0:x1] - target[y0:y1,x0:x1])[roi].mean()) for a in arrays]
        receipt['images'].append(dict(region=label, crop_integer_xyxy=box, alias=f'comparison-{alias}.png',
                                       record=str(dest), old899_mae=errors[0], native1520_mae=errors[1]))
    for group in range(6):
        ims = [Image.open(root / f'export/review/novel_{i:03d}.png').convert('RGB') for i in range(group*6,group*6+6)]
        contact = join([join(ims[:3]),join(ims[3:])], vertical=True)
        contact.save(out / f'novel-contact-{group:02d}.png')
    rt.save(out / 'result.json', receipt)
    print(dict(output=str(out), regions=receipt['images'], model_updates=0), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--baseline', required=True)
    a = p.parse_args(); main(a.root, a.baseline)
