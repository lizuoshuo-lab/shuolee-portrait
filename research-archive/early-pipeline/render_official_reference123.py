"""Read-only final reference-pose renders; no fitting, training, or new parameters."""
import argparse
import fcntl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def main(root, project):
    root, project = Path(root).resolve(), Path(project).resolve()
    sys.path.insert(0, str(project / 'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import camera, image
    terminal = rt.read(root / 'gaussian-launch/progress.json')
    result = rt.read(root / 'result.json')
    validation = rt.read(root / 'export/validation.json')
    if terminal['state'] != 'exported_quality_pending' or result['steps'] != 18000:
        raise RuntimeError('Use only the terminal completed first-cycle export.')
    source = root / 'export/point_cloud.ply'
    if not validation['format_passed'] or rt.sha(source) != validation['sha256']:
        raise RuntimeError('Export hash or native parity is not verified.')
    reference = validation['reference_frame_id']
    source_rows = rt.read(root / 'data/training-inputs.json')['records']
    target_rows = rt.read(root / 'export/cameras.json')['records']
    ids = [r['frame'] for r in source_rows]
    if ids.count(reference) != 1 or ids != [r['frame'] for r in target_rows]:
        raise RuntimeError('The named reference/camera roster differs.')
    slot = ids.index(reference)
    original, portable = source_rows[slot], target_rows[slot]
    if rt.sha(original['image']) != original['source_sha256']:
        raise RuntimeError('The reference source image changed.')
    out = root / 'export/reference-review'
    if out.exists():
        raise RuntimeError('Preserve the existing reference review; do not rerun it.')
    # Same nonblocking GPU lock as the training process. Never run alongside it.
    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        trained = rt.read(root / 'gaussian-run/result.json')
        snapshot = root / 'gaussian-run' / trained['final_snapshot']
        snap_receipt = rt.read(snapshot / 'native-ply-parity.json')
        if snap_receipt['reference_frame_id'] != reference or rt.sha(snapshot / 'world.ply') != snap_receipt['sha256']:
            raise RuntimeError('Static snapshot reference differs.')
        native = GaussianModel(3); native.load_ply(str(snapshot / 'world.ply'))
        preview = GaussianModel(3); preview.load_ply(str(source))
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        background = torch.zeros(3, device='cuda')
        out.mkdir()
        checks = []
        with Image.open(original['image']) as im:
            raw = im.convert('RGB')
        with Image.open(original['mask']) as im:
            raw_mask = im.convert('L')
        with torch.no_grad():
            for height in (899, original['wh'][1]):
                ca, native_camera = camera(original, slot, height, pose_index=slot)
                cb, portable_camera = camera(portable, slot, height, pose_index=slot)
                a = render(ca, native, pipe, background)['render']
                b = render(cb, preview, pipe, background)['render']
                delta = abs(a-b)
                check = dict(height=height, mean_abs=float(delta.mean()), max_abs=float(delta.max()),
                             native_camera=native_camera, portable_camera=portable_camera)
                if check['mean_abs'] >= 2e-6 or check['max_abs'] >= .005:
                    raise RuntimeError('Reference-pose native/export render parity failed.')
                image(a).save(out / f'{reference}_h{height}_native.png')
                image(b).save(out / f'{reference}_h{height}_viewer.png')
                raw.resize((ca.image_width, ca.image_height), Image.Resampling.LANCZOS).save(out / f'{reference}_h{height}_input.png')
                alpha = render(cb, preview, pipe, background,
                               override_color=torch.ones_like(preview.get_xyz))['render']
                image(alpha).save(out / f'{reference}_h{height}_alpha.png')
                target_mask = np.asarray(raw_mask.resize((ca.image_width, ca.image_height), Image.Resampling.BILINEAR), dtype=float) / 255
                rendered_alpha = alpha[0].cpu().numpy()
                difference = abs(rendered_alpha-target_mask)
                foreground = target_mask > .99
                check['silhouette_loss_scope'] = dict(mask_area_fraction=float(foreground.mean()),
                    global_mae=float(difference.mean()), foreground_mean=float(difference[foreground].mean()),
                    outside_mean=float(difference[~foreground].mean()),
                    existing_weighted_global_term=float(.1*difference.mean()),
                    meaning='Describes the unchanged existing loss normalization; does not prove a unique error cause.')
                checks.append(check)
        rt.save(out / 'result.json', dict(created=rt.now(), source_sha256=validation['sha256'],
            reference_frame_id=reference, source_frame=original['source_frame_index'], checks=checks,
            same_named_pose=True, same_source_camera=True, native_full_resolution_rendered=True,
            model_updates=0, quality_accepted=False, actual_visual_review=False))
        print(out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--project-root', default='/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
    args = p.parse_args()
    main(args.root, args.project_root)
