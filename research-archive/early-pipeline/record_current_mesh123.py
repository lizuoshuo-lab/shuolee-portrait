"""Record raw/current trained SMPL-X/Gaussian using existing forward and overlay.

Read-only with respect to model, cameras, poses and optimizer. This is a saved
result inspection command, not another fitting or reconstruction method.
"""
from pathlib import Path
import argparse
import fcntl
import json
import sys
import numpy as np
from PIL import Image
from review_full_sequence123 import mesh_renderer, relocate
from review_shared_vertex_offset123 import join
import run_official_gaussians123 as io


def main(root):
    root = Path(root).resolve()
    snapshot = root/'gaussian-run/step-001000'
    out = root/'comparison/current-mesh'
    out.mkdir(exist_ok=False)
    sys.path[:0] = [str(io.PROJECT/'src'), str(io.PROJECT/'src/avatar_core')]
    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [root/'fit/fitted.npz', root/'data/training-inputs.json',
               snapshot/'shared-static-offset.npz', snapshot/'world.ply',
               io.PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py']
    hashes = {str(p): io.sha(p) for p in sources}
    comparison = io.read(root/'comparison/comparison.json')
    assert comparison['source_sha256'] == hashes[str(snapshot/'world.ply')]
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as data:
        fit = dict(data)
    with np.load(snapshot/'shared-static-offset.npz', allow_pickle=False) as data:
        offset = data['static_offset']
    rows = {row['frame']: row for row in io.read(root/'data/training-inputs.json')['records']}
    slots = {str(name): i for i, name in enumerate(fit['frame_ids'])}
    records = Path('/mnt/d/00/记录/005')
    mesh, renderer = mesh_renderer()
    hashes[str(renderer)] = io.sha(renderer)
    receipt = []
    vertices, projected = [], []
    with open('/tmp/shuolee-native122-gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = SMPLXGaussianModel(3)
        model.load_fresh_meshes(root/'fit/fitted.npz')
        parity = model.enable_shared_static_offset(root/'fit/fitted.npz', io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
        with torch.no_grad():
            model.static_offset.copy_(torch.as_tensor(offset, device='cuda'))
            for cr in comparison['comparisons']:
                name = cr['frame']; row = rows[name]; i = slots[name]
                assert str(fit['image_sha256'][i]) == row['source_sha256']
                assert io.sha(relocate(row['image'])) == row['source_sha256']
                world = model.shared_offset_vertices(i).cpu().numpy().astype(np.float64)
                E = fit['w2c'][i]; K = fit['K_integer_centers'][i]
                camera = world @ E[:3, :3].T + E[:3, 3]
                q = camera @ K.T; uv = q[:, :2] / q[:, 2:]
                if not np.isfinite(uv).all() or camera[:, 2].min() <= 0:
                    raise ValueError('Invalid saved-pose projection: ' + name)
                vertices.append(camera); projected.append(uv)
                raw = Image.open(relocate(row['image'])).convert('RGB')
                gs = Image.open(snapshot/(name+'_render.png')).convert('RGB')
                assert raw.size == gs.size == (2704, 1520)
                overlay = mesh(raw, np.array([0, 0, *raw.size]), dict(
                    vertices_camera=camera[None], faces=fit['faces'], K_integer_centers=K))
                for region, values in cr['regions'].items():
                    box = tuple(values['crop'])
                    panels = [im.crop(box) for im in (raw, overlay, gs)]
                    image = join(panels)
                    dest = records/('3DGS' if region == 'body' else '分析')/(
                        '20261030_增密1000步_左原图_中当前SMPLX含共享偏移_右高斯_'+name+'_'+region+'.png')
                    if dest.exists():
                        raise FileExistsError(dest)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    image.save(dest)
                    image.save(out/(name+'_'+region+'.png'))
                    with Image.open(dest) as saved:
                        w, h = panels[0].size
                        assert saved.size == image.size
                        assert np.array_equal(np.asarray(saved)[:, :w], np.asarray(panels[0]))
                    delta = camera-fit['vertices_camera'][i]
                    receipt.append(dict(frame=name, source_frame_index=row['source_frame_index'],
                        region=region, crop_integer=list(box), panel_wh=list(panels[0].size),
                        path=str(dest), sha256=io.sha(dest),
                        posed_mesh_displacement_median_m=float(np.median(np.linalg.norm(delta, axis=1)))))
                print(json.dumps(dict(frame=name, record_images=3)), flush=True)
        del model
    np.savez_compressed(out/'saved-current-projection.npz',
        vertices_camera=np.stack(vertices), projected_native=np.stack(projected),
        frame_ids=np.array([r['frame'] for r in comparison['comparisons']]))
    for path, digest in hashes.items():
        assert io.sha(path) == digest
    result = dict(updated=io.now(), state='generated_actual_visual_review_pending',
        files=receipt, images=len(receipt), source_hashes=hashes,
        current_trained_shared_offset_used=True, zero_offset_bridge=parity,
        panels=['raw', 'current SMPL-X including trained shared static offset', 'same-pose trained Gaussian'],
        original_pixels_unchanged=True, resampling=False, image_text=False,
        renderer='Existing depth-sorted shaded triangle overlay; not a z-buffer exact renderer acceptance.',
        fitting_steps=0, Gaussian_training_steps=0, camera_changed=False,
        quality_accepted=False)
    io.save(out/'result.json', result)
    print(json.dumps(dict(images=len(receipt), output=str(out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    main(p.parse_args().root)
