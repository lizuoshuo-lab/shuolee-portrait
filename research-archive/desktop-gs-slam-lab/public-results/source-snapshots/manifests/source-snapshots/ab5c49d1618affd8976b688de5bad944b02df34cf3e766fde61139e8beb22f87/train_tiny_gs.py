"""Educational fixed-pose 3D Gaussian optimization with the open-source gsplat renderer.

This is a deliberately small training harness, not SplaTAM or paper-level 3DGS
reproduction: random initialization, fixed Gaussian count, RGB colors, no SH,
densification, pose optimization, or depth supervision. Views 100..105 are held out.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from gsplat import rasterization

ROOT = Path(__file__).resolve().parents[1]

def save_image(array, destination):
    Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(destination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--points', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', default='tiny-nerf-gsplat-001')
    args = parser.parse_args()
    if not args.run.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Run name must contain only letters, numbers, dashes and underscores')
    output = ROOT / 'outputs' / args.run
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = np.load(ROOT / 'data' / 'tiny-nerf' / 'tiny_nerf_data.npz', allow_pickle=False)
    images = torch.from_numpy(dataset['images']).float().cuda()
    poses_gl = torch.from_numpy(dataset['poses']).float().cuda()
    # OpenGL c2w (right, up, backward) -> OpenCV c2w (right, down, forward).
    convention = torch.diag(torch.tensor([1., -1., -1., 1.], device='cuda'))
    poses_cv = poses_gl @ convention
    viewmats = torch.linalg.inv(poses_cv).contiguous()
    height, width = images.shape[1:3]
    focal = float(dataset['focal'])
    intrinsics = torch.tensor([[focal, 0., width / 2.], [0., focal, height / 2.],
                              [0., 0., 1.]], device='cuda')[None]
    means = torch.nn.Parameter((torch.rand(args.points, 3, device='cuda') - .5) * 2.6)
    log_scales = torch.nn.Parameter(torch.full((args.points, 3), math.log(.045), device='cuda'))
    quats = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device='cuda').repeat(args.points, 1))
    colors = torch.nn.Parameter(torch.zeros(args.points, 3, device='cuda'))
    opacity = torch.nn.Parameter(torch.full((args.points,), -2., device='cuda'))
    params = {'means': means, 'log_scales': log_scales, 'quats': quats,
              'color_logits': colors, 'opacity_logits': opacity}
    optimizer = torch.optim.Adam([
        {'params': [means], 'lr': .008}, {'params': [log_scales], 'lr': .01},
        {'params': [quats], 'lr': .005}, {'params': [colors], 'lr': .03},
        {'params': [opacity], 'lr': .03}], eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9994)

    def render(view):
        return rasterization(means, torch.nn.functional.normalize(quats, dim=-1),
                             log_scales.exp(), opacity.sigmoid(), colors.sigmoid(),
                             viewmats[view:view+1], intrinsics, width, height,
                             packed=False)[0][0]

    train_indices = list(range(100))
    test_indices = list(range(100, len(images)))
    with torch.no_grad():
        initial = render(101).detach().cpu().numpy()
        initial_mse = float(((torch.from_numpy(initial).cuda() - images[101]) ** 2).mean())
    save_image(initial, output / 'initial_test101.png')
    save_image(images[101].cpu().numpy(), output / 'target_test101.png')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []
    for step in range(args.steps):
        view = random.choice(train_indices)
        predicted = render(view)
        loss = ((predicted - images[view]) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            log_scales.clamp_(math.log(.002), math.log(.3))
            opacity.clamp_(-10, 10)
        if step % 25 == 0 or step == args.steps - 1:
            history.append({'step': step + 1, 'view': view, 'mse': float(loss.detach())})
        if step % 250 == 0 or step == args.steps - 1:
            torch.cuda.synchronize()
            print(f'step={step+1}/{args.steps} mse={float(loss):.6f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated()
    with torch.no_grad():
        test_metrics = []
        for view in test_indices:
            prediction = render(view)
            mse = float(((prediction - images[view]) ** 2).mean())
            test_metrics.append({'view': view, 'mse': mse, 'psnr_db': -10. * math.log10(max(mse, 1e-12))})
            save_image(prediction.cpu().numpy(), output / f'predicted_test{view}.png')
        train_metrics = []
        for view in range(0, 100, 10):
            mse = float(((render(view) - images[view]) ** 2).mean())
            train_metrics.append({'view': view, 'psnr_db': -10. * math.log10(max(mse, 1e-12))})
    checkpoint = {'parameters': {name: value.detach().cpu() for name, value in params.items()},
                  'config': vars(args), 'width': width, 'height': height, 'focal': focal,
                  'training_views': train_indices, 'test_views': test_indices,
                  'camera_convention': 'OpenCV w2c from OpenGL c2w @ diag(1,-1,-1,1)'}
    torch.save(checkpoint, output / 'gaussians.pt')
    with (output / 'loss.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['step', 'view', 'mse'])
        writer.writeheader()
        writer.writerows(history)
    result = {'status': 'completed', 'scope': 'educational fixed-pose 3DGS; not SLAM',
              'config': vars(args), 'train_views': train_indices, 'test_views': test_indices,
              'resolution': [width, height], 'initial_test101_psnr_db': -10. * math.log10(initial_mse),
              'test_per_view': test_metrics, 'test_mean_psnr_db': float(np.mean([m['psnr_db'] for m in test_metrics])),
              'train_sample_psnr_db': float(np.mean([m['psnr_db'] for m in train_metrics])),
              'training_seconds_excludes_compile_and_final_evaluation': elapsed,
              'peak_torch_allocated_mib': peak_memory / 1024**2,
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'dataset_attribution': 'Tiny NeRF Lego data from the official NeRF tutorial; original Lego asset by Håvard Dalen / Heinzelnisse, CC BY-NC. See manifests/datasets/tiny_nerf.json.',
              'limitations': ['fixed known poses', '100x100 teaching data', 'one seed', 'no SH/densification', 'not paper benchmark']}
    (output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    panel = Image.new('RGB', (600, 240), 'white')
    draw = ImageDraw.Draw(panel)
    for index, (filename, label) in enumerate([('target_test101.png', 'Held-out target 101'),
                                              ('initial_test101.png', 'Initial Gaussian map'),
                                              ('predicted_test101.png', 'After optimization')]):
        with Image.open(output / filename) as frame:
            panel.paste(frame.resize((200, 200)), (200*index, 30))
        draw.text((200*index+5, 8), label, fill='black')
    panel.save(output / 'comparison.png')
    print(json.dumps({key: value for key, value in result.items() if key not in ['train_views', 'test_per_view']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
