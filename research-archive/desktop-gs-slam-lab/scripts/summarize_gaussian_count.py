"""Audit saved counts/splits/metrics, then plot the actual capacity comparison."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs' / 'gaussian-count-study-001'

def main():
    study = json.loads((OUTPUT / 'metrics.json').read_text(encoding='utf-8'))
    dataset = np.load(ROOT / 'data/tiny-nerf/tiny_nerf_data.npz', allow_pickle=False)
    audits = []
    for record in study['records']:
        folder = ROOT / 'outputs' / record['run']
        metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        checkpoint = torch.load(folder / 'gaussians.pt', map_location='cpu', weights_only=True)
        assert checkpoint['training_views'] == list(range(100))
        assert checkpoint['test_views'] == list(range(100, 106))
        assert not set(checkpoint['training_views']) & set(checkpoint['test_views'])
        for value in checkpoint['parameters'].values():
            assert value.shape[0] == record['points'] and bool(torch.isfinite(value).all())
        actual_mean = statistics.mean(row['psnr_db'] for row in metrics['test_per_view'])
        assert math.isclose(actual_mean, record['test_mean_psnr_db'], abs_tol=1e-8)
        quantization_delta = []
        for metric in metrics['test_per_view']:
            image = np.asarray(Image.open(folder / f"predicted_test{metric['view']}.png"), dtype=np.float64) / 255
            target = dataset['images'][metric['view']].astype(np.float64)
            png_psnr = -10 * math.log10(float(np.mean((image - target)**2)))
            quantization_delta.append(abs(png_psnr - metric['psnr_db']))
        assert max(quantization_delta) < .05, 'Saved renders do not match their float metrics'
        audits.append({'run': record['run'], 'parameter_tensors_finite': True,
                      'point_count_matches': True, 'split_disjoint': True,
                      'png_vs_float_max_psnr_difference_db': max(quantization_delta),
                      'checkpoint_sha256': hashlib.sha256((folder / 'gaussians.pt').read_bytes()).hexdigest()})
    validation = {'status': 'passed', 'runs': audits,
                  'note': 'PNG metrics differ slightly from float render metrics due to 8-bit quantization.'}
    (OUTPUT / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    fig, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    for index, group in enumerate(study['groups']):
        values = [r['test_mean_psnr_db'] for r in study['records'] if r['points'] == group['points']]
        axis.scatter(np.array([index]*3) + np.array([-.07, 0, .07]), values,
                     s=42, color='#2a6f97', zorder=3)
        axis.errorbar(index, statistics.mean(values), yerr=statistics.stdev(values),
                      fmt='s', color='#253a3e', capsize=6, markersize=7, linewidth=1.5)
    axis.set_xticks([0, 1], ['4,000 Gaussians', '8,000 Gaussians'])
    axis.set_xlim(-.5, 1.5)
    axis.set_ylabel('Mean held-out PSNR (dB)')
    axis.set_title('Fixed-pose map capacity: Tiny NeRF Lego')
    axis.grid(axis='y', alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
    axis.text(.01, -.20, 'Dots: seeds 0, 1, 2. Squares: mean ± sample SD.\n100×100 images; 1,500 steps; 6 held-out validation views.',
              transform=axis.transAxes, fontsize=9, color='#5b6466')
    fig.savefig(OUTPUT / 'capacity-comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'status': 'passed', 'runs_audited': len(audits),
                      'max_png_metric_difference_db': max(a['png_vs_float_max_psnr_difference_db'] for a in audits)}, indent=2))

if __name__ == '__main__':
    main()
