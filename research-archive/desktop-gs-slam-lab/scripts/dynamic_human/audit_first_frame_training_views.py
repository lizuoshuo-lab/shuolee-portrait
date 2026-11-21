"""Compare training-camera fit with known held-out first-frame diagnostics."""
import json
import math
import numpy as np
import torch
from adapter import DATA, load_official, dataset
from audit_first_frame_appearance import OUT, CHECKPOINTS, sha, make_panel, save


def main():
    assert not (OUT/'training-view-audit.json').exists()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.12)
    tr, helper = load_official()
    md = json.loads((DATA/'basketball/train_meta.json').read_text())
    rows = []
    for path in CHECKPOINTS:
        params = {k:v.cuda() for k,v in torch.load(path, map_location='cpu', weights_only=True).items()}
        for view in dataset(0, md):
            camera = int(view['filename'].split('/')[0])
            rgb, _, _ = tr.Renderer(view['cam'])(**helper.params2rendervar(params))
            gain = params['cam_m'][view['id']].exp()[:, None, None]
            bias = params['cam_c'][view['id']][:, None, None]
            pred = rgb*gain+bias
            fg = view['seg'][0] > .5
            gt = view['im']
            seg, _, _ = tr.Renderer(view['cam'])(**{**helper.params2rendervar(params), 'colors_precomp':params['seg_colors']})
            mask = seg[0] > .5
            row = dict(checkpoint=path.name, camera=camera, foreground_pixels=int(fg.sum()),
                train_foreground_psnr_raw=float(-10*math.log10(max(float(((rgb.clamp(0,1)-gt)[:,fg]**2).mean()),1e-12))),
                train_foreground_psnr_fitted_camera=float(-10*math.log10(max(float(((pred.clamp(0,1)-gt)[:,fg]**2).mean()),1e-12))),
                segmentation_iou=float((mask & fg).sum()/(mask | fg).sum()),
                foreground_contribution_mean_on_gt=float(seg[0][fg].mean()),
                foreground_fraction_contribution_below_half=float((seg[0][fg]<.5).float().mean()))
            rows.append(row)
            if path == CHECKPOINTS[-1] and camera in [14,16]:
                make_panel([a.permute(1,2,0).cpu().numpy() for a in [gt,rgb.clamp(0,1),pred.clamp(0,1)]],
                    ['原始观测', '训练相机原始渲染', '应用已拟合相机颜色校正'],
                    OUT/f'assets/train-camera{camera}-initial-10000.png',
                    f'训练相机 {camera} · 第一帧 · 仅训练相机采用其既有颜色校正；留出相机没有拟合校正')
        del params
        torch.cuda.empty_cache()
        print('TRAIN CAMERAS AUDITED', path.name, flush=True)
    summary = []
    for path in CHECKPOINTS:
        subset = [r for r in rows if r['checkpoint'] == path.name]
        summary.append(dict(checkpoint=path.name, sha256=sha(path), cameras=len(subset),
            mean_train_foreground_psnr_raw=float(np.mean([r['train_foreground_psnr_raw'] for r in subset])),
            mean_train_foreground_psnr_fitted_camera=float(np.mean([r['train_foreground_psnr_fitted_camera'] for r in subset])),
            mean_segmentation_iou=float(np.mean([r['segmentation_iou'] for r in subset]))))
    save(OUT/'training-view-audit.json', dict(status='training_fit_evaluated_visual_review_pending',
        summary=summary, per_view=rows, source_metadata_sha256=sha(DATA/'basketball/train_meta.json'),
        script_sha256=sha(__import__('pathlib').Path(__file__)),
        limitations=['Training foreground masks are author-provided and not held-out labels',
            'Training masked PSNR and held-out rectangular ROI PSNR have different supports; do not directly subtract',
            'Known data diagnosis, no generalization improvement claimed']))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    with torch.no_grad():
        main()
