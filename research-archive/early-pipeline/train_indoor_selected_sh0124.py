"""Indoor input handoff plus conservative lower-body supervision abstention.

Reuse the existing fitter output, Gaussian trainer and tested optional validity
loss. This is an exploratory partial-observation run, not recovered full legs.
"""
import argparse
import copy
import difflib
import inspect
import os
from pathlib import Path
import numpy as np
from PIL import Image
import train_sofa_selected_sh0124 as existing

gs = existing.gs
SUBJECT = 'indoor-selected-sh0-pipeline124'
OUT = gs.ROOT/SUBJECT
SOURCE = gs.ROOT/'indoor-selected-view-pose124'
REFERENCE = 'frame_000105'
CONTRACT = gs.BASE/'loss-validity-contract'


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert gs.read(SOURCE/'conclusion.json')['actual_visual_review']
    assert gs.read(SOURCE/'process-exit.json')['exit_code'] == 0
    assert gs.read(CONTRACT/'result.json')['state'] == 'isolated_CPU_contract_passed'
    before = inspect.getsource(existing.prepare)
    after = before
    for old, new in [
        ('assert n == 7', 'assert n == 6'),
        ("label='沙发人物7帧 SH0 全流程试跑'", "label='室内人物6帧 SH0 可见区域试跑'"),
        ("record_id='006'", "record_id='008'"),
        ("'assert len(training_slots)==7'", "'assert len(training_slots)==6'"),
        ("'training_frames=7'", "'training_frames=6'"),
        ('seven-frame-interface.diff', 'six-frame-interface.diff'),
        ('Seven preregistered sofa inputs', 'Six preregistered indoor inputs'),
        ('All seven preregistered', 'All six preregistered'),
        ('Path(__file__), Path(gs.__file__)', 'Path(__file__), Path(existing.__file__), Path(gs.__file__)')]:
        after = replace_once(after, old, new)
    ns = dict(existing.__dict__, SUBJECT=SUBJECT, OUT=OUT, SOURCE=SOURCE,
              REFERENCE=REFERENCE, existing=existing, __file__=__file__)
    exec(compile(after, __file__+':same-input-handoff', 'exec'), ns)
    ns['prepare']()
    (OUT/'input-handoff.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True))))
    (OUT/'loss_utils_validity.py').write_bytes((CONTRACT/'loss_utils_validity.py').read_bytes())
    p = gs.read(OUT/'plan.json')
    document = gs.read(OUT/'data/training-inputs.json')
    rows = document['records']
    with np.load(OUT/'fit/fitted.npz') as archive:
        fit = dict(archive)
    validity_dir = OUT/'data/validity'
    validity_dir.mkdir()
    support, tiles = [], []
    record = Path('/mnt/d/00/记录/008')
    for i, row in enumerate(rows):
        assert str(fit['frame_ids'][i]) == row['frame']
        mask = np.asarray(Image.open(row['mask']).convert('L'))
        raw = np.asarray(Image.open(row['image']).convert('RGB'))
        assert mask.shape == raw.shape[:2]
        hip = fit['observed_native_keypoints'][i, [9, 12]]
        assert np.isfinite(hip).all() and np.all(hip[:, 2] > 0)
        cut = int(np.floor(hip[:, 1].min()))
        assert 0 < cut < mask.shape[0]
        # Intentionally conservative, not a predicted occluder/depth map:
        # all positive source-mask pixels remain supervised; below the highest
        # observed hip, all remaining pixels are unknown, including real bg.
        valid = (np.arange(mask.shape[0])[:, None] < cut) | (mask > 0)
        assert valid[mask > 0].all() and valid[:cut].all() and (~valid).any()
        path = validity_dir/(row['frame']+'.png')
        Image.fromarray(valid.astype(np.uint8)*255).save(path)
        row.update(validity=str(path), validity_sha256=gs.sha(path))
        overlay = raw.copy()
        overlay[~valid] = np.round(.35*overlay[~valid]+.65*np.array([255, 0, 255])).astype(np.uint8)
        from review_full_sequence123 import crop_box, joined
        box = crop_box(Image.fromarray(mask).getbbox(), (raw.shape[1], raw.shape[0]), 50)
        panel = joined([Image.fromarray(raw).crop(box), Image.fromarray(overlay).crop(box)])
        picture = record/f'分析_室内可见区域124_{row["frame"]}_原图_未知监督区.png'
        assert not picture.exists()
        panel.save(picture)
        tile = panel.copy()
        tile.thumbnail((800, 600))
        tiles.append(tile)
        support.append(dict(frame=row['frame'], hip_y=hip[:, 1].tolist(), unknown_below_y=cut,
            known_pixels=int(valid.sum()), unknown_pixels=int((~valid).sum()),
            known_person_pixels=int((mask > 0).sum()), known_background_pixels=int(((mask == 0)&valid).sum()),
            mask_values=np.unique(mask).tolist(), source_mask_sha256=row['mask_sha256'],
            validity_sha256=row['validity_sha256'], evidence=str(picture), evidence_sha256=gs.sha(picture)))
    pages = []
    for start in (0, 3):
        page = Image.new('RGB', (2400, 600))
        for j, tile in enumerate(tiles[start:start+3]):
            page.paste(tile, (j*800, 0))
        path = OUT/f'validity-overview-{start//3+1}.png'
        page.save(path)
        pages.append(str(path))
    gs.save(OUT/'data/training-inputs.json', document)
    receipt = gs.read(OUT/'smplify-gaussian-interface-receipt.json')
    receipt['training_inputs_sha256'] = gs.sha(OUT/'data/training-inputs.json')
    gs.save(OUT/'smplify-gaussian-interface-receipt.json', receipt)
    gs.save(OUT/'validity-review.json', dict(updated=gs.now(), records=support, pages=pages,
        actual_visual_review=False, training_admitted=False, foreground_mask_unchanged=True,
        rule='All positive foreground pixels known; all pixels above min observed hip y known; lower non-foreground pixels unknown. Integer floor, no fitted geometry or tuned margin.',
        risks=['Some true lower background is deliberately ignored, so silhouette control is weaker.',
               'Original segmentation errors in supervised pixels remain.',
               'Unseen legs/feet remain unobserved, not reconstructed or completed by this mask.'],
        physical_occluder_segmentation=False, quality_accepted=False))
    source = (OUT/'adapter.py').read_text()
    patched = replace_once(source, '    from utils.loss_utils import l1_loss, ssim',
        "    from known_pixel_loss124 import KnownPixelLoss\n"
        "    validity_loss = module(OUT/'loss_utils_validity.py', 'indoor_known_pixel_loss')\n"
        "    photometric = KnownPixelLoss(validity_loss)\n"
        "    l1_loss, ssim = photometric.l1, photometric.ssim")
    patched = replace_once(patched, '    def render(cam, pc, pipe, background, **kwargs):',
        '    def render(cam, pc, pipe, background, **kwargs):\n        photometric.use_camera(cam)')
    patched = replace_once(patched, '        cams.append(cam); records.append(rec)',
        "        assert sha(row['validity']) == row['validity_sha256']\n"
        "        valid = np.asarray(Image.open(row['validity']).convert('L'))\n"
        "        assert set(np.unique(valid)) == {0, 255}\n"
        "        cam.supervision_validity = torch.as_tensor(valid == 255, device='cuda')[None]\n"
        "        cam.supervision_frame = row['frame']\n"
        "        cams.append(cam); records.append(rec)")
    patched = replace_once(patched, "    status('training_starting', 0)",
        "    assert read(OUT/'validity-review.json')['training_admitted']\n"
        "    assert read(OUT/'validity-cuda-preflight.json')['passed']\n"
        "    status('training_starting', 0)")
    patched = patched.replace("different subject inputs; same SH0 training configuration",
        "indoor inputs and conservative known-pixel L1/SSIM; original SH0 optimizer and density schedule")
    (OUT/'adapter.py').write_text(patched)
    (OUT/'validity-loss-interface.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), patched.splitlines(True))))
    p.update(updated=gs.now(), state='prepared_validity_review_pending',
        only_parameter_change='Indoor selected inputs plus optional known-pixel L1/SSIM. Fixed fit, SH0, offset lr, optimizer, original density schedule unchanged.',
        new_loss_validity_enabled=True, default_for_other_subjects_changed=False,
        supervision_scope='Visible source foreground and known upper background. Lower non-foreground unknown; no completed full legs/feet claim.',
        standalone_full_body_accepted=False, loss_validity_contract=str(CONTRACT/'result.json'),
        source_fit_review=str(SOURCE/'review/result.json'),
        comparison_attribution='Different subject and optional supervision jointly tested; no single-cause quality improvement claim.',
        budget_note='One from-scratch candidate, 1000-step actual review then at most18000; no pose refit, mask-margin sweep or best-frame mixing.',
        visibility_rule='min original observed left/right hip y; original positive foreground always retained; conservative unknown lower background.')
    p['source_hashes'] = {f:gs.sha(f) for f in p['source_hashes']}
    for f in [Path(existing.__file__), gs.HERE/'known_pixel_loss124.py', OUT/'loss_utils_validity.py',
              CONTRACT/'result.json', SOURCE/'review/result.json', OUT/'input-handoff.diff',
              OUT/'validity-loss-interface.diff']+[Path(r['validity']) for r in rows]:
        p['source_hashes'][str(f)] = gs.sha(f)
    gs.save(OUT/'plan.json', p)
    gs.save(OUT/'progress.json', dict(updated=gs.now(), state=p['state'], step=0, pid=None))
    print('INDOOR_VALIDITY_REVIEW_READY', pages, flush=True)


def preflight():
    import fcntl
    import torch
    from types import SimpleNamespace
    from known_pixel_loss124 import KnownPixelLoss
    assert not (OUT/'validity-cuda-preflight.json').exists()
    p = gs.read(OUT/'plan.json')
    for file, digest in p['source_hashes'].items():
        assert gs.sha(file) == digest, file
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    module = gs.module(OUT/'loss_utils_validity.py', 'indoor_validity_probe')
    loss = KnownPixelLoss(module)
    controls = []
    # All-valid/default parity on the same CUDA tensors, including gradients.
    x = torch.linspace(.1, .9, 3*37*49, device='cuda').reshape(3, 37, 49).requires_grad_(True)
    y = x.detach().flip(-1)
    for name in ('l1_loss', 'ssim'):
        fn = getattr(module, name)
        a = fn(x, y)
        ga = torch.autograd.grad(a, x)[0]
        b = fn(x, y, valid_mask=torch.ones((1,37,49), dtype=torch.bool, device='cuda'))
        gb = torch.autograd.grad(b, x)[0]
        assert torch.allclose(a, b, atol=2e-6, rtol=0) and torch.allclose(ga, gb, atol=2e-6, rtol=0)
        controls.append(dict(name=name+'_CUDA_all_valid_parity', value_error=float(abs(a-b)), gradient_max=float((ga-gb).abs().max())))
    rows = gs.read(OUT/'data/training-inputs.json')['records']
    checks = []
    for row in rows:
        valid = torch.as_tensor(np.array(Image.open(row['validity'])) == 255, device='cuda')[None]
        raw = torch.as_tensor(np.array(Image.open(row['image']).convert('RGB')), device='cuda').permute(2,0,1).float()/255
        mask = torch.as_tensor(np.array(Image.open(row['mask'])), device='cuda')[None].float()/255
        target = raw*mask
        prediction = torch.full_like(target, .37, requires_grad=True)
        loss.use_camera(SimpleNamespace(supervision_validity=valid, supervision_frame=row['frame']))
        assert loss.frame == row['frame']
        value = .8*loss.l1(prediction, target)+.2*(1-loss.ssim(prediction,target))
        grad = torch.autograd.grad(value,prediction)[0]
        changed = torch.where(valid, target, torch.full_like(target, float('nan')))
        prediction2 = prediction.detach().clone().requires_grad_(True)
        other = .8*loss.l1(prediction2,changed)+.2*(1-loss.ssim(prediction2,changed))
        othergrad = torch.autograd.grad(other,prediction2)[0]
        unknown = (~valid).expand_as(grad)
        assert torch.isfinite(value) and torch.isfinite(other)
        assert torch.equal(value,other) and torch.equal(grad,othergrad)
        assert torch.count_nonzero(grad[unknown]) == 0 and torch.count_nonzero(grad[~unknown]) > 0
        negative = module.l1_loss(prediction,target)
        negativegrad = torch.autograd.grad(negative,prediction)[0]
        assert torch.count_nonzero(negativegrad[unknown]) > 0
        checks.append(dict(frame=row['frame'],loss=float(value),unknown_gradient_nonzero=0,
            known_gradient_nonzero=int(torch.count_nonzero(grad[~unknown])),
            unmasked_negative_control_unknown_gradient_nonzero=int(torch.count_nonzero(negativegrad[unknown])),
            unknown_nan_target_invariance=True, known_ssim_windows=int(module._ssim_support(valid,11).sum())))
        del raw,mask,target,prediction,prediction2,grad,othergrad,negativegrad
    source = (gs.UP/'train.py').read_text()
    assert 'batch_size=None' in source
    assert source.index('render_pkg = render(viewpoint_cam') < source.index('losses[\'l1\'] = l1_loss(image, gt_image)')
    result = dict(updated=gs.now(),passed=True,checks=checks,controls=controls,
        actual_six_native_dimensions=True,unchanged_original_loss_weights=True,
        no_optimizer_or_gaussian_updates=True,training_started=False,quality_accepted=False,
        pixels_outside_validity_have_no_RGB_gradient=True,
        field_camera_link='Validity selected by the actual camera entering render; synchronous original batch_size=None and local num_workers=0.')
    gs.save(OUT/'validity-cuda-preflight.json',result)
    print('INDOOR_KNOWN_PIXEL_CUDA_PREFLIGHT_PASSED',len(checks),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','preflight','train'])
    args = parser.parse_args()
    {'prepare':prepare,'preflight':preflight,'train':lambda:gs.train(SUBJECT)}[args.mode]()
