"""Read the actual native training masks and exercise the unchanged RGB loss on CPU."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
import numpy as np
from PIL import Image
import torch

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle')
RUN = ROOT / 'native-resolution-1000'
OUT = ROOT / 'mask-supervision-check'
SRC = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src')
RECORD = Path('/mnt/d/00/记录/005/mask')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    assert not OUT.exists(), 'Do not repeat this completed bounded audit.'
    OUT.mkdir()
    script_hash = sha(Path(__file__))
    source = SRC / 'shuolee_portrait/training.py'
    launch = read(RUN / 'gaussian-launch/input.json')
    assert sha(source) == launch['interface_sources']['training.py']
    config = read(RUN / 'gaussian-run/config.json')
    inputs = RUN / 'data/training-inputs.json'
    rows = read(inputs)['records']
    assert len(rows) == 94 and config['training_height'] == 1520
    assert all(r['wh'] == [2704, 1520] for r in rows)
    selected = ['frame_000032', 'frame_000063', 'frame_000094']
    save(OUT / 'plan.json', dict(created=datetime.now(timezone.utc).isoformat(),
         operation='Read94 current masks, check image identity/size, test unchanged foreground_loss on three native integer crops.',
         training_source_sha256=sha(source), audit_source_sha256=script_hash,
         training_input_sha256=sha(inputs), frames=selected, GPU_calls=0,
         fit_changes=0, optimizer_updates=0, no_training=True,
         stop_condition='Stop on any input hash, dimensions, non-finite loss, RGB background dependence or outside RGB gradient failure.'))
    sys.path.insert(0, str(SRC))
    from shuolee_portrait.training import foreground_loss
    torch.set_num_threads(4)
    start = time.monotonic()
    records, tests = [], []
    RECORD.mkdir(parents=True, exist_ok=True)
    for r in rows:
        image_path, mask_path = Path(r['image']), Path(r['mask'])
        assert sha(image_path) == r['source_sha256']
        assert sha(mask_path) == r['mask_sha256']
        with Image.open(image_path) as image:
            assert list(image.size) == r['wh']
            rgb = np.array(image.convert('RGB')) if r['frame'] in selected else None
        with Image.open(mask_path) as image:
            assert list(image.size) == r['wh']
            mask = np.array(image.convert('L'))
        valid = mask > .99*255
        ys, xs = np.where(valid)
        assert len(xs) > 1000 and len(xs) < mask.size
        box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
        records.append(dict(frame_id=r['frame'], mask_sha256=r['mask_sha256'],
            image_sha256=r['source_sha256'], wh=r['wh'], foreground_pixels=int(valid.sum()),
            foreground_fraction=float(valid.mean()), nonbinary_pixels=int(((mask>0)&(mask<255)).sum()),
            bbox_xyxy=box, image_mask_dimensions_and_identity_passed=True))
        if rgb is None:
            continue
        x0, y0 = max(0, box[0]-24), max(0, box[1]-24)
        x1, y1 = min(mask.shape[1], box[2]+24), min(mask.shape[0], box[3]+24)
        raw = rgb[y0:y1,x0:x1]
        a = mask[y0:y1,x0:x1]
        mask_rgb = np.repeat(a[:,:,None],3,axis=2)
        portrait = np.rint(raw.astype(float)*a[:,:,None]/255).astype('uint8')
        panel = Image.fromarray(np.concatenate([raw,mask_rgb,portrait],axis=1))
        dest = RECORD / f'20261030_训练实际mask核验_{r["frame"]}_左原图_中mask_右mask内人像.png'
        assert not dest.exists()
        panel.save(dest)
        panel.save(OUT / f'{r["frame"]}.png')
        gt = torch.from_numpy(raw.copy()).permute(2,0,1).float()/255
        alpha = torch.from_numpy(a.copy())[None].float()/255
        pred = (gt*.7+.1).detach().requires_grad_(True)
        other = gt.clone()
        outside = alpha[0] <= .99
        other[:,outside] = 1-gt[:,outside]
        loss = foreground_loss(pred,gt,alpha)[0]
        gradient = torch.autograd.grad(loss,pred)[0]
        changed_loss = foreground_loss(pred,other,alpha)[0]
        changed_gradient = torch.autograd.grad(changed_loss,pred)[0]
        loss_delta = float(abs(loss-changed_loss))
        gradient_delta = float((gradient-changed_gradient).abs().max())
        outside_gradient = float(gradient[:,outside].abs().max())
        assert all(np.isfinite(v) for v in [loss_delta,gradient_delta,outside_gradient,float(loss)])
        assert loss_delta == 0 and gradient_delta == 0 and outside_gradient == 0
        assert float(gradient[:,~outside].abs().max()) > 0
        tests.append(dict(frame_id=r['frame'], crop_xyxy=[x0,y0,x1,y1], resized=False,
            background_RGB_replaced_pixels=int(outside.sum()),
            background_RGB_loss_change=loss_delta, background_RGB_gradient_change=gradient_delta,
            photometric_gradient_outside_mask_max=outside_gradient, actual_training_function=True,
            actual_render_used=False, test_prediction='CPU synthetic prediction only to isolate the actual loss dependency; not a quality test.',
            image=str(dest), image_sha256=sha(dest)))
    save(OUT/'result.json',dict(state='complete_visual_review_pending',seconds=time.monotonic()-start,
         training_source_sha256=sha(source),launch_source_hash_matches=True,
         preflight=read(RUN/'gaussian-run/preflight.json'),frames=94,records=records,tests=tests,
         RGB_outside_mask_leak_found=False,production_training_code_changed=False,
         fitting_changed=False,GPU_calls=0,training_started=False,
         actual_mask_visual_review=False,quality_accepted=False,
         limits='Loss exclusion does not prove masks contain no background, and does not correct uncertain boundaries, projection or SMPL-X alignment.'))
    print(json.dumps(dict(state='complete_visual_review_pending',frames=94,tests=tests,seconds=time.monotonic()-start),ensure_ascii=False))

if __name__ == '__main__':
    main()
