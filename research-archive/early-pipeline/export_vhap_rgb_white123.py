"""Replay saved parameters as pure RGB on white; no fitting or GT fill."""
import fcntl
import time
import numpy as np
import torch
from PIL import Image
import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
import preflight_vhap_direct_body123 as interface
from run_official_gaussians123 import read, save, sha, now

ROOT = base.mask.joint.ROOT
RUN = ROOT / 'smplify-vhap-monocular-lr'
PARAMS = tuple('body_pose' if k == 'pose_embedding' else k for k in bridge.PARAMS)


def main():
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert read(RUN/'result.json')['state'] == 'complete_pending_visual_review'
    output = RUN/'rgb-white-replay'
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32 = False
    t, rows = base.make_tracker(output)
    t.cfg.render.background_eval = 'white'
    t.cfg.render.background_train = 'white'
    t.cfg.model.use_vposer = False
    interface.OUT = output
    t.forward_smplx = interface.alternative_forward(t)
    images = []
    roster = read(ROOT/'smplify-vhap-monocular-lr-plan.json')['rgb_visual_roster']
    for label in ('initial', 'epoch-30'):
        checkpoint = RUN/label/'tracked-smplx-params.npz'
        with np.load(checkpoint, allow_pickle=False) as z:
            saved = dict(z)
        assert list(saved['frame_ids']) == [r['frame'] for r in rows]
        t.body_pose = torch.as_tensor(saved['body_pose'], device='cuda').clone()
        with torch.no_grad():
            for k in PARAMS:
                getattr(t,k).copy_(torch.as_tensor(saved[k], device='cuda'))
        before = {k:bridge.digest_tensor(getattr(t,k)) for k in PARAMS}
        for i,row in enumerate(rows):
            if row['frame'] not in roster:
                continue
            assert time.monotonic()-start < 180
            t.clear_cache()
            with torch.no_grad():
                total, logs, vertices, faces, joints, albedos, pred = t.compute_energy(t.dataset.one(i),stage=None)
            replay = float(np.max(np.abs(vertices.cpu().numpy()[0]-saved['vertices_camera'][i])))
            assert replay < 5e-6
            rgb = pred['pred_rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
            alpha = pred['pred_alpha'][0,0].clamp(0,1).cpu().numpy()
            assert np.max(np.abs(rgb[alpha==0]-1)) < 1e-6, 'Ground truth must not fill missing geometry.'
            raw = Image.open(row['image']).convert('RGB')
            mask = np.asarray(Image.open(row['mask']).convert('L'))/255.
            target = np.uint8(np.clip(np.asarray(raw)*mask[...,None]+255*(1-mask[...,None]),0,255))
            x0,y0,x1,y1 = [int(v) for v in row['initial_box_xyxy']]
            box = (max(0,x0-48),max(0,y0-48),min(raw.width,x1+48),min(raw.height,y1+48))
            tiles = [im.crop(box) for im in (raw,Image.fromarray(target),Image.fromarray(np.uint8(rgb*255)),Image.fromarray(np.uint8(alpha*255)).convert('RGB'))]
            panel = Image.new('RGB',(4*tiles[0].width,tiles[0].height))
            for j,im in enumerate(tiles):
                panel.paste(im,(j*im.width,0))
            path = output/(row['frame']+'_'+label+'_raw_target_white_rgb_alpha.png')
            panel.save(path)
            images.append(dict(frame=row['frame'],checkpoint=label,checkpoint_sha256=sha(checkpoint),path=str(path),sha256=sha(path),crop=list(box),replay_max_abs_m=replay,photo_loss_white=float(logs['photo']),background='white',zero_alpha_is_white=True))
        assert all(bridge.digest_tensor(getattr(t,k))==v for k,v in before.items())
    save(output/'result.json',dict(updated=now(),state='complete_read_only',optimizer_updates=0,parameters_unchanged=True,
        images=images,seconds=time.monotonic()-start,training_unchanged=True,actual_visual_review=False,
        replaces_for_presentation='rgb-initial/rgb-final use original evaluation target fill; kept as historical evidence, not pure render.',
        quality_accepted=False))
    print('PURE_WHITE_RGB_REPLAY_COMPLETE',flush=True)


if __name__=='__main__':
    main()
