"""All-frame, zero-update audit of actual VHAP surface loss gradients.

Replay the accepted-for-analysis direct-body checkpoint, never optimize it.
Keep current normalized Laplacian, full native images, and loss weights.
"""
import fcntl
import json
import os
import time
import traceback
import cv2
import numpy as np
import torch
from PIL import Image
import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
import preflight_vhap_direct_body123 as interface
from run_official_gaussians123 import read, save, sha, now

ROOT = base.mask.joint.ROOT
RUN = ROOT / 'smplify-vhap-direct-body'
OUT = ROOT / 'vhap-surface-force-audit'
PLAN = ROOT / 'vhap-surface-force-plan.json'
PARAMS = tuple('body_pose' if k == 'pose_embedding' else k for k in bridge.PARAMS)
PROBES = ('static_offset', 'body_pose', 'global_orient', 'transl')


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(minimum=float(x.min()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), maximum=float(x.max()))


def cosine(a, b):
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if d < 1e-20 else float(np.sum(a * b) / d)


def main():
    plan = read(PLAN)
    assert plan['source_sha256'] == sha(__file__)
    assert plan['optimizer_updates'] == 0 and not OUT.exists()
    for p, h in plan['input_hashes'].items():
        assert sha(p) == h, p
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir()
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32 = False
    t, rows = base.make_tracker(OUT)
    t.cfg.render.background_train = 'white'
    checkpoint = RUN / 'epoch-30/tracked-smplx-params.npz'
    with np.load(checkpoint, allow_pickle=False) as z:
        saved = dict(z)
    assert list(saved['frame_ids']) == [r['frame'] for r in rows] == plan['roster']
    t.body_pose = torch.tensor(saved['body_pose'], device='cuda', requires_grad=True)
    t.cfg.model.use_vposer = False
    interface.OUT = OUT
    t.forward_smplx = interface.alternative_forward(t)
    with torch.no_grad():
        for k in PARAMS:
            getattr(t, k).copy_(torch.as_tensor(saved[k], device='cuda'))
    groups = t.get_train_parameters('rgb_global_tracking')
    trainable = {id(p) for g in groups.values() for p in g}
    for k in PARAMS:
        getattr(t, k).requires_grad_(id(getattr(t, k)) in trainable)
    t.pose_embedding.requires_grad_(False)
    t.vposer.requires_grad_(False)
    t.vposer.eval()
    before = {k: bridge.digest_tensor(getattr(t, k)) for k in PARAMS}
    tensors = tuple(getattr(t, k) for k in PROBES)
    records, image_rows = [], []
    accumulated, accumulated_norm = {}, {}
    all_offset_grads = {}
    for i, row in enumerate(rows):
        assert time.monotonic() - start < plan['max_seconds']
        assert sha(row['image']) == row['source_sha256']
        assert sha(row['mask']) == row['mask_sha256']
        t.clear_cache()
        sample = t.dataset.one(i)
        total, logs, vertices, faces, joints, albedos, output = t.compute_energy(sample, stage='rgb_global_tracking')
        parity = float(np.max(np.abs(vertices.detach().cpu().numpy()[0] - saved['vertices_camera'][i])))
        assert parity < 5e-6, parity
        terms = {k: v for k, v in logs.items() if k != 'total'}
        assert all(k in terms for k in ('photo', 'silhouette', 'reg_offset', 'reg_offset_lap'))
        assert float((sum(terms.values()) - total).detach().abs()) < 5e-4
        pred = output['pred_alpha']
        target = sample['alpha_map'].to(pred)
        pa = pred.detach().cpu().numpy()[0, 0]
        ta = target.detach().cpu().numpy()[0, 0]
        binary = (pa > .5).astype(np.uint8)
        tb = (ta > .5).astype(np.uint8)
        dist_to_mesh = cv2.distanceTransform(1-binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        far = (tb > 0) & (pa < .001) & (dist_to_mesh > plan['far_pixels'])
        far_tensor = torch.as_tensor(far, device='cuda')[None, None]
        error = (pred-target).abs() * t.cfg.w.photo / target.sum().clamp_min(1)
        # Spatial partition is frozen at this state. Far alpha-error tests the
        # local capture range, not a proposed loss or target modification.
        separated = dict(silhouette_far_missing=(error * far_tensor).sum(),
                         silhouette_rest=(error * ~far_tensor).sum())
        assert float((sum(separated.values())-terms['silhouette']).detach().abs()) < 1e-5
        gradients, metrics = {}, {}
        for name, value in {**terms, **separated, 'total': total}.items():
            gs = torch.autograd.grad(value, tensors, retain_graph=True, allow_unused=True)
            current = {}
            for key, tensor, g in zip(PROBES, tensors, gs):
                array = np.zeros_like(tensor.detach().cpu().numpy()) if g is None else g.detach().cpu().numpy()
                assert np.isfinite(array).all()
                current[key] = array[0] if key == 'static_offset' else array[i]
            gradients[name] = current
            og = current['static_offset']
            metrics[name] = dict(loss=float(value.detach()), gradient_l2={k: float(np.linalg.norm(g)) for k, g in current.items()},
                                 offset_vertices_nonzero=int((np.linalg.norm(og, axis=-1) > 1e-12).sum()))
            accumulated[name] = accumulated.get(name, np.zeros_like(og, dtype=np.float64)) + og
            accumulated_norm[name] = accumulated_norm.get(name, 0.) + float(np.linalg.norm(og))
            all_offset_grads.setdefault(name, []).append(og.copy())
        sg = gradients['silhouette']['static_offset']
        assert np.max(np.abs(sg-gradients['silhouette_far_missing']['static_offset']-gradients['silhouette_rest']['static_offset'])) < 2e-4
        active = np.linalg.norm(sg, axis=-1) > 1e-12
        directions = {}
        for name, g in gradients.items():
            og = g['static_offset']
            directions[name] = dict(cosine_with_mask=cosine(sg, og),
                mask_directional_derivative_under_negative_gradient=float(-np.sum(sg*og)),
                mask_active_vertices_gradient_l2=float(np.linalg.norm(og[active])))
        edge = binary.astype(bool) & ~cv2.erode(binary, np.ones((3, 3), np.uint8)).astype(bool)
        targetedge = tb.astype(bool) & ~cv2.erode(tb, np.ones((3, 3), np.uint8)).astype(bool)
        dt_targetedge = cv2.distanceTransform((~targetedge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        fn = (tb > 0) & (binary == 0)
        mask_info = dict(iou=float(((binary > 0) & (tb > 0)).sum()/max(1, ((binary > 0) | (tb > 0)).sum())),
            target_pixels=int(tb.sum()), missed_pixels=int(fn.sum()), far_missed_pixels=int(far.sum()),
            far_fraction_of_missing=float(far.sum()/max(1, fn.sum())),
            predicted_boundary_distance_to_target_px=stats(dt_targetedge[edge]),
            predicted_boundary_far_fraction=float((dt_targetedge[edge] > plan['far_pixels']).mean()))
        rec = dict(frame=row['frame'], replay_max_abs_m=parity, mask=mask_info, losses=metrics,
                   offset_directions=directions)
        records.append(rec)
        if row['frame'] in plan['visual_roster']:
            x0,y0,x1,y1 = np.asarray(row['initial_box_xyxy'], dtype=int)
            box = (max(0,x0-48), max(0,y0-48), min(row['wh'][0],x1+48), min(row['wh'][1],y1+48))
            raw = Image.open(row['image']).convert('RGB')
            alpha = Image.fromarray(np.uint8(np.clip(pa,0,1)*255)).convert('RGB')
            mask = Image.fromarray(np.uint8(np.clip(ta,0,1)*255)).convert('RGB')
            # Computed measurement visualization, no painted correction.
            diff = np.zeros((*pa.shape,3),np.uint8)
            diff[(binary>0)&(tb>0)] = (180,180,180)
            diff[(binary>0)&(tb==0)] = (230,50,50)
            diff[fn] = (30,120,230)
            diff[far] = (240,180,0)
            tiles = [im.crop(box) for im in (raw,mask,alpha,Image.fromarray(diff))]
            panel = Image.new('RGB',(tiles[0].width*4,tiles[0].height))
            for j,im in enumerate(tiles): panel.paste(im,(j*im.width,0))
            path=OUT/(row['frame']+'_native_raw_target_mesh_mismatch.png')
            panel.save(path)
            image_rows.append(dict(frame=row['frame'],path=str(path),crop=list(box),sha256=sha(path)))
        save(OUT/'progress.json',dict(updated=now(),state='auditing',pid=os.getpid(),frames=i+1,
                                     optimizer_updates=0,seconds=time.monotonic()-start))
        print(json.dumps(dict(frame=row['frame'],seconds=time.monotonic()-start,
            far_missing_fraction=mask_info['far_fraction_of_missing'],mask_offset_norm=metrics['silhouette']['gradient_l2']['static_offset'],
            far_offset_norm=metrics['silhouette_far_missing']['gradient_l2']['static_offset'])),flush=True)
        del total,logs,terms,separated,vertices,faces,joints,albedos,output,gs,pred,target,error
    assert all(bridge.digest_tensor(getattr(t,k))==h for k,h in before.items())
    for p,h in plan['input_hashes'].items(): assert sha(p)==h,p
    summary={name:dict(loss=stats([r['losses'][name]['loss'] for r in records]),
        offset_gradient_l2=stats([r['losses'][name]['gradient_l2']['static_offset'] for r in records]),
        mean_offset_gradient_l2=float(np.linalg.norm(g/len(records))),
        offset_gradient_coherence=float(np.linalg.norm(g)/max(accumulated_norm[name],1e-20)),
        opposing_mask_frames=sum((r['offset_directions'][name]['cosine_with_mask'] or 0)<0 for r in records))
        for name,g in accumulated.items()}
    np.savez_compressed(OUT/'offset-gradients.npz',frame_ids=np.asarray(plan['roster']),
        **{name:np.asarray(g) for name,g in all_offset_grads.items()})
    result=dict(updated=now(),state='complete_read_only_pending_visual_review',frames=len(rows),optimizer_updates=0,
        parameters_unchanged=True,seconds=time.monotonic()-start,checkpoint_sha256=sha(checkpoint),
        loss_summary=summary,records=records,images=image_rows,far_pixels=plan['far_pixels'],
        far_missing_fraction=stats([r['mask']['far_fraction_of_missing'] for r in records]),
        mean_gradient_cosines_with_mask={k:cosine(accumulated['silhouette'],g) for k,g in accumulated.items()},
        actual_visual_review=False,quality_accepted=False,gaussian_training_started=False,
        limits=['This is the local gradient at one rejected checkpoint, not a fitted improvement or unique causal attribution.',
                'Mask is a segmentation observation, not independent physical surface truth.',
                'Gradient norms are compared within the same parameter units; Adam history/preconditioning are not simulated.',
                'The 8px spatial split is a declared diagnostic ruler, not a fitted threshold or new loss.',
                'All59 frames included; no correspondence or visibility rejection by error.'])
    save(OUT/'result.json',result)
    save(OUT/'progress.json',dict(updated=now(),state='complete_read_only',pid=None,frames=len(rows),optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','images')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as error:
        if OUT.exists():
            save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
            save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,optimizer_updates=0))
        raise
