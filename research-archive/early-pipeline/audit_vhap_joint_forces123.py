"""Read-only all-frame loss/pose-gradient accounting at the frozen VHAP result."""
import fcntl
import json
import os
import time
import traceback
import numpy as np
import torch
import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
from run_official_gaussians123 import read, save, sha, now

ROOT = base.mask.joint.ROOT
RUN = ROOT / 'smplify-vhap-normalized-laplacian'
OUT = RUN / 'joint-force-audit'
PLAN = ROOT / 'smplify-vhap-joint-force-plan.json'
GROUPS = {'face': 'lmk2d', 'body': 'lmk2d_body',
          'left_hand': 'lmk2d_lhand', 'right_hand': 'lmk2d_rhand'}
PARAMETERS = ('pose_embedding', 'left_hand_pose', 'right_hand_pose',
              'global_orient', 'transl')


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(median=float(np.median(x)), maximum=float(x.max()), minimum=float(x.min()))


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
    with np.load(checkpoint, allow_pickle=False) as z, torch.no_grad():
        assert list(z['frame_ids']) == [r['frame'] for r in rows] == plan['roster']
        for k in bridge.PARAMS:
            getattr(t, k).copy_(torch.as_tensor(z[k], device='cuda'))
    groups = t.get_train_parameters('rgb_global_tracking')
    trainable = {id(p) for g in groups.values() for p in g}
    for k in bridge.PARAMS:
        getattr(t, k).requires_grad_(id(getattr(t, k)) in trainable)
    t.vposer.requires_grad_(False)
    t.vposer.eval()
    before = {k: bridge.digest_tensor(getattr(t, k)) for k in bridge.PARAMS}
    tensors = tuple(getattr(t, k) for k in PARAMETERS)
    records = []
    for i, row in enumerate(rows):
        assert time.monotonic() - start < plan['max_seconds']
        t.clear_cache()
        sample = t.dataset.one(i)
        total, logs, vertices, faces, joints, albedos, output = t.compute_energy(sample, stage='rgb_global_tracking')
        # Reuse the same current geometry and the original loss helper. No new correspondences.
        _, _, j, face, _ = t.forward_smplx(sample['timestep_index'])
        separated = {}
        for name, key in GROUPS.items():
            only = {k: v for k, v in sample.items() if k not in GROUPS.values() or k == key}
            value, _ = t.compute_lmk_energy_smplx(only, j, lmks_face=face)
            separated['lmk_' + name] = t.cfg.w.landmark * value
        lmk_diff = float((sum(separated.values()) - logs['lmk']).detach().abs())
        assert lmk_diff < 2e-5, lmk_diff
        terms = {k: v for k, v in logs.items() if k not in ('total', 'lmk')}
        terms.update(separated)
        assert float((sum(terms.values()) - total).detach().abs()) < 5e-4
        body_target = sample['lmk2d_body'].cuda()
        native = (output['pred_lmk2d_body'] + 1) * torch.tensor(row['wh'], device='cuda') / 2
        error2 = ((native - body_target[..., :2]) ** 2).sum(-1)
        probes = {name: error2[0, index] for name, index in [('left_elbow', 6), ('right_elbow', 3)]}
        gradients = {}
        metrics = {}
        for name, value in {**terms, **probes}.items():
            grads = torch.autograd.grad(value, tensors, retain_graph=True, allow_unused=True)
            current = {}
            for key, tensor, g in zip(PARAMETERS, tensors, grads):
                current[key] = np.zeros(tensor.shape[1:], np.float32) if g is None else g[i].detach().cpu().numpy()
                assert np.isfinite(current[key]).all()
            gradients[name] = current
            metrics[name] = dict(loss=float(value.detach()), gradient_l2={k: float(np.linalg.norm(g)) for k, g in current.items()})
        # Compare directions only within the same 32-dimensional body-pose latent.
        directional = {}
        for probe in probes:
            target = gradients[probe]['pose_embedding']
            norm = float(np.linalg.norm(target))
            directional[probe] = {}
            for name in terms:
                g = gradients[name]['pose_embedding']
                gn = float(np.linalg.norm(g))
                directional[probe][name] = dict(cosine=float(np.dot(target, g) / max(norm * gn, 1e-20)),
                    negative_gradient_directional_derivative=float(-np.dot(target, g)))
        records.append(dict(frame=row['frame'],losses=metrics,elbow_px={k:float(v.detach().sqrt()) for k,v in probes.items()},
                            elbow_descent=directional,landmark_partition_abs_error=lmk_diff))
        save(OUT / 'progress.json', dict(updated=now(),state='auditing',pid=os.getpid(),frames=i+1,
                                         optimizer_updates=0,seconds=time.monotonic()-start))
        del total, logs, terms, separated, vertices, faces, joints, albedos, output, j, face, grads, gradients
    assert all(bridge.digest_tensor(getattr(t, k)) == h for k, h in before.items())
    for p, h in plan['input_hashes'].items():
        assert sha(p) == h, p
    summary = {name: dict(loss=stats([r['losses'][name]['loss'] for r in records]),
                         body_latent_gradient_l2=stats([r['losses'][name]['gradient_l2']['pose_embedding'] for r in records]),
                         left_elbow_opposing_frames=sum(r['elbow_descent']['left_elbow'][name]['cosine'] < 0 for r in records))
               for name in records[0]['elbow_descent']['left_elbow']}
    result = dict(updated=now(),state='complete_read_only',frames=len(records),optimizer_updates=0,
                  parameters_unchanged=True,seconds=time.monotonic()-start,checkpoint_sha256=sha(checkpoint),
                  loss_summary=summary,records=records,quality_accepted=False,
                  landmark_outer_weight=t.cfg.w.landmark,
                  landmark_inner_weights=dict(face=t.cfg.w.landmark,body=t.cfg.w.landmark_body,hand=t.cfg.w.landmark_hand),
                  limits=['Current-state local gradients do not identify the unique origin of a bad pose.',
                          'No confidence, correspondences, weights, optimizer, checkpoint or input was changed.',
                          'Gradient comparisons are within the same pose parameter, not across units.',
                          'Photometric disturbance uses a fixed seed; this is not an independent image-quality test.'])
    save(OUT / 'result.json', result)
    save(OUT / 'progress.json', dict(updated=now(),state='complete_read_only',pid=None,frames=len(records),optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            save(OUT / 'failure.json', dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        raise
