"""Measure actual saved body-parameter changes, without interpreting them as accuracy."""
from pathlib import Path
import json
import numpy as np
import torch

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
RUN = LAB / 'outputs/dynamic-human-street-smpl-joint-011/articulated'
OUT = LAB / 'outputs/dynamic-human-training-alignment-013'


def stats(value):
    return dict(mean=float(value.mean()), median=float(value.median()),
                p95=float(torch.quantile(value, .95)), maximum=float(value.max()))


def angle(a, b):
    a, b = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    return 2 * torch.acos((a * b).sum(-1).abs().clamp(0, 1)) * 180 / torch.pi


def main():
    torch.set_num_threads(3)
    target = next(r['model_index'] for r in json.loads((RUN / 'initialization.json').read_text())['instances']['SMPLNodes'] if r['author_instance'] == 79)
    snapshots = {}
    for updates in (500, 3000, 10000, 15000):
        path = RUN / f'checkpoint-{updates:05d}.pt'
        assert path.exists(), path
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert saved['step'] == updates - 1
        node = saved['models']['SMPLNodes']
        mask = node['points_ids'][:, 0] == target
        assert int(mask.sum()) == 6890
        snapshots[updates] = dict(trans=node['instances_trans'][:, target].clone(),
            root=node['instances_quats'][:, target].clone(), joints=node['smpl_qauts'][:, target].clone(),
            valid=node['instances_fv'][:, target].clone(), xyz=node['_means'][mask].clone(),
            dc=node['_features_dc'][mask].clone())
        del saved, node
    baseline = snapshots[500]
    valid = baseline['valid']
    assert int(valid.sum()) == 100
    training_valid = valid.clone()
    training_valid[10::10] = False
    rows = []
    for updates, current in snapshots.items():
        assert torch.equal(valid, current['valid'])
        translation = (current['trans'] - baseline['trans']).norm(dim=-1)
        root_rotation = angle(current['root'], baseline['root']).flatten()
        joint_rotation = angle(current['joints'], baseline['joints'])
        rows.append(dict(updates=updates, reference_updates=500,
            valid_training_frames=int(training_valid.sum()),
            translation_change_m=stats(translation[training_valid]),
            root_rotation_change_degrees=stats(root_rotation[training_valid]),
            local_joint_rotation_change_degrees=stats(joint_rotation[training_valid].flatten()),
            canonical_vertex_change_m=stats((current['xyz'] - baseline['xyz']).norm(dim=-1)),
            dc_feature_absolute_change=stats((current['dc'] - baseline['dc']).abs().flatten()),
            selected_training_frames=[dict(frame=f, translation_change_m=float(translation[f]),
                root_change_degrees=float(root_rotation[f]), joint_change_mean_degrees=float(joint_rotation[f].mean()))
                for f in (129, 131, 149, 151)]))
    report = dict(status='saved_parameter_changes_measured_no_accuracy_claim', target=79, model_index=target,
        rows=rows, device='cpu', optimized_steps=0,
        boundaries=['Changes are relative to500updates, not the exact initial model.',
            'Geodesic angles normalize quaternion sign; tiny float32 angles have limited precision.',
            'Small or large parameter updates alone do not establish correct geometry.',
            'No heldout RGB or additional fitting; model tensors read without changing training.'])
    destination = OUT / 'parameter-updates-through15000.json'
    assert not destination.exists(), 'Preserve previous diagnostic'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
