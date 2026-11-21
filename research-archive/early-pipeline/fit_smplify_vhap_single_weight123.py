"""Apply each declared landmark-group weight once, as original VHAP does."""
import types
import torch
import fit_smplify_vhap_normalized_laplacian123 as base
from run_official_gaussians123 import read, save, now

joint = base.mask.joint
joint.OUT = joint.ROOT / 'smplify-vhap-single-landmark-weight'
joint.PLAN = joint.ROOT / 'smplify-vhap-single-landmark-weight-plan.json'


def make_tracker(out):
    t, rows = base.make_tracker(out)
    original = t.compute_lmk_energy_smplx

    def group_energy(self, *args, **kwargs):
        # The SMPL-X helper already applies landmark/landmark_body/landmark_hand.
        # Cancel only the redundant outer face-weight multiplier in compute_energy.
        value, result = original(*args, **kwargs)
        assert self.cfg.w.landmark is not None and self.cfg.w.landmark > 0
        return value / self.cfg.w.landmark, result

    t.compute_lmk_energy_smplx = types.MethodType(group_energy, t)
    save(out / 'landmark-weight-contract.json', dict(updated=now(),
        original_vhap='Raw landmark mean is multiplied by landmark weight once.',
        inherited_smplx='Group-weighted means were multiplied by landmark face weight again.',
        correction='Keep original group weights and cancel only redundant outer multiplication.',
        face_weight=t.cfg.w.landmark, body_weight=t.cfg.w.landmark_body,
        hand_weight=t.cfg.w.landmark_hand, redundant_multiplier_removed=t.cfg.w.landmark,
        optimizer_updates=0, quality_accepted=False))
    # Exact local loss/gradient equivalence to the unchanged weighted-group helper.
    frame = read(joint.PLAN)['initialization_frame']
    index = next(i for i, r in enumerate(rows) if r['frame'] == frame)
    sample = t.dataset.one(index)
    _, _, joints, face, _ = t.forward_smplx(sample['timestep_index'])
    expected, _ = original(sample, joints, lmks_face=face)
    actual, _ = t.compute_lmk_energy_smplx(sample, joints, lmks_face=face)
    actual = t.cfg.w.landmark * actual
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    return t, rows


def main():
    plan = read(joint.PLAN)
    assert plan['single_landmark_weight'] and plan['restore_original_offset_initialization']
    joint.make_tracker = make_tracker
    joint.main()
    for name in ('result.json', 'progress.json'):
        d = read(joint.OUT / name)
        d.update(updated=now(), original_optimizer_loss_scheduler=False,
                 original_optimizer_and_scheduler=True,
                 restored_operator='Original VHAP normalized Laplacian retained from preceding trial',
                 landmark_weight_once=True,
                 scope='Only remove redundant landmark outer multiplier relative to normalized-Laplacian control. Same full59 inputs, zero offset, fresh optimizer, initialization and schedule.')
        save(joint.OUT / name, d)


if __name__ == '__main__':
    main()
