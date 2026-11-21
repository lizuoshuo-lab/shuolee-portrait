"""Optional adapter for the pinned SMPLify-X angle prior; never enabled implicitly.

The caller chooses a coefficient after checking loss units. This module does
not alter a tracker, camera, optimizer or source repository.
"""
import importlib.util
from pathlib import Path


def load_original_prior(source, dtype, device):
    source = Path(source)
    spec = importlib.util.spec_from_file_location('pinned_smplify_angle_prior123', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SMPLifyAnglePrior(dtype=dtype).to(device)


def decoded_body(tracker, timesteps):
    if tracker.cfg.model.use_vposer:
        return tracker.vposer.decode(tracker.pose_embedding[timesteps])['pose_body'].reshape(-1, 63)
    return tracker.body_pose[timesteps]


def bending_energy(tracker, timesteps, prior, coefficient):
    """Mean over frames, sum over four original elbow/knee terms, in radians."""
    pose = decoded_body(tracker, timesteps)
    if pose.ndim != 2 or pose.shape[1] != 63:
        raise ValueError('Expected 63D body axis-angle, excluding global rotation.')
    return prior(pose, with_global_pose=False).sum(-1).mean() * coefficient


def vposer_ratio_coefficient(vhap_weight, latent_dimensions, smplify_body_weight):
    """Preserve ONLY the original bending-to-VPoser ratio across sum/mean.

    This is a dimensional conversion, not a calibrated weight relative to RGB,
    silhouette or landmarks. It must not be described as a recommended weight.
    """
    if min(vhap_weight, latent_dimensions, smplify_body_weight) <= 0:
        raise ValueError('Positive weights and dimension are required.')
    common_scale = vhap_weight / (latent_dimensions * smplify_body_weight ** 2)
    return common_scale * (3.17 * smplify_body_weight)
