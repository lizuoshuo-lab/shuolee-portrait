"""Finite forward/gradient equivalence check; no optimizer or fitting run."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import torch
import smplx

project = Path('/mnt/d/Projects/Shuolee')
sys.path.insert(0, str(project / 'src'))
from shuolee_portrait.smplifyx_runtime import cache_static_topology

out = Path(__file__).with_name('static-topology-cuda-check.json')
model_path = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
body = smplx.SMPLX(str(model_path), batch_size=1, num_betas=10,
    num_expression_coeffs=10, use_pca=True, num_pca_comps=12,
    flat_hand_mean=False, use_face_contour=False, create_transl=False).cuda()
saved_parents = body.parents
float_tensors = {name: value for name, value in [*body.named_parameters(), *body.named_buffers()]
                 if value.is_floating_point()}
snapshots = {name: value.detach().clone() for name, value in float_tensors.items()}
K = torch.tensor([[1100., 3.7, 440.2], [0., 1050., 790.4], [0., 0., 1.]], device='cuda')


def evaluate(seed):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    fields = dict(betas=10, global_orient=3, body_pose=63, left_hand_pose=12,
                  right_hand_pose=12, expression=10, jaw_pose=3,
                  leye_pose=3, reye_pose=3)
    inputs = {name: (torch.randn((1, size), generator=gen, device='cuda') * .04).requires_grad_()
              for name, size in fields.items()}
    translation = torch.tensor([.03, -.02, 4.], device='cuda', requires_grad=True)
    result = body(**inputs)
    camera_joints = result.joints + translation
    homogeneous = camera_joints @ K.T
    pixels = homogeneous[..., :2] / homogeneous[..., 2:]
    loss = result.vertices.square().mean() + pixels.square().mean() * 1.e-6
    gradients = torch.autograd.grad(loss, [*inputs.values(), translation])
    values = dict(vertices=result.vertices, joints=result.joints, pixels=pixels, loss=loss)
    values.update({f'gradient:{name}': grad for name, grad in zip([*inputs, 'translation'], gradients)})
    if not all(torch.isfinite(value).all().item() for value in values.values()):
        raise AssertionError('Nonfinite equivalence state')
    return {name: value.detach().cpu() for name, value in values.items()}


states = []
try:
    for seed in (32, 63, 94):
        body.parents = saved_parents
        baseline = evaluate(seed)
        metadata = cache_static_topology(body)
        assert cache_static_topology(body) == metadata
        candidate = evaluate(seed)
        errors = {name: (baseline[name] - candidate[name]).abs().max().item()
                  for name in baseline}
        assert all(torch.equal(baseline[name], candidate[name]) for name in baseline), errors
        for name, value in [*body.named_parameters(), *body.named_buffers()]:
            if value.is_floating_point():
                assert value is float_tensors[name] and value.device.type == 'cuda'
                assert torch.equal(value.detach(), snapshots[name])
        assert torch.equal(body.parents, saved_parents.cpu())
        states.append(dict(seed=seed, exact_equal=True, max_abs_errors=errors))
finally:
    body.parents = saved_parents

source = project / 'src/shuolee_portrait/smplifyx_runtime.py'
report = dict(updated=datetime.now(timezone.utc).isoformat(), passed=True,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), torch=torch.__version__,
    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), states=states,
    actual_smplx_model=True, synthetic_parameter_states=True, CUDA_forwards=6,
    CUDA_gradient_evaluations=6, optimizer_steps=0, fitting_runs=0, gaussian_training_runs=0,
    floating_tensor_identities_devices_values_unchanged=True, rollback_exact=True,
    speed_measured=False, full94_verified=False, server_verified=False,
    runtime_hook_connected_to_product_CLI=False)
out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ('passed', 'CUDA_forwards', 'CUDA_gradient_evaluations',
    'optimizer_steps', 'fitting_runs', 'floating_tensor_identities_devices_values_unchanged',
    'runtime_hook_connected_to_product_CLI')}))
