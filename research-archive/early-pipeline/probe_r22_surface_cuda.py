"""Verify native R22 gradients and a bounded geometry-triggered split on real R21 inputs."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import launch_unified_cold_016_006 as cold

PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-splat-distance-20261111')
OUT=Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round22/input-audit'
sys.path.insert(0,str(PROJECT/'src'))

from shuolee_portrait import runtime as rt
from shuolee_portrait.bound_replay import RAW_PARAMETERS, _inputs, _load_model
from shuolee_portrait.sh0_main import Constraints, DCColorProjection
from shuolee_portrait.surface_attachment import SurfaceAttachment
from shuolee_portrait.surface_densification import SURFACE_DENSIFICATION, refine_surface_distributions
from shuolee_portrait.training import _frozen_smpl_surface, _native_foreground_target, camera, render_pose, smpl_sh0_effective_config
from shuolee_portrait.world_scale import WorldScale
from gaussian_renderer import render
from diff_gaussian_rasterization import _C
from utils.sh_utils import C0


def subject_probe(subject):
    root=cold.LAB/f'shuolee-smpl-delivery-round21-{subject}/run'
    snapshot=root/'gaussian-run/step-018000'
    path=snapshot/'bound-replay.json'
    manifest=rt.read(path)
    asset=rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    fit,mesh,facts=_inputs(root,snapshot,asset)
    assert manifest['inputs_sha256']==facts['inputs_sha256']
    assert manifest['bound_ply_sha256']==rt.sha(snapshot/'bound-local.ply')
    model,_=_load_model(snapshot/'bound-local.ply',snapshot,fit,mesh,facts,asset)
    model.binding = model.binding.long()  # Canonical fresh-training index dtype.
    model.posed_vertices=lambda slot: model.vertices_sequence[slot]
    model.static_offset=torch.nn.Parameter(torch.from_numpy(mesh['static_offset']).cuda(),requires_grad=False)
    model.spatial_lr_scale=1.
    for name in RAW_PARAMETERS:
        getattr(model,name).requires_grad_(True)
    options=smpl_sh0_effective_config()
    model.training_setup(SimpleNamespace(**options['optimization']))
    model.max_radii2D = model._xyz.new_zeros(len(model._xyz))
    rows=rt.read(root/'data/training-inputs.json')['records']
    slots=[f['pose_index'] for f in facts['frames']]
    ref=facts['reference_pose_index']
    cameras=[camera(row,i,pose_index=slot)[0] for i,(row,slot) in enumerate(zip(rows,slots,strict=True))]
    _,_,clothing,protected,_=_frozen_smpl_surface(fit,asset)
    head_faces=torch.as_tensor(protected,device='cuda')[model.faces].all(1)
    bins=rt.read(root/'gaussian-run/config.json')['training_source_bins']
    targets=[]
    for row in rows:
        rgb,alpha,box=_native_foreground_target(row)
        targets.append((rgb.cuda(),alpha.cuda(),box))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda')
    schedule=options['unified_main']
    extent=float(np.linalg.norm(np.ptp(fit['vertices_world'][ref],axis=0)))
    with torch.no_grad():
        scale=WorldScale(model,extent,ref,schedule['world_scale'])
        constraints=Constraints(model,rows,slots,cameras,fit,pipe,bg,targets,bins,extent,schedule)
        skin=SurfaceAttachment(model,constraints,head_faces,fit['observed_native_keypoints'],
                               projected=fit['joints_projected_native'], distribution=True)
        measured=skin.measure()
    sample=max(range(len(rows)),key=lambda i: measured['views'][i]['supported_rows'])
    assert measured['views'][sample]['supported_rows']>0
    dc=DCColorProjection(model._features_dc,C0)
    random.seed(4521); np.random.seed(731); torch.manual_seed(814)

    step = 3000
    model.update_learning_rate(step)
    pkg = render_pose(render, cameras[sample], model, pipe, bg, slots[sample])
    prior = skin.loss(sample, pkg['visibility_filter'])
    assert prior.requires_grad and prior.isfinite() and prior.item() > 0
    isolated = torch.autograd.grad(prior, (model._xyz, model._scaling, model._rotation))
    assert all(bool(grad.isfinite().all()) and grad.abs().sum().item() > 0 for grad in isolated)
    measured_prior = skin.loss(sample, pkg['visibility_filter'], audit_step=step)
    (measured_prior*.02+pkg['render'].mean()*.001).backward()
    model.optimizer.step()
    skin.record_optimizer_step()
    dc.apply(model, after_step=True)
    model.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        scale.project()
    attempted = set()
    split = refine_surface_distributions(model, skin, scale, step, attempted_ids=attempted,
        diagnostic_dir=OUT/f'archived-split-canonical-binding-{subject}')
    assert split
    return dict(subject=subject, native_views=len(rows), frozen_mesh_sha256=rt.sha(snapshot/'reference-mesh.npz'),
        bound_ply_sha256=manifest['bound_ply_sha256'], native_pose_sample=rows[sample]['frame'],
        isolated_prior_gradient_abs_sum=dict(zip(('xyz','scaling','rotation'),(float(g.abs().sum()) for g in isolated))),
        training_support=skin.report(), archive_split_gate=split, attempted_ids=sorted(attempted))


def main():
    output=OUT/'cuda-surface-distribution-proof.json'
    if output.exists():
        raise FileExistsError(output)
    OUT.mkdir(parents=True,exist_ok=True)
    rt.configure(cold.ASSETS)
    rows=[]
    for subject in ('016','006'):
        print('Verifying R22 geometry on '+subject,flush=True)
        rows.append(subject_probe(subject))
    names=('surface_attachment.py','surface_densification.py','splat_surface_distance.py',
           'render_support.py','size_child_commit.py','size_child_optimization.py','training.py')
    rt.save(output,dict(state='passed',schema='shuolee-r22-cuda-surface-proof/1',gpu=torch.cuda.get_device_name(),
        fixture_sha256=rt.sha(Path(__file__)),runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
            raster_extension_sha256=rt.sha(Path(_C.__file__))),
        source_sha256={f'src/shuolee_portrait/{name}':rt.sha(PROJECT/'src/shuolee_portrait'/name) for name in names},
        split_policy=SURFACE_DENSIFICATION, subjects=rows))
    print(output,rt.sha(output),flush=True)

if __name__=='__main__':
    main()
