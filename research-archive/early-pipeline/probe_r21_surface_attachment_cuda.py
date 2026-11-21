"""Verify native face guidance and exact Adam/CUDA checkpoint resume on real R20 inputs."""
import hashlib
import io
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import launch_unified_cold_016_006 as cold

PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-head-support-20261111')
OUT=Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round21/input-audit'
sys.path.insert(0,str(PROJECT/'src'))

from shuolee_portrait import runtime as rt
from shuolee_portrait.bound_replay import RAW_PARAMETERS, _inputs, _load_model
from shuolee_portrait.duration_checkpoint import capture_duration_checkpoint, restore_duration_checkpoint
from shuolee_portrait.sh0_main import Constraints, DCColorProjection
from shuolee_portrait.surface_attachment import SurfaceAttachment
from shuolee_portrait.training import _frozen_smpl_surface, _native_foreground_target, camera, render_pose, smpl_sh0_effective_config
from shuolee_portrait.world_scale import WorldScale
from gaussian_renderer import render
from diff_gaussian_rasterization import _C
from utils.sh_utils import C0


def subject_probe(subject):
    root=cold.LAB/f'shuolee-smpl-delivery-round20-{subject}/run'
    snapshot=root/'gaussian-run/step-010000'
    path=snapshot/'bound-replay.json'
    manifest=rt.read(path)
    asset=rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    fit,mesh,facts=_inputs(root,snapshot,asset)
    assert manifest['inputs_sha256']==facts['inputs_sha256']
    assert manifest['bound_ply_sha256']==rt.sha(snapshot/'bound-local.ply')
    model,_=_load_model(snapshot/'bound-local.ply',snapshot,fit,mesh,facts,asset)
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
                               projected=fit['joints_projected_native'])
        measured=skin.measure()
    sample=max(range(len(rows)),key=lambda i: measured['views'][i]['supported_rows'])
    assert measured['views'][sample]['supported_rows']>0
    dc=DCColorProjection(model._features_dc,C0)
    random.seed(4521); np.random.seed(731); torch.manual_seed(814)
    def update(step):
        model.update_learning_rate(step)
        pkg=render_pose(render,cameras[sample],model,pipe,bg,slots[sample])
        g=skin.loss(sample,pkg['visibility_filter'])
        grads=torch.autograd.grad(g,(model._xyz,model._scaling,model._rotation),retain_graph=True)
        assert g.isfinite() and all(bool(torch.isfinite(d).all()) for d in grads)
        (g*.02 + pkg['render'].mean()*.001).backward()
        fixed = {group['name']:group['params'][0].grad.detach().clone()
                 for group in model.optimizer.param_groups if group['params'][0].grad is not None}
        model.optimizer.step()
        dc.apply(model,after_step=True)
        model.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scale.project()
        return float(g.detach()),[float(x.abs().sum()) for x in grads],fixed
    loss,grad,fixed=update(10000)
    assert grad[0]>0 and (grad[1]>0 or grad[2]>0), (subject,grad)
    state=capture_duration_checkpoint(model,scale,dc)
    stream=io.BytesIO()
    torch.save(state,stream)
    serialized=hashlib.sha256(stream.getvalue()).hexdigest()
    stream.seek(0)
    restored=torch.load(stream,map_location='cpu',weights_only=True)
    def step_fixed():
        model.update_learning_rate(10001)
        for group in model.optimizer.param_groups:
            if group['name'] in fixed:
                group['params'][0].grad=fixed[group['name']].clone()
        model.optimizer.step()
        dc.apply(model,after_step=True)
        model.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scale.project()

    draws=(random.random(),np.random.random(),torch.rand(8),torch.rand(8,device='cuda'))
    step_fixed()
    after={name:getattr(model,name).detach().clone() for name in RAW_PARAMETERS}
    adam={g['name']:int(model.optimizer.state[g['params'][0]]['step']) if model.optimizer.state.get(g['params'][0]) else 0
          for g in model.optimizer.param_groups}
    restore_duration_checkpoint(model,scale,dc,restored)
    now=(random.random(),np.random.random(),torch.rand(8),torch.rand(8,device='cuda'))
    assert draws[:2]==now[:2] and all(torch.equal(a,b) for a,b in zip(draws[2:],now[2:],strict=True))
    # Repeat the same CUDA Adam update using exact saved native gradients.
    # We do not claim a separate raster backward run is deterministic.
    for name,packed in restored['parameters'].items():
        assert torch.equal(getattr(model,name).detach().cpu(),packed['value']['tensor'])
    step_fixed()
    max_parameter_difference=max(float((after[k]-getattr(model,k)).abs().max()) for k in after if after[k].numel())
    assert max_parameter_difference==0.,(subject,max_parameter_difference)
    return dict(subject=subject,diagnostic_native_views=len(rows),frozen_archive_sha256=manifest['bound_ply_sha256'],
        support=skin.report(),visible_skin_metrics=measured,native_pose_sample=rows[sample]['frame'],
        attachment_loss=loss,absolute_grad_sum=dict(center=grad[0],scale=grad[1],rotation=grad[2]),
        checkpoint=dict(schema=restored['schema'],serialized_sha256=serialized,scheduled_xyz_lr=restored['optimizer']['groups']['xyz']['lr']['value'],
            weights_only_load=True,cuda_cpu_and_numpy_python_rng_exact=True,adam_steps_after_next=adam,
            next_update_max_abs_parameter_difference=max_parameter_difference))


def main():
    output=OUT/'cuda-surface-attachment-proof.json'
    if output.exists():
        raise FileExistsError(output)
    rt.configure(cold.ASSETS)
    rows=[]
    for subject in ('016','006'):
        print('Verifying native surface loss and full state for '+subject,flush=True)
        rows.append(subject_probe(subject))
    names=('surface_attachment.py','duration_checkpoint.py','clothing_surface_guide.py')
    rt.save(output,dict(state='passed',schema='shuolee-r21-cuda-surface-proof/1',gpu=torch.cuda.get_device_name(),
        fixture_sha256=rt.sha(Path(__file__)), runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
            raster_extension_sha256=rt.sha(Path(_C.__file__))),
        source_sha256={f'src/shuolee_portrait/{name}':rt.sha(PROJECT/'src/shuolee_portrait'/name) for name in names},
        subjects=rows,cloth_proofs={s:rt.sha(OUT/f'cuda-clothing-guide-{s}.json') for s in ('016','006')}))
    print(output,rt.sha(output),flush=True)

if __name__=='__main__':
    main()
