"""Fresh ordinary static world Gaussians from the observed clothing envelope.

The avatar mesh is a proposal extent/scale, never a position target. This local
route reuses foreground-only image supervision and the exact camera contract.
"""
import argparse
import json
import time
from types import SimpleNamespace
import numpy as np
from . import runtime as rt
from .training import Views,foreground_loss,image
from .clothed_surface import make_model,center_statistics


def train_clothed(root,cfg):
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from arguments import OptimizationParams
    init=rt.read(root/'clothing-init/audit.json')
    assert init['point_sha256']==rt.sha(root/'clothing-init/points.npz')
    assert init['fit_sha256']==rt.sha(root/'fit/fitted.npz') and init['points']==200000
    assert not init['body_attraction'] and not init['per_triangle_quota']
    out=root/'gaussian-run';out.mkdir()
    fit=dict(np.load(root/'fit/fitted.npz'));rows=rt.read(root/'data/training-inputs.json')['records']
    assert np.max(abs(fit['vertices_world']-fit['vertices_world'][0]))==0
    seed=cfg['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    views=Views(rows,cfg['training_height'],cfg['image_cache_frames'])
    rt.save(out/'training-cameras.json',dict(records=list(views.records),source_sha256=rt.sha(root/'data/training-inputs.json'),interpolated_cameras=0))
    model=make_model(root);scale=model.world_scale
    parser=argparse.ArgumentParser();options=OptimizationParams(parser).extract(parser.parse_args([]))
    steps=cfg['iterations'];options.iterations=steps;options.position_lr_max_steps=steps
    # The old avatar rates are in triangle-local units. These ordinary3DGS
    # rates operate in world units, multiplied by this fresh body's scale.
    options.position_lr_init=.00016;options.position_lr_final=.0000016
    options.scaling_lr=.005;options.lambda_xyz=0.;options.lambda_scale=0.
    options.densify_until_iter=steps;model.training_setup(options)
    assert model.binding is None and model.binding_counter is None and len(model.optimizer.state)==0
    initial_xyz=model.get_xyz.detach().clone()
    low=initial_xyz.min(0).values-.1*scale;high=initial_xyz.max(0).values+.1*scale
    extent=float(torch.linalg.norm(initial_xyz.max(0).values-initial_xyz.min(0).values))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    rt.save(out/'config.json',dict(**cfg,route='local_static_clothed_world',upstream_complete_reproduction=False,
        execution_kind='finite_clothing_distribution_research',formal_training_admitted=False,
        from_scratch=True,old_subject_checkpoint_loaded=False,optimizer_initial_state_entries=0,
        body_attraction=False,per_triangle_quota=False,canonical_xyz_free_world_coordinates=True,
        initialization_sha256=init['point_sha256'],fit_sha256=init['fit_sha256'],
        position_lr_init=options.position_lr_init,position_lr_final=options.position_lr_final,
        spatial_lr_scale=scale,scaling_lr=options.scaling_lr,
        position_constraint='Only broad surface bounding box+0.1m; no attraction to body or initial points.',
        scale_constraint='Penalty on standard deviation above0.03model-m, no display/kernel inflation.',
        source_sha256=rt.sha(__file__)))
    cam,gt,alpha=views.get(0);pkg=render(cam,model,pipe,bg)
    photo,_,_=foreground_loss(pkg['render'],gt,alpha);other=gt.clone();other[:,alpha[0]<=.99]=.37
    outside_effect=float(abs(photo-foreground_loss(pkg['render'],other,alpha)[0]))
    if outside_effect>1e-7:raise RuntimeError('Background RGB leaked into foreground loss.')
    aa=render(cam,model,pipe,bg,override_color=torch.ones_like(model.get_xyz))['render'][:1]
    (photo+.1*(aa-alpha).abs().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
    nonzero_gradient=int((model._xyz.grad.norm(dim=1)>0).sum())
    assert nonzero_gradient>0
    model.optimizer.zero_grad(set_to_none=True)
    rt.save(out/'preflight.json',dict(foreground_external_RGB_effect=outside_effect,finite_backward=True,
        nonzero_world_position_gradients=nonzero_gradient,optimizer_updates=0,binding_is_none=True,
        get_xyz_is_exact_world_parameter=torch.equal(model._xyz,model.get_xyz)))
    del pkg,photo,aa,other
    rng=np.random.default_rng(seed);stack=[];counts=np.zeros(len(rows),np.int32);start=time.monotonic()
    selected=np.unique(np.linspace(0,len(rows)-1,6).astype(int))
    with (out/'loss.jsonl').open('x') as log:
        for step in range(1,steps+1):
            model.update_learning_rate(step)
            if step%1000==0:model.oneupSHdegree()
            if not stack:stack=rng.permutation(len(rows)).tolist()
            i=stack.pop();counts[i]+=1;cam,gt,alpha=views.get(i)
            pkg=render(cam,model,pipe,bg);photo,l1,ssim=foreground_loss(pkg['render'],gt,alpha)
            aa=render(cam,model,pipe,bg,override_color=torch.ones_like(model.get_xyz))['render'][:1]
            silhouette=(aa-alpha).abs().mean();visible=pkg['visibility_filter']
            if not visible.any():raise RuntimeError('No visible points in a fixed real camera.')
            bounds=(torch.relu(low-model._xyz)+torch.relu(model._xyz-high)).mean()/scale
            large=torch.relu(model.get_scaling/scale-.03).square().mean()
            loss=photo+.1*silhouette+.01*bounds+.1*large
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at{step}')
            loss.backward()
            with torch.no_grad():
                model.max_radii2D[visible]=torch.maximum(model.max_radii2D[visible],pkg['radii'][visible])
                if step<steps:model.add_densification_stats(pkg['viewspace_points'],visible)
                model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
                if options.densify_from_iter<step<steps and step%options.densification_interval==0:
                    model.densify_and_prune(options.densify_grad_threshold,.005,extent,None)
                    if not 1000<len(model._xyz)<650000:raise RuntimeError('Registered point budget exceeded.')
                assert model.binding is None
            if step==1 or step%100==0:
                progress=dict(updated=rt.now(),stage='training',step=step,target_steps=steps,points=len(model.get_xyz),
                    loss=float(loss),foreground_l1=float(l1),foreground_ssim=float(ssim),alpha_mae=float(silhouette),
                    seconds=time.monotonic()-start,views_seen=int((counts>0).sum()),quality_accepted=False,
                    representation='shared_static_world_clothed_surface',execution_kind='finite_clothing_distribution_research')
                rt.save(root/'progress.json',progress);log.write(json.dumps(progress)+'\n');log.flush();print(progress,flush=True)
            if step in [1000,6000,12000,steps]:
                with torch.no_grad():
                    snapshot=out/f'step-{step:06d}';snapshot.mkdir();model.export_world_ply(snapshot/'world.ply',0)
                    reloaded=GaussianModel(3);reloaded.load_ply(str(snapshot/'world.ply'));reloaded.active_sh_degree=model.active_sh_degree
                    checks=[]
                    for j in selected:
                        cam,gt,alpha=views.get(int(j));native=render(cam,model,pipe,bg)['render']
                        delta=abs(native-render(cam,reloaded,pipe,bg)['render'])
                        checks.append(dict(frame=rows[j]['frame'],mean_abs=float(delta.mean()),max_abs=float(delta.max())))
                        image(gt).save(snapshot/f'{rows[j]["frame"]}_input.png')
                        image(gt*alpha).save(snapshot/f'{rows[j]["frame"]}_masked_input.png')
                        image(native).save(snapshot/f'{rows[j]["frame"]}_render.png')
                    passed=all(r['mean_abs']<2e-6 and r['max_abs']<.005 for r in checks)
                    rt.save(snapshot/'native-ply-parity.json',dict(passed=passed,checks=checks,sha256=rt.sha(snapshot/'world.ply')))
                    if not passed:raise RuntimeError('World PLY does not reproduce native geometry.')
                    xyz=model.get_xyz.cpu().numpy();stats=center_statistics(xyz,fit,rows)
                    if len(xyz)==len(initial_xyz):
                        movement=torch.linalg.norm(model.get_xyz-initial_xyz,dim=1)/scale
                        stats['center_movement_model_m']={str(q):float(torch.quantile(movement,q)) for q in [.5,.9,.99]}
                    std=model.get_scaling/scale
                    stats['kernel_max_std_model_m']={str(q):float(torch.quantile(std.max(1).values,q)) for q in [.5,.9,.99]}
                    stats['binding_is_none']=model.binding is None
                    rt.save(snapshot/'center-distribution.json',stats);del reloaded
    rt.save(out/'result.json',dict(state='completed',
        from_scratch=True,steps=step,schedule_steps=steps,execution_kind='finite_clothing_distribution_research',
        formal_training_admitted=False,per_frame_updates=counts.tolist(),final_snapshot=f'step-{step:06d}',quality_accepted=False,
        representation='shared_static_world_clothed_surface',old_checkpoint_loaded=False,binding=False))
