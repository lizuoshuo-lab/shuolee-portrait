"""Frozen native renderer contribution census; no optimizer, fitting or pruning."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
sys.path[:0] = ['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src',
                '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS
from audit_body_anisotropy123 import fields, ply, unit, triangle_distance, summary, PARTS, MODEL

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
B = ROOT/'indoor-selected-sh0-pipeline124'
D = ROOT/'indoor-zoom-clock18k124'
OUT = ROOT/'indoor-directional-support124'
CANDIDATES = [('original18k', B), ('zoom_clock18k', D)]
INDICES = (0, 9, 18, 27)


def contribution_summary(values, weights):
    report = summary(values, weights)
    for key in ('median', 'p90'):
        report['contribution_weighted_'+key] = report.pop('opacity_weighted_'+key)
    return report


def prepare():
    assert not OUT.exists(), 'One registered audit only.'
    paths = [Path(__file__), Path(__file__).with_name('audit_body_anisotropy123.py'), MODEL,
             B/'data/training-inputs.json', B/'delivery/step-018000/novel/cameras.json']
    for _, root in CANDIDATES:
        paths += [root/'gaussian-run/step-018000'/n for n in
                  ('world.ply', 'bound-local.ply', 'binding.npz', 'current-meshes.npz', 'result.json')]
        paths += [root/'fit/fitted.npz', root/'delivery/step-018000/point_cloud.ply']
    for row in rt.read(B/'data/training-inputs.json')['records']:
        paths += [Path(row[k]) for k in ('image', 'mask', 'validity')]
    core = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core')
    paths += [core/n for n in ('gaussian_renderer/__init__.py', 'scene/gaussian_model.py', 'scene/smplx_gaussian_model.py')]
    OUT.mkdir()
    rt.save(OUT/'plan.json', dict(updated=rt.now(), source_hashes={str(p):rt.sha(p) for p in paths},
        candidates=[n for n, _ in CANDIDATES], budget_seconds=180,
        questions=['Are actual known-foreground contributions concentrated in one source view?',
                   'Do visible Gaussians remain close to their own bound triangle and mesh normal?',
                   'Do the same four frozen novel views expose weakly supported surface regions?'],
        measurement='Derivative of summed native RGB with respect to per-point override color equals alpha-compositing contribution. Three channels accumulate known foreground, known background and unknown pixels separately. Six original-pixel cameras for both models; not their exact loss weighting.',
        controls=['Fixed geometry and all model tensors require_grad=False; only diagnostic colors require gradients.',
                  'Sum per-point derivatives equals summed alpha per region; finite color perturbation validates linearity.',
                  'Independently reconstruct bound-local centers/scales in world space and check existing export.',
                  'No selected frames removed; all six source views and four pre-existing uniform novel cameras.'],
        statistics=['Normalize each view contribution by known foreground pixel count; per-point effective views =1/sum(q*q).',
                    'Fixed dominance>0.9 and per-point source fraction>=0.05 are descriptive, never pruning/training gates.',
                    'Anatomy groups use mean triangle skinning weights; own-triangle distance is not distance to true clothing.'],
        limitations=['Observed directional support is not independent physical correspondence or ground truth geometry.',
                     'Flat surface Gaussians are not by themselves errors. Snapshot point sets differ after densification.',
                     'The zoom/clock model differs from original18k in two factors; do not attribute all change to zoom.',
                     'Excluded indoor legs and hidden regions remain unknown; no new negative supervision.'],
        optimizer_updates=0, training_changed=False, quality_accepted=False))
    print('DIRECTIONAL_SUPPORT_PLAN_READY', flush=True)


def verify(plan):
    for p, h in plan['source_hashes'].items():
        assert rt.sha(p) == h, p


def run():
    plan = rt.read(OUT/'plan.json'); verify(plan)
    assert not (OUT/'progress.json').exists()
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rt.save(OUT/'progress.json', dict(state='running_readonly', pid=os.getpid(), updated=rt.now()))
    started = time.monotonic()
    def deadline(*_): raise TimeoutError('Registered read-only budget exceeded')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(plan['budget_seconds'])
    try:
        rt.configure(ASSETS)
        import torch
        from scene.gaussian_model import GaussianModel
        from scene.smplx_gaussian_model import SMPLXGaussianModel
        from gaussian_renderer import render
        from shuolee_portrait.training import camera, image
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        class Bound(GaussianModel):
            select_mesh_by_timestep = SMPLXGaussianModel.select_mesh_by_timestep
            get_covariance = SMPLXGaussianModel.get_covariance
        def freeze(model):
            for val in vars(model).values():
                if isinstance(val, torch.Tensor) and val.is_leaf:
                    val.requires_grad_(False)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        bg = torch.zeros(3, device='cuda')
        rows = rt.read(B/'data/training-inputs.json')['records']; assert len(rows) == 6
        orbit = rt.read(B/'delivery/step-018000/novel/cameras.json')['records']
        weights = np.load(MODEL, allow_pickle=False)['weights']
        reports, arrays, panels = [], {}, {}
        for label, root in CANDIDATES:
            snap = root/'gaussian-run/step-018000'
            mesh = dict(np.load(snap/'current-meshes.npz', allow_pickle=False))
            init = dict(np.load(root/'fit/fitted.npz', allow_pickle=False))
            assert list(mesh['frame_ids']) == [r['frame'] for r in rows] == list(init['frame_ids'])
            verts = np.asarray([(v-np.asarray(r['w2c'])[:3,3]) @ np.asarray(r['w2c'])[:3,:3]
                                for v, r in zip(mesh['vertices_camera'], rows)])
            model = Bound(0); model.load_ply(str(snap/'bound-local.ply')); model.active_sh_degree = 0
            model.vertices_sequence = torch.as_tensor(verts, dtype=torch.float32, device='cuda')
            model.faces = torch.as_tensor(mesh['faces'], dtype=torch.long, device='cuda')
            model.static_offset = None; freeze(model); model.select_mesh_by_timestep(0)
            world = ply(snap/'world.ply'); local = ply(snap/'bound-local.ply')
            xyz = fields(world, ['x','y','z']); loc = fields(local, ['x','y','z'])
            scales = np.exp(fields(world, ['scale_0','scale_1','scale_2']))
            bind = np.load(snap/'binding.npz', allow_pickle=False)['triangle_ids']
            assert np.array_equal(bind, model.binding.cpu().numpy())
            tri = verts[0][mesh['faces']][bind]
            e0 = unit(tri[:,1]-tri[:,0]); normal = unit(np.cross(e0, tri[:,2]-tri[:,0])); e2 = -unit(np.cross(normal,e0))
            orient = np.stack((e0, normal, e2), -1)
            factor = .5*(np.linalg.norm(tri[:,1]-tri[:,0], axis=1)+np.abs(np.sum(e2*(tri[:,2]-tri[:,0]),1)))
            reconstructed = np.einsum('nij,nj->ni', orient, loc)*factor[:,None]+tri.mean(1)
            xe = float(np.abs(reconstructed-xyz).max())
            se = float(np.abs(np.exp(fields(local, ['scale_0','scale_1','scale_2']))*factor[:,None]-scales).max())
            assert xe < 2e-6 and se < 2e-6 and float((model.get_xyz-torch.tensor(xyz, device='cuda')).abs().max()) < 3e-6
            ordered = np.sort(scales,axis=1); dist, plane, inside = triangle_distance(xyz,tri)
            raw, tests, directions = [], [], []
            for i, row in enumerate(rows):
                model.select_mesh_by_timestep(i); cam,_ = camera(row,i)
                mask = np.asarray(Image.open(row['mask']).convert('L')) > 127
                known = np.asarray(Image.open(row['validity']).convert('L')) == 255
                regions = torch.as_tensor(np.stack((mask & known, ~mask & known, ~known)), dtype=torch.float32, device='cuda')
                colors = torch.ones((len(xyz),3),device='cuda',requires_grad=True)
                alpha = render(cam,model,pipe,bg,override_color=colors)['render']
                grad, = torch.autograd.grad((alpha*regions).sum(), colors)
                assert torch.isfinite(grad).all() and float(grad.min()) >= -1e-7
                summed = (alpha.detach()*regions).sum((1,2)); accumulated = grad.sum(0)
                err = float(((summed-accumulated).abs()/summed.clamp_min(1)).max())
                assert err < 1e-4, err
                test = dict(frame=row['frame'],alpha_gradient_sum_relative_error=err,known_foreground_pixels=int((mask&known).sum()))
                if i == 0:
                    generator = torch.Generator(device='cuda').manual_seed(124)
                    delta = .125*(2*torch.rand(colors.shape, generator=generator, device='cuda')-1)
                    with torch.no_grad(): shifted = render(cam,model,pipe,bg,override_color=colors.detach()+delta)['render']
                    observed = ((shifted-alpha.detach())*regions).sum((1,2)); predicted = (grad*delta).sum(0)
                    linerr = float(((observed-predicted).abs()/summed.clamp_min(1)).max())
                    assert linerr < 1e-5, linerr
                    test['color_perturbation_sum_relative_error'] = linerr
                raw.append(grad.detach().cpu().numpy())
                # Triangle-local ray directions permit posed-frame comparisons without claiming true material trajectories.
                t = verts[i][mesh['faces']][bind]
                u = unit(t[:,1]-t[:,0]); n = unit(np.cross(u,t[:,2]-t[:,0])); v = -unit(np.cross(n,u))
                basis = np.stack((u,n,v),-1)
                ray = unit(cam.camera_center.detach().cpu().numpy()-model.get_xyz.detach().cpu().numpy())
                directions.append(np.einsum('nji,nj->ni',basis,ray))
                tests.append(test)
                del colors,alpha,grad
            raw = np.asarray(raw,dtype=np.float64)
            fg = raw[:,:,0]/np.asarray([t['known_foreground_pixels'] for t in tests])[:,None]
            total = fg.sum(0); live = total > 0
            q = np.divide(fg,total[None,:],out=np.zeros_like(fg),where=live[None,:])
            effective = np.divide(1,(q*q).sum(0),out=np.zeros(len(xyz)),where=live)
            dominance = q.max(0)
            directional = np.linalg.norm((q[:,:,None]*np.asarray(directions)).sum(0),axis=-1)
            dispersion = np.degrees(np.arccos(np.clip(directional,0,1)))
            joint = weights[mesh['faces']].mean(1).argmax(1)[bind]
            values = dict(elongation=ordered[:,2]/ordered[:,1],flattening=ordered[:,1]/ordered[:,0],
                          sigma_max_mm=ordered[:,2]*1000,bound_triangle_distance_mm=dist*1000,bound_plane_distance_mm=plane*1000,
                          effective_views=effective,dominant_view_share=dominance,triangle_local_ray_dispersion_deg=dispersion)
            regions_out = {}
            for name, ids in {'all':list(range(55)),**PARTS}.items():
                sel = np.isin(joint,ids); supported = sel & live
                regions_out[name] = dict(points=int(sel.sum()),zero_known_foreground_contribution=int((sel&~live).sum()),
                    contribution_share=float(total[sel].sum()/total.sum()),
                    contribution_carried_by_dominance_over09=float(total[sel & (dominance>.9)].sum()/max(total[sel].sum(),1e-30)),
                    metrics={k:contribution_summary(val[supported],total[supported]) for k,val in values.items()} if supported.any() else {},
                    center_projection_inside_bound_triangle_fraction=float(inside[sel].mean()))
            arrays[label+'_contribution_by_view_and_region'] = raw
            arrays[label+'_effective_views'] = effective
            arrays[label+'_binding'] = bind
            arrays[label+'_anatomy_joint'] = joint
            arrays.update({label+'_'+key:val for key,val in values.items()})
            # Frozen +Y export has the same Gaussian ordering, checked explicitly.
            plain = GaussianModel(0); plain.load_ply(str(root/'delivery/step-018000/point_cloud.ply')); freeze(plain)
            assert np.max(np.abs(plain.get_xyz.cpu().numpy()-xyz*np.array([1,-1,-1]))) < 3e-6
            amount = np.clip((effective-1)/5,0,1)
            color = np.stack((1-amount,amount,np.full_like(amount,.12)),1).astype(np.float32)
            color[~live] = [.3,.3,.3]
            color = torch.as_tensor(color,device='cuda')
            panels[label] = []
            with torch.no_grad():
                for index in INDICES:
                    cam,_ = camera(orbit[index],index)
                    native = image(render(cam,plain,pipe,bg)['render'])
                    support = image(render(cam,plain,pipe,bg,override_color=color)['render'])
                    pair = Image.new('RGB',(native.width*2,native.height))
                    pair.paste(native,(0,0));pair.paste(support,(native.width,0))
                    pair.save(OUT/f'{label}-orbit-{index:02d}-rgb-support.png');panels[label].append(pair)
            delta_mesh = np.linalg.norm(mesh['vertices_camera']-init['vertices_camera'],axis=-1)*1000
            reports.append(dict(name=label,points=len(xyz),regions=regions_out,controls=tests,
                                world_center_max_abs_m=xe,world_scale_max_abs_m=se,
                                mesh_displacement_mm_all6=summary(delta_mesh.ravel())))
            print('DIRECTIONAL_SUPPORT_SNAPSHOT',label,len(xyz),regions_out['all'],flush=True)
            del model,plain
        verify(plan)
        np.savez_compressed(OUT/'point-metrics.npz',**arrays)
        rows_images = []
        for index in range(len(INDICES)):
            parts = [panels[label][index] for label,_ in CANDIDATES]
            row = Image.new('RGB',(parts[0].width*2,parts[0].height))
            for i,p in enumerate(parts):row.paste(p,(i*p.width,0))
            s = min(1,1800/row.width);rows_images.append(row.resize((round(row.width*s),round(row.height*s)),Image.Resampling.LANCZOS))
        sheet = Image.new('RGB',(rows_images[0].width,sum(p.height for p in rows_images)+44),(22,22,22))
        draw = ImageDraw.Draw(sheet)
        draw.text((12,8),'Old RGB | old support | zoom+clock RGB | support. Red: 1 effective view; green: 6; gray: no known-foreground contribution.',fill='white')
        y = 44
        for row in rows_images:sheet.paste(row,(0,y));y+=row.height
        sheet.save(OUT/'four-orbits-support.png')
        target = Path('/mnt/d/00/记录/008/分析_室内放大124_四方向_原图与实际视角贡献.png')
        assert not target.exists();sheet.save(target)
        rt.save(OUT/'result.json',dict(updated=rt.now(),state='readonly_completed',reports=reports,
            source_hashes_unchanged=True,optimizer_updates=0,elapsed_seconds=time.monotonic()-started,
            actual_visual_review=False,quality_accepted=False,limitations=plan['limitations']))
        rt.save(OUT/'progress.json',dict(state='completed',pid=None,updated=rt.now()))
        print('DIRECTIONAL_SUPPORT_AUDIT_COMPLETE',flush=True)
    except BaseException as error:
        rt.save(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        rt.save(OUT/'progress.json',dict(state='failed_preserved',pid=None,updated=rt.now()))
        raise
    finally:signal.alarm(0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
