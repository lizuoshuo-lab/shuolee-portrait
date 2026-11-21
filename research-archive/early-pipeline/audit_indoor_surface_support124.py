"""One read-only native alpha/binding audit; no fit, gradient or optimizer."""
import fcntl
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path[:0]=['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src',
             '/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
B=ROOT/'indoor-selected-sh0-pipeline124';Z=ROOT/'indoor-person-zoom2x124'
OUT=ROOT/'indoor-surface-support-audit124'

def main():
    assert not OUT.exists(),'One audit only; preserve errors rather than restart.'
    OUT.mkdir()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=rt.read(B/'data/training-inputs.json')['records']
    candidates=[('base1000',B,1000),('base18000',B,18000),('zoom1000',Z,1000)]
    paths=[Path(__file__),B/'plan.json',Z/'plan.json']
    for _,root,step in candidates:
        paths += [root/'gaussian-run'/f'step-{step:06d}'/n for n in ('bound-local.ply','world.ply','current-meshes.npz','binding.npz','result.json')]
    hashes={str(p):rt.sha(p) for p in paths}
    rt.save(OUT/'plan.json',dict(updated=rt.now(),source_hashes=hashes,
        purpose='Differentiate observed-view opacity gaps and point allocation from unobserved-view failure before new training.',
        inputs='Same frozen indoor6 source views; baseline1000, baseline18000 and ROI2x1000.',
        tests='Same native camera RGB replay, alpha via all-white foreground over black,10px interior-mask coverage, per-triangle binding and opacity, original clock exposure.',
        thresholds='Diagnostic alpha0.5 and0.9 fixed in advance; not training or pruning gates.',
        budget_seconds=120,optimizer_updates=0,quality_claim=False))
    rt.configure(ASSETS)
    import torch
    import cv2
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera,image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;cv2.setNumThreads(1)
    class Bound(GaussianModel):
        select_mesh_by_timestep=SMPLXGaussianModel.select_mesh_by_timestep
        get_covariance=SMPLXGaussianModel.get_covariance
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');results=[];alpha_by={};started=time.monotonic()
    with np.load(B/'fit/surface-cache.npz',allow_pickle=False) as z:surface=dict(z)
    with torch.no_grad():
        for name,root,step in candidates:
            snap=root/'gaussian-run'/f'step-{step:06d}';source_rows=rt.read(root/'data/training-inputs.json')['records']
            with np.load(snap/'current-meshes.npz',allow_pickle=False) as z:mesh=dict(z)
            assert list(mesh['frame_ids'])==[r['frame'] for r in source_rows]
            worlds=[]
            for i,row in enumerate(source_rows):
                E=np.asarray(row['w2c']);worlds.append((mesh['vertices_camera'][i]-E[:3,3])@E[:3,:3])
            model=Bound(0);model.load_ply(str(snap/'bound-local.ply'));model.active_sh_degree=0
            model.vertices_sequence=torch.as_tensor(np.asarray(worlds),dtype=torch.float32,device='cuda')
            model.faces=torch.as_tensor(mesh['faces'],device='cuda',dtype=torch.long)
            model.static_offset=None;model.select_mesh_by_timestep(0)
            plain=GaussianModel(0);plain.load_ply(str(snap/'world.ply'))
            center_error=float((model.get_xyz-plain.get_xyz).abs().max())
            assert center_error<3e-6,center_error
            bindings=model.binding.long().cpu().numpy();op=model.get_opacity[:,0].cpu().numpy()
            counts=np.bincount(bindings,minlength=len(mesh['faces']))
            clothing=np.any(surface['clothing_vertices'][mesh['faces']],axis=1)
            protected=np.any(surface['protected_face_vertices'][mesh['faces']],axis=1)
            areas=np.linalg.norm(np.cross(worlds[0][mesh['faces'][:,1]]-worlds[0][mesh['faces'][:,0]],
                worlds[0][mesh['faces'][:,2]]-worlds[0][mesh['faces'][:,0]]),axis=1)*.5
            regions={}
            for region,ids in [('clothing_triangles',clothing),('protected_face_triangles',protected),('other_triangles',~(clothing|protected))]:
                point_ids=ids[bindings]
                regions[region]=dict(triangles=int(ids.sum()),points=int(point_ids.sum()),
                    point_fraction=float(point_ids.mean()),mesh_area_fraction=float(areas[ids].sum()/areas.sum()),
                    empty_triangles=int((counts[ids]==0).sum()),median_points_per_triangle=float(np.median(counts[ids])),
                    opacity_under005=int((op[point_ids]<.005).sum()),median_opacity=float(np.median(op[point_ids])))
            stats=[]
            for i,(source,row) in enumerate(zip(source_rows,rows)):
                assert source['frame']==row['frame']
                model.select_mesh_by_timestep(i);cam,_=camera(source,i)
                rgb=render(cam,model,pipe,bg)['render']
                saved=np.asarray(Image.open(snap/(source['frame']+'_gaussian.png')).convert('RGB'),dtype=np.int16)
                replay=np.asarray(image(rgb),dtype=np.int16);error=np.abs(saved-replay)
                assert error.mean()<.02 and error.max()<=3,(name,source['frame'],float(error.mean()),int(error.max()))
                # Return to the identical original-pixel source camera for all candidates.
                cam,_=camera(row,i)
                white=torch.ones((len(model._xyz),3),device='cuda')
                alpha3=render(cam,model,pipe,bg,override_color=white)['render']
                assert float((alpha3[0]-alpha3[1]).abs().max())==0
                alpha=alpha3[0].clamp(0,1).cpu().numpy();alpha_by[(name,row['frame'])]=alpha
                mask=np.asarray(Image.open(row['mask']).convert('L'))
                known=np.asarray(Image.open(row['validity']).convert('L'))==255
                interior=(cv2.erode((mask>127).astype(np.uint8),np.ones((21,21),np.uint8))>0)&known
                assert interior.any()
                stats.append(dict(frame=row['frame'],interior_known_pixels=int(interior.sum()),
                    alpha_under05=int((alpha[interior]<.5).sum()),alpha_under09=int((alpha[interior]<.9).sum()),
                    alpha_under05_fraction=float((alpha[interior]<.5).mean()),alpha_under09_fraction=float((alpha[interior]<.9).mean()),
                    mean_alpha=float(alpha[interior].mean()),same_native_png_mean_abs_byte=float(error.mean()),
                    same_native_png_max_abs_byte=int(error.max())))
                assert time.monotonic()-started<120
            opt=rt.read(root/'gaussian-run/effective-optimization.json')
            events=[s for s in range(1,step) if s<opt['densify_until_iter'] and s>opt['densify_from_iter'] and s%opt['densification_interval']==0]
            results.append(dict(name=name,points=len(model._xyz),step=step,regions=regions,frames=stats,
                original_clock_densify_events_before_saved_step=events,world_ply_center_replay_max_m=center_error))
            del model,plain
        pages=[]
        for row in sorted(rows,key=lambda r:r['frame']):
            frame=row['frame'];raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
            x0,y0,x1,y1=mask.getbbox();box=(max(0,x0-20),max(0,y0-20),min(raw.width,x1+20),min(raw.height,y1+20))
            pics=[raw.crop(box),mask.convert('RGB').crop(box)]
            for name,_,_ in candidates:pics.append(Image.fromarray((alpha_by[name,frame]*255).round().astype(np.uint8)).convert('RGB').crop(box))
            panel=Image.new('RGB',(pics[0].width*5,pics[0].height),'black')
            for j,pic in enumerate(pics):panel.paste(pic,(j*pic.width,0))
            filename=f'分析_室内破碎支持124_{frame}_原图_mask_原1000alpha_原18000alpha_放大1000alpha.png'
            target=Path('/mnt/d/00/记录/008')/filename;assert not target.exists();panel.save(target)
            s=min(1,350/panel.height,1600/panel.width)
            pages.append(panel.resize((round(panel.width*s),round(panel.height*s)),Image.Resampling.LANCZOS))
        contact=Image.new('RGB',(max(p.width for p in pages),sum(p.height for p in pages)),(30,30,30));y=0
        for page in pages:contact.paste(page,(0,y));y+=page.height
        contact.save(OUT/'alpha-all6.png')
    assert all(rt.sha(p)==digest for p,digest in hashes.items())
    rt.save(OUT/'result.json',dict(updated=rt.now(),state='readonly_audit_completed',results=results,
        source_hashes_unchanged=True,optimizer_updates=0,seconds=time.monotonic()-started,
        alpha_scope='Accumulated opacity; not depth correctness, tissue visibility or full material correspondences.',
        regions_overlap_possible=True,occluded_pixels_excluded=True,actual_visual_review=False,quality_accepted=False))
    print('SURFACE_SUPPORT_AUDIT_COMPLETE',[(r['name'],r['points']) for r in results],flush=True)

if __name__=='__main__':main()
