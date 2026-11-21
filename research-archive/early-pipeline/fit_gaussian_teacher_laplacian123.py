"""One new registered comparison: original normalized Laplacian replaces weak edge penalty.

No production source mutation, no Gaussian optimizer, no pose refit.
"""
from pathlib import Path
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,SNAP,OUT as AUDIT,process,verify

CONTRACT=AUDIT/'frozen-skeleton-contract'
OUT=AUDIT/'teacher-surface-laplacian-100'
LAP=AUDIT/'upstream-laplacian-contract'
PREVIOUS=AUDIT/'teacher-surface-100'
HELPER=Path(__file__).with_name('gaussian_surface_guidance123.py')
VHAP=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')


def prepare():
    verify();result=io.read(CONTRACT/'result.json');assert result['frames']==59 and result['skeletal_joints_frozen']
    assert io.sha(CONTRACT/'frozen-lbs-cache.npz')==result['cache_sha256']
    assert io.read(PREVIOUS/'result.json')['state']=='stopped_by_geometry_guard'
    lap=io.read(LAP/'result.json');assert lap['state']=='passed_original_laplacian_cpu_contract'
    assert io.sha(LAP/'original-laplacian.npz')==lap['sparse_cache_sha256']
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),HELPER,CONTRACT/'result.json',CONTRACT/'frozen-lbs-cache.npz',
        VHAP/'vhap/util/render_nvdiffrast.py',VHAP/'vhap/util/vector_ops.py',
        LAP/'original-laplacian.npz',LAP/'result.json',VHAP/'vhap/model/tracker.py',VHAP/'vhap/config/base.py']
    io.save(OUT/'plan.json',dict(created=io.now(),frames=59,max_steps=100,max_seconds=300,
        hashes={str(p):io.sha(p) for p in files},source_step=120000,gaussian_parameters_frozen=True,
        stage='Shared clothing-surface distillation pilot from an immutable Gaussian teacher; NOT Gaussian continuation or pose optimization.',
        teacher_policy=['Original frame identity, native pixels, full K and masks.',
            'Current bound triangle retained; use only centers whose perpendicular projection is inside that triangle.',
            'Positive depth, inside native mask, original raster frustum visibility; opacity supplies a fixed continuous weight.',
            'All three bound vertices must be in the predefined clothing-only anatomical group.',
            'Normalize teacher point weights within a triangle; no density-based dominance.',
            'Teacher visibility remains a proxy, not independently verified physical surface.'],
        trainable_dominant_joint_ids=[0,1,2,3,4,5,6,9,13,14,16,17,18,19],
        protected='Head/neck, hands, feet and original skeleton are frozen. Boundary edges remain in the smoothness term.',
        optimizer=dict(name='Adam',lr=.0001,betas=[.9,.999],eps=1e-8),
        guide_beta_m=.005,guide_weight=1.,laplacian_weight=1e6,
        loss='mean per-frame robust teacher guide +1e6 original uniform Laplacian-coordinate delta squared, summed XYZ and averaged all10475 vertices',
        changed_factor='Only replace weak local edge regularizer with exact upstream Laplacian formula and its VHAP default weight. Same teacher, zero initial delta, Adam, LR, masks,59frames, checks and limits.',
        unit_audit='At the rejected20-step delta, this regularizer is157.9 times the teacher objective. It may prevent deformation or overconstrain; no coefficient search is permitted in this comparison.',
        trainable_initialization='zero additional shared canonical10475x3 displacement; empty optimizer',
        frozen=['source mesh','skeletal transforms','pose','shape','expression','K','w2c','Gaussian center/shape/color/opacity/binding'],
        checks_every=20,stop=dict(nonfinite=True,new_normal_reversal_count=0,min_face_area_ratio=.1,max_added_offset_m=.02),
        admission=dict(guide_relative_reduction=.05,max_frame_mask_iou_regression=.005,max_frame_mesh_outside_increase=.005,
            new_normal_reversal_count=0,actual_all59_visual_review_required=True),
        no_alpha_or_RGB_training_term=True,no_SH_change=True,no_threshold_sweep=True,no_automatic_new_run=True,quality_accepted=False))
    print('TEACHER_LAPLACIAN_CONTROL_REGISTERED',flush=True)


def checked():
    verify();p=io.read(OUT/'plan.json')
    for f,h in p['hashes'].items():assert io.sha(f)==h,f
    return p


def replay(pid):
    p=checked();assert process(pid)['state']=='T';start=time.monotonic()
    import torch
    from PIL import Image
    from gaussian_surface_guidance123 import closest_bound_barycentrics
    sys.path.insert(0,str(io.PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(io.ASSETS)
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera
    from types import SimpleNamespace
    sys.path.insert(0,str(VHAP))
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with np.load(CONTRACT/'frozen-lbs-cache.npz',allow_pickle=False) as z:cache=dict(z)
    with np.load(LAP/'original-laplacian.npz',allow_pickle=False) as z:lap_cache=dict(z)
    with np.load(io.ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True) as z:dominant=np.asarray(z['weights']).argmax(1)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];slots={str(f):i for i,f in enumerate(cache['frame_ids'])}
    base=torch.tensor(cache['base_vertices'],device='cuda');A=torch.tensor(cache['linear'],device='cuda')
    faces=torch.tensor(cache['faces'],device='cuda',dtype=torch.long)
    learn=torch.tensor(np.isin(dominant,p['trainable_dominant_joint_ids']),device='cuda')
    assert learn.shape==(10475,) and 0<int(learn.sum())<len(learn)
    model=SMPLXGaussianModel(3);model.load_ply(str(SNAP/'bound-local.ply'))
    # PLY stores binding as int32; scatter accumulation requires int64 indices.
    model.binding=model.binding.long()
    model.vertices_sequence=base;model.faces=faces;model.num_timesteps=59;model.static_offset=None;model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    targets=[];barys=[];weights=[];support_records=[]
    perface_train=learn[faces].all(-1)
    normal_support=[]
    for row in rows:
        assert time.monotonic()-start<p['max_seconds']
        assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
        slot=slots[row['frame']];model.select_mesh_by_timestep(slot);cam,_=camera(row,slot,pose_index=slot)
        with torch.no_grad():
            xyz=model.get_xyz.detach().clone();tri=base[slot][faces[model.binding]]
            bary,valid,inside=closest_bound_barycentrics(tri,xyz)
            visible=render(cam,model,pipe,bg)['visibility_filter']
            E=torch.tensor(row['w2c'],device='cuda',dtype=xyz.dtype);K=torch.tensor(row['K'],device='cuda',dtype=xyz.dtype)
            vc=xyz@E[:3,:3].T+E[:3,3];q=vc@K.T;uv=q[:,:2]/q[:,2:].clamp_min(1e-9)
            ix=uv[:,0].floor().long();iy=uv[:,1].floor().long();w,h=row['wh']
            mask=torch.as_tensor(np.asarray(Image.open(row['mask']).convert('L')).copy(),device='cuda').float()/255
            inimage=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(vc[:,2]>0)
            use=valid&inside&visible&inimage&perface_train[model.binding]
            weight=use*model.get_opacity[:,0].detach()*mask[iy.clamp(0,h-1),ix.clamp(0,w-1)]
            sums=torch.zeros(len(faces),device='cuda');sums.scatter_add_(0,model.binding,weight)
            weight=weight/sums[model.binding].clamp_min(1e-20)
            assert float(weight.sum())>0
            weight=weight/weight.sum()
            normal=torch.nn.functional.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
            signed=((xyz-(tri*bary[...,None]).sum(1))*normal).sum(-1)
            normal_support.append(dict(frame=row['frame'],positive_normal_weight=float(weight[signed>0].sum()),
                negative_normal_weight=float(weight[signed<0].sum()),weighted_signed_distance_m=float((weight*signed).sum()),
                caveat='Mesh-normal relative sign only; no proof of clothing visibility or anatomical outwardness.'))
            targets.append(xyz);barys.append(bary);weights.append(weight)
            support_records.append(dict(frame=row['frame'],points=int((weight>0).sum()),triangles=int((sums>0).sum())))
    targets=torch.stack(targets);barys=torch.stack(barys);weights=torch.stack(weights)
    order=torch.tensor([slots[r['frame']] for r in rows],device='cuda')
    base=base[order];A=A[order];tri_vid=faces[model.binding]
    L=torch.sparse_coo_tensor(torch.tensor(lap_cache['indices'],device='cuda'),
        torch.tensor(lap_cache['values'],device='cuda'),tuple(lap_cache['shape']),device='cuda').coalesce()
    assert np.array_equal(lap_cache['faces'],cache['faces'])
    init_tri=base[:,faces];init_norm=torch.cross(init_tri[:,:,1]-init_tri[:,:,0],init_tri[:,:,2]-init_tri[:,:,0],dim=-1)
    init_area=init_norm.norm(dim=-1)
    delta=torch.nn.Parameter(torch.zeros((10475,3),device='cuda'));optimizer=torch.optim.Adam([delta],**{k:v for k,v in p['optimizer'].items() if k!='name'})
    def verts():return base+(A@(delta*learn[:,None])[None,:,:,None]).squeeze(-1)
    def losses():
        v=verts();surface=(v[:,tri_vid]*barys[:,:,:,None]).sum(2)
        raw=torch.nn.functional.smooth_l1_loss(surface,targets,beta=p['guide_beta_m'],reduction='none').sum(-1)
        guide=(raw*weights).sum(1).mean()
        lap_delta=torch.sparse.mm(L,delta*learn[:,None])
        smooth=lap_delta.square().sum(-1).mean()
        return guide,smooth,guide+p['laplacian_weight']*smooth
    def geometry():
        v=verts();tri=v[:,faces];normal=torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1)
        valid=init_area>1e-12
        return dict(new_normal_reversal_count=int(((normal*init_norm).sum(-1)[valid]<=0).sum()),
            min_face_area_ratio=float((normal.norm(dim=-1)[valid]/init_area[valid]).min()),
            max_added_offset_m=float((delta*learn[:,None]).norm(dim=-1).max()),
            protected_vertex_max_abs_m=float((v[:,~learn]-base[:,~learn]).abs().max()))
    with torch.no_grad():g0,s0,l0=map(float,losses());initial_geometry=geometry()
    history=[];stop_reason=None;completed=0
    for step in range(1,p['max_steps']+1):
        assert time.monotonic()-start<p['max_seconds']
        optimizer.zero_grad();guide,smooth,total=losses();assert torch.isfinite(total);total.backward()
        assert torch.isfinite(delta.grad).all() and not torch.any(delta.grad[~learn])
        optimizer.step();completed=step
        if step%p['checks_every']==0 or step==p['max_steps']:
            with torch.no_grad():gg,ss,ll=map(float,losses());gate=geometry()
            history.append(dict(step=step,guide=gg,smoothness=ss,total=ll,geometry=gate))
            if gate['new_normal_reversal_count']>p['stop']['new_normal_reversal_count']:stop_reason='new_triangle_normal_reversal'
            elif gate['min_face_area_ratio']<p['stop']['min_face_area_ratio']:stop_reason='triangle_area_collapse'
            elif gate['max_added_offset_m']>p['stop']['max_added_offset_m']:stop_reason='surface_displacement_limit'
            assert gate['protected_vertex_max_abs_m']==0
            io.save(OUT/'progress.json',dict(state='fitting_surface',pid=os.getpid(),step=step,guide=gg,stop_reason=stop_reason,elapsed_seconds=time.monotonic()-start))
            print('TEACHER_SURFACE_STEP',step,gg,gate,flush=True)
            if stop_reason:break
    with torch.no_grad():candidate=verts().detach();final_losses=list(map(float,losses()));final_geometry=geometry()
    np.savez_compressed(OUT/'candidate-diagnostic-only.npz',frame_ids=np.array([r['frame'] for r in rows]),
        vertices_before=base.cpu().numpy(),vertices_after=candidate.cpu().numpy(),faces=cache['faces'],
        additional_canonical_delta=(delta*learn[:,None]).detach().cpu().numpy(),learnable_vertices=learn.cpu().numpy(),
        skeleton=cache['skeletal_joints'][order.cpu().numpy()],K=np.array([r['K'] for r in rows]),w2c=np.array([r['w2c'] for r in rows]))
    # Real upstream triangle rasterization, full K and native image dimensions.
    renderer=NVDiffRenderer();cv_to_gl=torch.diag(torch.tensor([1.,-1.,-1.,1.],device='cuda'))
    images=[];evaluation=[];folder=OUT/'review';folder.mkdir();record=Path('/mnt/d/00/记录/demo');record.mkdir(exist_ok=True)
    with torch.no_grad():
        for i,row in enumerate(rows):
            assert time.monotonic()-start<p['max_seconds']
            raw=np.asarray(Image.open(row['image']).convert('RGB'));mask=np.asarray(Image.open(row['mask']).convert('L'))>127
            E=torch.tensor(row['w2c'],device='cuda',dtype=base.dtype);RT=(cv_to_gl@E)[None]
            K=torch.tensor(row['K'],device='cuda',dtype=base.dtype)[None];w,h=row['wh']
            panels=[raw];masks=[]
            for v in (base[i],candidate[i]):
                rast=renderer.rasterize(v[None],faces,RT,K,(h,w),use_cache=False)['rast_out'][0].flip(0)
                visible=rast[:,:,-1]>0;masks.append(visible.cpu().numpy());ids=(rast[:,:,-1].long()-1).clamp_min(0)
                vc=v@E[:3,:3].T+E[:3,3];tri=vc[faces];n=torch.nn.functional.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
                shade=.3+.7*(n@torch.tensor([.3,-.2,-.9327379],device='cuda')).abs()
                rgb=shade[ids,None]*torch.tensor([80.,210.,220.],device='cuda')
                overlay=raw.copy();sel=masks[-1];rendered=rgb.cpu().numpy();overlay[sel]=(.6*rendered[sel]+.4*raw[sel]).clip(0,255).astype(np.uint8);panels.append(overlay)
            def metrics(m):return dict(iou=float((m&mask).sum()/max(1,(m|mask).sum())),outside=float((m&~mask).sum()/max(1,m.sum())))
            before,after=map(metrics,masks)
            evaluation.append(dict(frame=row['frame'],before=before,after=after,iou_change=after['iou']-before['iou'],outside_change=after['outside']-before['outside']))
            yy,xx=np.where(mask);box=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21)]
            crops=[Image.fromarray(x).crop(box) for x in panels];sheet=Image.new('RGB',(crops[0].width*3,crops[0].height))
            for k,img in enumerate(crops):sheet.paste(img,(k*img.width,0))
            name=f'SMPL-X_高斯引导原库Laplacian_{completed}步_{row["frame"]}_原图_原网格_引导网格.png'
            path=folder/name;sheet.save(path);destination=record/name;assert not destination.exists();destination.write_bytes(path.read_bytes())
            images.append(dict(frame=row['frame'],path=str(path),record_path=str(destination),sha256=io.sha(path),crop_integer_xyxy=box,no_resampling=True,no_text_added=True))
            print('TEACHER_SURFACE_REVIEW',i+1,row['frame'],flush=True)
    checked()
    numerical_pass=(stop_reason is None and final_losses[0]<=g0*(1-p['admission']['guide_relative_reduction']) and
        min(r['iou_change'] for r in evaluation)>=-p['admission']['max_frame_mask_iou_regression'] and
        max(r['outside_change'] for r in evaluation)<=p['admission']['max_frame_mesh_outside_increase'])
    result=dict(state='stopped_by_geometry_guard' if stop_reason else 'bounded_pilot_complete_pending_visual_review',updated=io.now(),steps=completed,stop_reason=stop_reason,
        frames=59,trainable_vertices=int(learn.sum()),protected_vertices=int((~learn).sum()),initial_guide_loss=g0,final_guide_loss=final_losses[0],
        guide_reduction_fraction=1-final_losses[0]/g0,history=history,geometry=final_geometry,teacher_support=support_records,teacher_normal_support=normal_support,
        regularizer_source='Exact original PyTorch3D uniform Laplacian and VHAP coordinate-change squared loss;1e6default. Shared additional static-delta use is a local adapter.',
        mask_evaluation=evaluation,images=images,numerical_candidate_passed=numerical_pass,actual_visual_review=False,
        elapsed_seconds=time.monotonic()-start,skeleton_pose_shape_camera_unchanged=True,teacher_Gaussians_unchanged=True,
        no_Gaussian_training=True,no_production_source_changed=True,PLY_exported=False,quality_accepted=False,
        interpretation='Surface teacher fit is not physical clothing truth. Original pose errors and protected head/hand/foot geometry are not repaired by this pilot. No held-out reconstruction quality or15-minute E2E acceptance.')
    io.save(OUT/'result.json',result);io.save(OUT/'progress.json',dict(state=result['state'],pid=None,steps=completed,stop_reason=stop_reason))
    print('TEACHER_LAPLACIAN_CONTROL_FINISHED',result['state'],result['guide_reduction_fraction'],flush=True)


def run(pid):
    checked();assert not (OUT/'result.json').exists()
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(pid);assert before['state'] not in ('T','Z') and 'train_demo_body_upstream_offset123.py train' in before['command']
    io.save(OUT/'gpu-handoff.json',dict(started=io.now(),baseline=before,state='suspending'))
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)['state']=='T':break
            time.sleep(.05)
        assert process(pid)['state']=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--pid',str(pid)],check=True,timeout=360)
    except BaseException as error:
        io.save(OUT/'failure.json',dict(updated=io.now(),error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        assert process(pid)['start_identity']==before['start_identity']
        os.kill(pid,signal.SIGCONT)
        io.save(OUT/'gpu-handoff.json',dict(finished=io.now(),baseline_before=before,baseline_after=process(pid),state='resumed'))
        print('BASELINE_RESUMED',pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay']);parser.add_argument('--pid',type=int);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:globals()[args.mode](args.pid)
