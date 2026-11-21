"""Bounded clothing-only mesh control using corrected original AA derivatives.

No Gaussian center teacher, new pose/camera solver, or facial mesh updates.
The only optimized state is the existing shared canonical clothing delta.
"""
from pathlib import Path
import argparse,fcntl,os,signal,subprocess,sys,time,traceback
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,OUT as AUDIT,process,verify
from fit_gaussian_teacher_import123 import active_gpu_pids
from test_isolated_aa_grad123 import OUT as FIX,load_variants

OUT=AUDIT/'mask-clothing-surface-590'
CACHE=AUDIT/'original-import-teacher-contract/original-import-lbs-cache.npz'
LAP=AUDIT/'upstream-laplacian-contract/original-laplacian.npz'
FACE=AUDIT/'face-landmark-boundary'
VHAP=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')


def prepare():
    verify();OUT.mkdir(exist_ok=False)
    contract=io.read(FIX/'all59-mask-contract/result.json')
    assert contract['state']=='passed_native_mask_alpha_gradient_contract' and contract['frames']==59
    assert max(r['protected_gradient_max_abs'] for r in contract['frames_result'])==0
    boundary=io.read(FACE/'result.json');assert boundary['overlap_vertices']==0
    paths=[Path(__file__),CACHE,LAP,FACE/'vertex-boundary.npz',FACE/'face-geometry-policy.json',
        FIX/'synthetic-result.json',FIX/'build-result.json',FIX/'all59-mask-contract/result.json',
        ROOT/'data/training-inputs.json',Path(__file__).with_name('native_mesh_mask123.py'),
        Path(__file__).with_name('facial_geometry_boundary123.py'),Path(__file__).with_name('test_isolated_aa_grad123.py'),
        VHAP/'vhap/util/render_nvdiffrast.py']
    build=io.read(FIX/'build-result.json')
    paths.extend(Path(x['module']) for x in build['builds'])
    rows=io.read(ROOT/'data/training-inputs.json')['records']
    assert len(rows)==59 and len({r['frame'] for r in rows})==59
    io.save(OUT/'plan.json',dict(created=io.now(),state='registered',hashes={str(p):io.sha(p) for p in paths},
        hypothesis='Direct original-image silhouette gradients may correct clothing outlines that a Gaussian-center teacher shrank. This does not establish depth or solve frozen bone/pose defects.',
        factor='Replace the failed Gaussian-center teaching objective with native original-mask L1 using the verified AA backward repair. Retain original VHAP mesh initialization, shared canonical offset, fixed original LBS, optimizer and delta-Laplacian. This is an explicitly different objective, not a claim that K alone improved.',
        loss='One chronological frame per update: sum(abs(alpha-original mask))/fixed foreground area +1e6*mean_vertices(sum_xyz((L*masked_delta)^2)).',
        optimizer=dict(name='Adam',lr=.0001,betas=[.9,.999],eps=1e-8),laplacian_weight=1e6,
        max_steps=590,full_passes=10,frame_cycle=sorted(r['frame'] for r in rows),frames=59,
        max_training_seconds=180,max_total_seconds=300,max_parent_seconds=330,checks_every=20,
        initialization='Original VHAP import, zero additional delta, empty Adam; no rejected geometry or old optimizer.',
        frozen=['all cameras','body/hand/head poses','shape/expression','skeleton','all Gaussian parameters','head/neck/face','hands','feet'],
        face_rule='Every clothing loss and smoothness term uses clothing_only_offset with explicit head/landmark exclusion. Independent future facial fitting is allowed; not permanently frozen.',
        input_rule='All59 original native2704x1520 frames and masks, fullK, identity-based joins. No resize, warp, mask relabeling, residual pruning or selected-frame fitting.',
        stop=dict(nonfinite=True,new_normal_reversal_count=0,min_face_area_ratio=.1,max_added_offset_m=.02),
        admission=dict(mean_mask_loss_relative_reduction=.05,max_frame_mask_iou_regression=.005,
            max_frame_mesh_outside_increase=.005,protected_vertex_motion_m=0,all59_visual_review_required=True),
        no_SH_change=True,no_Gaussian_center_teacher=True,no_parameter_sweep=True,no_automatic_repeat=True,
        installed_runtime_modified=False,quality_accepted=False))
    print('CLOTHING_MASK_CONTROL_REGISTERED',flush=True)


def checked():
    verify();plan=io.read(OUT/'plan.json')
    for f,h in plan['hashes'].items():assert io.sha(f)==h,f
    return plan


def replay(pid):
    p=checked();assert process(pid)['state']=='T' and active_gpu_pids()<={pid,os.getpid()}
    begin=time.monotonic()
    import torch
    import nvdiffrast.torch.ops as ops
    from PIL import Image
    from facial_geometry_boundary123 import clothing_only_offset
    from native_mesh_mask123 import native_alpha,foreground_normalized_mask_loss
    sys.path.insert(0,str(VHAP))
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    modules=load_variants();original_backward=ops._nvdiffrast_c.antialias_grad
    ops._nvdiffrast_c.antialias_grad=modules['direct'].antialias_grad
    with np.load(CACHE,allow_pickle=False) as z:cache=dict(z)
    with np.load(LAP,allow_pickle=False) as z:lap=dict(z)
    with np.load(FACE/'vertex-boundary.npz',allow_pickle=False) as z:boundary=dict(z)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];ids=[r['frame'] for r in rows]
    assert ids==list(cache['frame_ids'])
    assert np.array_equal(cache['faces'],lap['faces']) and np.array_equal(cache['faces'],boundary['faces'])
    base=torch.tensor(cache['base_vertices'],device='cuda');A=torch.tensor(cache['linear'],device='cuda')
    faces=torch.tensor(cache['faces'],device='cuda',dtype=torch.long)
    learn=torch.tensor(boundary['clothing_vertices'],device='cuda',dtype=torch.bool)
    face=torch.tensor(boundary['protected_face_vertices'],device='cuda',dtype=torch.bool)
    assert not torch.any(face & learn)
    delta=torch.nn.Parameter(torch.zeros((10475,3),device='cuda'))
    optimizer=torch.optim.Adam([delta],**{k:v for k,v in p['optimizer'].items() if k!='name'})
    L=torch.sparse_coo_tensor(torch.tensor(lap['indices'],device='cuda'),torch.tensor(lap['values'],device='cuda'),tuple(lap['shape']),device='cuda').coalesce()
    renderer=NVDiffRenderer(lighting_type='constant')
    cv_to_gl=torch.diag(torch.tensor([1.,-1.,-1.,1.],device='cuda'))
    Ks=[];RTs=[];masks=[]
    for row in rows:
        assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
        w,h=row['wh'];mask=np.asarray(Image.open(row['mask']).convert('L'))>127
        assert mask.shape==(h,w)
        masks.append(torch.from_numpy(mask[None,:,:,None]))
        Ks.append(torch.tensor(row['K'],device='cuda',dtype=base.dtype)[None])
        RTs.append((cv_to_gl@torch.tensor(row['w2c'],device='cuda',dtype=base.dtype))[None])
    initial_tri=base[:,faces];initial_normal=torch.cross(initial_tri[:,:,1]-initial_tri[:,:,0],initial_tri[:,:,2]-initial_tri[:,:,0],dim=-1)
    initial_area=initial_normal.norm(dim=-1);valid=initial_area>1e-12
    def safe():return clothing_only_offset(delta,learn,face)
    def verts(i=None):
        d=safe()
        return base+(A@d[None,:,:,None]).squeeze(-1) if i is None else base[i]+(A[i]@d[:,:,None]).squeeze(-1)
    @torch.no_grad()
    def geometry():
        v=verts();t=v[:,faces];n=torch.cross(t[:,:,1]-t[:,:,0],t[:,:,2]-t[:,:,0],dim=-1)
        return dict(new_normal_reversal_count=int(((n*initial_normal).sum(-1)[valid]<=0).sum()),
            min_face_area_ratio=float((n.norm(dim=-1)[valid]/initial_area[valid]).min()),
            max_added_offset_m=float(safe().norm(dim=-1).max()),
            face_displacement_max_abs_m=float((v[:,face]-base[:,face]).abs().max()),
            protected_vertex_max_abs_m=float((v[:,~learn]-base[:,~learn]).abs().max()))
    def alpha_for(v,i):
        w,h=rows[i]['wh'];return native_alpha(renderer,v[None],faces,RTs[i],Ks[i],(h,w))
    @torch.no_grad()
    def metric(v,i):
        alpha,rast=alpha_for(v,i);target=masks[i].to(device='cuda',dtype=base.dtype)
        visible=rast['rast_out'][0,...,-1].flip(0)>0;gt=target[0,:,:,0]>0
        result=dict(mask_loss=float(foreground_normalized_mask_loss(alpha,target)),
            iou=float((visible & gt).sum()/((visible | gt).sum().clamp_min(1))),
            outside=float((visible & ~gt).sum()/visible.sum().clamp_min(1)))
        return result
    initial=[]
    with torch.no_grad():
        for i,row in enumerate(rows):initial.append(metric(base[i],i))
    assert geometry()['protected_vertex_max_abs_m']==0
    order=[ids.index(fid) for fid in p['frame_cycle']];history=[];seen=set();completed=0;stop_reason=None;fit_start=time.monotonic()
    for step in range(1,p['max_steps']+1):
        if time.monotonic()-fit_start>p['max_training_seconds']:
            stop_reason='training_time_budget';break
        if step%20==1:assert active_gpu_pids()<={pid,os.getpid()}
        i=order[(step-1)%len(order)];seen.add(ids[i]);optimizer.zero_grad(set_to_none=True)
        alpha,_=alpha_for(verts(i),i);target=masks[i].to(device='cuda',dtype=base.dtype)
        mask_loss=foreground_normalized_mask_loss(alpha,target)
        smooth=torch.sparse.mm(L,safe()).square().sum(-1).mean()
        loss=mask_loss+p['laplacian_weight']*smooth
        assert torch.isfinite(loss);loss.backward()
        assert torch.isfinite(delta.grad).all() and not torch.any(delta.grad[~learn]) and not torch.any(delta.grad[face])
        optimizer.step();completed=step
        if step%p['checks_every']==0 or step==p['max_steps']:
            gate=geometry()
            if gate['new_normal_reversal_count']>0:stop_reason='new_triangle_normal_reversal'
            elif gate['min_face_area_ratio']<p['stop']['min_face_area_ratio']:stop_reason='triangle_area_collapse'
            elif gate['max_added_offset_m']>p['stop']['max_added_offset_m']:stop_reason='surface_displacement_limit'
            assert gate['protected_vertex_max_abs_m']==0 and gate['face_displacement_max_abs_m']==0
            item=dict(step=step,frame=ids[i],sample_mask_loss=float(mask_loss.detach()),smooth=float(smooth.detach()),geometry=gate)
            history.append(item);io.save(OUT/'progress.json',dict(updated=io.now(),state='fitting_clothing_mask',pid=os.getpid(),step=step,seen_frames=len(seen),geometry=gate,stop_reason=stop_reason))
            print('MASK_FIT',step,ids[i],gate,flush=True)
            if stop_reason:break
        del alpha,target,mask_loss,smooth,loss
    with torch.no_grad():candidate=verts().detach();final_geometry=geometry()
    fitting_seconds=time.monotonic()-fit_start
    np.savez_compressed(OUT/'candidate-diagnostic-only.npz',frame_ids=np.array(ids),faces=cache['faces'],
        vertices_before=base.cpu().numpy(),vertices_after=candidate.cpu().numpy(),
        additional_canonical_delta=safe().detach().cpu().numpy(),learnable_vertices=learn.cpu().numpy(),
        protected_face_vertices=face.cpu().numpy(),skeleton=cache['skeletal_joints'],K=np.array([r['K'] for r in rows]),w2c=np.array([r['w2c'] for r in rows]))
    evaluation=[];images=[];folder=OUT/'review';folder.mkdir();record=Path('/mnt/d/00/记录/demo')
    with torch.no_grad():
        for i,row in enumerate(rows):
            assert time.monotonic()-begin<p['max_total_seconds']
            assert active_gpu_pids()<={pid,os.getpid()}
            raw=np.asarray(Image.open(row['image']).convert('RGB'));mask=masks[i][0,:,:,0].numpy()
            panels=[raw]
            for v in [base[i],candidate[i]]:
                rast=renderer.rasterize(v[None],faces,RTs[i],Ks[i],tuple(reversed(row['wh'])),use_cache=False)['rast_out'][0].flip(0)
                visible=rast[...,-1]>0;triid=(rast[...,-1].long()-1).clamp_min(0)
                E=torch.tensor(row['w2c'],device='cuda',dtype=base.dtype);vc=v@E[:3,:3].T+E[:3,3];t=vc[faces]
                n=torch.nn.functional.normalize(torch.cross(t[:,1]-t[:,0],t[:,2]-t[:,0],dim=-1),dim=-1)
                light=.3+.7*(n@torch.tensor([.3,-.2,-.9327379],device='cuda')).abs()
                rgb=(light[triid,None]*torch.tensor([80.,210.,220.],device='cuda')).cpu().numpy()
                panel=raw.copy();sel=visible.cpu().numpy();panel[sel]=(.6*rgb[sel]+.4*raw[sel]).clip(0,255).astype(np.uint8);panels.append(panel)
            after=metric(candidate[i],i);before=initial[i]
            evaluation.append(dict(frame=row['frame'],before=before,after=after,
                iou_change=after['iou']-before['iou'],outside_change=after['outside']-before['outside']))
            yy,xx=np.where(mask);w,h=row['wh'];box=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21)]
            crops=[Image.fromarray(p).crop(box) for p in panels];sheet=Image.new('RGB',(crops[0].width*3,crops[0].height))
            for j,crop in enumerate(crops):sheet.paste(crop,(j*crop.width,0))
            name=f'SMPL-X_修复反传衣物轮廓_{completed}步_{row["frame"]}_原图_原网格_轮廓网格.png'
            dest=record/name;assert not dest.exists();path=folder/name;sheet.save(path);dest.write_bytes(path.read_bytes())
            images.append(dict(frame=row['frame'],path=str(path),record_path=str(dest),sha256=io.sha(path),no_resampling=True,no_text=True))
            print('MASK_REVIEW',i+1,row['frame'],evaluation[-1]['iou_change'],flush=True)
    ops._nvdiffrast_c.antialias_grad=original_backward
    checked()
    mean_before=float(np.mean([x['before']['mask_loss'] for x in evaluation]));mean_after=float(np.mean([x['after']['mask_loss'] for x in evaluation]))
    numerical=(stop_reason is None and len(seen)==59 and mean_after<=mean_before*.95
        and min(x['iou_change'] for x in evaluation)>=-.005 and max(x['outside_change'] for x in evaluation)<=.005)
    result=dict(updated=io.now(),state='stopped_by_geometry_guard' if stop_reason else 'bounded_mask_pilot_complete_pending_visual_review',
        steps=completed,frames_seen=len(seen),seen_frame_ids=sorted(seen),stop_reason=stop_reason,history=history,geometry=final_geometry,
        mean_mask_loss_before=mean_before,mean_mask_loss_after=mean_after,mask_loss_reduction_fraction=1-mean_after/mean_before,
        mask_evaluation=evaluation,images=images,numerical_candidate_passed=numerical,
        fitting_seconds=fitting_seconds,elapsed_seconds=time.monotonic()-begin,frames=59,
        max_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),actual_all59_visual_review=False,quality_accepted=False,
        frozen_face_skeleton_camera=True,no_Gaussian_training=True,installed_runtime_modified=False,PLY_exported=False,
        scope='Mask loss verifies image outlines, not physical depth, limb pose correction or new-video robustness. No face/hand/foot repair is claimed.')
    io.save(OUT/'result.json',result);io.save(OUT/'progress.json',dict(updated=io.now(),state=result['state'],pid=None,step=completed,stop_reason=stop_reason))
    print('CLOTHING_MASK_FINISHED',result['state'],result['mask_loss_reduction_fraction'],numerical,flush=True)


def run(pid):
    p=checked();assert not (OUT/'result.json').exists() and not (OUT/'started.json').exists()
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(pid);assert pid==331 and before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert 'train_demo_body_upstream_offset123.py train' in before['command'];assert active_gpu_pids()<={pid}
    io.save(OUT/'started.json',dict(updated=io.now(),before=before,script_sha256=io.sha(__file__)))
    child=None
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(50):
            if process(pid)['state']=='T':break
            time.sleep(.05)
        assert process(pid)['state']=='T'
        child=subprocess.Popen([sys.executable,'-B',__file__,'replay','--pid',str(pid)])
        code=child.wait(timeout=p['max_parent_seconds']);assert code==0,code
    except BaseException as error:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
        io.save(OUT/'failure.json',dict(updated=io.now(),error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        assert process(pid)['start_identity']==before['start_identity']
        foreign=active_gpu_pids()-{pid}-(set() if child is None else {child.pid})
        if not foreign:os.kill(pid,signal.SIGCONT)
        io.save(OUT/'gpu-handoff.json',dict(updated=io.now(),before=before,after=process(pid),foreign=sorted(foreign)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:globals()[args.mode](args.pid)
