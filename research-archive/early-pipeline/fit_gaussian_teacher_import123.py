"""One bounded pilot on original VHAP import geometry, with fixed120k teacher.

No Gaussian training, residual-based teacher filtering or new camera/pose solver.
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

CONTRACT=AUDIT/'original-import-teacher-contract'
OUT=AUDIT/'teacher-import-surface-1000'
NATIVE=AUDIT/'native-alpha-contribution'
LAP=AUDIT/'upstream-laplacian-contract'
PREVIOUS=AUDIT/'teacher-visible-surface-100'
HELPER=Path(__file__).with_name('gaussian_surface_guidance123.py')
VHAP=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx')


def active_gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    return {int(x.strip()) for x in value.splitlines() if x.strip().isdigit()}


def prepare():
    verify();result=io.read(CONTRACT/'result.json');assert result['frames']==59 and result['skeletal_joints_frozen']
    assert result['numerical_contract_passed'] and result['coverage']['all59_nonempty_support']
    assert io.read(PREVIOUS/'result.json')['state']=='stopped_by_geometry_guard'
    native=io.read(NATIVE/'result.json');assert native['state']=='native_contribution_audit_complete_no_optimization' and native['frames']==59
    assert max(x['conservation_relative'] for x in native['frames_result'])<1e-4
    lap=io.read(LAP/'result.json');assert lap['state']=='passed_original_laplacian_cpu_contract'
    assert io.sha(LAP/'original-laplacian.npz')==lap['sparse_cache_sha256']
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),HELPER,CONTRACT/'result.json',CONTRACT/'original-import-lbs-cache.npz',
        VHAP/'vhap/util/render_nvdiffrast.py',VHAP/'vhap/util/vector_ops.py',
        LAP/'original-laplacian.npz',LAP/'result.json',VHAP/'vhap/model/tracker.py',VHAP/'vhap/config/base.py',
        NATIVE/'native-contribution-weights.npz',NATIVE/'result.json',CONTRACT/'teacher-support-diagnostic-only.npz']
    io.save(OUT/'plan.json',dict(created=io.now(),frames=59,max_steps=1000,max_seconds=300,
        hashes={str(p):io.sha(p) for p in files},source_step=120000,gaussian_parameters_frozen=True,
        stage='Exploratory shared clothing-surface distillation from original VHAP import and immutable120k teacher; not Gaussian training or a physical-surface acceptance.',
        teacher_policy=['Original frame identity, native pixels, full K and masks.',
            'Current bound triangle retained; use only centers whose perpendicular projection is inside that triangle.',
            'Same positive depth/native center mask and measured foreground T*alpha visibility, recomputed inside-bound-triangle support on original import.',
            'All three bound vertices must be in the predefined clothing-only anatomical group.',
            'Normalize teacher point weights within a triangle; no density-based dominance.',
            'Teacher visibility remains a proxy, not independently verified physical surface.'],
        trainable_dominant_joint_ids=[0,1,2,3,4,5,6,9,13,14,16,17,18,19],
        protected='Head/neck, hands, feet and original skeleton are frozen. Boundary edges remain in the smoothness term.',
        optimizer=dict(name='Adam',lr=.0001,betas=[.9,.999],eps=1e-8),
        guide_beta_m=.005,guide_weight=1.,laplacian_weight=1e6,
        loss='mean per-frame robust teacher guide +1e6 original uniform Laplacian-coordinate delta squared, summed XYZ and averaged all10475 vertices',
        changed_factor='Original VHAP import mesh and its original frozen LBS replace the degraded120k mesh. Same immutable120k target positions and Gaussian IDs; inside-bound-triangle support and footpoints are consistently recomputed on this reference. This coupled initialization/association change is explicit. Same Adam,LR,1e6 additional-delta Laplacian,59frames and geometric gates.1000step ceiling, not a claim of matched-step causal quality improvement.',
        unit_audit='1e6 additional-delta Laplacian preserves the imported learned clothing shape. The failed total-offset shrinkage control is not imported. No coefficient/LR sweep.1000step ceiling allows slow coherent motion; all20step guards and2cm displacement stop remain.',
        teacher_uncertainty='Only2604/6921clothing faces supported; canonical descriptor dispersion median1.588mm. This admits a guarded research pilot only, not physical correspondence or production training.',
        trainable_initialization='originalVHAP import plus zero additional shared canonical10475x3 displacement; empty optimizer',
        frozen=['source mesh','skeletal transforms','pose','shape','expression','K','w2c','Gaussian center/shape/color/opacity/binding'],
        checks_every=20,stop=dict(nonfinite=True,new_normal_reversal_count=0,min_face_area_ratio=.1,max_added_offset_m=.02),
        admission=dict(guide_relative_reduction=.05,max_frame_mask_iou_regression=.005,max_frame_mesh_outside_increase=.005,
            new_normal_reversal_count=0,actual_all59_visual_review_required=True),
        no_alpha_or_RGB_training_term=True,no_SH_change=True,actual_foreground_alpha_contribution_weights=True,no_threshold_sweep=True,no_automatic_new_run=True,quality_accepted=False))
    print('IMPORT_TEACHER_PILOT_REGISTERED',flush=True)


def checked():
    verify();p=io.read(OUT/'plan.json')
    for f,h in p['hashes'].items():assert io.sha(f)==h,f
    return p


def replay(pid):
    p=checked();assert process(pid)['state']=='T';assert active_gpu_pids()<={pid,os.getpid()};start=time.monotonic()
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
    with np.load(CONTRACT/'original-import-lbs-cache.npz',allow_pickle=False) as z:cache=dict(z)
    with np.load(LAP/'original-laplacian.npz',allow_pickle=False) as z:lap_cache=dict(z)
    with np.load(CONTRACT/'teacher-support-diagnostic-only.npz',allow_pickle=False) as z:teacher=dict(z)
    with np.load(io.ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True) as z:dominant=np.asarray(z['weights']).argmax(1)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];slots={str(f):i for i,f in enumerate(cache['frame_ids'])}
    for row in rows:
        assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
    base=torch.tensor(cache['base_vertices'],device='cuda');A=torch.tensor(cache['linear'],device='cuda')
    faces=torch.tensor(cache['faces'],device='cuda',dtype=torch.long)
    learn=torch.tensor(np.isin(dominant,p['trainable_dominant_joint_ids']),device='cuda')
    assert learn.shape==(10475,) and 0<int(learn.sum())<len(learn)
    assert np.array_equal(teacher['frame_ids'],[r['frame'] for r in rows])
    binding=torch.tensor(teacher['binding'],device='cuda',dtype=torch.long)
    targets=torch.tensor(teacher['targets'],device='cuda')
    barys=torch.tensor(teacher['barycentric_per_frame'],device='cuda')
    weights=torch.tensor(teacher['weights'],device='cuda')
    contract_result=io.read(CONTRACT/'result.json')
    support_records=[dict(frame=r['frame'],points=r['points'],triangles=r['faces']) for r in contract_result['frames_result']]
    normal_support=[dict(frame=r['frame'],positive_normal_weight=r['positive_normal_weight'],
        caveat='Original-import mesh normal sign; not anatomical outward or physical clothing truth.') for r in contract_result['frames_result']]
    assert torch.allclose(weights.sum(1),torch.ones(59,device='cuda'),atol=1e-6)
    assert torch.isfinite(targets).all() and torch.isfinite(barys).all()
    order=torch.tensor([slots[r['frame']] for r in rows],device='cuda')
    base=base[order];A=A[order];tri_vid=faces[binding]
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
        if step%20==1:assert active_gpu_pids()<={pid,os.getpid()},'Foreign GPU process appeared'
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
            name=f'SMPL-X_原VHAP网格高斯引导_{completed}步_{row["frame"]}_原图_原网格_引导网格.png'
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
        regularizer_source='Original PyTorch3D uniform Laplacian applied to new shared delta relative to frozen VHAP import,1e6weight; this is a local fine-tuning prior, not a claim of original VHAP total-offset semantics.',
        mask_evaluation=evaluation,images=images,numerical_candidate_passed=numerical_pass,actual_visual_review=False,
        elapsed_seconds=time.monotonic()-start,skeleton_pose_shape_camera_unchanged=True,teacher_Gaussians_unchanged=True,
        no_Gaussian_training=True,no_production_source_changed=True,PLY_exported=False,quality_accepted=False,
        interpretation='Surface teacher fit is not physical clothing truth. Original pose errors and protected head/hand/foot geometry are not repaired by this pilot. No held-out reconstruction quality or15-minute E2E acceptance.')
    io.save(OUT/'result.json',result);io.save(OUT/'progress.json',dict(state=result['state'],pid=None,steps=completed,stop_reason=stop_reason))
    print('IMPORT_TEACHER_PILOT_FINISHED',result['state'],result['guide_reduction_fraction'],flush=True)


def run(pid):
    checked();assert not (OUT/'result.json').exists();lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(pid);assert pid==331 and before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert 'train_demo_body_upstream_offset123.py train' in before['command'];assert active_gpu_pids()<={pid}
    io.save(OUT/'gpu-handoff.json',dict(started=io.now(),baseline_before=before,state='suspending'))
    child=None
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(50):
            if process(pid)['state']=='T':break
            time.sleep(.05)
        assert process(pid)['state']=='T'
        child=subprocess.Popen([sys.executable,'-B',__file__,'replay','--pid',str(pid)])
        code=child.wait(timeout=300);assert code==0,code
    except BaseException as err:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
        io.save(OUT/'failure.json',dict(updated=io.now(),error=repr(err),traceback=traceback.format_exc()));raise
    finally:
        assert process(pid)['start_identity']==before['start_identity']
        foreign=active_gpu_pids()-{pid}-(set() if child is None else {child.pid})
        if not foreign:os.kill(pid,signal.SIGCONT)
        io.save(OUT/'gpu-handoff.json',dict(finished=io.now(),baseline_before=before,baseline_after=process(pid),
            state='resumed' if not foreign else 'left_suspended_due_to_foreign_GPU',foreign_gpu_pids=sorted(foreign)))
        print('BASELINE_STATE',process(pid),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay']);parser.add_argument('--pid',type=int);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:globals()[args.mode](args.pid)
