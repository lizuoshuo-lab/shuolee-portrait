"""Native zero-update contribution audit using the existing color override API.

d(sum foreground RGB_R)/d(carrier_R_i) = sum(mask * T_i * alpha_i).
The carrier is diagnostic only; no saved SH, opacity, geometry or optimizer is
changed. Depth moments remain Gaussian-center moments, not measured surfaces.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,fcntl,os,signal,subprocess,sys,time,traceback
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,SNAP,OUT as AUDIT,verify,process

OUT=AUDIT/'native-alpha-contribution'
CPU=AUDIT/'teacher-surface-definition'
RENDER=io.PROJECT/'src/avatar_core/gaussian_renderer/__init__.py'


def active_gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    return {int(x.strip()) for x in value.splitlines() if x.strip().isdigit()}


def prepare():
    verify();assert io.read(CPU/'result.json')['frames']==59
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),RENDER,SNAP/'current-meshes.npz',SNAP/'bound-local.ply',
        CPU/'result.json',CPU/'center-support-analysis.npz',ROOT/'data/training-inputs.json']
    io.save(OUT/'plan.json',dict(created=io.now(),hashes={str(p):io.sha(p) for p in paths},frames=59,max_seconds=240,
        optimizer_updates=0,source_step=120000,source_vertices='Literal current-meshes.npz keyed by frame_id; no CPU LBS replay as substitute.',
        mechanism='Original render override_color + original CUDA backward; color carrier derivative equals foreground transmittance-alpha contribution.',
        controls=['Pure CPU analytic carrier derivative conservation.', 'Native constant carrier [1,-1,2] control.',
            'Every original RGB render agrees with saved120k native PNG within quantization.',
            'Sum individual foreground carrier gradients equals foreground rendered alpha.',
            'Only first carrier channel receives gradient; learned parameters frozen and unchanged.',
            'All59 original frames and masks; preserve prior anatomical/inside-bound-triangle/center-in-mask admission.'],
        thresholds=dict(alpha_conservation_relative=1e-4,constant_carrier_max_abs=2e-6,native_mean_u8=.02,native_max_u8=2,variance_min_m2=-2e-6),
        changing_for_analysis='Replace detached opacity/frustum proxy weights by actual foreground T*alpha contribution, then retain equalized per-triangle weighting.',
        no_candidate_fit=True,no_new_model_or_PLY=True,quality_accepted=False,
        limitation='Visible contribution is rendering evidence, not independent physical geometry. Moment depth is an alpha-weighted center statistic, not a hard Gaussian surface.',
        resource_rule='Require no foreign CUDA process before suspension and throughout replay. Only suspend/resume known331 identity9229; do not touch other jobs.'))
    print('NATIVE_CONTRIBUTION_REGISTERED',flush=True)


def checked():
    verify();p=io.read(OUT/'plan.json')
    for f,h in p['hashes'].items():assert io.sha(f)==h,f
    return p


def replay(pid):
    p=checked();assert process(pid)['state']=='T';assert active_gpu_pids()<={pid,os.getpid()}
    started=time.monotonic()
    import torch
    from PIL import Image
    sys.path.insert(0,str(io.PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(io.ASSETS)
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    # Exact alpha-compositing carrier identity independent of the native code.
    al=torch.tensor([[.2,.5],[.7,.1],[.1,.8]],dtype=torch.float64)
    T=torch.cumprod(torch.cat([torch.ones_like(al[:1]),1-al[:-1]],0),dim=0);weights=T*al
    carrier=torch.ones(3,dtype=torch.float64,requires_grad=True);pixelmask=torch.tensor([1.,.25],dtype=torch.float64)
    composed=(weights*carrier[:,None]).sum(0);grad=torch.autograd.grad((composed*pixelmask).sum(),carrier)[0]
    assert torch.allclose(grad,(weights*pixelmask).sum(1),atol=1e-15,rtol=0)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:saved=dict(z)
    with np.load(CPU/'center-support-analysis.npz',allow_pickle=False) as z:proxy=dict(z)
    rows=io.read(ROOT/'data/training-inputs.json')['records'];slots={str(x):i for i,x in enumerate(saved['frame_ids'])};pslots={str(x):i for i,x in enumerate(proxy['frame_ids'])}
    model=SMPLXGaussianModel(3);model.load_ply(str(SNAP/'bound-local.ply'));model.binding=model.binding.long()
    model.vertices_sequence=torch.tensor(saved['vertices_camera'],device='cuda');model.faces=torch.tensor(saved['faces'].astype('int64'),device='cuda');model.num_timesteps=59;model.static_offset=None;model.active_sh_degree=3
    learned=[model._xyz,model._scaling,model._rotation,model._opacity,model._features_dc,model._features_rest]
    for param in learned:param.requires_grad_(False)
    initial=[x.detach().clone() for x in learned]
    assert np.array_equal(model.binding.cpu().numpy(),proxy['binding'])
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    records=[];native_weights=[];all_contributions=[];images=[];folder=OUT/'review';folder.mkdir();record=Path('/mnt/d/00/记录/demo')
    constant_control=None
    for i,row in enumerate(rows):
        assert time.monotonic()-started<p['max_seconds'];assert active_gpu_pids()<={pid,os.getpid()},'Foreign GPU task appeared; stop diagnostic.'
        assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
        slot=slots[row['frame']];model.select_mesh_by_timestep(slot);cam,_=camera(row,slot,pose_index=slot)
        mask=torch.tensor(np.asarray(Image.open(row['mask']).convert('L')).copy(),device='cuda').float()/255
        E=torch.tensor(row['w2c'],device='cuda',dtype=torch.float32);xyz=model.get_xyz.detach();z=(xyz@E[:3,:3].T+E[:3,3])[:,2]
        z0=float((model.vertices_sequence[slot]@E[:3,:3].T+E[:3,3])[:,2].median());d=z-z0
        with torch.no_grad():
            rgb=render(cam,model,pipe,bg)['render'];original=np.asarray(Image.open(SNAP/(row['frame']+'_gaussian.png')).convert('RGB')).copy()
            quant=(rgb.clamp(0,1)*255).byte().permute(1,2,0);difference=(quant.float()-torch.tensor(original,device='cuda')).abs()
            native_parity=dict(mean_u8=float(difference.mean()),max_u8=float(difference.max()))
            assert native_parity['mean_u8']<p['thresholds']['native_mean_u8'] and native_parity['max_u8']<=p['thresholds']['native_max_u8'],native_parity
            if i==0:
                const=torch.tensor([1.,-1.,2.],device='cuda').repeat(len(xyz),1)
                cc=render(cam,model,pipe,bg,override_color=const)['render']
                constant_control=dict(negative_channel_max_abs=float((cc[1]+cc[0]).abs().max()),double_channel_max_abs=float((cc[2]-2*cc[0]).abs().max()))
                assert max(constant_control.values())<p['thresholds']['constant_carrier_max_abs'],constant_control
        carrier=torch.stack((torch.ones_like(d),d,d.square()),dim=1).detach().requires_grad_()
        moments=render(cam,model,pipe,bg,override_color=carrier)['render'];loss=(moments[0]*mask).sum()
        derivative=torch.autograd.grad(loss,carrier)[0];contribution=derivative[:,0].detach()
        assert (contribution>=0).all() and torch.isfinite(contribution).all() and not torch.any(derivative[:,1:])
        conservation=float((contribution.sum()-loss.detach()).abs()/loss.detach().clamp_min(1e-20))
        assert conservation<p['thresholds']['alpha_conservation_relative'],conservation
        with torch.no_grad():
            alpha=moments[0].detach();den=alpha.clamp_min(1e-12);mean=moments[1].detach()/den;variance=moments[2].detach()/den-mean.square()
            # Fixed reporting domain only; it does not select or reject teacher points.
            meaningful=(mask>0)&(alpha>1/255);assert meaningful.any()
            minvariance=float(variance[meaningful].min());assert minvariance>=p['thresholds']['variance_min_m2'],minvariance
            old=torch.tensor(proxy['weights'][pslots[row['frame']]],device='cuda');support=old>0
            weight=contribution*support;sums=torch.zeros(len(model.faces),device='cuda');sums.scatter_add_(0,model.binding,weight)
            weight=weight/sums[model.binding].clamp_min(1e-20);weight=weight/weight.sum()
            signed=torch.tensor(proxy['normal_distance_m'][pslots[row['frame']]],device='cuda')
            rec=dict(frame=row['frame'],native_png_parity=native_parity,conservation_relative=conservation,
                foreground_alpha_pixel_sum=float(loss.detach()),foreground_contributing_Gaussians=int((contribution>0).sum()),
                proxy_points=int(support.sum()),proxy_points_with_zero_foreground_contribution=int((support&(contribution==0)).sum()),
                proxy_weight_on_zero_contribution=float(old[contribution==0].sum()),
                active_teacher_points=int((weight>0).sum()),active_teacher_triangles=int((sums>0).sum()),
                proxy_positive_normal_weight=float(old[signed>0].sum()),native_positive_normal_weight=float(weight[signed>0].sum()),
                normalized_weight_L1_change=float((weight-old).abs().sum()),
                native_weighted_normal_distance_mm=float((weight*signed).sum())*1000,
                depth_center_anchor_m=z0,center_depth_standard_deviation_median_mm=float(variance[meaningful].clamp_min(0).sqrt().median())*1000,
                variance_min_m2=minvariance,moment_depth_is_not_physical_surface=True)
            records.append(rec);native_weights.append(weight.cpu().numpy());all_contributions.append(contribution.cpu().numpy())
            raw=Image.open(row['image']).convert('RGB');coverage=np.repeat((alpha.clamp(0,1)*255).byte().cpu().numpy()[...,None],3,axis=-1)
            yy,xx=np.where(mask.cpu().numpy()>.5);w,h=row['wh'];box=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21)]
            panels=[raw.crop(box),Image.fromarray(quant.cpu().numpy()).crop(box),Image.fromarray(coverage).crop(box)];sheet=Image.new('RGB',(panels[0].width*3,panels[0].height))
            for j,img in enumerate(panels):sheet.paste(img,(j*img.width,0))
            name=f'分析_高斯可见贡献_{row["frame"]}_原图_120k渲染_不透明度.png';path=folder/name;sheet.save(path);dest=record/name;assert not dest.exists();dest.write_bytes(path.read_bytes())
            images.append(dict(frame=row['frame'],path=str(path),record_path=str(dest),sha256=io.sha(path),crop_integer_xyxy=box,no_resampling=True,no_text_added=True))
        io.save(OUT/'progress.json',dict(state='native_zero_update_audit',pid=os.getpid(),frames=i+1,elapsed_seconds=time.monotonic()-started))
        print('NATIVE_CONTRIBUTION_FRAME',i+1,row['frame'],conservation,rec['proxy_points_with_zero_foreground_contribution'],flush=True)
        del moments,loss,derivative,carrier,rgb
    assert all(torch.equal(x,y) for x,y in zip(initial,learned)) and model.optimizer is None
    checked();np.savez_compressed(OUT/'native-contribution-weights.npz',frame_ids=np.array([x['frame'] for x in rows]),
        binding=model.binding.cpu().numpy(),foreground_contribution=np.array(all_contributions),normalized_teacher_weights=np.array(native_weights))
    r=dict(state='native_contribution_audit_complete_no_optimization',updated=io.now(),frames=59,source_step=120000,
        CPU_analytic_identity_passed=True,native_constant_carrier_control=constant_control,frames_result=records,images=images,
        optimizer_updates=0,learned_parameters_unchanged=True,new_mesh_or_PLY=False,actual_visual_review=False,quality_accepted=False,
        elapsed_seconds=time.monotonic()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        interpretation='Actual T*alpha visibility/contribution established for this saved renderer and masks; neither geometric truth nor successful mesh distillation established.')
    io.save(OUT/'result.json',r);io.save(OUT/'progress.json',dict(state=r['state'],frames=59,pid=None));print('NATIVE_CONTRIBUTION_COMPLETE',flush=True)


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
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','replay']);p.add_argument('--pid',type=int);args=p.parse_args()
    if args.mode=='prepare':prepare()
    else:globals()[args.mode](args.pid)
