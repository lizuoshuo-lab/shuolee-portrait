"""One registered native two-pose training-entry control and 200k layout step."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,hashlib,time,traceback,importlib.util

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    tmp=Path(str(p)+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)
def need(ok,msg):
    if not ok:raise RuntimeError(msg)

out=Path(sys.argv[1]).resolve();plan=json.loads((out/'plan.json').read_text())
need(not (out/'progress.json').exists() and not (out/'result.json').exists(),'Already started: inspect existing process/result')
need(plan['purpose']=='synthetic_native_training_entry' and plan['steps']==16 and plan['performance_points']==200000,'Changed registered work')
for p,h in plan['source_sha256'].items():need(sha(p)==h,'Source changed:'+p)
sys.path[:0]=[plan['project_src'],plan['gsplat_repo'],str(Path(__file__).parent)]
start=time.monotonic()
r=dict(passed=False,started=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),plan_sha256=sha(out/'plan.json'),
    checks=[],native_forward_attempted=0,native_forward_completed=0,backward_attempted=0,backward_completed=0,
    optimizer_steps=0,real_person_images=0,real_fitting_steps=0,real_gaussian_steps=0,synthetic_only=True,
    physical_quality_accepted=False,source_sha256=plan['source_sha256'])
def progress(stage):
    save(out/'progress.json',dict(stage=stage,pid=os.getpid(),elapsed=time.monotonic()-start,
        native_forward_attempted=r['native_forward_attempted'],native_forward_completed=r['native_forward_completed'],
        optimizer_steps=r['optimizer_steps']))
def check(name,ok,**metrics):
    r['checks'].append(dict(name=name,passed=bool(ok),**metrics));need(ok,'Control failed:'+name)
progress('import_verified_runtime')
try:
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    import gsplat
    need(torch.__version__=='2.7.1+cu128' and gsplat.__version__=='1.5.3','Unexpected backend versions')
    need(sha(plan['extension'])==plan['extension_sha256'],'Extension changed')
    spec=importlib.util.spec_from_file_location('gsplat_cuda',plan['extension'])
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    # Use gsplat's documented precompiled-module branch, with the exact already
    # built binary. This prevents a source-cache mismatch from launching JIT.
    sys.modules['gsplat.csrc']=extension;gsplat.csrc=extension
    from gsplat.cuda._backend import _C
    need(Path(_C.__file__).resolve()==Path(plan['extension']).resolve(),'Wrong loaded binary')
    from native_portrait_renderer122_v3 import render_native,REVISION
    from native_residual_camera122 import ResidualRenderContract
    from avatar_core.utils.sh_utils import eval_sh
    from shuolee_portrait.free_clothing_torch import FreeClothingTorch
    from shuolee_portrait.framewise_native_training import FramewiseNativeTrainer,TrainingConfig,foreground_native_loss
    import native_training_fixture123 as f
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    contract=ResidualRenderContract('synthetic_known_camera','pixel_corners')
    zeros=[torch.zeros((1,n),device=device,dtype=torch.float32) for n in (6,2,4)]
    r['runtime']=dict(python=sys.executable,torch=torch.__version__,gsplat=gsplat.__version__,GPU=torch.cuda.get_device_name(device),
        extension=str(_C.__file__),extension_sha256=sha(plan['extension']),loader='Existing precompiled gsplat.csrc alias; no JIT')
    def native(*args,**kwargs):
        need(time.monotonic()-start<plan['wall_seconds'],'Wall budget exhausted')
        need(r['native_forward_attempted']<plan['max_native_forwards'],'Forward budget exhausted')
        r['native_forward_attempted']+=1;progress('native_forward')
        result=render_native(*args,**kwargs);torch.cuda.synchronize()
        r['native_forward_completed']+=1
        return result
    target={};images={};masks={}
    with torch.no_grad():
        for frame in f.FRAME_IDS:
            args=f.analytic_render_args(frame,torch,device,eval_sh)
            rgb,a,_=native(*args,*f.WH,contract=contract,radial_coeffs=zeros[0],tangential_coeffs=zeros[1],thin_prism_coeffs=zeros[2],sh_degree=None)
            target[frame]=(rgb.detach().clone(),a.detach().clone())
            images[frame]=rgb[0].cpu().numpy();masks[frame]=a[0,:,:,0].cpu().numpy()
    bundle=f.write_bundle(out/'synthetic-input',images,masks)
    from PIL import Image
    validity=out/'synthetic-input/known-valid.png';Image.new('L',f.WH,255).save(validity)
    specs={frame:dict(target_wh=f.WH,radial=[0.]*6,tangential=[0.]*2,thin_prism=[0.]*4,
        alpha_valid_path=str(validity),alpha_valid_sha256=sha(validity),validity_note='Known unoccluded synthetic scene; soft mask is quantized native coverage, not a person silhouette.') for frame in f.FRAME_IDS}
    provenance=dict(adapter_revision=REVISION,adapter_sha256=sha(Path(__file__).parent/'native_portrait_renderer122_v3.py'),extension_sha256=plan['extension_sha256'])
    initprov=dict(kind='synthetic_known',old_subject_checkpoint_loaded=False,fixture='native-entry-two-pose-six-free-gaussians-v1')
    def make(count=6,perturbed=False,run='known-six',steps=16):
        need(count==6,'Six-point fixture only;200k must originate from the surface initializer')
        src=f.source_transport(count)
        consumer=FreeClothingTorch.from_cpu(src,camera_length_contract=f.LENGTH,
            frame_source_w2c={frame:f.frame_math(frame)[3] for frame in f.FRAME_IDS},device=device,dtype=torch.float32)
        p=f.tensor_initial(torch,device,perturbed=perturbed);p['sh']=p['sh'].transpose(1,2).contiguous()
        cfg=TrainingConfig(max_steps=steps,initial_points=count,sh_degree=3,xyz_lr=.003,quaternion_lr=.001,
            log_scale_lr=.005,opacity_lr=.01,sh_lr=.01,alpha_weight=.1,rgb_foreground_threshold=.05)
        trainer=FramewiseNativeTrainer(bundle,consumer,p,gaussian_ids=src.gaussian_ids,frame_specs=specs,
            render_native=native,render_contract=contract,renderer_provenance=provenance,
            initialization_provenance=initprov,run_id=run,config=cfg)
        return trainer
    bridge=make()
    check('fresh_empty_optimizer',len(bridge.optimizer.state)==0)
    comparisons=[]
    with torch.no_grad():
        for frame in f.FRAME_IDS:
            got=bridge.render(frame);expected=target[frame]
            rgbmax=float((got['rgb']-expected[0][0]).abs().max());amax=float((got['alpha']-expected[1][0,:,:,0]).abs().max())
            comparisons.append(dict(frame=frame,rgb_max=rgbmax,alpha_max=amax))
    check('new_consumer_to_native_independent_two_pose_images',all(x['rgb_max']<2e-5 and x['alpha_max']<2e-5 for x in comparisons),frames=comparisons)
    del bridge
    trainer=make(perturbed=True,run='bounded-six-step-entry')
    observations={frame:trainer.load_observation(frame) for frame in f.FRAME_IDS}
    def losses(model):
        values=[]
        with torch.no_grad():
            for frame in f.FRAME_IDS:
                o=observations[frame]; im=model.render(frame)
                values.append(float(foreground_native_loss(im['rgb'],im['alpha'],o.rgb,o.foreground,o.rgb_valid,o.alpha_valid,alpha_weight=.1)['loss']))
        return values
    before=losses(trainer)
    initial={k:v.detach().clone() for k,v in trainer.field.named_parameters()}
    history=[]
    for step in range(plan['steps']):
        frame=f.FRAME_IDS[(step+1)%2];r['backward_attempted']+=1
        tick=trainer.step(frame,observations[frame]);torch.cuda.synchronize()
        r['backward_completed']+=1;r['optimizer_steps']+=1
        history.append(dict(step=tick['step'],frame=frame,loss=float(tick['loss']),rgb_l1=float(tick['rgb_l1']),alpha_l1=float(tick['alpha_l1'])))
    after=losses(trainer)
    delta={k:float((v-initial[k]).abs().max()) for k,v in trainer.field.named_parameters()}
    check('fixed16steps_shared_field_updates',trainer.step_count==16 and trainer.report()['view_counts']=={frame:8 for frame in f.FRAME_IDS} and all(v>0 for v in delta.values()),parameter_max_changes=delta)
    check('joint_two_pose_loss_decreased',float(np.mean(after))<float(np.mean(before)),before=before,after=after,no_monotonic_or_convergence_claim=True)
    trainer.save_checkpoint(out/'synthetic-checkpoint.pt');trainer.save_reference_state(out/'named-reference.pt')
    ref=trainer.reference_state()
    check('named_reference_not_last_frame_or_average',ref['frame_id']==f.REFERENCE_ID and not ref['posed_clouds_averaged'] and torch.equal(ref['parameters']['xyz'],trainer.field.xyz.detach().cpu()),reference=f.REFERENCE_ID)
    recovered=make(perturbed=True,run='bounded-six-step-entry')
    recovered.load_checkpoint(out/'synthetic-checkpoint.pt',resume_same_run=True)
    recover_delta=max(float((a-b).abs().max()) for a,b in zip(trainer.field.parameters(),recovered.field.parameters()))
    check('same_run_checkpoint_parameter_and_optimizer_recovery',recover_delta==0 and recovered.step_count==16 and len(recovered.optimizer.state)==5,max_difference=recover_delta)
    with torch.no_grad():
        recovered_images={frame:recovered.render(frame) for frame in f.FRAME_IDS}
    for i,frame in enumerate(f.FRAME_IDS):
        Image.fromarray(np.rint(np.clip(recovered_images[frame]['rgb'].cpu().numpy(),0,1)*255).astype('uint8')).save(out/f'03_synthetic_trained_{i}.png')
    r['training']=dict(before=before,after=after,history=history,report=trainer.report(),named_reference=f.REFERENCE_ID)
    del recovered,trainer,observations,initial,ref,target
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    progress('fixed_200000_layout_single_step')
    from shuolee_portrait.framewise_clothed_surface import ReferenceGrid,initialize_framewise_surface
    from shuolee_portrait.framewise_native_training import fresh_reference_parameters
    perf_start=time.monotonic()
    cube=f.write_cube_bundle(out/'cube-input'); cube_ids=tuple(cube.report()['frame_ids'])
    valid={name:np.ones((65,65),dtype=bool) for name in cube_ids}
    notes={name:'Known analytic moving cube and unoccluded static background; no person data.' for name in cube_ids}
    grid=ReferenceGrid.from_bounds([-.8,-.8,3.2],[.8,.8,4.8],spacing_m=.1,reference_frame_id='reference',max_voxels=100000)
    seeds=initialize_framewise_surface(cube,grid,reference_frame_id='reference',valid_masks=valid,validity_notes=notes,
        camera_length_contract=f.LENGTH,min_valid_views=6,vote_fraction=.9,seed=123,initial_points=200000)
    save(out/'initializer-result.json',seeds.report())
    np.savez_compressed(out/'synthetic-initializer.npz',xyz_ref=seeds.xyz_ref,
        surface_vertices_ref=seeds.surface_vertices_ref,surface_faces=seeds.surface_faces,
        positive=seeds.sample_positive_counts,valid=seeds.sample_valid_counts)
    check('actual_initializer_to_fixed200k_consumer',len(seeds.xyz_ref)==200000 and seeds.report()['camera_translation_scale_applications']==0,
        sampled_valid=seeds.report()['sampled_points_with_sufficient_views'],sampled_vote_rule=seeds.report()['sampled_points_meeting_vote_rule'],
        physical_surface_quality=False,sample_votes_not_rewritten=True)
    valid_path=out/'cube-input/known-valid.png';Image.new('L',(65,65),255).save(valid_path)
    cube_specs={name:dict(target_wh=(65,65),radial=[0.]*6,tangential=[0.]*2,thin_prism=[0.]*4,
        alpha_valid_path=str(valid_path),alpha_valid_sha256=sha(valid_path),validity_note=notes[name]) for name in cube_ids}
    c=FreeClothingTorch.from_cpu(seeds.transport,camera_length_contract=f.LENGTH,
        frame_source_w2c={name:cube.frame(name).w2c for name in cube_ids},device=device,dtype=torch.float32)
    p=fresh_reference_parameters(seeds.xyz_ref,initial_scale=.003,initial_opacity=.05,initial_rgb=[.6,.4,.3],device=device)
    perf=FramewiseNativeTrainer(cube,c,p,gaussian_ids=seeds.gaussian_ids,frame_specs=cube_specs,render_native=native,
        render_contract=contract,renderer_provenance=provenance,initialization_provenance=dict(kind='fresh_clothing_surface',old_subject_checkpoint_loaded=False,initializer_sha256=sha(out/'initializer-result.json')),
        run_id='known-cube-200k-full-entry',config=TrainingConfig(max_steps=1,initial_points=200000))
    setup_seconds=time.monotonic()-perf_start
    need(not perf.optimizer.state,'Performance run must also start empty')
    observation=perf.load_observation('right');torch.cuda.synchronize()
    start_step=time.monotonic();r['backward_attempted']+=1
    tick=perf.step('right',observation);torch.cuda.synchronize()
    r['backward_completed']+=1;r['optimizer_steps']+=1
    step_seconds=time.monotonic()-start_step
    memory=torch.cuda.max_memory_allocated(device)
    check('200000_free_layout_single_native_training_step',len(perf.gaussian_ids)==200000 and perf.step_count==1 and len(perf.optimizer.state)==5,
        points=200000,width=65,height=65,setup_seconds=setup_seconds,step_seconds=step_seconds,peak_allocated_bytes=memory,
        native_resolution_performance_validated=False,real_quality_validated=False)
    r['performance']=dict(report=perf.report(),points=200000,resolution=[65,65],setup_seconds=setup_seconds,step_seconds=step_seconds,peak_allocated_bytes=memory,
        initial_point_origin='six-view moved silhouette reference envelope; actualproduction initializer, no baremesh attraction',scene_quality_accepted=False)
    for p,h in plan['source_sha256'].items():need(sha(p)==h,'Source changed during execution:'+p)
    need(sha(plan['extension'])==plan['extension_sha256'],'Extension changed during execution')
    r['passed']=True
except BaseException as e:
    r.update(error=type(e).__name__,message=str(e),traceback=traceback.format_exc())
finally:
    r.update(elapsed_seconds=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
    save(out/'result.json',r);progress('completed' if r['passed'] else 'failed')
    print(json.dumps({k:r.get(k) for k in ('passed','error','message','elapsed_seconds','native_forward_completed','backward_completed','optimizer_steps','checks')},indent=2),flush=True)
raise SystemExit(0 if r['passed'] else 1)
