"""Bounded R7 engineering controls; no person inference, fit or training.

Each phase is once-only. Synthetic fixtures are never person admissions.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse, hashlib, json, os, signal, subprocess, sys, time, traceback
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from PIL import Image

H=Path(__file__).resolve().parent
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-pipeline'
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def put(p,d):save(p,d);return ref(p)
def linux(p):return Path('/mnt/'+p[0].lower()+p[2:].replace('\\','/')) if len(p)>2 and p[1]==':' else Path(p)

def camera(i=0):
    from native_residual_camera122 import projection
    e=np.eye(4);e[0,3]=-.20*i
    return dict(id=str(i),image_name=f'synthetic-{i}.png',width=48,height=40,
        K=[[49.,0,22.7],[0,51.,18.2],[0,0,1.]],w2c=e.tolist(),
        residual_projection=projection([.06,-.003,.0001,.002,0.,0.],[.001,-.002],[.001,0.,-.001,0.],input_domain='synthetic_known_camera'))

def pixel_fixture():
    p=O/'synthetic-pixels';p.mkdir(exist_ok=True)
    frame={'frame_id':0}
    for name in ['rgb','alpha','rgb_valid','observation_valid']:
        f=p/(name+'.png')
        if not f.exists():
            y,x=np.indices((40,48))
            a=np.stack([x*3,y*4,(x+y)*2],axis=-1).astype(np.uint8) if name=='rgb' else np.full((40,48),255,np.uint8)
            Image.fromarray(a).save(f)
        frame[name]=ref(f)
    return p,frame,camera()

def baseline():
    from train_native_person122_v2 import PixelCache
    from native_training_policy122 import digest
    root,frame,c=pixel_fixture();cache=PixelCache(root,'cpu')
    first=cache.get(frame,c);second=cache.get(frame,c)
    state=dict(updated=now(),scope='synthetic-control-only',source=ref(H/'train_native_person122_v2.py'),
        expected_key=[frame['frame_id'],digest(c),digest(frame)],actual_key=cache.current,
        repeated_same_frame_reuses_object=first is second,
        native_rgb_equal=bool(np.array_equal(first.rgb.numpy(),second.rgb.numpy())),
        expected_behavior_passed=first is second,
        consequence='Repeated image/mask reads and tensor allocations; not evidence of a projection or alignment cause.',
        GPU_imported=False,person_training=False)
    save(O/'cache-initial-failure.json',state)
    assert not state['expected_behavior_passed'], 'Observed source no longer reproduces the registered cache issue'
    save(O/'progress.json',dict(updated=now(),state='cache_defect_reproduced_repair_pending',test_executed=True,person_training=False))
    return state

def cpu_controls():
    import torch
    from train_native_person122_v2 import PixelCache,DiskView,load_fresh_initialization
    from native_training122_v4 import assert_frozen_view
    from native_training_policy122 import digest,cycle_schedule,complete_graph,validate_reviews
    from native_pipeline_binding122 import pipeline_sources,verify_preparation,sampling_for_input
    checks=[]
    def check(name,condition,**detail):
        checks.append(dict(name=name,passed=bool(condition),**detail));save(O/'cpu-checks.json',checks)
        if not condition:raise AssertionError(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,FileNotFoundError,KeyError) as e:check(name,True,error=str(e));return
        raise AssertionError('Unexpected acceptance: '+name)
    root,frame,c=pixel_fixture();cache=PixelCache(root,'cpu')
    first=cache.get(frame,c);second=cache.get(frame,c)
    check('Same frame reuses one native RGB and mask allocation after cache repair',first is second and cache.current==(0,digest(c),digest(frame)))
    expected=np.asarray(Image.open(root/'rgb.png')).astype(np.float32)/255.
    disk=DiskView(frame,c,cache,'cpu')
    check('Disk view retains exact native raster and complete residual camera',np.array_equal(disk.rgb.numpy(),expected) and disk.camera==c and disk.alpha.shape==(40,48,1) and disk.K.shape==(1,3,3))
    next_frame=deepcopy(frame);next_frame['frame_id']=1
    third=cache.get(next_frame,camera(1))
    check('Switching frame replaces the single cached view',third is not first and cache.current[0]==1 and cache.value is third)
    missing=deepcopy(c);missing.pop('residual_projection')
    reject('No silent zero distortion when complete camera is absent',lambda:cache.get(frame,missing))
    disk.camera['residual_projection']['radial'][0]+=.01
    reject('Camera coefficient mutation violates the frozen view',lambda:assert_frozen_view(disk))
    wrong=deepcopy(frame);wrong['rgb']['sha256']='f'*64
    reject('Changed image identity rejects before pixel tensor creation',lambda:cache.get(wrong,c))
    check('CPU pixel controls did not initialize CUDA',not torch.cuda.is_initialized())

    cameras=[camera(i) for i in range(4)]
    single=dict(kind='uniform_single',density='fixed_free_centers')
    pair=dict(kind='covisible_pair',density='fixed_free_centers')
    schedule=cycle_schedule(cameras,single,122)
    check('Single-view cycle deterministic with every declared camera once',schedule==cycle_schedule(cameras,single,122) and sorted(x[0] for x in schedule)==['0','1','2','3'])
    observations=[dict(frame_id=c['id'],family_id=str(f),split='fit',group='foreground',visible=True,identity='confirmed',static='confirmed',evidence_sha256='a'*64) for c in cameras for f in range(3)]
    graph=complete_graph(cameras,observations,source_sha256='b'*64,min_shared=2)
    cycle=cycle_schedule(cameras,pair,122,graph=graph)
    from collections import Counter
    check('Supported two-view cycle balances all four camera exposures',dict(Counter(x for batch in cycle for x in batch))==dict.fromkeys(['0','1','2','3'],2))
    mixed=deepcopy(observations);mixed[0]['split']='heldout'
    reject('Whole physical-family split leakage rejects CVO',lambda:complete_graph(cameras,mixed,source_sha256='b'*64,min_shared=2))
    empty=deepcopy(graph);empty['edges']=[]
    reject('Unsupported CVO cannot silently fall back to random pairs',lambda:cycle_schedule(cameras,pair,122,graph=empty))
    altered=deepcopy(cameras);altered[0]['K'][0][2]+=.5
    reject('CVO support bound to exact full K E D',lambda:cycle_schedule(altered,pair,122,graph=graph))
    reject('1000 step inspection cannot be omitted',lambda:validate_reviews(18000,[],None,None))

    # Explicitly synthetic provenance fixtures. They test a consumer, not a fit.
    f=O/'synthetic-provenance';f.mkdir(exist_ok=False)
    evidence=put(f/'known-fixture.json',dict(scope='synthetic-control-only',human_fit_executed=False,purpose='Known zero-step arrays and fabricated fixture receipts for consumer rejection controls only'))
    m=dict(scope='synthetic-control-only',run_id='R7_known_initial_arrays',source={'asset':evidence},cameras=evidence,frame_plan=evidence,
        frames=[frame],observations=evidence,physical_evidence=evidence,training=dict(sampling_policy=single,seed=122))
    sources=pipeline_sources();parent=put(f/'input.json',m)
    n=200000;arrays=dict(means=np.zeros((n,3),np.float32),quats=np.tile(np.array([1,0,0,0],np.float32),(n,1)),
        scales=np.full((n,3),.01,np.float32),opacities=np.full(n,.1,np.float32),sh=np.zeros((n,16,3),np.float32))
    arrays['means'][:,2]=3
    np.savez_compressed(f/'known-initial.npz',**arrays);ar=ref(f/'known-initial.npz')
    fit_source=ref(H/'fitting_output_residual122.py')
    fit=dict(scope='synthetic-control-only',run_id=m['run_id'],source_sha256=evidence['sha256'],cameras_sha256=evidence['sha256'],frame_plan_sha256=evidence['sha256'],
        pipeline_sources_sha256=digest(sources),from_scratch=True,old_individual_geometry_loaded=False,executed_command=['synthetic_fixture_only'],exit_code=0,
        fit_arrays_sha256=evidence['sha256'],fit_source_sha256=fit_source['sha256'],**{k:evidence for k in ['new_prior_receipt','new_prior_prediction','original_init','fit_result','fit_protocol']})
    fitref=put(f/'synthetic-fit-receipt.json',fit)
    binding=dict(scope='synthetic-control-only',run_id=m['run_id'],source_sha256=evidence['sha256'],cameras_sha256=evidence['sha256'],frame_plan_sha256=evidence['sha256'],
        frames_sha256=digest(m['frames']),observations_sha256=evidence['sha256'],material_evidence_sha256=evidence['sha256'],source_manifest=parent,pipeline_sources=sources,
        fit_review=put(f/'synthetic-fit-review.json',dict(scope='synthetic-control-only',fitted_sha256=evidence['sha256'],actual_same_camera_fit_reviewed=True,allow_surface=True,review_evidence=evidence)),
        geometry_gate=put(f/'synthetic-geometry.json',dict(scope='synthetic-control-only',fit_sha256=evidence['sha256'],coarse_initialization_gate_passed=True)),
        surface_result=put(f/'synthetic-surface.json',dict(scope='synthetic-control-only',arrays_sha256=ar['sha256'])))
    initial=dict(scope='synthetic-control-only',schema='shuolee-fresh-clothing-initialization-v2-output-residual',pipeline_binding=put(f/'binding.json',binding),
        run_id=m['run_id'],source_sha256=evidence['sha256'],cameras_sha256=evidence['sha256'],frame_plan_sha256=evidence['sha256'],representation='free_world_clothing_surface',world_frame='right_handed_world_y_up',
        old_individual_geometry_loaded=False,old_gaussians_loaded=False,old_optimizer_loaded=False,training_steps=0,initial_points=n,
        fresh_fit_receipt=fitref,fit_arrays=evidence,fit_source=fit_source,initializer_source=ref(H/'native_surface122_v2.py'),arrays=ar,executed_command=['synthetic_fixture_only'],exit_code=0)
    m['fresh_initialization']=put(f/'initialization.json',initial)
    m['surface_review']=put(f/'synthetic-surface-review.json',dict(scope='synthetic-control-only',arrays_sha256=ar['sha256'],actual_same_camera_surface_reviewed=True,actual_center_distribution_reviewed=True,allow_training=True,review_evidence=evidence))
    loaded,receipt=load_fresh_initialization(m,f,digest(m))
    check('200k complete zero-step consumer preserves every synthetic attribute',all(np.array_equal(loaded[k],arrays[k]) for k in arrays),scope='synthetic-control-only',human_fit_executed=False)
    del loaded
    def altered_initial(tag,updates):
        mm=deepcopy(m);rr=deepcopy(initial);rr.update(updates);mm['fresh_initialization']=put(f/(tag+'.json'),rr);return mm
    mm=altered_initial('old-optimizer',dict(old_optimizer_loaded=True))
    reject('Old optimizer provenance cannot masquerade as fresh',lambda:load_fresh_initialization(mm,f,digest(mm)))
    mm=altered_initial('nonzero-step',dict(training_steps=1))
    reject('Already trained initialization rejects before optimizer',lambda:load_fresh_initialization(mm,f,digest(mm)))
    mm=deepcopy(m);mm['frames'][0]['alpha']['sha256']='f'*64
    reject('Different training mask violates preparation binding',lambda:verify_preparation(mm,f))
    bb=deepcopy(binding);bb['pipeline_sources']['train_native_person122_v2.py']['sha256']='f'*64
    mm=altered_initial('changed-code',dict(pipeline_binding=put(f/'changed-code-binding.json',bb)))
    reject('Preparation and training source mismatch cannot dispatch',lambda:verify_preparation(mm,f))
    mm=deepcopy(m);mm['surface_review']=put(f/'not-reviewed.json',dict(arrays_sha256=ar['sha256'],actual_same_camera_surface_reviewed=False,actual_center_distribution_reviewed=True,allow_training=True,review_evidence=evidence))
    reject('Review assertion cannot omit actual surface inspection',lambda:verify_preparation(mm,f))
    # Same real failed input, different newly wired entries. No optimizer imported.
    actual=R/'native-residual-admission/real-diagnostic-input.json'
    for script,stage,out,status_name in [('prepare_native_person122_v3.py','fit','actual-preparation','preparation-status.json'),('train_native_person122_v2.py','train','actual-training','status.json')]:
        command=[sys.executable,str(H/script),stage,'--input',str(actual),'--output',str(O/out)]
        child=subprocess.run(command,capture_output=True,text=True,timeout=150)
        (O/(out+'.log')).write_text(child.stdout+child.stderr,encoding='utf-8')
        state=read(O/out/status_name);audit=read(O/out/'admission.json')
        check('Actual failed person rejected before GPU at '+script,child.returncode==3 and state['state']=='input_rejected' and state['GPU_imported'] is False and state['training_started'] is False,
            targets=audit['material_geometry']['targets'],face=audit['heldout_face'],command=command)
        assert audit['material_geometry']['targets']==102 and audit['heldout_face']['observations']==13 and audit['heldout_face']['accepted']==0
    check('CPU suite never allocated a GPU context',not torch.cuda.is_initialized())
    save(O/'cache-repair.json',dict(updated=now(),initial_failure=ref(O/'cache-initial-failure.json'),corrected_source=ref(H/'train_native_person122_v2.py'),
        change='Separate immutable cache identity from per-channel loop variable; no image or camera value changed.',
        baseline_field_note='Initial failure GPU_imported=false means no CUDA context; torch was imported for CPU tensors.',alignment_cause_proven=False))
    return dict(updated=now(),state='cpu_controls_complete_gpu_pending',checks=checks,scope='engineering only',person_fit=False,person_training=False,physical_quality_accepted=False)

def gpu_controls():
    import torch, ast
    from native_training122_v5 import FreshNativeGaussians,create_view,render_view,accumulate_update,optimize_fresh,export_native_checkpoint
    from native_training122 import foreground_objective
    from native_training_policy122 import complete_graph,digest
    from native_residual_camera122 import ResidualRenderContract
    from native_scene_package122_v5 import load_scene,render_recorded
    from native_training_admission122_v2 import checked_asset
    checks=[];start=time.monotonic()
    def check(name,condition,**detail):
        checks.append(dict(name=name,passed=bool(condition),**detail));save(O/'gpu-checks.json',checks)
        if not condition:raise AssertionError(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError) as e:check(name,True,error=str(e));return
        raise AssertionError('Unexpected acceptance: '+name)
    assert read(O/'cpu-result.json')['state']=='cpu_controls_complete_gpu_pending'
    contract=ResidualRenderContract('synthetic_known_camera');cameras=[camera(i) for i in range(4)]
    cameras[1]['w2c'][0][3]=-1.4;cameras[2]['w2c'][0][3]=.7;cameras[3]['w2c'][1][3]=.6
    rng=np.random.default_rng(122);n=96
    initial=dict(means=rng.uniform([-.7,-.5,2.1],[.7,.5,2.7],(n,3)).astype(np.float32),
        quats=rng.normal(size=(n,4)).astype(np.float32),scales=rng.uniform(.025,.08,(n,3)).astype(np.float32),
        opacities=rng.uniform(.2,.5,n).astype(np.float32),sh=rng.normal(0,.015,(n,16,3)).astype(np.float32))
    initial['quats']/=np.linalg.norm(initial['quats'],axis=1,keepdims=True)
    initial['sh'][:,0,:]=rng.uniform(-.3,.4,(n,3))
    target=FreshNativeGaussians(initial,device='cuda');views=[]
    blank=np.zeros((40,48,3),np.float32);alpha=np.zeros((40,48,1),np.float32);valid=np.ones((40,48,1),bool)
    images=O/'synthetic-native-images';images.mkdir(exist_ok=False)
    with torch.no_grad():
        for c in cameras:
            dummy=create_view(c,blank,alpha,valid,valid,device='cuda')
            rgb,a,_=render_view(target,dummy,contract)
            v=create_view(c,rgb.cpu().numpy(),a.cpu().numpy(),valid,valid,device='cuda');views.append(v)
            Image.fromarray(np.rint(np.clip(rgb.cpu().numpy(),0,1)*255).astype(np.uint8)).save(images/f'target-{c["id"]}.png')
    seed=deepcopy(initial);seed['means'][:,0]+=.025;seed['sh'][:,0,1]+=.035
    rates=dict(means=.002,quats=.001,log_scales=.001,opacity_logits=.001,sh=.002)
    oracle=FreshNativeGaussians(seed,device='cuda');reference_optimizer=oracle.optimizer(rates)
    predicted_losses=[];expected_counts=torch.zeros(n,dtype=torch.int32,device='cuda');expected_radius=torch.zeros(n,device='cuda');expected_norm=torch.zeros(n,device='cuda');vis=[]
    for v in views[:2]:
        rgb,a,meta=render_view(oracle,v,contract);loss,_=foreground_objective(rgb,a,v);predicted_losses.append(loss)
        world_grad=torch.autograd.grad(loss,oracle.means,retain_graph=True)[0]
        seen=(meta['radii'][0]>0).all(-1);radius=meta['radii'][0].amax(-1).float();vis.append(seen)
        expected_counts+=seen.to(torch.int32);expected_radius=torch.maximum(expected_radius,torch.where(seen,radius,0))
        expected_norm+=torch.where(seen,torch.linalg.vector_norm(world_grad,dim=-1),0)
    check('Two-view test includes different visible point sets',bool(torch.any(vis[0]!=vis[1])),visible_counts=[int(v.sum()) for v in vis])
    joint=torch.stack(predicted_losses).mean();joint.backward()
    expected_grad={k:p.grad.detach().clone() for k,p in oracle.named_parameters()};reference_optimizer.step()
    actual=FreshNativeGaussians(seed,device='cuda');opt=actual.optimizer(rates)
    stats=accumulate_update(actual,views[:2],contract,opt,capture_gradients=True)
    err=max(float((stats['gradients'][k]-v).abs().max()) for k,v in expected_grad.items())
    parameter_err=max(float((dict(actual.named_parameters())[k]-p).abs().max()) for k,p in oracle.named_parameters())
    check('Actual sequential two-view update equals independent joint-loss Adam step',err<1e-6 and parameter_err<1e-6,gradient_max_abs=err,parameter_max_abs=parameter_err)
    norm_error=float((stats['world_gradient_norm_sum']-expected_norm).abs().max())
    check('Both views contribute visibility radius and unscaled world gradient',torch.equal(stats['visible_count'],expected_counts) and torch.equal(stats['max_projected_radius'],expected_radius) and norm_error<1e-6,world_gradient_stat_max_abs=norm_error)
    graph=complete_graph(cameras,[dict(frame_id=c['id'],family_id=str(i),split='fit',group='foreground',visible=True,identity='confirmed',static='confirmed',evidence_sha256='a'*64) for c in cameras for i in range(3)],source_sha256='b'*64,min_shared=2)
    policy=dict(kind='covisible_pair',density='fixed_free_centers')
    models=[];reports=[];histories=[];checkpoint_receipts=[];continuations=[]
    def snapshot_factory(label):
        def snapshot(step,model):
            scene=export_native_checkpoint(O/f'{label}-step-{step:03d}',model,views,contract,scope='synthetic-control-only')
            origin=dict(scope='synthetic-control-only',step=step,complete_cameras_sha256=digest(cameras),scene_sha256=sha(scene),physical_quality_accepted=False)
            receipt=dict(scene=str(scene),scene_sha256=sha(scene),origin=put(scene.parent/'training-origin.json',origin))
            checkpoint_receipts.append(receipt);return receipt
        return snapshot
    for label in ['repeat-a','repeat-b']:
        model=FreshNativeGaussians(seed,device='cuda')
        def approve_known(step,receipt):
            continuations.append(dict(step=step,receipt=receipt,model_id=id(model),scope='synthetic-control-only'))
            return True # Known-scene fixture only; never a person review.
        report,history=optimize_fresh(model,views,contract,steps=4,learning_rates=rates,seed=122,output=O/label,policy=policy,graph=graph,review_steps=[2],review=approve_known,checkpoint=snapshot_factory(label))
        models.append(model);reports.append(report);histories.append(history)
    repeat_error=max(float((dict(models[0].named_parameters())[k]-p).abs().max()) for k,p in models[1].named_parameters())
    check('Repeated fresh nonzero-D runs retain identical schedules and parameters',repeat_error<1e-6 and [r['frames'] for r in histories[0]]==[r['frames'] for r in histories[1]],parameter_max_abs=repeat_error,steps_each=4)
    check('Accepted known-scene review continues the same optimizer through step4',all(r['steps']==4 and r['optimizer_initial_state_entries']==0 and r['optimizer_final_state_entries']==5 for r in reports) and all(r['view_update_counts']==dict.fromkeys(['0','1','2','3'],2) for r in reports))
    reject('Completed model cannot construct another fresh optimizer',lambda:models[0].optimizer(rates))
    stopped_model=FreshNativeGaussians(seed,device='cuda')
    stopped,_=optimize_fresh(stopped_model,views,contract,steps=4,learning_rates=rates,seed=122,output=O/'review-stop',policy=policy,graph=graph,review_steps=[2],review=lambda step,receipt:False,checkpoint=snapshot_factory('review-stop'))
    check('No review approval stops before further optimization',stopped['state']=='stopped_at_review' and stopped['steps']==2 and len(stopped['snapshots'])==1)

    # Execute the exact person-entry review function, without creating a person,
    # optimizer or waiting. Fixture timestamps bound the missing-file case.
    tree=ast.parse((H/'train_native_person122_v2.py').read_text(encoding='utf-8'))
    execute=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='execute')
    node=next(x for x in ast.walk(execute) if isinstance(x,ast.FunctionDef) and x.name=='review')
    review_dir=O/'review-binding-fixture';review_dir.mkdir(exist_ok=False)
    env=dict(output=review_dir,status={},now=now,save=save,read=read,time=time,checked_asset=checked_asset)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual person review function','exec'),env)
    rp=review_dir/'review-000002.json';receipt=checkpoint_receipts[0]
    decision=dict(scope='synthetic-control-only',snapshot=receipt,actual_same_camera_face_body_review=True,actual_center_distribution_review=True,review_evidence=ref(O/'plan.json'),continue_same_run=True)
    save(rp,decision);check('Actual review consumer verifies exact snapshot and evidence',env['review'](2,receipt) is True)
    bad=deepcopy(decision);bad['snapshot']['scene_sha256']='f'*64;save(rp,bad)
    reject('Review for a different snapshot is rejected',lambda:env['review'](2,receipt))
    bad=deepcopy(decision);bad['review_evidence']['sha256']='f'*64;save(rp,bad)
    reject('Review with changed evidence is rejected',lambda:env['review'](2,receipt))
    class NoWaitClock:
        def __init__(self):self.t=0
        def monotonic(self):self.t+=2000;return self.t
        def sleep(self,seconds):raise AssertionError('No wait expected in bounded missing-file fixture')
    env['time']=NoWaitClock();check('Missing review times out without default approval',env['review'](3,receipt) is False)
    final=Path(reports[0]['snapshots'][4]['scene']);loaded=load_scene(final)
    params=models[0].numpy_linear();array_error=max(float(np.abs(loaded.parameters[k]-v).max()) for k,v in params.items())
    check('Final native NPZ and per-frame full K E D roundtrip exactly',array_error==0 and [loaded.camera(c['id']) for c in cameras]==cameras,arrays_max_abs=array_error)
    image_error=0
    with torch.no_grad():
        for v in views:
            rgb,a,_=render_view(models[0],v,contract);rr,aa,_=render_recorded(loaded,v.camera['id'])
            image_error=max(image_error,float(np.abs(rgb.cpu().numpy()-rr).max()),float(np.abs(a.cpu().numpy()-aa).max()))
            Image.fromarray(np.rint(np.clip(rr,0,1)*255).astype(np.uint8)).save(images/f'readback-{v.camera["id"]}.png')
    check('Recorded-camera readback preserves native RGB alpha response',image_error<1e-6,max_abs=image_error)
    frozen=read(O/'plan.json')['frozen'];check('Registered old sources remain unchanged',all(sha(linux(k))==v for k,v in frozen.items()))
    torch.cuda.synchronize()
    return dict(updated=now(),state='bounded_engineering_suite_complete_pending_visual_review',checks=checks,seconds_cuda=time.monotonic()-start,
        gaussian_points=96,total_optimizer_updates=12,source=ref(H/'native_training122_v5.py'),scope='synthetic-control-only',person_fit=False,person_training=False,physical_quality_accepted=False,
        limitations=['No new calibrated person cameras, material evidence or SMPL-X fit.', 'Synthetic review assertions only exercise code; they are not human inspection.', 'Fixed-count free centers; adaptive split/prune remains out of scope.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['baseline','cpu','gpu']);a=p.parse_args()
    phase=O/(a.phase+'-registration.json')
    if phase.exists():raise FileExistsError('Phase already executed; inspect retained evidence')
    plan=read(O/'plan.json')
    assert all(sha(linux(k))==v for k,v in plan['frozen'].items())
    if not (O/'execution-registration.json').exists():
        save(O/'execution-registration.json',dict(updated=now(),plan=ref(O/'plan.json'),command=sys.argv,
            source_preflight=ref(O/'source-preflight-20261029.json'),scope='synthetic engineering plus actual failed-input refusal',person_training=False))
    save(phase,dict(updated=now(),pid=os.getpid(),command=sys.argv,driver=ref(__file__)))
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('Registered bounded phase budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120 if a.phase=='gpu' else 560)
    try:
        if a.phase=='baseline':result=baseline()
        elif a.phase=='cpu':result=cpu_controls()
        else:result=gpu_controls()
        result['seconds']=time.monotonic()-started
        save(O/(a.phase+'-result.json'),result)
        print(json.dumps(dict(phase=a.phase,seconds=result['seconds'],state=result.get('state'),checks=len(result.get('checks',[])))))
    except Exception as e:
        save(O/(a.phase+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started))
        raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
