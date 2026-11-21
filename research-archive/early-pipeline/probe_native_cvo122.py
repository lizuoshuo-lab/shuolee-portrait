"""T8 preregistered component controls; never runs a person or estimates a camera."""
from pathlib import Path
from datetime import datetime, timezone
import copy, gc, hashlib, json, os, signal, sys, time, traceback

H = Path(__file__).resolve().parent
R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O = R / 'native-cvo-control'
G = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
sys.path.insert(0, str(G))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def now(): return datetime.now(timezone.utc).isoformat()
def mounted(p):
    p = p.replace('\\','/')
    return Path('/mnt/'+p[0].lower()+p[2:]) if len(p)>2 and p[1]==':' else Path(p)


def small_fixture():
    import numpy as np
    rng = np.random.default_rng(1228)
    xx, yy = np.meshgrid(np.linspace(-1.5,1.5,12), np.linspace(-.7,.7,8))
    means = np.column_stack([xx.ravel(),yy.ravel(),2+.02*np.sin(xx.ravel())]).astype(np.float32)
    n = len(means); q = rng.normal(size=(n,4)).astype(np.float32); q /= np.linalg.norm(q,axis=1,keepdims=True)
    sh = (rng.normal(size=(n,16,3))*.01).astype(np.float32)
    sh[:,0] = np.array([.3,.05,-.1],np.float32)
    p = dict(means=means,quats=q,scales=np.tile(np.array([.08,.06,.04],np.float32),(n,1)),
             opacities=np.full(n,.6,np.float32),sh=sh)
    cams = []
    for i in range(4):
        e = np.eye(4); e[0,3] = -i*.7
        cams.append(dict(id='known-'+str(i),image_name='known-'+str(i)+'.png',width=160,height=128,
                         K=[[110.,0.,93.8+i*2.5],[0.,112.,67.2-i*.75],[0.,0.,1.]],w2c=e.tolist()))
    return p, cams


def known_observations(cams):
    rows=[]
    # Synthetic ground truth, not a designation of any real person observation.
    for i,c in enumerate(cams):
        for f in range(16):
            if f>=8 and (i+f)%3==0: continue
            rows.append(dict(frame_id=c['id'],family_id='known-physical-'+str(f),split='fit',group='foreground',
                             visible=True,identity='confirmed',static='confirmed',evidence_sha256='c'*64))
        for f in range(3):
            rows.append(dict(frame_id=c['id'],family_id='known-heldout-'+str(f),split='heldout',group='foreground',
                             visible=True,identity='confirmed',static='confirmed',evidence_sha256='d'*64))
    return rows


def cpu(check, reject, plan):
    import numpy as np
    import native_linear_archive122 as archive
    import native_covisibility122 as cvo
    import native_scene_package122_v3 as package
    from native_portrait_renderer122_v2 import NativeCameraContractV2
    p, cams = small_fixture(); rows=known_observations(cams)
    graph=cvo.build_graph(cams,rows,source_sha256='a'*64,min_shared=4)
    save(O/'known-fit-graph.json',graph)
    check('all_six_known_edges',len(graph['edges'])==6)
    check('heldout_families_never_enter_pairing',not any('heldout' in f for e in graph['edges'] for f in e['families']))
    mixed=copy.deepcopy(rows);mixed[-1]['family_id']=rows[0]['family_id']
    reject('mixed_physical_family_split_rejected',lambda:cvo.build_graph(cams,mixed,source_sha256='a'*64,min_shared=4))
    reject('duplicate_descriptor_not_extra_material',lambda:cvo.build_graph(cams,rows+[rows[0]],source_sha256='a'*64,min_shared=4))
    bad=copy.deepcopy(rows);bad[0]['identity']='contradicted'
    badgraph=cvo.build_graph(cams,bad,source_sha256='a'*64,min_shared=4)
    check('one_contradiction_excludes_entire_family',not any(rows[0]['family_id'] in e['families'] for e in badgraph['edges']))
    unknown=copy.deepcopy(rows)
    for x in unknown:x['identity']='unknown'
    reject('unsupported_graph_cannot_sample',lambda:cvo.select_cycle(cvo.build_graph(cams,unknown,source_sha256='a'*64,min_shared=4),seed=42,policy='covisible'))
    a=cvo.select_cycle(graph,seed=42,policy='covisible'); b=cvo.select_cycle(graph,seed=42,policy='random-qualified')
    ca=cvo.validate_schedule(graph,a);cb=cvo.validate_schedule(graph,b)
    check('paired_policies_match_all_view_exposures_and_updates',ca==cb and set(ca.values())=={2} and len(a)==len(b)==4)
    save(O/'known-schedules.json',dict(covisible=a,random_qualified=b,exposures=ca,training_comparison_executed=False))
    reject('missing_frame_schedule_rejected',lambda:cvo.validate_schedule(graph,a[:-1]))
    # Independent world gauge rotation: w2c changes by right multiplication.
    angle=.7; q=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    angled=copy.deepcopy(cams)
    for i,c in enumerate(angled):
        x=.2*i;rr=np.array([[np.cos(x),0,np.sin(x)],[0,1,0],[-np.sin(x),0,np.cos(x)]])
        e=np.asarray(c['w2c']);e[:3,:3]=rr;c['w2c']=e.tolist()
    rotated=copy.deepcopy(angled)
    for c in rotated:
        e=np.asarray(c['w2c']);e[:3,:3]=e[:3,:3]@q.T;c['w2c']=e.tolist()
    g0=cvo.build_graph(angled,rows,source_sha256='a'*64,min_shared=4)
    g1=cvo.build_graph(rotated,rows,source_sha256='a'*64,min_shared=4)
    angle_error=max(abs(x['world_axis_angle_deg']-y['world_axis_angle_deg']) for x,y in zip(g0['edges'],g1['edges']))
    check('world_axis_angles_invariant_to_world_gauge',angle_error<1e-10,max_abs_degrees=angle_error)
    archive.write(O/'cpu-linear.npz',p);restored=archive.read(O/'cpu-linear.npz',vertices=96,degree=3)
    check('linear_arrays_bit_exact_all_attributes',all(np.array_equal(p[k],restored[k]) for k in archive.FIELDS))
    reject('archive_wrong_count_rejected',lambda:archive.read(O/'cpu-linear.npz',vertices=95,degree=3))
    qbad=copy.deepcopy(p);qbad['quats']*=2
    reject('archive_does_not_renormalize_bad_input',lambda:archive.validate(qbad))
    f64=copy.deepcopy(p);f64['scales']=f64['scales'].astype(np.float64)
    reject('archive_no_hidden_float64_conversion',lambda:archive.validate(f64))
    extra=copy.deepcopy(p);extra['optimizer']=np.zeros(1)
    reject('archive_is_not_optimizer_resume',lambda:archive.validate(extra))
    contract=NativeCameraContractV2('pinhole','synthetic_known_camera')
    scene_path=package.write_scene(O/'cpu-package',p,cams,contract=contract,scope='synthetic-control-only')
    scene=package.load_scene(scene_path)
    check('all_camera_principal_points_and_transforms_retained',list(scene.cameras.values())==cams)
    check('package_exact_native_parameters',all(np.array_equal(p[k],scene.parameters[k]) for k in archive.FIELDS))
    original=read(scene_path)
    def bad_manifest(name, mutate):
        m=copy.deepcopy(original);mutate(m);dest=scene_path.parent/(name+'.json');save(dest,m)
        return lambda:package.load_scene(dest)
    reject('reject_native_asset_hash_mismatch',bad_manifest('wrong-hash',lambda m:m['model']['native_asset'].update(sha256='f'*64)))
    reject('reject_native_path_escape',bad_manifest('escape',lambda m:m['model']['native_asset'].update(path='../cpu-linear.npz')))
    reject('reject_renderer_reinterpretation',bad_manifest('renderer',lambda m:m['renderer'].update(gaussian_evaluation='screen-2d')))
    reject('reject_inference_claim_as_resume',bad_manifest('resume',lambda m:m.update(optimizer='restore-me')))
    reject('reject_synthetic_as_person',bad_manifest('person',lambda m:m.update(scope='person-reconstruction')))
    return dict(graph_edges=len(graph['edges']),exposures=ca,native_array_count=96,cpu_only=True,
                human_training=False,quality_accepted=False)


def gpu(check, reject, plan):
    import numpy as np
    import torch
    from PIL import Image
    import native_covisibility122 as cvo
    import native_linear_archive122 as archive
    import native_scene_package122_v3 as package
    from native_gaussian_ply122 import read_standard_ply
    from native_portrait_renderer122_v2 import NativeCameraContractV2, render_native
    from native_training122 import NativeTrainingView, foreground_objective
    from native_training122_v2 import FreshNativeGaussians
    contract=NativeCameraContractV2('pinhole','synthetic_known_camera')
    check('registered_cpu_controls_passed',read(O/'cpu-result.json')['state']=='passed')
    check('exact_cuda_runtime',torch.__version__=='2.7.1+cu128' and package.runtime_record()['version']=='1.5.3')
    save(O/'runtime.json',dict(backend=package.runtime_record(),device=torch.cuda.get_device_name(0)))
    small,cams=small_fixture(); cams=cams[:2]
    def make_views(p,cameras,large=False):
        target={k:v.copy() for k,v in p.items()}
        target['means'][:,0]+=.012 if not large else .002
        target['sh'][:,0,:]+=.025
        views=[]
        for camera in cameras:
            rgb,alpha,_=package.render_parameters(target,camera,contract=contract)
            mask=(alpha>.05);valid=np.ones_like(mask,dtype=bool)
            h,w=mask.shape[:2]
            # A known synthetic external occluder is ignored, never empty-space supervision.
            valid[h//2-3:h//2+3,w//2-3:w//2+3]=False;mask &= valid
            view=NativeTrainingView.create(camera,np.clip(rgb,0,1),alpha,mask,valid,device='cuda')
            views.append(view)
        return views
    views=make_views(small,cams)
    model=FreshNativeGaussians(small,device='cuda');opt=model.optimizer(plan['optimizer'])
    reference=FreshNativeGaussians(small,device='cuda');ropt=reference.optimizer(plan['optimizer'])
    ropt.zero_grad(set_to_none=True); independent=[];meanloss=0
    for v in views:
        p=reference.parameters_linear()
        rgb,alpha,meta=render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],v.w2c,v.K,
            v.camera['width'],v.camera['height'],contract=contract,sh_degree=3)
        loss,_=foreground_objective(rgb[0],alpha[0],v)
        g=torch.autograd.grad(loss,reference.means,retain_graph=True)[0]
        independent.append(dict(visible=(meta['radii'][0]>0).all(-1).detach(),radius=meta['radii'][0].amax(-1).float().detach(),grad=g.detach()))
        meanloss=meanloss+loss/2
    meanloss.backward();rgrads={k:p.grad.detach().clone() for k,p in reference.named_parameters()};ropt.step()
    actual=cvo.accumulate_update(model,views,contract,opt)
    error={k:float((actual['gradients'][k]-rgrads[k]).abs().max()) for k in rgrads}
    check('CVO_mean_gradient_matches_joint_objective',all(torch.allclose(actual['gradients'][k],rgrads[k],atol=2e-6,rtol=2e-4) for k in rgrads),max_abs=error)
    paramerror=max(float((p-dict(reference.named_parameters())[k]).abs().max()) for k,p in model.named_parameters())
    check('CVO_single_Adam_matches_reference_update',paramerror<2e-6,max_abs=paramerror)
    expected_counts=sum(x['visible'].to(torch.int32) for x in independent)
    expected_radii=torch.maximum(*[torch.where(x['visible'],x['radius'],0) for x in independent])
    expected_g=sum(torch.where(x['visible'],torch.linalg.vector_norm(x['grad'],dim=-1),0) for x in independent)
    check('visibility_counts_cover_both_views',torch.equal(actual['visible_count'],expected_counts))
    check('projected_radius_max_covers_both_views',torch.equal(actual['max_projected_radius'],expected_radii))
    check('world_gradient_statistics_cover_both_views',torch.allclose(actual['world_gradient_norm_sum'],expected_g,atol=2e-6,rtol=2e-4))
    only_first=int((independent[0]['visible']&~independent[1]['visible']).sum())
    only_second=int((independent[1]['visible']&~independent[0]['visible']).sum())
    check('control_has_points_missed_by_either_anchor_only',only_first>0 and only_second>0,first_only=only_first,second_only=only_second)
    # Masked pixels may contain arbitrary colors/alpha without affecting the loss.
    v=views[0];before_rgb=v.rgb.clone();before_alpha=v.alpha.clone()
    p=model.parameters_linear();rr,aa,_=render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],v.w2c,v.K,
        v.camera['width'],v.camera['height'],contract=contract,sh_degree=3)
    loss0,_=foreground_objective(rr[0],aa[0],v)
    v.rgb[~v.rgb_valid[...,0]]=1-v.rgb[~v.rgb_valid[...,0]]
    v.alpha[~v.observation_valid]=1-v.alpha[~v.observation_valid]
    loss1,_=foreground_objective(rr[0],aa[0],v)
    check('unknown_occlusion_and_blank_RGB_do_not_supervise',torch.equal(loss0,loss1))
    v.rgb.copy_(before_rgb);v.alpha.copy_(before_alpha)
    save(O/'small-controls.json',dict(gradient_max_abs=error,parameter_max_abs=paramerror,
        first_only_visible=only_first,second_only_visible=only_second,steps_per_fresh_model=1,adaptive_density=False))
    del model,opt,reference,ropt,views,independent,actual,meanloss,p,rr,aa,loss0,loss1,rgrads,g,loss,rgb,alpha,meta,v
    gc.collect();torch.cuda.empty_cache()
    # Full-size known geometry. Same frozen200k arrays, two-view ONE update.
    fixture=R/'native-training-integration/subpixel-culling-diagnosis'
    with np.load(fixture/'frozen-large-arrays.npz',allow_pickle=False) as z:initial={k:z[k] for k in z.files}
    cameras=plan['native_cameras'];views=make_views(initial,cameras,large=True)
    model=FreshNativeGaussians(initial,device='cuda');optimizer=model.optimizer(plan['optimizer'])
    check('large_optimizer_initially_empty',len(optimizer.state)==0)
    before=model.means.detach().clone();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    output=cvo.accumulate_update(model,views,contract,optimizer)
    torch.cuda.synchronize();elapsed=time.monotonic()-tick
    moved=int((model.means.detach()!=before).any(-1).sum())
    check('200k_centers_have_actual_fresh_update',moved>0,moved=moved)
    parameters=model.numpy_linear()
    # Save exact PRE-EXPORT linear arrays before invoking either scene codec.
    archive.write(O/'preexport-linear.npz',parameters)
    native_refs=[]
    for c in cameras:
        rgb,alpha,meta=package.render_parameters(parameters,c,contract=contract)
        np.savez_compressed(O/(c['id']+'-preexport-float.npz'),rgb=rgb,alpha=alpha)
        Image.fromarray(np.uint8(np.clip(rgb,0,1)*255+.5)).save(O/(c['id']+'-reference.png'))
        native_refs.append(dict(camera=c,metadata=meta))
    peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
    stat=dict(exposures=[v.camera['id'] for v in views],visible_by_view=[x['visible_count'] for x in output['per_view']],
              visible_union=int((output['visible_count']>0).sum()),visible_both=int((output['visible_count']==2).sum()),
              statistic=output['statistic'],adaptive_density_implemented=False)
    del model,optimizer,views,output,before
    gc.collect();torch.cuda.empty_cache()
    path=package.write_scene(O/'exact-native-package',parameters,cameras,contract=contract,scope='synthetic-control-only')
    loaded=package.load_scene(path)
    check('full200k_native_attributes_bit_exact',all(np.array_equal(parameters[k],loaded.parameters[k]) for k in archive.FIELDS))
    preview=read_standard_ply(path.parent/'point_cloud.ply')
    delta={k:float(np.max(np.abs(parameters[k]-preview[k]))) for k in archive.FIELDS}
    comparisons=[]
    for item in native_refs:
        c=item['camera']
        with np.load(O/(c['id']+'-preexport-float.npz')) as z:ref={k:z[k].copy() for k in z.files}
        rgb,alpha,receipt=package.render_recorded(loaded,c['id'],'pinhole')
        dr=float(np.max(np.abs(rgb-ref['rgb'])));da=float(np.max(np.abs(alpha-ref['alpha'])))
        check(c['id']+':exact_native_fullfloat_replay',max(dr,da)<=plan['limits']['native_roundtrip_max_abs'],rgb=dr,alpha=da)
        np.savez_compressed(O/(c['id']+'-replay-float.npz'),rgb=rgb,alpha=alpha)
        pr,pa,_=package.render_parameters(preview,c,contract=contract)
        dpr=float(np.max(np.abs(pr-ref['rgb'])));dpa=float(np.max(np.abs(pa-ref['alpha'])))
        # Preview is measured separately, never promoted by exact-native success.
        comparisons.append(dict(camera_id=c['id'],native_rgb_max=dr,native_alpha_max=da,
            preview_rgb_max=dpr,preview_alpha_max=dpa,preview_within_prior_2e5=max(dpr,dpa)<=2e-5,receipt=receipt))
        Image.fromarray(np.uint8(np.clip(pr,0,1)*255+.5)).save(O/(c['id']+'-preview.png'))
        Image.fromarray(np.uint8(np.clip(alpha[...,0],0,1)*255+.5)).save(O/(c['id']+'-alpha.png'))
    return dict(steps=1,view_exposures=2,points=200000,moved_centers=moved,update_seconds=elapsed,
                torch_peak=peak,statistics=stat,comparisons=comparisons,preview_parameter_max_abs=delta,
                scope='synthetic-component-only',human_training=False,physical_quality_accepted=False,
                adaptive_density_implemented=False,sampler_quality_comparison_executed=False,
                native_package=str(path),preexport_arrays_saved=True,preexport_float_references_saved=True)


def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ('--cpu','--gpu'): raise ValueError('One registered stage required')
    stage=sys.argv[1][2:];dest=O/(stage+'-result.json')
    if dest.exists() or (O/(stage+'-progress.json')).exists(): raise RuntimeError('Already attempted; inspect evidence, never duplicate')
    plan=read(O/'plan.json');reg=read(O/'execution-registration.json')
    if reg['plan_sha256']!=sha(O/'plan.json') or any(sha(mounted(p))!=s for p,s in reg['sources'].items()):
        raise RuntimeError('Changed registration/implementation before run')
    if stage=='gpu' and os.environ.get('SHUOLEE_T8_SHARED_LOCK')!='held-by-flock': raise RuntimeError('Registered GPU flock required')
    start=time.monotonic();checks=[]
    def check(name, condition, **details):
        checks.append(dict(name=name,passed=bool(condition),**details))
        if not condition: raise RuntimeError('Control failed: '+name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,FileNotFoundError):check(name,True)
        else:check(name,False)
    def timeout(a,b):raise TimeoutError('Registered stage wall budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['limits'][stage+'_wall_seconds'])
    save(O/(stage+'-progress.json'),dict(state='running',pid=os.getpid(),updated=now(),stage=stage))
    try:
        result=(cpu if stage=='cpu' else gpu)(check,reject,plan)
        check('registered_sources_unchanged',all(sha(mounted(p))==s for p,s in reg['sources'].items()))
        result.update(state='passed',checks=checks,elapsed=time.monotonic()-start,updated=now())
        save(dest,result);save(O/(stage+'-progress.json'),dict(state='complete',pid=None,updated=now()))
        print(json.dumps(dict(stage=stage,state='passed',checks=len(checks),elapsed=result['elapsed'])),flush=True)
    except BaseException as e:
        save(dest,dict(state='failed',stage=stage,checks=checks,error=repr(e),traceback=traceback.format_exc(),elapsed=time.monotonic()-start,
                       human_training=False,physical_quality_accepted=False))
        save(O/(stage+'-progress.json'),dict(state='failed',pid=None,error=repr(e),updated=now()))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
