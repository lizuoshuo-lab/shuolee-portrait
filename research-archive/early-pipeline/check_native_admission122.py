"""T3 once-only entry integration: CPU fixtures plus the unchanged real failure."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse,json,sys,subprocess,signal,time,traceback
import numpy as np
from PIL import Image
from native_training_admission122 import audit_input,assert_snapshot,sha,read,save,native_camera,FACE_REGIONS,BODY_REGIONS,SCENE_REGIONS
from material_observations122 import Observation
from train_native_person122 import load_fresh_initialization

H=Path(__file__).resolve().parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-training-entry';I=R/'video-update-20261027'
def now():return datetime.now(timezone.utc).isoformat()
def ref(p):return dict(path=str(p.resolve()),sha256=sha(p))
def put(p,value):save(p,value);return ref(p)

def register():
    if O.exists():raise RuntimeError('T3 already registered; inspect current state')
    if read(R/'queue-state.json')['running'] is not None:raise RuntimeError('Existing task requires inspection')
    files=[H/n for n in ['native_training_admission122.py','train_native_person122.py','check_native_admission122.py','run_native_admission122.sh',
        'native_training122_v2.py','native_training122.py','material_observations122.py','material_geometry122.py',
        'native_pixel_convention122.py','native_portrait_renderer122_v2.py','native_scene_package122_v2.py']]
    inputs=[I/'clip03-scene-support-plan.json',I/'clip03-scene-support/result.json',I/'clip03-scene-support/decode-command.json',
        I/'clip03-fresh-geometry/result.json',I/'clip03-roof-review/reviewed-support.json',
        I/'material-geometry-consumer/gauge-consistent-repair/evaluated-observations.json',
        I/'material-geometry-consumer/gauge-consistent-repair/result.json']
    source=read(inputs[0])['source'];inputs.append(Path(source['path']))
    inputs += [I/'clip03-scene-support'/f'frame_{f:06d}.png' for f in read(inputs[0])['frames']]
    old=[Path('/mnt/d/00/工作/人体重建/src/shuolee_portrait')/n for n in ['admission.py','cli.py','training.py']]
    old += [H/n for n in ['native_interactive122.py','native_interactive122_v2.py','native_interactive122_v2.html']]
    O.mkdir()
    save(O/'plan.json',dict(stage='T3',registered=now(),kind='actual_training_entry_integration_no_person_training',
        problem='V2 optimizer was callable without consuming complete source/frame/mask/material evidence. Diagnostic7frames are not a preregistered training manifest.',
        change='Add CPU native input audit called unconditionally by train entry before initialization/GPU. Reuse N3/N13 observations and target-excluded geometry. Lazyoneview cache feeds unchanged v2 optimizer; fresh fit/surface provenance and1000review required.',
        actual_case='Existing clip03 source464encodedframes,7diagnostic cameras and102unchanged observations. Do not invent a full-frame plan, occlusion masks or physical evidence. Actual train command must exit3 before initializer/torch/optimizer.',
        controls=['Known exact3D synthetic fourview/fullK/region/image-mask fixture passes CPUinput audit only.',
            'Missing/fullframe mismatches,interpolated/invalid camera,unknown material/staticity/uncertainty,development exposure,family leakage and>1pxface eachreject.',
            'Imagehash/size/boolean/SSIM/support andgroup-region mislabel controls,missing scopedphysicalevidence,stalesnapshot andnooutputoverwrite.',
            'Explicitinteger-to-corner fullK mapping andoneview diskcache CPUread; noCUDAkernel.',
            'Fresh200k initializer binding positive andoldfit/camera/step/pointcount negative; arraysknownsynthetic, not personfit.',
            'Actualsevenframefailure recomputed by sameN13consumer; exactly102targets retained andsame13face errors; actual CLI train exits3withoutGPU/model.'],
        limits='Prepared-input entry only. Raw-video calibration/freshfit-surface executor are upstream, no claim of their integration orphysicaltruth. Physicalreviewrecords requireindependentevidence; hashesonlybindthem.',
        budget=dict(runs=1,CPU_wall_seconds=300,GPU=0,optimizer_steps=0,new_features=0,new_cameras=0),
        stop='Preserve anyfailure, repaironlydemonstratedcode errors. Nochange todata/K/threshold, nohumantrain regardlessoffixturepass.',
        source_hashes={str(p):sha(p) for p in files},input_hashes={str(p):sha(p) for p in inputs},historical_hashes={str(p):sha(p) for p in old}))
    q=read(R/'queue-state.json');q.update(updated=now(),phase='T3_native_training_entry_registered',running=None,
        next='Run registered300s CPUentry controls once, then reviewactualinputrejection. NoGPU/persontraining.')
    save(R/'queue-state.json',q);print('T3_REGISTERED')

def fixture(root):
    root.mkdir();source=root/'source.bin';source.write_bytes(b'T3 known synthetic source, not a captured person')
    src=ref(source);K=np.array([[25.,0.,17.25],[0.,27.,14.75],[0.,0.,1.]])
    cams=[]
    for f in range(4):
        E=np.eye(4);E[0,3]=-.45+f*.3
        cams.append(dict(frame_id=f,w2c=E.tolist(),registration='observed'))
    regions=sorted(FACE_REGIONS|BODY_REGIONS|SCENE_REGIONS)
    truth=dict(scope='known_synthetic_component_only',K=K.tolist(),cameras=cams,points=[])
    obs=[];annotations=[]
    for track in range(16):
        point=np.array([.10*np.sin(track),.08*np.cos(track),3.+.05*track]);truth['points'].append(point.tolist())
        fit=track>=12;region=None if fit else regions[track]
        group='static_scene' if fit or region in SCENE_REGIONS else 'face_candidate' if region in FACE_REGIONS else 'body_candidate'
        split='fit_background' if fit else 'heldout_background' if group=='static_scene' else 'heldout_human'
        for c in cams:
            E=np.array(c['w2c']);x=K@(E[:3,:3]@point+E[:3,3]);uv=tuple(x[:2]/x[2])
            o=Observation(src['sha256'],c['frame_id'],track,str(track),split,group,uv,32,32,True,
                'visible','confirmed','bounded_static',.01,('known_exact_synthetic_scene',),False)
            obs.append(o.record())
            if not fit:annotations.append(dict(key=list(o.key),region=region))
    ev=put(root/'known-evidence.json',truth)
    frameplan=put(root/'frame-plan.json',dict(kind='preregistered_training_frames',source_sha256=src['sha256'],frame_ids=list(range(4))))
    camera=put(root/'cameras.json',dict(source_sha256=src['sha256'],pixel_convention='integer_centers',K=K.tolist(),cameras=cams,fit_families=[str(i) for i in range(12,16)]))
    frames=[];dec=[]
    for f in range(4):
        rgb=np.zeros((32,32,3),np.uint8);rgb[:,:,f%3]=30+f*40
        Image.fromarray(rgb).save(root/f'rgb{f}.png')
        alpha=np.zeros((32,32),np.uint8);alpha[6:26,6:26]=255
        Image.fromarray(alpha).save(root/f'alpha{f}.png')
        Image.fromarray(np.full((32,32),255,np.uint8)).save(root/f'valid{f}.png')
        frames.append(dict(frame_id=f,rgb=ref(root/f'rgb{f}.png'),alpha=ref(root/f'alpha{f}.png'),rgb_valid=ref(root/f'alpha{f}.png'),observation_valid=ref(root/f'valid{f}.png')))
        dec.append(dict(frame_id=f,rgb_sha256=frames[-1]['rgb']['sha256'],width=32,height=32))
    decode=put(root/'decode.json',dict(source_sha256=src['sha256'],geometric_resampling=False,rotation_applied=False,frames=dec))
    observations=put(root/'observations.json',obs)
    for a in annotations:a['evidence']=ev
    coverage=put(root/'coverage.json',dict(source_sha256=src['sha256'],frame_plan_sha256=frameplan['sha256'],observations=annotations,
        requirements=[dict(frame_id=f,region=r,visibility='visible',evidence=ev) for f in range(4) for r in regions]))
    physical=dict(bindings=dict(source_sha256=src['sha256'],cameras_sha256=camera['sha256'],observations_sha256=observations['sha256'],
        frame_plan_sha256=frameplan['sha256'],decode_receipt_sha256=decode['sha256'],
        frames_sha256=__import__('hashlib').sha256(json.dumps(frames,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    for n in ['output_projection','pose_depth_support','static_scene_control','subject_staticity','mask_occluder_support','world_y_up']:
        physical[n]=dict(state='supported',method='exact constructed synthetic control',limitations=['Not physical evidence for any captured person'],evidence=ev)
    manifest=dict(schema='shuolee-native-training-input-v1',scope='synthetic-control-only',run_id='T3_fixture',
        input_domain='already_rectified_resized_cropped',projection='pinhole',radial_coefficients=None,pixel_convention='integer_centers',
        world_frame='right_handed_world_y_up',source=dict(asset=src,width=32,height=32,encoded_frame_count=4),frame_plan=frameplan,
        cameras=camera,decode_receipt=decode,frames=frames,observations=observations,coverage_plan=coverage,
        physical_evidence=put(root/'physical.json',physical),fresh_initialization=None,
        training=dict(steps=18000,seed=12231,learning_rates=dict(means=.003,quats=.002,log_scales=.005,opacity_logits=.02,sh=.03)))
    put(root/'input.json',manifest);return manifest

def run():
    if (O/'progress.json').exists():raise RuntimeError('Existing T3 run; inspect rather than repeat')
    plan=read(O/'plan.json');frozen={**plan['source_hashes'],**plan['input_hashes'],**plan['historical_hashes']}
    if any(sha(p)!=h for p,h in frozen.items()):raise RuntimeError('Pre-registered input/source changed')
    started=time.monotonic();tests=[]
    def check(name,condition,**details):
        tests.append(dict(name=name,passed=bool(condition),**details))
        save(O/'checks-progress.json',tests)
        if not condition:raise AssertionError(name)
    def expired(*_):raise TimeoutError('T3 bounded CPU budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(300)
    save(O/'progress.json',dict(state='running',pid=os.getpid(),updated=now()))
    try:
        root=O/'synthetic';m=fixture(root);report,_=audit_input(root/'input.json')
        check('Known complete synthetic input passes CPU audit only',report['training_admitted'] and report['heldout_face']['maximum_px']<1e-8)
        check('CPU input audit did not import torch', 'torch' not in sys.modules)
        save(O/'synthetic-positive.json',report)
        expected_issues={
            'missing-frame-plan':'invalid_asset','missing-image-frame':'image_frame_set_mismatch',
            'missing-camera':'camera_frame_set_mismatch','interpolated-camera':'unobserved_or_interpolated_camera',
            'scaled-extrinsic':'invalid_camera_contract','wrong-pixel-origin':'native_integer_pixel_measurements_required',
            'second-fisheye':'captured_export_requires_explicit_verified_output_projection',
            'bare-accepted-boolean':'physical_assessment_incomplete',
            **{k:'independent_material_geometry_failed' for k in ['material','staticity','uncertainty','development','contradiction','face-error']},
            'family-leakage':'invalid_material_observations','mask-hash':'invalid_asset',
            'nonbinary-mask':'nonboolean_supervision_mask','occluder-supervised':'foreground_outside_observation_support',
            'region-material-mismatch':'region_annotation_not_matching_heldout_material_group',
            'resized-image':'invalid_native_image','no-SSIM-support':'no_complete_foreground_SSIM_window'}
        def variant(name,patch):
            mm=deepcopy(m);patch(mm)
            # Synthetic controls rebind their declared fixture evidence so a
            # stale assessment hash cannot be the reason a different guard passes.
            if name!='bare-accepted-boolean':
                physical=read(m['physical_evidence']['path'])
                physical['bindings'].update(cameras_sha256=mm['cameras']['sha256'],
                    observations_sha256=mm['observations']['sha256'],
                    frame_plan_sha256=(mm['frame_plan'] or {}).get('sha256'),
                    frames_sha256=__import__('hashlib').sha256(json.dumps(mm['frames'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
                mm['physical_evidence']=put(O/f'{name}-physical.json',physical)
            p=O/f'case-{name}.json';save(p,mm);rr,_=audit_input(p)
            check(name+' rejected for the intended condition',not rr['training_admitted'] and expected_issues[name] in rr['issue_counts']
                and rr['source_observation_records']==64,expected_issue=expected_issues[name],issue_kinds=sorted(rr['issue_counts']))
            return rr
        variant('missing-frame-plan',lambda x:x.update(frame_plan=None))
        variant('missing-image-frame',lambda x:x['frames'].pop())
        def camera_patch(x,change,name):
            d=read(x['cameras']['path']);change(d);x['cameras']=put(O/f'{name}-camera.json',d)
        variant('missing-camera',lambda x:camera_patch(x,lambda d:d['cameras'].pop(),'missing'))
        variant('interpolated-camera',lambda x:camera_patch(x,lambda d:d['cameras'][0].update(registration='interpolated'),'interp'))
        variant('scaled-extrinsic',lambda x:camera_patch(x,lambda d:d['cameras'][0]['w2c'][0].__setitem__(0,2),'scale'))
        variant('wrong-pixel-origin',lambda x:x.update(pixel_convention='pixel_corners'))
        variant('second-fisheye',lambda x:x.update(projection='opencv_fisheye4',radial_coefficients=[0]*4))
        variant('bare-accepted-boolean',lambda x:x.update(physical_evidence=put(O/'bare.json',dict(accepted=True))))
        def obs_patch(x,patch,name):
            d=read(x['observations']['path']);patch(d);x['observations']=put(O/f'{name}-observations.json',d)
        for name,field,value in [('material','material_identity_evidence','unknown'),('staticity','static_evidence','unknown'),
            ('uncertainty','measurement_uncertainty_px',None),('development','development_exposure',True),('contradiction','material_identity_evidence','contradicted')]:
            rr=variant(name,lambda x,field=field,value=value,name=name:obs_patch(x,lambda d:d[0].update({field:value}),name))
            expected_reason={'material':'material_unknown','staticity':'target_staticness_unknown','uncertainty':'measurement_uncertainty_unknown',
                'development':'method_development_exposed','contradiction':'family_has_material_contradiction'}[name]
            check(name+' reason reaches the actual geometry consumer',any(expected_reason in r['reasons'] for r in rr['material_geometry']['observations']))
        variant('family-leakage',lambda x:obs_patch(x,lambda d:d[0].update(frozen_family_id='12'),'leak'))
        face_index=next(i for i,o in enumerate(read(m['observations']['path'])) if o['group']=='face_candidate')
        rr=variant('face-error',lambda x:obs_patch(x,lambda d:d[face_index].update(raw_uv=[d[face_index]['raw_uv'][0]+4,d[face_index]['raw_uv'][1]]),'error'))
        check('Failed face residual remains visible in entry report',rr['heldout_face']['maximum_px']>=1 and rr['heldout_face']['observations']==16)
        variant('mask-hash',lambda x:x['frames'][0]['rgb_valid'].update(sha256='0'*64))
        Image.fromarray(np.full((32,32),1,np.uint8)).save(O/'not-binary.png')
        variant('nonbinary-mask',lambda x:x['frames'][0].update(rgb_valid=ref(O/'not-binary.png')))
        Image.fromarray(np.zeros((32,32),np.uint8)).save(O/'empty-valid.png')
        variant('occluder-supervised',lambda x:x['frames'][0].update(observation_valid=ref(O/'empty-valid.png')))
        Image.fromarray(np.zeros((16,16,3),np.uint8)).save(O/'resized.png')
        variant('resized-image',lambda x:x['frames'][0].update(rgb=ref(O/'resized.png')))
        tiny=np.zeros((32,32),np.uint8);tiny[10:15,10:15]=255;Image.fromarray(tiny).save(O/'tiny.png')
        variant('no-SSIM-support',lambda x:x['frames'][0].update(rgb_valid=ref(O/'tiny.png')))
        def wrong_annotation(x):
            d=read(x['coverage_plan']['path']);d['observations'][0]['region']='nose' if d['observations'][0]['region'] in SCENE_REGIONS else 'far_scene'
            x['coverage_plan']=put(O/'wrong-region.json',d)
        variant('region-material-mismatch',wrong_annotation)
        # A completed audit is not a transferable permission after input changes.
        save(O/'stale-input.json',m);stale,_=audit_input(O/'stale-input.json')
        changed=deepcopy(m);changed['run_id']='different';save(O/'stale-input.json',changed)
        try:assert_snapshot(stale,O/'stale-input.json')
        except RuntimeError:rejected=True
        else:rejected=False
        check('Changed manifest cannot reuse previous admission',rejected)
        camera,bridge=native_camera(read(m['cameras']['path'])['cameras'][0],read(m['cameras']['path']),32,32,'rgb0.png')
        check('FullK integer-to-corner bridge changes only principal point by0.5',camera['K'][0][2]==17.75 and camera['K'][1][2]==15.25 and camera['K'][0][0]==25 and bridge['extrinsics_modified'] is False)
        # Build a diagnostic input bundle from existing bytes, not a new dataset.
        real=O/'real-input';real.mkdir();source=read(I/'clip03-scene-support-plan.json')['source']
        geo=read(I/'clip03-fresh-geometry/result.json');old=read(I/'material-geometry-consumer/gauge-consistent-repair/evaluated-observations.json')
        obsref=put(real/'observations.json',[r['observation'] for r in old['observations']])
        fit=[str(t['family']) for t in read(I/'clip03-roof-review/reviewed-support.json')['tracks'] if t['split']=='fit']
        camera_ref=put(real/'cameras.json',dict(source_sha256=source['sha256'],pixel_convention='integer_centers',K=geo['K_integer_centers'],
            fit_families=fit,cameras=[dict(frame_id=c['frame'],w2c=c['w2c'],registration='observed') for c in geo['cameras']]))
        rows=read(I/'clip03-scene-support/result.json')['rows'];frame_rows=[];decode_rows=[]
        for row in rows:
            f=row['frame'];image=I/'clip03-scene-support'/f'frame_{f:06d}.png'
            if sha(image)!=row['source_sha256']:raise RuntimeError('Existing decode image mismatch')
            frame_rows.append(dict(frame_id=f,rgb=ref(image),alpha=None,rgb_valid=None,observation_valid=None))
            decode_rows.append(dict(frame_id=f,rgb_sha256=sha(image),width=source['width'],height=source['height']))
        decode_ref=put(real/'decode.json',dict(source_sha256=source['sha256'],geometric_resampling=False,rotation_applied=False,frames=decode_rows,
            original_evidence=ref(I/'clip03-scene-support/result.json'),command=ref(I/'clip03-scene-support/decode-command.json')))
        actual=deepcopy(m);actual.update(scope='person-reconstruction',run_id='T3_clip03_diagnostic_only',
            world_frame='unvalidated_diagnostic_gauge',source=dict(asset=ref(Path(source['path'])),width=source['width'],height=source['height'],encoded_frame_count=source['frames']),
            cameras=camera_ref,frame_plan=None,decode_receipt=decode_ref,frames=frame_rows,observations=obsref,coverage_plan=None,physical_evidence=None,fresh_initialization=None)
        actual_path=real/'input.json';save(actual_path,actual)
        command=[sys.executable,str(H/'train_native_person122.py'),'train','--input',str(actual_path),'--output',str(O/'actual-entry')]
        with (O/'actual-entry.log').open('x') as log:proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        actual_report=read(O/'actual-entry/admission.json');status=read(O/'actual-entry/status.json')
        check('Actual train CLI rejects before torch initializer or optimizer',proc.returncode==3 and not status['GPU_imported'] and not status['initializer_loaded'] and not status['optimizer_constructed'] and not status['training_started'])
        check('Actual raw464count kept separate from missing training plan and7diagnostic cameras',actual_report['encoded_source_frames']==464 and not actual_report['declared_training_frames'] and len(actual_report['registered_camera_frames'])==7)
        actual_geo=actual_report['material_geometry'];before={tuple(r['prediction']['target_key']):r['error_px'] for r in old['observations']}
        delta=max(abs(r['error_px']-before[tuple(r['prediction']['target_key'])]) for r in actual_geo['observations'] if r['error_px'] is not None)
        check('All102actualtargets and13faceobservations retained with unchanged numeric result',actual_geo['targets']==102 and actual_report['heldout_face']['observations']==13 and delta<1e-10,max_numeric_difference_px=delta)
        state_sha=sha(O/'actual-entry/status.json')
        repeated=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        check('Existing run output refuses overwrite without changing its status',repeated.returncode!=0 and sha(O/'actual-entry/status.json')==state_sha and not (O/'actual-entry/failure.json').exists())
        check('Entry rejection never creates a Gaussian directory',not (O/'actual-entry/gaussian-run').exists())
        check('No torch before all physical input refusal controls','torch' not in sys.modules)
        # Exercise actual disk-view bridge on CPU, no rendering or optimization.
        from train_native_person122 import PixelCache,DiskView
        cache=PixelCache(root,'cpu');d0=DiskView(m['frames'][0],camera,cache,'cpu');a=d0.rgb
        c1,_=native_camera(read(m['cameras']['path'])['cameras'][1],read(m['cameras']['path']),32,32,'rgb1.png')
        d1=DiskView(m['frames'][1],c1,cache,'cpu');b=d1.rgb
        check('Disk views use exact native RGB and one image cache',a.shape==(32,32,3) and b.shape==a.shape and cache.current==1 and float(a[0,0,0])>0 and float(b[0,0,1])>0 and float(b[0,0,0])==0)
        check('CPU views retain complete explicit renderer K without camera gradients',float(d0.K[0,0,2])==17.75 and not d0.K.requires_grad and not d0.w2c.requires_grad)
        # Real initialization loader exercised with declared synthetic200k arrays only.
        initroot=O/'init-controls';initroot.mkdir();arrays=initroot/'initial.npz'
        np.savez_compressed(arrays,means=np.zeros((200000,3),np.float32),quats=np.tile(np.array([1,0,0,0],np.float32),(200000,1)),
            scales=np.full((200000,3),.01,np.float32),opacities=np.full(200000,.1,np.float32),sh=np.zeros((200000,16,3),np.float32))
        fit_arrays=put(initroot/'known-fit.json',dict(synthetic=True,not_a_person_fit=True))
        fit_source=ref(Path(__file__));fitreceipt=dict(run_id=m['run_id'],source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],frame_plan_sha256=m['frame_plan']['sha256'],
            from_scratch=True,old_individual_geometry_loaded=False,executed_command=['synthetic-control'],exit_code=0,fit_arrays_sha256=fit_arrays['sha256'],fit_source_sha256=fit_source['sha256'])
        init=dict(schema='shuolee-fresh-clothing-initialization-v1',run_id=m['run_id'],source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],frame_plan_sha256=m['frame_plan']['sha256'],
            representation='free_world_clothing_surface',world_frame='right_handed_world_y_up',old_individual_geometry_loaded=False,old_gaussians_loaded=False,old_optimizer_loaded=False,
            training_steps=0,initial_points=200000,fresh_fit_receipt=put(initroot/'fit-receipt.json',fitreceipt),fit_arrays=fit_arrays,fit_source=fit_source,initializer_source=fit_source,arrays=ref(arrays),executed_command=['synthetic-control'],exit_code=0)
        mm=deepcopy(m);mm['fresh_initialization']=put(initroot/'init.json',init)
        values,_=load_fresh_initialization(mm,root,'a'*64)
        check('Actual initialization loader reads complete synthetic200k attributes',len(values['means'])==200000 and values['sh'].shape==(200000,16,3))
        for name,key,val in [('old-fit','old_individual_geometry_loaded',True),('trained-points','training_steps',1),('wrong-camera','cameras_sha256','b'*64),('wrong-count','initial_points',199999)]:
            bad=dict(init);bad[key]=val;mm['fresh_initialization']=put(initroot/f'{name}.json',bad)
            try:load_fresh_initialization(mm,root,'a'*64)
            except ValueError:rejected=True
            else:rejected=False
            check('Initializer rejects '+name,rejected)
        check('Frozen inputs old113entries and live native sources unchanged',all(sha(p)==h for p,h in frozen.items()))
        import torch
        check('No CUDA context initialized',not torch.cuda.is_initialized())
        result=dict(stage='T3',state='complete_pending_actual_review',updated=now(),seconds=time.monotonic()-started,checks=tests,
            all_controls_passed=True,actual_entry=dict(command=command,exit_code=3,report='actual-entry/admission.json',status=status,
            issue_counts=actual_report['issue_counts'],source_encoded_frames=464,diagnostic_camera_frames=7,training_frame_plan_established=False,
            targets=102,heldout_face=actual_report['heldout_face'],numeric_difference_from_N13_px=delta),
            GPU_used=False,optimizer_steps=0,new_person_training_started=False,physical_training_admission=False,
            entry_integrated=True,entry_scope='Prepared-native-input audit/streaming and dispatch to existingv2trainer. Raw-video calibration and freshfit/surface executor remain upstream; positive realperson training not tested.',
            remaining=['No physically admitted fullframeperson input.','Actual newfit/surface generator integration andfullresolutioncapacity remain unverified.','Staticreconstruction appearance,novelviews,fourpeople andunseen generalization notaccepted.'])
        save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_actual_review',pid=None,updated=now()))
        print(json.dumps(dict(checks=len(tests),all_controls_passed=True,actual_face=result['actual_entry']['heldout_face'],no_GPU=True)))
    except BaseException as e:
        save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),checks=tests))
        save(O/'progress.json',dict(state='failed',pid=None,updated=now()));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','run']);args=parser.parse_args();globals()[args.stage]()
