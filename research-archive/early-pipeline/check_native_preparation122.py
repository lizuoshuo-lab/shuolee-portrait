"""T4 bounded preparation integration and known-envelope causal controls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse,json,sys,subprocess,signal,time,traceback
import numpy as np
from PIL import Image
from native_training_admission122 import read,save,sha
from native_surface122 import evaluate_support,fit_proposal,initialize_surface
from prepare_native_person122 import build_fit_inputs,SURFACE_CONFIG,dispatch_training
H=Path(__file__).parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-fresh-preparation';PRE=R/'native-training-entry'
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
def now():return datetime.now(timezone.utc).isoformat()
def ref(p):return dict(path=str(p.resolve()),sha256=sha(p))
def put(p,v):save(p,v);return ref(p)

def register():
    if O.exists() or read(R/'queue-state.json')['running'] is not None:raise RuntimeError('Existing stage; inspect rather than duplicate')
    source=[H/n for n in ['native_surface122.py','prepare_native_person122.py','check_native_preparation122.py',
        'native_training_admission122.py','train_native_person122.py','native_training122_v2.py','native_training122.py','native_portrait_renderer122_v2.py',
        'native_interactive122.py','native_interactive122_v2.py']]
    historical=[PROJECT/'src/shuolee_portrait'/n for n in ['fitting.py','fit_gate.py','prior.py','runtime.py','clothed_surface.py','cli.py']]
    inputs=[PRE/'real-input/input.json',PRE/'actual-entry/admission.json',PRE/'synthetic/input.json']
    O.mkdir()
    save(O/'plan.json',dict(stage='T4',registered=now(),kind='prepared_input_freshfit_surface_orchestration_and_observed_support',
        codefinding='Old119surface divides positive silhouette votes by all frames. Out-of-image/behindcamera/occlusion can become zero negative votes; it has no observed-valid mask.',
        hypothesis='In a controlled static silhouette with known occlusion, normalization by observed-valid votes preserves a supported clothing volume while the all-frame denominator wrongly rejects it.',
        scope='Integrate fresh local official prior/sharedSMPLX fit with fullK+.5/native observations, actualfit review, free observed-support clothing surface, actualsurface review then existingT3train entry. Oldfitter/renderer notmodified.',
        fixed=dict(surface=SURFACE_CONFIG,fit_steps=1200,source_K_not_fitted=True,physical_gate_unchanged=True),
        no_claim='This controls initialization occlusion behavior, not the unique cause of oldface ghosting. Positive actual person prior/fit/training unavailable without physical camera/material admission.',
        tests=['Native observation/fitter bridge preserves exact pixels,K,E,frame/time identity,oldpersonalizedinput refusal.',
            'Known eightview clothedbox,offaxis fullK: obscured pixels abstain; actualbackground rejects; outofframe/behindcamera/insufficientsupport unknown.',
            'New20万 unbound seeds lie outside innerbarebox and on known outer silhouette envelope; fullNPZ attributes and static/worldtransform preserved.',
            'Unknown boundary and clipped proposal stop rather than invent surfaces. No threshold tuning.',
            'Actual rejected whiteT input calls newfit command and exits3 before any prior/fit/GPU; unchanged102face/body/bg evidence.',
            'Missing actualsurface review cannotdispatch training. Existing frozen sources preserved.'],
        budget=dict(runs=1,CPU_seconds=300,GPU=0,person_fit=0,Gaussian_optimizer_steps=0,max_grid_voxels=5000000),
        stop='Preserve failure; fixonlydemonstratedimplementation errors, no threshold/data/K relaxation.',
        sources={str(p):sha(p) for p in source},historical={str(p):sha(p) for p in historical},inputs={str(p):sha(p) for p in inputs}))
    q=read(R/'queue-state.json');q.update(updated=now(),phase='T4_fresh_preparation_registered',running=None,next='Run registered T4CPU controls once; actualperson prior/fit/train remain denied.');save(R/'queue-state.json',q)
    print('T4_REGISTERED')

def fit_fixture():
    root=O/'fit-adapter';root.mkdir();m=deepcopy(read(PRE/'synthetic/input.json'))
    obs=root/'raw-landmarks.npz'
    np.savez_compressed(obs,frame_ids=np.arange(4),boxes=np.tile(np.array([6,6,26,26.]),(4,1)),keypoints=np.full((4,133,2),16.),scores=np.ones((4,133)),flip_disagreement=np.zeros((4,133)))
    evidence=put(root/'fixture-evidence.json',dict(scope='synthetic_coordinate_adapter_only',not_anatomical_or_physical_person_truth=True))
    spec=dict(schema='shuolee-native-fit-input-v1',source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],frame_plan_sha256=m['frame_plan']['sha256'],
        pixel_convention='integer_centers',observations=ref(obs),frame_clock=[dict(frame_id=i,time_seconds=i/30) for i in range(4)],
        clock_evidence=evidence,observation_evidence=evidence,old_individual_fit_loaded=False)
    m['fit_inputs']=put(root/'fit-inputs.json',spec)
    return root,m,spec

def known_scene():
    root=O/'known-scene';root.mkdir();K=np.array([[110.,0,71.3],[0,108.,57.7],[0,0,1]])
    w,h=144,128;yy,xx=np.mgrid[:h,:w];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    inner_lo=np.array([-.22,-.6,-.12]);inner_hi=-inner_lo;outer_lo=np.array([-.32,-.6,-.25]);outer_hi=-outer_lo
    vertices=np.array([[x,y,z] for x in [inner_lo[0],inner_hi[0]] for y in [inner_lo[1],inner_hi[1]] for z in [inner_lo[2],inner_hi[2]]])
    fit=dict(vertices_world=np.repeat(vertices[None],8,axis=0),root_rotation=np.repeat(np.eye(3)[None],8,axis=0),world_translation=np.zeros((8,3)),world_scale=np.array(1.))
    views=[];images=[]
    for index,theta in enumerate(np.arange(8)*np.pi/4):
        c=np.array([2*np.sin(theta),.2,2*np.cos(theta)]);z=-c/np.linalg.norm(c);x=np.cross(z,[0,1,0]);x/=np.linalg.norm(x);y=np.cross(z,x)
        rot=np.stack([x,y,z]);E=np.eye(4);E[:3,:3]=rot;E[:3,3]=-rot@c
        d=rays@rot
        with np.errstate(divide='ignore',invalid='ignore'):
            v0=(outer_lo-c)/d;v1=(outer_hi-c)/d
        enter=np.minimum(v0,v1).max(2);leave=np.maximum(v0,v1).min(2)
        mask=(leave>=np.maximum(enter,0)).astype(np.uint8)*255
        valid=np.full((h,w),255,np.uint8)
        if index<2:valid[h//2:,:]=0
        Image.fromarray(mask).save(root/f'alpha{index}.png');Image.fromarray(valid).save(root/f'valid{index}.png')
        views.append(dict(w2c=E.tolist(),alpha=ref(root/f'alpha{index}.png'),observation_valid=ref(root/f'valid{index}.png')))
        images.append(mask)
    save(root/'known-geometry.json',dict(K_integer_centers=K.tolist(),inner_box=[inner_lo.tolist(),inner_hi.tolist()],clothing_box=[outer_lo.tolist(),outer_hi.tolist()],
        cameras=views,known_occlusion='Lower half of first two views deliberately unknown; all other views known. This is synthetic only.'))
    return root,K,fit,views,images

def run():
    if (O/'progress.json').exists():raise RuntimeError('Existing run; inspect not repeat')
    plan=read(O/'plan.json');frozen={**plan['sources'],**plan['historical'],**plan['inputs']}
    if any(sha(p)!=h for p,h in frozen.items()):raise RuntimeError('Frozen inputs/source changed')
    started=time.monotonic();checks=[]
    def check(name,value,**details):
        checks.append(dict(name=name,passed=bool(value),**details));save(O/'checks.json',checks)
        if not value:raise AssertionError(name)
    def rejects(name,fn,reason):
        try:fn()
        except (ValueError,RuntimeError,FileNotFoundError) as e:check(name,reason in str(e),error=str(e))
        else:check(name,False)
    def timeout(*_):raise TimeoutError('T4CPU budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    save(O/'progress.json',dict(state='running',pid=os.getpid(),updated=now()))
    try:
        root,m,spec=fit_fixture();adapt=root/'prepared';adapt.mkdir();binding=build_fit_inputs(m,root,adapt)
        rows=read(adapt/'data/inputs.json')['records'];oldcam=read(m['cameras']['path'])
        check('Raw landmark NPZ bytes unchanged for new fitter',binding['observation_bytes_unchanged'])
        check('Frames slots clock and image hashes exact', [r['frame'] for r in rows]==list(range(4)) and [r['official_prediction_slot'] for r in rows]==list(range(4)) and [r['time_seconds'] for r in rows]==[i/30 for i in range(4)] and all(r['source_sha256']==m['frames'][i]['rgb']['sha256'] for i,r in enumerate(rows)))
        check('FullK bridge matches oldfitter-minus-half and prior-integer conventions',all(np.array_equal(np.array(r['K'])-np.array([[0,0,.5],[0,0,.5],[0,0,0]]),oldcam['K']) and r['w2c']==oldcam['cameras'][i]['w2c'] for i,r in enumerate(rows)))
        check('One shared static body config and no camera estimation',read(adapt/'config.json')['fit_steps']==1200 and not binding['cameras_reestimated'] and not binding['old_individual_geometry_loaded'])
        for name,field,value,reason in [('different-camera','cameras_sha256','a'*64,'binding mismatch'),('old-fit','old_individual_fit_loaded',True,'personalized'),
            ('wrong-clock','frame_clock',[dict(frame_id=i,time_seconds=0) for i in range(4)],'Monotonic')]:
            changed=dict(spec);changed[field]=value;mm=deepcopy(m);mm['fit_inputs']=put(root/f'{name}.json',changed);out=root/name;out.mkdir()
            rejects(name,lambda mm=mm,out=out:build_fit_inputs(mm,root,out),reason)
        mm=deepcopy(m);mm['source']['asset']['path']='relative.bin';out=root/'relative';out.mkdir()
        rejects('Relative reference cannot silently change physical binding',lambda:build_fit_inputs(mm,root,out),'absolute asset')
        check('Adapter has not imported torch','torch' not in sys.modules)
        scene,K,fit,views,images=known_scene()
        queries=np.array([[0,-.4,0],[0,.3,0],[.55,0,0],[100,100,100]],float)
        support=evaluate_support(queries,views,K,scene,tolerance_px=2,min_views=3,vote_fraction=.97)
        check('Occluded lower-region point survives known foreground support',support['inside'][0] and support['observed'][0]==6 and support['positive'][0]==6,
            positive=int(support['positive'][0]),observed=int(support['observed'][0]),views=8)
        check('Same positive votes with old all-view denominator wrongly reject',support['legacy_all_view_fraction'][0]<.97,old_fraction=float(support['legacy_all_view_fraction'][0]),new_fraction=float(support['values'][0]))
        check('Actual observed background rejects outside point',not support['inside'][2] and support['observed'][2]>=3 and support['positive'][2]<support['observed'][2])
        check('Outside camera support is unknown not negative',not support['known'][3] and not support['inside'][3])
        angle=.31;G=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]);t=np.array([.2,.1,-.3]);Hinv=np.eye(4);Hinv[:3,:3]=G.T;Hinv[:3,3]=-G.T@t
        moved=[dict(v,w2c=(np.asarray(v['w2c'])@Hinv).tolist()) for v in views]
        second=evaluate_support(queries@G.T+t,moved,K,scene,tolerance_px=2,min_views=3,vote_fraction=.97)
        check('Rigid worldgauge changes preserve votes',all(np.array_equal(support[k],second[k]) for k in ['observed','positive','inside','known']))
        invalid=deepcopy(fit);invalid['vertices_world'][1,0,0]+=.01
        rejects('Dynamic per-frame fit cannot initialize static model',lambda:fit_proposal(invalid,spacing_m=.012,padding_m=.4,max_voxels=5000000),'Static one-pose')
        invalid=deepcopy(fit);invalid['world_scale']=np.array(-1.)
        rejects('Invalid scale cannot initialize',lambda:fit_proposal(invalid,spacing_m=.012,padding_m=.4,max_voxels=5000000),'positive scale')
        result=initialize_surface(fit,views,K,scene,O/'surface-positive',SURFACE_CONFIG)
        with np.load(O/'surface-positive/initial.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        check('Exactly200k complete free world Gaussians initialized',arrays['means'].shape==(200000,3) and arrays['sh'].shape==(200000,16,3) and (arrays['scales']>0).all() and np.array_equal(arrays['quats'][:,0],np.ones(200000)))
        check('Centers spread beyond bare proposal onto clothing envelope',result['centers_beyond_bare_proposal_box']>50000 and result['max_distance_outside_bare_box_model_m']>.05,
            outside_bare_box=result['centers_beyond_bare_proposal_box'],max_outside_m=result['max_distance_outside_bare_box_model_m'])
        # Finite proposal clipping must be detected, not hidden by generating points.
        clipped=dict(SURFACE_CONFIG,padding_m=.001)
        rejects('Clipped proposal stops before seeding',lambda:initialize_surface(fit,views,K,scene,O/'surface-clipped',clipped),'proposal boundary')
        unknown=scene/'unknown.png';Image.fromarray(np.zeros((128,144),np.uint8)).save(unknown)
        unknown_views=[dict(v,observation_valid=ref(unknown)) for v in views]
        s=evaluate_support(queries,unknown_views,K,scene,tolerance_px=2,min_views=3,vote_fraction=.97)
        check('All occluded observations yield no occupied or background evidence',not s['known'].any() and not s['inside'].any() and not s['observed'].any())
        full=scene/'full-observed.png';top=scene/'top-observed.png'
        Image.fromarray(np.full((128,144),255,np.uint8)).save(full)
        topmask=np.full((128,144),255,np.uint8);topmask[64:]=0;Image.fromarray(topmask).save(top)
        partial=[dict(v,observation_valid=ref(full if i<2 else top if i==2 else unknown)) for i,v in enumerate(views)]
        rejects('Unknown-support border is not closed into a fake surface',lambda:initialize_surface(fit,partial,K,scene,O/'surface-unknown-border',SURFACE_CONFIG),'unknown support')
        # Real projected center maps, no Gaussian rasterizer or image annotation.
        canvas=Image.new('RGB',(144*4,128*2),(16,20,18))
        means=arrays['means'].astype(float)
        for column,index in enumerate([0,2,4,6]):
            original=np.repeat(images[index][...,None],3,2);canvas.paste(Image.fromarray(original),(column*144,0))
            E=np.asarray(views[index]['w2c']);x=means@E[:3,:3].T+E[:3,3];q=x@K.T;uv=np.floor(q[:,:2]/q[:,2:]+.5).astype(int)
            good=(x[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<144)&(uv[:,1]>=0)&(uv[:,1]<128)
            frame=np.zeros((128,144,3),np.uint8);frame[uv[good,1],uv[good,0]]=[40,190,150];canvas.paste(Image.fromarray(frame),(column*144,128))
        canvas.save(O/'known-mask-centers.png')
        command=[sys.executable,str(H/'prepare_native_person122.py'),'fit','--input',str(PRE/'real-input/input.json'),'--output',str(O/'actual-person-denied')]
        with (O/'actual-person.log').open('x') as log:proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        actual=read(O/'actual-person-denied/preparation-status.json');actual_audit=read(O/'actual-person-denied/admission.json')
        check('Actual freshfit command denies bad person before priorfitGPU',proc.returncode==3 and actual['state']=='input_rejected' and not any(actual[k] for k in ['GPU_imported','prior_started','fit_started','surface_started','training_started']))
        check('Real material failures remain same102targets and13face max',actual_audit['source_observation_records']==102 and actual_audit['heldout_face']==read(PRE/'actual-entry/admission.json')['heldout_face'])
        not_reviewed=O/'review-guard';not_reviewed.mkdir();save(not_reviewed/'preparation-status.json',dict(state='actual_fit_review_pending'))
        rejects('No surface review means no training dispatch',lambda:dispatch_training(not_reviewed),'Only the new surface')
        check('All frozen old and new source inputs unchanged',all(sha(p)==h for p,h in frozen.items()))
        check('No torch or CUDA imported for T4','torch' not in sys.modules)
        save(O/'result.json',dict(stage='T4',state='complete_pending_visual_review',updated=now(),seconds=time.monotonic()-started,checks=checks,
            all_controls_passed=True,source_finding='Old all-frame denominator can erase independently observed clothing when another view is unknown.',
            controlled_example=dict(positive=6,observed=6,total_views=8,old_fraction=.75,new_fraction=1.),surface=result,
            actual_person_command=command,actual_person_status=actual,actual_person_exit=3,physical_admission=False,person_fit_executed=False,
            GPU_used=False,Gaussian_optimizer_steps=0,quality_accepted=False,
            limits=['Actual public-prior+SMPLX fit and positive GPUtraining execution are not tested here.','Known synthetic box is not a person or subpixel geometry proof.','Occlusion normalization diagnosis does not establish cause of historical face ghosting.','Rawvideo calibration,physical materials andcompleteframe admission stillmissing.']))
        save(O/'progress.json',dict(state='complete_pending_visual_review',pid=None,updated=now()))
        print(json.dumps(dict(checks=len(checks),all_controls_passed=True,points=result['points'],old_fraction=.75,new_fraction=1.,person_fit=False,GPU=False)))
    except BaseException as e:
        save(O/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),checks=checks,updated=now()));save(O/'progress.json',dict(state='failed',pid=None,updated=now()));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);a=p.parse_args();globals()[a.stage]()
