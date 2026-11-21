"""Once-only R5 CPU controls, independent OpenCV truth and real refusal."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
from dataclasses import replace
import json,sys,time,signal,traceback,subprocess,hashlib
import numpy as np
import cv2
from PIL import Image
from native_training_admission122_v2 import audit_input,assert_snapshot,native_camera,observation,read,save,sha
from material_geometry122_v2 import evaluate_material_geometry,predict_target
from material_geometry122 import evaluate_material_geometry as old_geometry
from native_residual_camera122 import projection,project_world
from check_native_admission122 import fixture as old_fixture

H=Path(__file__).resolve().parent
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-admission'
def now():return datetime.now(timezone.utc).isoformat()
def ref(p):return dict(path=str(p.resolve()),sha256=sha(p))
def put(p,v):save(p,v);return ref(p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def hostpath(p):
    if p.startswith('C:/') or p.startswith('C:\\'):return Path('/mnt/c/'+p[3:].replace('\\','/'))
    return Path(p)

def rebind(mm,root,tag):
    ph=read(Path(mm['physical_evidence']['path']))
    ph['bindings'].update(cameras_sha256=mm['cameras']['sha256'],observations_sha256=mm['observations']['sha256'],
        frame_plan_sha256=mm['frame_plan']['sha256'],decode_receipt_sha256=mm['decode_receipt']['sha256'],frames_sha256=digest(mm['frames']))
    mm['physical_evidence']=put(root/(tag+'-physical.json'),ph)

def fixture(root):
    m=old_fixture(root)
    truth=read(root/'known-evidence.json');original_obs=read(Path(m['observations']['path']))
    cameras=[];matrices={};points={};cvDs={}
    for f in range(4):
        K=np.array([[145.+f*4,0,160.25+f*.4],[0,148.+f*2,118.75-f*.3],[0,0,1.]])
        E=np.eye(4);E[:3,:3]=cv2.Rodrigues(np.array([.013*f,-.02*f,.006*f]))[0];E[0,3]=-.45+.3*f;E[1,3]=.02*f
        # OpenCV's documented order is independent of the production split fields.
        D=np.array([.16*(-1 if f%2 else 1),.012,.005*(f+1),-.004,.002,.006,-.001,.0002,.003,-.0004,-.002,.0003])
        P=projection([D[0],D[1],D[4],D[5],D[6],D[7]],[D[2],D[3]],D[8:],input_domain='synthetic_known_camera')
        Kn=K.copy();Kn[:2,2]+=.5
        c=dict(id=str(f),image_name=f'rgb{f}.png',width=320,height=240,K=Kn.tolist(),w2c=E.tolist(),residual_projection=P)
        cameras.append(dict(frame_id=f,registration='observed',camera=c));matrices[f]=(K,E);cvDs[f]=D
    for track in range(16):
        points[track]=np.array([1.12*np.sin(track*1.3),.82*np.cos(track*1.1),2.35+.075*track])
    obs=[]
    for o in original_obs:
        K,E=matrices[o['frame_id']];X=points[o['track_id']]
        uv=cv2.projectPoints(X[None],cv2.Rodrigues(E[:3,:3])[0],E[:3,3],K,cvDs[o['frame_id']])[0].reshape(2)
        assert (uv>0).all() and uv[0]<320 and uv[1]<240
        o.update(raw_uv=uv.tolist(),width=320,height=240);obs.append(o)
    m['schema']='shuolee-native-training-input-v2-output-residual';m['projection']='opencv_brown12_output';m.pop('radial_coefficients')
    m['run_id']='R5_known_output_residual';m['source'].update(width=320,height=240)
    m['cameras']=put(root/'cameras-residual.json',dict(schema='shuolee-output-residual-cameras-v1',source_sha256=m['source']['asset']['sha256'],
        measurement_pixel_convention='integer_centers',camera_pixel_convention='pixel_corners',fit_families=[str(i) for i in range(12,16)],cameras=cameras))
    m['observations']=put(root/'observations-residual.json',obs)
    dec=[]
    for f,row in enumerate(m['frames']):
        for k in ['rgb','alpha','rgb_valid','observation_valid']:
            a=np.full((240,320,3),60+f*30,np.uint8) if k=='rgb' else np.full((240,320),255,np.uint8)
            p=root/f'{k}-{f}-native.png';Image.fromarray(a).save(p);row[k]=ref(p)
        dec.append(dict(frame_id=f,rgb_sha256=row['rgb']['sha256'],width=320,height=240))
    m['decode_receipt']=put(root/'decode-residual.json',dict(source_sha256=m['source']['asset']['sha256'],geometric_resampling=False,rotation_applied=False,frames=dec))
    truth.update(K=None,cameras=cameras,points={str(k):v.tolist() for k,v in points.items()},
        oracle='OpenCV projectPoints with independently specified documented12-coefficient layout; no production projector used to create observations.',
        oracle_coefficients={str(k):v.tolist() for k,v in cvDs.items()})
    # Preserve the first fixture's evidence file; updated synthetic truth has separate binding.
    ev=put(root/'residual-known-evidence.json',truth)
    cv=read(Path(m['coverage_plan']['path']))
    for a in cv['observations']+cv['requirements']:a['evidence']=ev
    m['coverage_plan']=put(root/'coverage-residual.json',cv)
    ph=read(Path(m['physical_evidence']['path']))
    for name in ['output_projection','pose_depth_support','static_scene_control','subject_staticity','mask_occluder_support','world_y_up']:ph[name]['evidence']=ev
    m['physical_evidence']=put(root/'physical-base-residual.json',ph)
    rebind(m,root,'base');put(root/'input-residual.json',m)
    return m,points,matrices

def main():
    if (O/'progress.json').exists():raise RuntimeError('R5 already executed; inspect current state instead of repeating')
    plan=read(O/'plan.json');frozen={**plan['historical_hashes'],**plan['input_hashes']}
    assert all(sha(hostpath(p))==v for p,v in frozen.items())
    sources=[H/n for n in ['material_geometry122_v2.py','native_training_admission122_v2.py','check_residual_admission122.py',
        'check_native_admission122.py','native_scene_package122.py','native_pixel_convention122.py','shared_background122.py']]
    source_hashes={str(p):sha(p) for p in sources}
    save(O/'execution-registration.json',dict(updated=now(),plan_sha256=sha(O/'plan.json'),source_hashes=source_hashes,command=sys.argv,CPU_only=True))
    checks=[];started=time.monotonic()
    def check(name,condition,**detail):
        checks.append(dict(name=name,passed=bool(condition),**detail));save(O/'checks-progress.json',checks)
        if not condition:raise AssertionError(name)
    def deadline(*_):raise TimeoutError('R5 CPU budget300s exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
    save(O/'progress.json',dict(state='running',pid=os.getpid(),updated=now()))
    try:
        m,truth,matrices=fixture(O/'synthetic');root=O/'synthetic'
        report,_=audit_input(root/'input-residual.json');save(O/'synthetic-positive.json',report)
        check('Known nonzero-D full input passes CPU admission only',report['training_admitted'] and report['heldout_face']['maximum_px']<1e-8,face=report['heldout_face'])
        rows=report['material_geometry']['observations'];camdoc=read(Path(m['cameras']['path']));cameras={c['frame_id']:c['camera'] for c in camdoc['cameras']}
        xyz=max(np.linalg.norm(np.array(r['prediction']['point_world_diagnostic'])-truth[r['observation']['track_id']]) for r in rows)
        check('Support-only inverse-ray DLT recovers independent knownXYZ',xyz<1e-8,max_world_error=float(xyz))
        check('No torch or CUDA imported for input admission','torch' not in sys.modules)
        for record in camdoc['cameras']:
            c,receipt=native_camera(record,camdoc,320,240,f"rgb{record['frame_id']}.png")
            check('Renderer camera identity '+str(record['frame_id']),c==record['camera'] and receipt['renderer_camera_unchanged'] and receipt['coefficients_modified'] is False)
        def variant(name,patch,expected=None):
            mm=deepcopy(m);patch(mm)
            if name!='physical_unknown':rebind(mm,O,name)
            p=O/f'{name}-input.json';save(p,mm);rr,_=audit_input(p)
            save(O/f'{name}-audit.json',rr)
            if expected:check(name+' rejects intended defect',not rr['training_admitted'] and expected in rr['issue_counts'],issues=list(rr['issue_counts']))
            return rr
        def camerachange(mm,name,patch):
            d=read(Path(mm['cameras']['path']));patch(d);mm['cameras']=put(O/f'{name}-cameras.json',d)
        invalid=[
            ('missingD',lambda d:d['cameras'][0]['camera'].pop('residual_projection')),
            ('raw_fisheye',lambda d:d['cameras'][0]['camera']['residual_projection'].update(input_domain='raw_fisheye')),
            ('inverse_mapping',lambda d:d['cameras'][0]['camera']['residual_projection'].update(mapping='observed_to_ideal')),
            ('wrong_count',lambda d:d['cameras'][0]['camera']['residual_projection'].update(radial=[0]*8)),
            ('wrong_origin',lambda d:d.update(camera_pixel_convention='integer_centers')),
            ('scaled_E',lambda d:d['cameras'][0]['camera']['w2c'][0].__setitem__(0,2)),
            ('folded_D',lambda d:d['cameras'][0]['camera']['residual_projection'].update(radial=[-5,0,0,0,0,0])),
            ('wrong_source',lambda d:d.update(source_sha256='0'*64)),
            ('camera_id',lambda d:d['cameras'][0]['camera'].update(id='2')),
        ]
        for name,patch in invalid:
            variant(name,lambda mm,name=name,patch=patch:camerachange(mm,name,patch),'invalid_camera_contract')
        variant('interpolated',lambda mm:camerachange(mm,'interp',lambda d:d['cameras'][0].update(registration='interpolated')),'unobserved_or_interpolated_camera')
        variant('missing_camera',lambda mm:camerachange(mm,'missing',lambda d:d['cameras'].pop()),'camera_frame_set_mismatch')
        variant('physical_unknown',lambda mm:mm.update(physical_evidence=put(O/'unknown-physical.json',{'accepted':True})),'physical_assessment_incomplete')
        def obschange(mm,name,patch):
            d=read(Path(mm['observations']['path']));patch(d);mm['observations']=put(O/f'{name}-observations.json',d)
        for name,field,value in [('identity','material_identity_evidence','unknown'),('motion','static_evidence','moving'),('uncertainty','measurement_uncertainty_px',None),('development','development_exposure',True)]:
            variant(name,lambda mm,name=name,field=field,value=value:obschange(mm,name,lambda d:d[0].update({field:value})),'independent_material_geometry_failed')
        variant('leaked_family',lambda mm:obschange(mm,'leaked',lambda d:d[0].update(frozen_family_id='12')),'invalid_material_observations')
        def zeroD(d):
            for x in d['cameras']:x['camera']['residual_projection']=projection([0]*6,[0]*2,[0]*4,input_domain='synthetic_known_camera')
        wrong=variant('ignored_D',lambda mm:camerachange(mm,'ignored_D',zeroD),'independent_material_geometry_failed')
        check('NonzeroD is materially consumed; no observation removed',wrong['material_geometry']['targets']==len(rows) and wrong['heldout_face']['maximum_px']>1.,face_max=wrong['heldout_face']['maximum_px'])
        held=[observation(r['observation']) for r in rows];target=held[0];track=[o for o in held if o.track_id==target.track_id]
        pred=predict_target(cameras,target,track,camdoc['fit_families'],m['cameras']['sha256'])
        changed=replace(target,raw_uv=(target.raw_uv[0]+4,target.raw_uv[1]))
        pred2=predict_target(cameras,changed,[changed if o.key==target.key else o for o in track],camdoc['fit_families'],m['cameras']['sha256'])
        check('Changing target does not change target-excluded XYZ orprediction',pred['uv']==pred2['uv'] and pred['point_world_diagnostic']==pred2['point_world_diagnostic'] and list(target.key) not in pred['triangulation_observation_keys'])
        faceidx=next(i for i,x in enumerate(read(Path(m['observations']['path']))) if x['group']=='face_candidate')
        perturbed=variant('face_error',lambda mm:obschange(mm,'face_error',lambda d:d[faceidx].update(raw_uv=[d[faceidx]['raw_uv'][0]+4,d[faceidx]['raw_uv'][1]])),'independent_material_geometry_failed')
        check('Failed face observation remains in full report',perturbed['material_geometry']['targets']==len(rows) and perturbed['heldout_face']['maximum_px']>=1.)
        # Known zero-D, commonK and deterministic noise must reproduce original N13 exactly.
        zero=deepcopy(cameras);K=np.array([[145.,0,160.25],[0,148.,118.75],[0,0,1.]])
        zobs=[];Es={}
        for f,c in zero.items():
            c['K']=K.copy().tolist();c['K'][0][2]+=.5;c['K'][1][2]+=.5
            c['residual_projection']=projection([0]*6,[0]*2,[0]*4,input_domain='synthetic_known_camera');Es[f]=np.asarray(c['w2c'])
        for o in held:
            E=Es[o.frame_id]
            uv=cv2.projectPoints(truth[o.track_id][None],cv2.Rodrigues(E[:3,:3])[0],E[:3,3],K,np.zeros(12))[0].reshape(2)
            uv+=.04*np.array([np.sin(o.frame_id+o.track_id),np.cos(o.frame_id-o.track_id)])
            zobs.append(replace(o,raw_uv=tuple(uv)))
        old=old_geometry(K,Es,zobs,camdoc['fit_families'],'syntheticzero',list(range(4)),m['source']['asset']['sha256'])
        new=evaluate_material_geometry(zero,zobs,camdoc['fit_families'],'syntheticzero',list(range(4)),m['source']['asset']['sha256'])
        diff=max(abs(x['error_px']-y['error_px']) for x,y in zip(old['observations'],new['observations']))
        check('Explicit D0 keeps noisy old pixel-DLT numeric semantics',diff<1e-9,max_error_difference_px=diff)
        Q=cv2.Rodrigues(np.array([.2,-.1,.3]))[0];scale=2.7;shift=np.array([3.,-1.,.8]);gauged=deepcopy(cameras)
        for c in gauged.values():
            E=np.asarray(c['w2c']);rr=E[:3,:3]@Q.T;E[:3,3]=scale*E[:3,3]-rr@shift;E[:3,:3]=rr;c['w2c']=E.tolist()
        gr=evaluate_material_geometry(gauged,held,camdoc['fit_families'],'gauge',list(range(4)),m['source']['asset']['sha256'])
        gd=max(np.linalg.norm(np.array(x['prediction']['uv'])-y['prediction']['uv']) for x,y in zip(rows,gr['observations']))
        check('Full residual target predictions are world-Sim3 invariant',gd<1e-8,max_difference_px=float(gd))
        # Snapshot binding includes D asset bytes, not just the top-level manifest.
        stale=deepcopy(m);d=read(Path(m['cameras']['path']));stale['cameras']=put(O/'stale-cameras.json',d);rebind(stale,O,'stale')
        stale_path=O/'stale-input.json';save(stale_path,stale);sr,_=audit_input(stale_path)
        d['cameras'][0]['camera']['residual_projection']['tangential'][0]+=.001;save(O/'stale-cameras.json',d)
        try:assert_snapshot(sr,stale_path)
        except RuntimeError:rejected=True
        else:rejected=False
        check('Changed D asset cannot reuse an accepted report',sr['training_admitted'] and rejected)
        # Old train entry must refuse this new camera document rather than stripD.
        command=[sys.executable,str(H/'train_native_person122.py'),'train','--input',str(root/'input-residual.json'),'--output',str(O/'old-entry-refusal')]
        proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45)
        (O/'old-entry-refusal.log').write_text(proc.stdout)
        status=read(O/'old-entry-refusal/status.json')
        check('Legacy training command rejects new schema before GPU',proc.returncode==3 and not status['GPU_imported'] and not status['training_started'])
        # One read-only transport of the already failed real T3 diagnostic input.
        original=read(R/'native-training-entry/real-input/input.json');actual=deepcopy(original);cd=read(Path(original['cameras']['path']))
        realcams=[];Ki=np.asarray(cd['K']);Kn=Ki.copy();Kn[:2,2]+=.5
        for c in cd['cameras']:
            image=next(f for f in original['frames'] if f['frame_id']==c['frame_id'])
            camera=dict(id=str(c['frame_id']),image_name=Path(image['rgb']['path']).name,width=actual['source']['width'],height=actual['source']['height'],K=Kn.tolist(),w2c=c['w2c'],
                residual_projection=projection([0]*6,[0]*2,[0]*4,input_domain='processed_output_pixels'))
            realcams.append(dict(frame_id=c['frame_id'],registration=c['registration'],camera=camera))
        actual.update(schema='shuolee-native-training-input-v2-output-residual',projection='opencv_brown12_output',run_id='R5_readonly_zeroD_transport_not_new_candidate')
        actual.pop('radial_coefficients')
        actual['cameras']=put(O/'real-diagnostic-cameras.json',dict(schema='shuolee-output-residual-cameras-v1',source_sha256=cd['source_sha256'],measurement_pixel_convention='integer_centers',camera_pixel_convention='pixel_corners',fit_families=cd['fit_families'],cameras=realcams))
        actual_path=O/'real-diagnostic-input.json';save(actual_path,actual)
        command=[sys.executable,str(H/'native_training_admission122_v2.py'),'--input',str(actual_path),'--output',str(O/'actual-entry')]
        proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        (O/'actual-entry.log').write_text(proc.stdout)
        ar=read(O/'actual-entry/admission.json');st=read(O/'actual-entry/status.json')
        check('Actual person diagnostic refused before any GPU or training',proc.returncode==3 and not st['GPU_imported'] and not st['training_started'] and not st['initializer_loaded'])
        prior=read(R/'native-training-entry/actual-entry/admission.json');oldrows={tuple(r['prediction']['target_key']):r for r in prior['material_geometry']['observations']}
        deltas=[]
        for r in ar['material_geometry']['observations']:
            old=oldrows[tuple(r['prediction']['target_key'])]
            assert r['observation']==old['observation'] and r['reasons']==old['reasons']
            if r['error_px'] is not None:deltas.append(abs(r['error_px']-old['error_px']))
        check('All real102targets and13face failures unchanged',ar['material_geometry']['targets']==102 and ar['heldout_face']['observations']==13 and max(deltas)<1e-9,max_error_delta_px=max(deltas),face=ar['heldout_face'])
        check('464source frames not confused with7diagnostic cameras',ar['encoded_source_frames']==464 and len(ar['registered_camera_frames'])==7 and not ar['declared_training_frames'])
        check('No optimizer checkpoint or Gaussian output created',not (O/'actual-entry/gaussian-run').exists() and not st['optimizer_constructed'])
        check('Original sources and frozen inputs remain byte identical',all(sha(hostpath(p))==v for p,v in frozen.items()) and all(sha(Path(p))==v for p,v in source_hashes.items()))
        check('Complete test process remained CPU only','torch' not in sys.modules)
        result=dict(stage='R5',state='completed_pending_review',updated=now(),seconds=time.monotonic()-started,checks=checks,
            synthetic_face=report['heldout_face'],synthetic_world_error=float(xyz),ignored_D_face_max_px=wrong['heldout_face']['maximum_px'],
            noisy_zeroD_regression_px=diff,world_gauge_difference_px=float(gd),
            actual_entry=dict(exit_code=3,status=st,face=ar['heldout_face'],targets=102,old_numeric_max_difference_px=max(deltas),source_frames=464,diagnostic_cameras=7),
            physical_quality_accepted=False,real_coefficients_fitted=False,person_training=False,GPU_used=False,
            positive_person_training_integrated=False,scope='Versioned full-residual material evaluator and prepared-input CPU audit. Native camera accessor exact. Old training entry correctly refuses new schema; positive fresh-fit/training orchestration still separate.')
        save(O/'result.json',result);save(O/'progress.json',dict(state='completed_pending_review',pid=None,updated=now()))
        print(json.dumps(dict(checks=len(checks),seconds=result['seconds'],synthetic_face=result['synthetic_face'],wrongD_face=result['ignored_D_face_max_px'],actual_face=result['actual_entry']['face'])))
    except BaseException as e:
        save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),checks=checks))
        save(O/'progress.json',dict(state='failed',pid=None,updated=now()));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
