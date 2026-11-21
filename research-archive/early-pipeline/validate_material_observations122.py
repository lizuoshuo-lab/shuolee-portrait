from pathlib import Path,PureWindowsPath
from dataclasses import replace
from datetime import datetime,timezone
import collections,hashlib,json,os,signal,time,traceback
import numpy as np
from material_observations122 import Observation,camera_fit_permission,independent_face_admission,audit_coverage,audit_family_splits,validate_observation

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
N=R/'native-temporal-material';M=R/'material-visibility';O=R/'observation-admission'
def read(p):return json.loads(p.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mapped(s):
    w=PureWindowsPath(s);return Path('/mnt')/w.drive[0].lower()/Path(*w.parts[1:])
def expired(*_):raise TimeoutError('N3 CPU budget exhausted')

def main():
    plan=read(R/'observation-admission-plan.json');assert not O.exists(),'N3 already exists, inspect instead of repeating'
    for p,h in plan['immutable_inputs'].items():assert sha(mapped(p))==h
    O.mkdir();save(O/'progress.json',dict(stage='running',pid=os.getpid(),updated=now()))
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['budget']['CPU_seconds']);t0=time.monotonic()
    tests=[]
    def check(name,condition):
        tests.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    try:
        seeds=read(N/'seeds.json')['seeds'];z=np.load(N/'observations.npz');n1plan=read(R/'native-temporal-material-plan.json')
        source=n1plan['input']['sha256'];negative={(c['frame'],c['track']) for c in read(N/'visual-review.json')['counterexamples']}
        def observation(frame,track):
            s=seeds[track];bad=(frame,track) in negative
            return Observation(source,frame,track,s['family'],s['split'],s['group'],tuple(float(x) for x in z['uv'][frame,track]),
                1520,2704,bool(z['valid'][frame,track]),visibility_evidence='occluded' if bad else 'unknown',
                material_identity_evidence='contradicted' if bad else 'unknown',
                evidence_source=('native-temporal-material/visual-review.json',) if bad else (),development_exposure=True)
        check('all raw557x307 observations kept by reference',z['uv'].shape==(557,307,2) and z['valid'].shape==(557,307))
        check('frozen physical families do not cross fit/holdout',audit_family_splits([observation(278,i) for i in range(307)])['passed'])
        counterexamples=[]
        for frame,track in sorted(negative):
            o=observation(frame,track);fb=float(z['forward_backward_px'][frame,track])
            fit=camera_fit_permission(o);face=independent_face_admission(o,None)
            check(f'real counterexample{frame}/{track} rejected despiteFB={fb:.8f}',o.tracker_numeric_validity and fb<.5 and not fit['allowed'] and not face['accepted'] and 'material_contradicted' in face['reasons'])
            counterexamples.append(dict(observation=o.record(),adjacent_FB_native_px=fb,camera_fit=fit,face_validation=face))
        p266=np.load(M/'frame000266.npz');passes=np.flatnonzero(p266['eligible'][:9]).tolist()
        check('N2 retained7nearby faces never promotes physical identity',len(passes)==7 and all(not independent_face_admission(observation(266,i),None)['accepted'] for i in passes))
        # Synthetic provenance fixtures only, never evidence for this video's accuracy.
        base=Observation('a'*64,10,1,'face_family','heldout_human','face_candidate',(20.,30.),100,100,True,
            'visible','confirmed','bounded_static',.1,('synthetic_known_geometry',),False)
        pred=dict(target_key=list(base.key),uv=[20.3,30.4],camera_fit_families=['background_fit'],
            triangulation_observation_keys=[['a'*64,8,1],['a'*64,12,1]],frozen_camera_evidence='synthetic_camera')
        check('known synthetic independent0.5px observation passes necessary point checks',independent_face_admission(base,pred)['accepted'])
        for name,o in [('unknown material',replace(base,material_identity_evidence='unknown')),
            ('unknown visibility',replace(base,visibility_evidence='unknown')),('occluded',replace(base,visibility_evidence='occluded')),
            ('uncertainty unknown',replace(base,measurement_uncertainty_px=None)),('development exposed',replace(base,development_exposure=True)),
            ('numeric invalid',replace(base,tracker_numeric_validity=False)),('outside image',replace(base,raw_uv=(-1.,30.)))]:
            check(name+' fails independent face admission',not independent_face_admission(o,pred)['accepted'])
        for name,patch in [('target triangulated',{'triangulation_observation_keys':pred['triangulation_observation_keys']+[list(base.key)]}),
            ('family leaked',{'camera_fit_families':['face_family']}),('wrong source target',{'target_key':['b'*64,10,1]}),
            ('wrong material support',{'triangulation_observation_keys':[['a'*64,8,2],['a'*64,12,1]]}),
            ('only one support view',{'triangulation_observation_keys':[['a'*64,8,1]]}),('missing camera evidence',{'frozen_camera_evidence':None}),
            ('exactly1px',{'uv':[21.,30.]}),('over1px',{'uv':[21.01,30.]}),('NaN prediction',{'uv':[float('nan'),30.]})]:
            check(name+' rejected',not independent_face_admission(base,dict(pred,**patch))['accepted'])
        background=replace(base,frozen_split='fit_background',frozen_family_id='groundfit',group='static_scene')
        check('known static synthetic fit observation permitted',camera_fit_permission(background)['allowed'])
        check('ground heldout never fits even when numerically perfect',not camera_fit_permission(replace(background,frozen_split='heldout_background'))['allowed'])
        check('unbounded staticness does not fit',not camera_fit_permission(replace(background,static_evidence='unknown'))['allowed'])
        check('duplicate family across split rejected',not audit_family_splits([background,replace(background,track_id=2,frozen_split='heldout_background')])['passed'])
        try:validate_observation(replace(base,evidence_source=()));bad=False
        except ValueError:bad=True
        check('claims without provenance rejected',bad)
        try:validate_observation(replace(base,native_pixel_center_convention='pixel_corners'));bad=False
        except ValueError:bad=True
        check('silent pixel convention relabel rejected',bad)
        check('empty observation set cannot pass coverage',not audit_coverage([0],['eyes','nose','mouth'],[0],[])['accepted'])
        complete=[dict(frame_id=0,region=r,accepted=True) for r in ['eyes','nose','mouth']]
        check('synthetic complete coverage passes only coverage control',audit_coverage([0],['eyes','nose','mouth'],[0],complete)['accepted'])
        check('missing camera rejected',not audit_coverage([0],['eyes','nose','mouth'],[],complete)['accepted'])
        check('missing mouth rejected',not audit_coverage([0],['eyes','nose','mouth'],[0],complete[:2])['accepted'])
        check('one failed heldout cannot hide behind passing aggregate',not audit_coverage([0],['eyes','nose','mouth'],[0],complete+[dict(frame_id=0,region='eyes',accepted=False)])['accepted'])
        manifest=dict(schema='shuolee-observation-set-v1',updated=now(),source_sha256=source,width=1520,height=2704,frames=557,
            coordinates='integer_centers',array_source=dict(path=str(N/'observations.npz'),sha256=sha(N/'observations.npz')),
            frozen_tracks=dict(path=str(N/'seeds.json'),sha256=sha(N/'seeds.json')),
            raw_observation_slots=557*307,numerically_valid_observations=int(z['valid'].sum()),
            fields_reference={'raw_uv':'observations.npz/uv','tracker_numeric_validity':'observations.npz/valid','frame_id':'array first index','track_id':'array second index',
                'frozen_family_id':'seeds.json/seeds[id].family','frozen_split':'seeds.json/seeds[id].split'},
            default_physical_evidence=dict(visibility='unknown',material_identity='unknown',measurement_uncertainty_px=None,staticness='unknown'),
            explicit_contradictions=[dict(frame=f,track=i,evidence='counterexamples.json') for f,i in sorted(negative)],
            development_exposure=True,development_scope='Used to diagnose failure modes; retained camera holdout is not a new blind generalization benchmark.',
            camera_fit_permission_count=0,independent_face_admission_count=0,raw_points_removed=0,raw_coordinates_changed=0,
            registered_training_frames=n1plan['preregistered_future_training_frames'],actual_registered_cameras=[],
            missing_camera_count=len(n1plan['preregistered_future_training_frames']),training_admitted=False)
        save(O/'observation-set.json',manifest);save(O/'counterexamples.json',counterexamples)
        check('all source files remain byte-identical',all(sha(mapped(p))==h for p,h in plan['immutable_inputs'].items()))
        result=dict(stage='N3',state='completed_engineering_controls_not_physical_admission',updated=now(),checks=tests,passed=all(t['passed'] for t in tests),
            source_sha256=sha(Path(__file__)),module_sha256=sha(Path(__file__).parent/'material_observations122.py'),elapsed_seconds=time.monotonic()-t0,
            raw_observation_slots=manifest['raw_observation_slots'],numerically_valid_observations=manifest['numerically_valid_observations'],
            physical_identity_contradictions=6,camera_fit_permitted=0,independent_face_admitted=0,raw_points_removed=0,
            no_new_inference=True,no_new_K_or_pose=True,no_training=True)
        save(O/'result.json',result);save(O/'progress.json',dict(stage='completed',pid=None,updated=now()))
        print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    except BaseException as e:
        save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),completed_checks=tests));save(O/'progress.json',dict(stage='failed',pid=None,updated=now()));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
