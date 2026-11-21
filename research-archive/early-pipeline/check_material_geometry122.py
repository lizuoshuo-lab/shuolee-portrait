"""N13 bounded integration of existing observation evidence with actual LOO geometry."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import argparse, hashlib, json, time, sys, signal, traceback
import numpy as np
from material_observations122 import Observation
from material_geometry122 import evaluate_material_geometry
from shared_background122 import evaluate

H=Path(__file__).resolve().parent
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027'; D=I/'clip03-fresh-geometry'; G=I/'clip03-grid-coverage/array-cache-repair'
F=I/'clip03-roof-review'; A=I/'clip03-head-level-support'
P=I/'material-geometry-consumer-plan.json'; O=I/'material-geometry-consumer'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

def register():
    if P.exists() or O.exists(): raise RuntimeError('N13 already registered/executed; inspect it instead')
    inputs=[D/'result.json',D/'validation.json',F/'reviewed-support.json',G/'visual-index.json',G/'visual-review.json',
        A/'visual-review.json',A/'observation-contract-check.json',I/'clip03-scene-support-plan.json']
    old=Path('/mnt/d/00/工作/人体重建/src/shuolee_portrait')
    source=read(I/'clip03-scene-support-plan.json')['source']
    inputs.append(Path(source['path']))
    code=[Path(__file__),H/'material_geometry122.py',H/'material_observations122.py',H/'shared_background122.py']
    repair='gauge-consistent-repair' in str(O)
    if repair:
        prior=read(O.parent/'parity-diagnosis.json')
        if prior['max_error_difference_restored216']>1e-8: raise RuntimeError('Parity cause not localized')
        inputs+=[O.parent/'failure.json',O.parent/'parity-diagnosis.json']
    save(P,dict(stage='N13-repair' if repair else 'N13',registered=now(),kind='engineering_consumer_validation_no_new_experiment',
       audit_findings=[
        'Existing122 N3 point permission is used by diagnostics/publication only; no completed122 training consumer exists.',
        'Old D:/00/工作/人体重建 shuolee/113 admission always rejects before model allocation; its code and company00 project must remain unchanged.',
        'Existing necessary face-point checks do not by themselves certify every supporting observation. Add a consumer, retain the old schema/module and all original measurements.'],
       change='Use existing Observation dataclasses and N3 point checks in target-excluded fixed-camera geometry. Propagate any explicit family contradiction to every target, validate all support, preserve every numeric result and frozen split. No separate observation schema.',
       actual_case='All15N12g held background families,2held face families andN12i4936 counterexample; expected102targets. Read fixed candidate only for diagnostic projection; no initialization/update/export.',
       controls='Known synthetic offcenterK rigid-track positive component, support contradiction/unknown, cross-track family contamination, fit/holdout leak, duplicate IDs, unknown uncertainty/staticity, development exposure, missing camera, changed source, and malformed pose. Old99numeric predictions restored in original216gauge for parity. New canonical-DLT predictions must agree under world-origin/rotation/scale changes, with changed numeric values explicitly reported.',
       implementation_repair='Retain initial failed consumer evidence. Homogeneous DLT depends on input-world gauge under pixel noise. For each target, use earliest-support camera coordinates and max-support-baseline scale before DLT; transform XYZ back. No target leakage, camera fitting, coordinate changes or nonlinear point optimization.' if repair else None,
       scope='One necessary material/geometry component. No claim of complete camera/body/staticity/training validation, no renderer/GPU/model allocation.',
       frame_scope='Seven diagnostic frames only. A full training-frame manifest is not established for this clip; do not invent one or claim464registered.',
       budget=dict(runs=1,CPU_wall_seconds=120,GPU=0,training=0),
       stop='Any identity/hash/numeric parity failure stops with evidence. No feature extraction, BA rerun, parameter sweep or automatic training.',
       input_hashes={str(p):sha(p) for p in inputs},code_hashes={str(p):sha(p) for p in code},
       historical_entry_hashes={str(old/n):sha(old/n) for n in ['admission.py','cli.py','training.py']},
       K_changed=False,new_camera=False,training=False))
    print('N13_REGISTERED')

def controls():
    tests=[]
    def check(n,b):
        tests.append(dict(name=n,passed=bool(b)))
        if not b: raise AssertionError(n)
    K=np.array([[210.,0,124.25],[0,230.,85.75],[0,0,1.]])
    cams={i:np.eye(4) for i in range(4)}
    for i,E in cams.items(): E[:3,3]=[i*.3, .04*i,0]
    xyz=np.array([.1,.2,4.]); source='a'*64; obs=[]
    for f,E in cams.items():
        q=K@(E[:3,:3]@xyz+E[:3,3]);uv=tuple(q[:2]/q[2])
        obs.append(Observation(source,f,1,'face','heldout_human','face_candidate',uv,320,240,True,
          'visible','confirmed','bounded_static',.05,('synthetic_known_point_and_pixel_measurement',),False))
    def evaluate(items=obs,mat=K,views=cams,fit=('background',),frames=(0,1,2,3),src=source):
        return evaluate_material_geometry(mat,views,items,fit,'synthetic_camera_control',frames,src)
    good=evaluate();check('Known synthetic full track passes component and has <1e-8px numeric error',good['material_geometry_component_accepted'] and good['numerical_max_px']<1e-8)
    check('A component pass never claims camera/whole training acceptance',not good['physical_camera_accepted'] and not good['training_admitted'])
    for name,patch in [('support material contradicted',dict(material_identity_evidence='contradicted')),
            ('support material unknown',dict(material_identity_evidence='unknown')),
            ('support uncertainty unknown',dict(measurement_uncertainty_px=None)),
            ('support staticness unknown',dict(static_evidence='unknown')),
            ('support occluded',dict(visibility_evidence='occluded')),
            ('support exposed in development',dict(development_exposure=True))]:
        changed=[replace(o,**patch) if o.frame_id==3 else o for o in obs];r=evaluate(changed)
        target=next(x for x in r['observations'] if x['observation']['frame_id']==0)
        check('Credible target rejected when '+name,not target['material_geometry_component_accepted'] and 'triangulation_support_not_physically_validated' in target['reasons'])
        check(name+' does not erase observations or change numeric predictions',r['targets']==4 and r['numerical_max_px']==good['numerical_max_px'])
    # One duplicate physical family on another track must contaminate the entire family.
    extra=replace(obs[0],track_id=2,material_identity_evidence='contradicted')
    r=evaluate(obs+[extra]);check('Contradiction on another track in same family reaches all original targets',all('family_has_material_contradiction' in x['reasons'] for x in r['observations'] if x['observation']['track_id']==1))
    r=evaluate(fit=('face',));check('Family used by camera fitter cannot validate heldout geometry',not r['material_geometry_component_accepted'] and all('heldout_family_used_for_camera_fit' in x['reasons'] for x in r['observations']))
    r=evaluate(views={i:E for i,E in cams.items() if i!=3});check('Missing actual registered camera rejects and retains targets',r['targets']==4 and r['missing_camera_frames']==[3] and not r['material_geometry_component_accepted'])
    for name,call in [
        ('duplicate observation ID',lambda:evaluate(obs+[obs[0]])),
        ('cross-split physical family',lambda:evaluate(obs+[replace(obs[0],track_id=2,frozen_split='fit_background')])),
        ('wrong source video',lambda:evaluate(src='b'*64)),
        ('same track different family',lambda:evaluate([replace(o,frozen_family_id='other') if o.frame_id==3 else o for o in obs])),
        ('non-rigid camera',lambda:evaluate(views={i:(E*2 if i==3 else E) for i,E in cams.items()})),
        ('unrepresentable skewK',lambda:evaluate(mat=np.array([[210.,1.,124.25],[0,230.,85.75],[0,0,1.]]))),
        ('duplicate required frame',lambda:evaluate(frames=[0,1,2,3,3]))]:
        try:call()
        except ValueError:rejected=True
        else:rejected=False
        check(name+' raises before measurement admission',rejected)
    empty=evaluate(items=[]);check('Empty collection does not pass',not empty['material_geometry_component_accepted'])
    noisy=[replace(o,raw_uv=(o.raw_uv[0]+.12*(o.frame_id%2),o.raw_uv[1]-.09*o.frame_id)) for o in obs]
    base=evaluate(items=noisy)
    Q=np.array([[np.cos(.4),-np.sin(.4),0],[np.sin(.4),np.cos(.4),0],[0,0,1.]])
    for scale,shift in [(1.,np.array([3.,-2.,5.])),(2.5,np.array([-4.,2.,8.])),(.01,np.array([.1,.2,-.3]))]:
        transformed={}
        for f,E in cams.items():
            M=np.eye(4);M[:3,:3]=E[:3,:3]@Q.T;M[:3,3]=scale*E[:3,3]-M[:3,:3]@shift;transformed[f]=M
        other=evaluate(items=noisy,views=transformed)
        change=max(np.linalg.norm(np.array(a['prediction']['uv'])-b['prediction']['uv']) for a,b in zip(base['observations'],other['observations']))
        check(f'Noisy offcenterK component invariant under similarity scale{scale}',change<1e-8)
    return tests

def run():
    if O.exists(): raise RuntimeError('N13 output already exists; do not repeat')
    plan=read(P)
    def hashes_unchanged():
        return all(sha(Path(p))==h for p,h in {**plan['input_hashes'],**plan['code_hashes'],**plan['historical_entry_hashes']}.items())
    if not hashes_unchanged(): raise RuntimeError('Plan/source hash mismatch')
    O.mkdir();started=time.monotonic()
    def expired(*_): raise TimeoutError('N13 CPU wall budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(120)
    save(O/'progress.json',dict(state='running',pid=os.getpid(),updated=now()))
    try:
        tests=controls();base=read(D/'result.json');visual=read(G/'visual-index.json');review=read(G/'visual-review.json')
        source=read(I/'clip03-scene-support-plan.json')['source'];support=read(F/'reviewed-support.json')['tracks']
        held=[t for t in support if t['split']=='heldout']
        faces=[visual['rows'][r['row']] for r in review['rows'] if r['reviewed_material']=='face_surface_coarse' and r['split']=='heldout']
        if len(held)!=15 or len(faces)!=2: raise RuntimeError('Frozen evaluation selection changed')
        observations=[]
        for group,tracks in [('static_scene',held),('face_candidate',faces)]:
            for t in tracks:
                for a in t['observations']:
                    observations.append(Observation(source['sha256'],a['frame'],t['track'],str(t['family']),
                      'heldout_background' if group=='static_scene' else 'heldout_human',group,tuple(a['uv']),1520,2704,True,
                      evidence_source=('N12g coarse review only, not certified physical identity or uncertainty',),development_exposure=True))
        for a in read(A/'observation-contract-check.json')['observations']:
            record={k:v for k,v in a.items() if k!='schema'};record['raw_uv']=tuple(record['raw_uv']);record['evidence_source']=tuple(record['evidence_source'])
            observations.append(Observation(**record))
        if len(observations)!=102: raise RuntimeError('Expected all99existing targets plus3counterexample targets')
        cameras={c['frame']:np.array(c['w2c']) for c in base['cameras']}
        out=evaluate_material_geometry(np.array(base['K_integer_centers']),cameras,observations,
            [str(t['family']) for t in support if t['split']=='fit'],str(D/'result.json')+'#sha256='+sha(D/'result.json'),
            base['registered_frames'],source['sha256'])
        reference=read(D/'validation.json')['final'];ref={(r['track'],r['target']):r for g in ['background','face_heldout'] for r in reference[g]['observations']}
        # Keep prior numeric evidence exactly reproducible in its declared gauge.
        original216={f:E@np.linalg.inv(cameras[216]) for f,E in cameras.items()}
        legacy=evaluate(np.array(base['K_integer_centers']),original216,held+faces)
        legacy_ref={(r['track'],r['target']):r for r in legacy['observations']}
        delta=[]
        for r in out['observations']:
            o=r['observation'];key=(o['track_id'],o['frame_id'])
            if key in ref:
                old=ref[key]
                if list(o['raw_uv'])!=old['observed']: raise RuntimeError('Native coordinate changed')
                if r['error_px'] is None or not old['valid_projection']: raise RuntimeError('Numeric status changed')
                delta.append(abs(legacy_ref[key]['error_px']-old['error_px']))
        if len(delta)!=99 or max(delta)>1e-8: raise RuntimeError('Recomputed prior projection mismatch')
        gauge=evaluate_material_geometry(np.array(base['K_integer_centers']),original216,observations,
            [str(t['family']) for t in support if t['split']=='fit'],str(D/'result.json')+'#sha256='+sha(D/'result.json'),base['registered_frames'],source['sha256'])
        gauge_delta=max(float(np.linalg.norm(np.array(a['prediction']['uv'])-b['prediction']['uv'])) for a,b in zip(out['observations'],gauge['observations']) if a['prediction']['uv'] is not None and b['prediction']['uv'] is not None)
        if gauge_delta>1e-8: raise RuntimeError('Canonical DLT remains world-gauge dependent')
        counter=[r for r in out['observations'] if r['observation']['frozen_family_id']=='4936']
        if len(counter)!=3 or not all('family_has_material_contradiction' in r['reasons'] for r in counter): raise RuntimeError('Actual counterexample failed whole-family rejection')
        if not hashes_unchanged(): raise RuntimeError('Input/code changed during validation')
        save(O/'evaluated-observations.json',out)
        face=[r for r in out['observations'] if r['observation']['group']=='face_candidate']
        result=dict(stage='N13-repair',state='complete_pending_review',updated=now(),elapsed_seconds=time.monotonic()-started,
           checks=tests,all_controls_passed=all(t['passed'] for t in tests),actual_targets=102,
           prior_numeric_targets_recomputed=99,prior_numeric_max_difference_px=max(delta),
           canonical_prediction_world_gauge_max_difference_px=gauge_delta,
           canonical_vs_prior_error_difference_max_px=max(abs(r['error_px']-ref[(r['observation']['track_id'],r['observation']['frame_id'])]['error_px']) for r in out['observations'] if (r['observation']['track_id'],r['observation']['frame_id']) in ref),
           face_heldout=dict(targets=len(face),below1=sum(r['error_px']<1 for r in face),max_px=max(r['error_px'] for r in face),physical_component_accepted=sum(r['material_geometry_component_accepted'] for r in face)),
           counterexample=dict(family=4936,targets=3,all_targets_rejected_by_family_identity=True,numeric_errors=[r['error_px'] for r in counter],previously_used_for_camera_fit=False),
           all_targets_physically_accepted=False,all_inputs_and_old_entries_unchanged=True,whole_training_entry_integrated=False,
           concrete_progress='Actual fixed-camera geometric evaluation now consumes the existing observation identity and every triangulation support record, instead of only testing a target point in isolation.',
           remaining=['Independent physical evidence is still absent for current targets/support.',
            'The122formaltraining consumer is not implemented; old113entry remains unchanged andalwaysrejecting.',
            'Full training-frame manifest/registration and independent face-region coverage, camera/staticity/body/novel-view validation are incomplete.'],
           camera_changed=False,new_features=False,new_matches=False,new_solve=False,training_started=False,GPU_used=False,
           source_hashes={p:sha(Path(p)) for p in plan['code_hashes']})
        if 'torch' in sys.modules: raise RuntimeError('Unexpected GPU framework imported')
        save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=now()))
        print(json.dumps({k:v for k,v in result.items() if k not in ['checks','source_hashes']},ensure_ascii=False))
    except BaseException as e:
        save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        save(O/'progress.json',dict(state='failed',pid=None,updated=now()));raise
    finally: signal.alarm(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['register','run','register-repair','run-repair']);args=parser.parse_args()
    action=args.stage
    if action.endswith('-repair'):
        P=O/'repair-plan.json'; O=O/'gauge-consistent-repair'; action=action.removesuffix('-repair')
    globals()[action]()
