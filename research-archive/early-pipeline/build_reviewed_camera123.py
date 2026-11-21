"""Install a separate frozen-observation entry; keep historical solver bytes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,ast

H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'camera-observation-entry'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
plan=read(REC/'camera-observation-entry-plan.json')
old=H/'local_surface123_camera.py';orchestrator=H/'local_surface123.py'
assert sha(old)==plan['existing_camera_source_sha256']
assert not OUT.exists()
OUT.mkdir()
shutil.copy2(old,OUT/'historical-camera.py');shutil.copy2(orchestrator,OUT/'historical-orchestrator.py')
text=old.read_text(encoding='utf-8')
geometry=text[text.index('    seed_pairs=[]'):]
prefix='''"""Fixed-K candidate geometry consuming frozen reviewed native observations.

The legacy geometry body is preserved below. No detector or matcher exists in
this entry. A candidate is not physical calibration or permission to fit SMPL-X.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from local_surface123 import PROJECT,status as state,now
from local_surface123_frozen_observations import solver_tracks,ObservationContractError,sha
import sys
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt

def camera(root,bundle):
    train,held=solver_tracks(bundle)
    # The native input was validated before preparing a new run. Bind it again
    # to the copied decoded roster and image bytes before solver imports/calls.
    root=Path(root)
    bindings={k:h for k,p,h in bundle.bound_files}
    if sha(root/'decoded.json')!=bindings['decoded']:
        raise ObservationContractError('solver_decoded_roster_changed')
    decoded=rt.read(root/'decoded.json')['records'];n=len(decoded)
    if root.name!=bundle.subject_id:
        raise ObservationContractError('solver_subject_changed')
    for row,frame in zip(decoded,bundle.frames):
        if sha(root/'raw'/row['name'])!=frame['image_sha256']:
            raise ObservationContractError('solver_native_image_changed')
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.sparse import lil_matrix
    start=time.monotonic();out=root/'camera';subject=root.name
    K=np.asarray(rt.read(root/'intrinsic-contract.json')['raw_K_integer'])
    xy=bundle.xy
    tracks=[dict(r['nodes']) for r in train]
    def status(stage,**kw):
        if time.monotonic()-start>1800:raise RuntimeError('1800-second camera budget exceeded.')
        state(subject,state=stage,run_root=str(root),training_running=False,cameras_total=n,seconds=time.monotonic()-start,**kw)
    def project(E,X):
        q=X@E[:3,:3].T+E[:3,3];uv=q@K.T
        return uv[:,:2]/uv[:,2,None],q[:,2]
    def summary(e):
        a=np.asarray(e);return dict(count=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(count=0)
    rt.save(out/'frozen-observation-audit.json',bundle.audit())
    rt.save(out/'plan.json',dict(created=now(),source_sha256=sha(Path(__file__)),pixels='integer full native decoded pixels',
        frame_count=n,observation_manifest_sha256=bundle.manifest_sha256,source_video_sha256=bundle.source_video_sha256,
        detector_called=False,matcher_called=False,original_partitions_preserved=True,
        source_segment_ids=[r['id'] for r in train],source_fit_families=sorted({r['family_id'] for r in train}),
        minimum_seed_direct_matches=160,seed_cheirality=80,min_pnp_inliers=50,min_BA_obs_per_camera=50,
        ransac_px=1.5,pnp_px=3,triangulation_px=3,min_parallax_degrees=1,max_BA_points=5000,max_BA_nfev=60,
        time_budget_seconds=1800,K_fixed=True,old_poses_loaded=False,training_started=False))
    cv2.setNumThreads(6);cv2.setRNGSeed(118)
    if len(held)<30:raise RuntimeError('Fewer than30 original whole-track holdouts; no geometry solve.')
'''
new=H/'local_surface123_camera_reviewed.py';assert not new.exists()
new.write_text(prefix+geometry,encoding='utf-8',newline='\n')
ast.parse(new.read_text(encoding='utf-8'))
# The math, loss, threshold, gauge and held-evaluation body are byte-for-byte
# unchanged. This is an interface repair, not a new geometric candidate.
assert new.read_text(encoding='utf-8')[new.read_text(encoding='utf-8').index('    seed_pairs=[]'):]==geometry
source=orchestrator.read_text(encoding='utf-8')
assert 'def prepare(subject):\n    import numpy as np' in source
source=source.replace('def prepare(subject):\n    import numpy as np',
'''def prepare(subject):
    # Fail before creating a run or decoding when the reviewed observation
    # contract is absent. The old implicit SIFT route is no longer a fallback.
    from local_surface123_frozen_observations import load_subject_input,solver_tracks
    frozen=load_subject_input(subject)
    solver_tracks(frozen)
    import numpy as np''',1)
source=source.replace("video=SOURCE/subject['video']\n    if sha(video)","video=frozen.source_video\n    if sha(video)",1)
source=source.replace("camera_estimator='Fresh native fixed-K OpenCV essential/PnP/BA research'", "camera_estimator='Frozen reviewed native fixed-K OpenCV essential/PnP/BA research'",1)
before="""    status(subject['id'],'decoding_native_frames',run_root=str(root),training_running=False)
    from shuolee_portrait.video_sfm import extract
    extract(video,root,cfg)
    decoded=read(root/'decoded.json')['records'];w,h=decoded[0]['wh']"""
after="""    status(subject['id'],'copying_frozen_native_frames',run_root=str(root),training_running=False)
    (root/'raw').mkdir()
    for row,frame in zip(frozen.decoded['records'],frozen.frames):
        target=root/'raw'/row['name']
        shutil.copy2(frame['image'],target)
        if sha(target)!=frame['image_sha256']:raise RuntimeError('Frozen native image copy differs.')
    bindings={k:p for k,p,h in frozen.bound_files}
    shutil.copy2(bindings['decoded'],root/'decoded.json')
    save(root/'frozen-observation-input.json',frozen.audit())
    decoded=read(root/'decoded.json')['records'];w,h=decoded[0]['wh']"""
assert before in source;source=source.replace(before,after,1)
assert '    from local_surface123_camera import camera\n    camera(root)' in source
source=source.replace('    from local_surface123_camera import camera\n    camera(root)',
    '    from local_surface123_camera_reviewed import camera\n    camera(root,frozen)',1)
# Keep the legacy hash check (history), and separately pin the new code actually
# executed. The new manifest is written only after the engineering tests.
needle="    if sha(p['source_camera']['round_source'])!=p['source_camera']['sha256']:raise RuntimeError('Camera source changed.')"
assert needle in source
source=source.replace(needle,needle+"\n    execution=read(REC/'camera-observation-entry/code-manifest.json')\n    for item in execution['files']:\n        if sha(HERE/item['name'])!=item['sha256']:raise RuntimeError('Reviewed entry source changed: '+item['name'])",1)
ast.parse(source)
orchestrator.write_text(source,encoding='utf-8',newline='\n')
receipt=dict(updated=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(REC/'camera-observation-entry-plan.json'),
    old_camera_sha256=sha(old),historical_orchestrator_sha256=sha(OUT/'historical-orchestrator.py'),
    new_orchestrator_sha256=sha(orchestrator),new_camera_sha256=sha(new),
    geometry_body_sha256=hashlib.sha256(geometry.encode()).hexdigest(),geometry_body_identical=True,
    detector_or_matcher_in_new_entry=False,tests_executed=False,camera_solved=False,training_started=False)
(OUT/'implementation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
