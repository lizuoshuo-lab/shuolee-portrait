"""Collect the actual 113 source and observation-only transfer supplements."""
from pathlib import Path
import ast, hashlib, json, shutil

P = Path('D:/00/工作/人体重建')
L = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B = L/'outputs/mini-splatting2-mip-depth-001/shuolee'
R = B/'current/person-static-native-align-fresh-113'
PG = L/'repos/PGSR-shuolee'
AV = L/'repos/GaussianAvatars_SMPLX-shuolee'
DEST = P/'src/route113'
entries=[]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        assert sha(src)==sha(dst),dst
    else: shutil.copy2(src,dst)
    entries.append(dict(path=str(dst.relative_to(P)).replace('\\','/'),source=str(src),sha256=sha(dst)))

for folder in ['arguments','gaussian_renderer','scene','utils','lpipsPyTorch']:
    for f in (PG/folder).rglob('*.py'):
        if '__pycache__' not in f.parts: cp(f,DEST/'pgsr'/f.relative_to(PG))
for name in ['LICENSE.md','camera_contract.py','observed_correspondence.py','shuolee_pixel_center.py']:
    cp(PG/name,DEST/'pgsr'/name)
for name in ['shuolee_static.py','native_person_alignment_113.py','check_native_person_113.py']:
    cp(R/'source-final'/name,DEST/'pgsr'/name)
native=PG/'submodules/diff-plane-rasterization'
for f in native.rglob('*'):
    if not f.is_file() or any(x in f.parts for x in ['.git','build','__pycache__','doc','test','util']):continue
    if f.suffix in ['.py','.cpp','.cu','.cuh','.h','.hpp','.inl','.txt','.md'] or f.name=='CMakeLists.txt':
        cp(f,P/'third_party/pgsr113-rasterizer'/f.relative_to(native))
names=['fit_head_surface_082.py','initialize_ncc_pair_093.py','initialize_static_clothing_cloud.py',
       'prepare_fullscene_clothing_075.py','prepare_native_geometry_113.py','project_human_surface_083.py']
for name in names: cp(AV/name,DEST/'preparation'/name)
cp(L/'repos/GVHMR-shuolee/shuolee/fit_fresh_sequence.py',DEST/'preparation/fit_fresh_sequence.py')
cp(L/'repos/hierarchical-3d-gaussians/preprocess/read_write_model.py',DEST/'preparation/read_write_model.py')

# Preserve exact historical bytes separately; portable changes are applied below.
for name in ['shuolee_static.py','native_person_alignment_113.py','check_native_person_113.py']:
    cp(R/'source-final'/name,DEST/'historical'/name)

def patch(rel,old,new):
    f=DEST/rel;s=f.read_text(encoding='utf-8-sig');assert old in s,(rel,old[:80]);
    f.write_text(s.replace(old,new),encoding='utf-8',newline='\n')

patch('pgsr/shuolee_static.py',"sys.path[1:1]=['/opt/gs-slam-lab/shuolee-pgsr/target',str(PRIVATE/'python-deps'),\n    '/opt/gs-slam-lab/shuolee-ga-raster/target','/opt/gs-slam-lab/mini-splatting2-native-001/deps']",'# Runtime dependencies must come from the active server environment.')
# Remove only comparisons against historical fitted arrays. Fresh calculations,
# model loss, seeds, visibility tests, sampling and numerical gates are retained.
patch('preparation/initialize_ncc_pair_093.py',"    old=c.BASE/'current/human-head-surface-initialization-082/initialization.npz'\n    a=h.load(h.O/'initialization.npz');ref=h.load(old)\n    err=float(np.max(np.abs(a['vertices']-ref['vertices'])))\n    assert err<1e-7 and np.array_equal(a['visibility'],ref['visibility'])\n    c.save(h.O/'initialization-review.json',dict(allow_bounded_fit=True,fresh_calculation=True,comparison_only_to082=True,vertices_max_difference=err,gaussian_training_allowed=False))", "    a=h.load(h.O/'initialization.npz')\n    assert np.isfinite(a['vertices']).all()\n    c.save(h.O/'initialization-review.json',dict(allow_bounded_fit=True,fresh_calculation=True,historical_array_comparison_performed=False,gaussian_training_allowed=False))")
patch('preparation/initialize_ncc_pair_093.py',"    assert np.array_equal(old,load(reference/'clothing-init/hull-before-head.npz')['xyz'])", "    assert len(old)==200000 and np.isfinite(old).all()")
patch('preparation/initialize_ncc_pair_093.py',"    equal083=np.array_equal(xyz,load(reference/'clothing-init/points.npz')['xyz']);assert equal083", "    equal083=None  # Historical learned arrays are not deployment inputs.")
patch('preparation/initialize_ncc_pair_093.py',"ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',16)","ImageFont.load_default()")
patch('preparation/fit_head_surface_082.py',"repo/'flame_model/assets/flame/landmark_embedding_with_eyes.npy'", "c.PRIVATE/'models/flame/landmark_embedding_with_eyes.npy'")
patch('preparation/fit_head_surface_082.py',"repo/'flame_model/assets/flame/head_template_mesh.obj'", "c.PRIVATE/'models/flame/head_template_mesh.obj'")
patch('preparation/prepare_fullscene_clothing_075.py',"    sys.path.insert(0,'/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/hierarchical-3d-gaussians/preprocess')", "    # read_write_model is vendored beside this module.")

# Save only raw observation fields, never prior optimized XYZ.
data=P/'data/hover175/route113';data.mkdir(exist_ok=True)
for source,dest in [(R/'head-fit/observations.npz','head-observations.npz'),
                    (R/'head-fit/input-manifest.json','head-input-manifest.json'),
                    (B/'current/full-scene-localization-074/cameras.json','sfm/cameras.json'),
                    (B/'current/fullscene-person-constraint-audit-076/observed-bank/pairs.npz','observed-pairs.npz')]:
    out=data/dest;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out)
for name in ['cameras.bin','images.bin','points3D.bin']:
    out=data/'sfm/refined_sparse'/name;out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(B/'current/full-scene-localization-074/refined_sparse'/name,out)
# NumPy-only extraction is performed by the existing Linux environment.
extract={'source':str(B/'current/shared-point-bundle-convergence-101/input.npz'),
         'output':str(data/'shared-observations.npz'),
         'keys':['observations','offsets','held','original_track_ids','K','w2c','face_ids','face_observations','face_point_indices','face_held']}
(P/'provenance/route113-extract-plan.json').write_text(json.dumps(extract,indent=2),encoding='utf-8')
for entry in entries:
    target=P/entry['path'];entry['portable_sha256']=sha(target);entry['modified_for_portability']=entry['sha256']!=entry['portable_sha256']
doc=dict(route='113',upstream='https://github.com/zju3dv/PGSR',upstream_commit='de24f1a38b350387e8d8fe381b2cd70c1ae946e7',
         historical_source_sha_verified=True,quality_accepted=False,remote_server_verified=False,
         portability_changes=['Remove local environment sys.path injection','Bind runtime paths via the public launcher',
             'Remove 082/083 historical-array comparison dependencies; recompute fresh geometry',
             'Use portable preview font; training tensors unchanged'],files=entries)
(P/'provenance/route113-sources.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for f in DEST.rglob('*.py'): ast.parse(f.read_text(encoding='utf-8-sig'))
print(json.dumps({'source_files':len(entries),'bytes':sum((P/e['path']).stat().st_size for e in entries),'route':str(DEST)}))
