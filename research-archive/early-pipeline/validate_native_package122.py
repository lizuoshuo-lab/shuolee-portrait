"""Execute the pre-registered R1c loader suite once; no people or training."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback

H = Path(__file__).parent
R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
OLD = R / 'covariance-sh-ply'
OUT = R / 'native-package-loader'
REPO = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
sys.path.insert(0, str(REPO))
OUT.mkdir(exist_ok=True)
if (OUT/'progress.json').exists() or (OUT/'result.json').exists():
    raise RuntimeError('R1c already has progress/result; inspect its process, do not repeat')
plan = json.loads((R/'native-package-loader-plan.json').read_text(encoding='utf-8'))
registration = json.loads((R/'native-package-loader-implementation.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, digest in registration['sources'].items():
    if sha(H/name) != digest:
        raise RuntimeError('Registered code differs: '+name)
for name, digest in registration['inputs'].items():
    if sha(R/name) != digest:
        raise RuntimeError('Registered input differs: '+name)
start = time.monotonic()
def save(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    tmp.replace(path)
def progress(stage):
    save(OUT/'progress.json', {'pid':os.getpid(), 'stage':stage,
        'updated':datetime.now(timezone.utc).isoformat(), 'seconds':time.monotonic()-start,
        'training_started':False})
def timeout(signum, frame):
    raise TimeoutError('Registered R1c 300-second budget exhausted')
signal.signal(signal.SIGALRM, timeout); signal.alarm(plan['budget']['wall_seconds'])
result = {'stage':'R1c', 'passed':False, 'checks':[], 'training_started':False,
    'physical_camera_or_face_accepted':False, 'native_interactive_GUI_verified':False,
    'plan_sha256':sha(R/'native-package-loader-plan.json'), 'registration_sha256':sha(R/'native-package-loader-implementation.json')}
def check(name, passed, **detail):
    result['checks'].append(dict(name=name, passed=bool(passed), **detail))
    if not passed:
        raise AssertionError('Failed: '+name)
progress('loading_frozen_fixture_and_building_package')
try:
    import numpy as np
    import torch
    from native_scene_package122 import (load_scene, render_recorded, save_render,
        runtime_record, COORDINATES, RENDERER, SCHEMA)
    from native_portrait_renderer122 import NativeCameraContract, render_native
    old = json.loads((OLD/'native-renderer.json').read_text(encoding='utf-8'))
    for name, digest in old['files'].items():
        check('immutable_R1b_'+name, sha(OLD/name)==digest)
    with np.load(OLD/'original-parameters.npz', allow_pickle=False) as z:
        frozen = {name: z[name].copy() for name in z.files}
    check('fixture_dimensions', frozen['means'].shape==(6,3) and frozen['viewmats'].shape==(3,4,4) and frozen['Ks'].shape==(3,3,3))
    package = OUT/'fixture-package'; package.mkdir(exist_ok=False)
    shutil.copyfile(OLD/'point_cloud.ply', package/'point_cloud.ply')
    prior_cameras = json.loads((OLD/'cameras.json').read_text(encoding='utf-8'))
    cameras = []
    for i,c in enumerate(prior_cameras):
        check('camera_source_identity_'+str(i), np.array_equal(c['w2c'], frozen['viewmats'][i]) and np.array_equal(c['K'], frozen['Ks'][i]))
        cameras.append({'id':str(c['id']), 'image_name':c['img_name'], 'width':c['width'], 'height':c['height'], 'K':c['K'], 'w2c':c['w2c']})
    save(package/'cameras.json', cameras)
    descriptor = lambda p: {'path':p.name, 'sha256':sha(p)}
    m = {'schema':SCHEMA, 'scope':'synthetic-control-only', 'coordinates':COORDINATES,
         'input_domain':'synthetic_known_camera',
         'model':{'asset':descriptor(package/'point_cloud.ply'), 'vertices':6, 'sh_degree':3},
         'cameras':descriptor(package/'cameras.json'), 'backend':runtime_record(), 'renderer':RENDERER,
         'projections':{p['contract']['cameraProjection']:{'radial_coefficients':p['radial_coefficients']} for p in old['projections']}}
    check('frozen_backend_continuity', m['backend']['version']==old['runtime']['gsplat'] and m['backend']['torch']==old['runtime']['torch']
          and m['backend']['adapter_sha256']==old['runtime']['adapter_sha256'] and m['backend']['rendering_sha256']==old['runtime']['backend_rendering_sha256'])
    save(package/'scene.json', m)
    scene = load_scene(package/'scene.json')
    errors = {name:float(np.abs(scene.parameters[name]-frozen[name]).max()) for name in ['means','quats','scales','opacities','sh']}
    check('loaded_camera_and_attribute_identity', all(np.array_equal(scene.camera(str(i))['w2c'],frozen['viewmats'][i]) and np.array_equal(scene.camera(str(i))['K'],frozen['Ks'][i]) for i in range(3)),
          attribute_max_abs=errors, camera_count=len(scene.cameras), original_PLY_bytes_preserved=sha(package/'point_cloud.ply')==sha(OLD/'point_cloud.ply'))
    progress('CPU_contract_rejection_controls')
    rejected = []
    def negative(name, mutate, expected, camera_mutate=None, action=None):
        with tempfile.TemporaryDirectory(prefix='shuolee-r1c-') as directory:
            p = Path(directory)
            shutil.copyfile(package/'point_cloud.ply',p/'point_cloud.ply')
            cc = copy.deepcopy(cameras)
            if camera_mutate is not None: camera_mutate(cc)
            save(p/'cameras.json',cc)
            mm = copy.deepcopy(m); mm['cameras']['sha256']=sha(p/'cameras.json')
            mutate(mm); save(p/'scene.json',mm)
            try:
                loaded = load_scene(p/'scene.json')
                if action is not None: action(loaded)
            except ValueError as e:
                if expected not in str(e): raise
                rejected.append({'case':name,'reason':str(e)}); return
            raise AssertionError('Invalid package was accepted: '+name)
    noop = lambda value: None
    negative('model_SHA_mismatch',lambda a:a['model']['asset'].update(sha256='0'*64),'SHA256 mismatch')
    negative('camera_SHA_mismatch',lambda a:a['cameras'].update(sha256='0'*64),'SHA256 mismatch')
    negative('path_escape',lambda a:a['model']['asset'].update(path='../point_cloud.ply'),'path escapes')
    negative('absolute_path',lambda a:a['model']['asset'].update(path='/tmp/point_cloud.ply'),'path escapes')
    negative('missing_recorded_frame',noop,'camera ID not found',action=lambda s:s.camera('missing'))
    negative('duplicate_frame',noop,'Duplicate camera/frame',camera_mutate=lambda c:c[1].update(id=c[0]['id']))
    negative('ambiguous_pose',noop,'camera: missing, extra or ambiguous',camera_mutate=lambda c:c[0].update(rotation=np.eye(3).tolist()))
    negative('nonrigid_pose',noop,'rigid OpenCV',camera_mutate=lambda c:c[0]['w2c'][0].__setitem__(0,2))
    negative('missing_K',noop,'camera: missing, extra or ambiguous',camera_mutate=lambda c:c[0].pop('K'))
    negative('negative_focal',noop,'Unsupported K',camera_mutate=lambda c:c[0]['K'][0].__setitem__(0,-1))
    negative('missing_pixel_convention',lambda a:a['coordinates'].pop('pixels'),'coordinate system')
    negative('wrong_world_basis',lambda a:a['coordinates'].update(world_up='+Z'),'coordinate system')
    negative('backend_version',lambda a:a['backend'].update(version='0.0'),'backend version')
    negative('backend_source',lambda a:a['backend'].update(adapter_sha256='0'*64),'Backend/source/version mismatch')
    negative('screen_space_evaluation',lambda a:a['renderer'].update(with_eval3d=False),'renderer settings')
    negative('changed_culling',lambda a:a['renderer'].update(radius_clip=1),'renderer settings')
    negative('wrong_SH_degree',lambda a:a['model'].update(sh_degree=2),'PLY vertex/SH count')
    negative('rectified_plus_fisheye',lambda a:a.update(input_domain='rectified'),'second fisheye projection')
    negative('pinhole_distortion',lambda a:a['projections']['pinhole'].update(radial_coefficients=[0,0,0,0]),'accepts no distortion')
    negative('KB8_not_KB4',lambda a:a['projections']['opencv_fisheye4'].update(radial_coefficients=[0]*8),'KB4 coefficients')
    negative('unclear_projection',noop,'Projection not recorded',action=lambda s:s.projection('equidistant'))
    # A rectified person package is admissible structurally ONLY in pinhole mode;
    # this does not admit a real camera or start human training.
    with tempfile.TemporaryDirectory(prefix='shuolee-r1c-positive-') as directory:
        p=Path(directory)
        for name in ['point_cloud.ply','cameras.json']:shutil.copyfile(package/name,p/name)
        mm=copy.deepcopy(m);mm['scope']='person-reconstruction';mm['input_domain']='rectified'
        mm['projections']={'pinhole':mm['projections']['pinhole']};save(p/'scene.json',mm)
        positive=load_scene(p/'scene.json')
        check('rectified_pinhole_contract',len(positive.manifest['projections'])==1)
    check('invalid_contracts_rejected_before_GPU',len(rejected)==21,cases=rejected)
    progress('native_recorded_camera_render')
    t=lambda v:torch.as_tensor(v,dtype=torch.float32,device='cuda')
    result['runtime']={'backend':m['backend'],'GPU':torch.cuda.get_device_name()}
    result['renders']=[]
    for projection in m['projections']:
        for i in range(3):
            rgb,alpha,receipt=render_recorded(scene,str(i),projection)
            radial=m['projections'][projection]['radial_coefficients']
            with torch.no_grad():
                ref,refa,meta=render_native(t(frozen['means']),t(frozen['quats']),t(frozen['scales']),
                    t(frozen['opacities']),t(frozen['sh']),t(frozen['viewmats'][i:i+1]),t(frozen['Ks'][i:i+1]),
                    cameras[i]['width'],cameras[i]['height'],contract=NativeCameraContract(projection,'synthetic_known_camera'),
                    radial_coeffs=t([radial]) if radial is not None else None,sh_degree=3)
            torch.cuda.synchronize()
            reference=ref[0].cpu().numpy();reference_alpha=refa[0].cpu().numpy()
            er=float(np.abs(rgb-reference).max());ea=float(np.abs(alpha-reference_alpha).max())
            visible=receipt['all_gaussians_visible'] and bool((meta['radii']>0).all())
            check(f'render_{projection}_{i}',er<plan['fixed_limits']['native_rgb_max_abs'] and ea<plan['fixed_limits']['native_alpha_max_abs'] and visible,
                  RGB_max_abs=er,alpha_max_abs=ea,all_six_visible=visible)
            d=OUT/f'{projection}_{i}'
            saved=save_render(d,rgb,alpha,receipt)
            np.savez_compressed(d/'frozen-reference.npz',rgb=reference,alpha=reference_alpha)
            result['renders'].append({'projection':projection,'camera':str(i),'RGB_max_abs':er,'alpha_max_abs':ea,
                'output':d.name,'reference_sha256':sha(d/'frozen-reference.npz'),'native_files':saved['files']})
    check('all_registered_inputs_unchanged',all(sha(R/name)==digest for name,digest in registration['inputs'].items()))
    result['passed']=True;result['state']='completed_headless_native_loader'
except Exception as exc:
    result['state']='failed';result['error']=str(exc);result['traceback']=traceback.format_exc()
    print(result['traceback'],flush=True)
finally:
    signal.alarm(0)
    result['seconds']=time.monotonic()-start
    result['updated']=datetime.now(timezone.utc).isoformat()
    save(OUT/'result.json',result);progress(result['state'])
    print(json.dumps({'state':result['state'],'passed':result['passed'],'checks':len(result['checks']),'seconds':result['seconds']},indent=2),flush=True)
if not result['passed']:sys.exit(1)
