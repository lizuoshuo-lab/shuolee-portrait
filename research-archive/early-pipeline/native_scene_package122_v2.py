"""Explicit v2 packages; legacy v1 is rejected rather than reinterpreted."""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,re
import numpy as np
from native_gaussian_ply122 import read_standard_ply,write_standard_ply
from native_scene_package122 import (COORDINATES,NativeScene,read_json,keys,positive_int,array,
    assert_constant,checked_asset,validate_camera,save_render,sha)
from native_portrait_renderer122_v2 import NativeCameraContractV2,render_native,REVISION,BOUNDS,UT_SETTINGS

SCHEMA='shuolee-native-scene-v2'
RENDERER=dict(adapter_revision=REVISION,bounds=BOUNDS,projection='unscented-transform',gaussian_evaluation='ray-3d',
    opacity_compensation=False,packed=False,eps2d=.3,tile_size=16,near_plane=.01,far_plane=1e10,radius_clip=0.,
    render_mode='RGB',rasterize_mode='classic',background=[0,0,0],ut_parameters=UT_SETTINGS,
    sh_direction='world point minus camera center',rolling_shutter='global')

def write_json(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def runtime_record():
    import torch,gsplat
    from gsplat.cuda import _wrapper
    import native_portrait_renderer122_v2 as adapter
    import native_scene_package122 as validation
    import native_portrait_renderer122 as camera_contract
    import native_gaussian_ply122 as codec
    root=Path(gsplat.__file__).parent/'cuda'
    files=sorted(p for d in [root/'csrc',root/'include'] for p in d.rglob('*') if p.suffix in ('.cu','.cuh','.cpp','.h'))
    if not files:raise RuntimeError('Backend kernel source unavailable')
    tree=hashlib.sha256()
    for p in files:tree.update((p.relative_to(root).as_posix()+'\0'+sha(p)+'\n').encode())
    return dict(name='gsplat',version=gsplat.__version__,torch=torch.__version__,adapter_revision=REVISION,
        source_sha256=dict(adapter=sha(adapter.__file__),loader=sha(__file__),wrapper=sha(_wrapper.__file__),
            codec=sha(codec.__file__),validation=sha(validation.__file__),camera_contract=sha(camera_contract.__file__),kernel_tree=tree.hexdigest()))

def validate_runtime(record):
    from gsplat.cuda._wrapper import UnscentedTransformParameters
    if record!=runtime_record() or record['version']!='1.5.3' or record['torch']!='2.7.1+cu128':raise ValueError('V2 backend/source mismatch; no substitution')
    assert_constant(asdict(UnscentedTransformParameters()),UT_SETTINGS,'UT parameters')

def load_scene(path,*,check_runtime=True):
    path=Path(path).resolve(strict=True);initial=sha(path);m=read_json(path)
    keys(m,['schema','scope','coordinates','input_domain','model','cameras','backend','renderer','projections'],'scene')
    if m['schema']!=SCHEMA or m['scope'] not in ('synthetic-control-only','person-reconstruction'):raise ValueError('Explicit v2 schema/scope required; old packages retain old renderer')
    assert_constant(m['coordinates'],COORDINATES,'coordinates');assert_constant(m['renderer'],RENDERER,'renderer')
    keys(m['backend'],['name','version','torch','adapter_revision','source_sha256'],'backend')
    for name,val in [('name','gsplat'),('version','1.5.3'),('torch','2.7.1+cu128'),('adapter_revision',REVISION)]:assert_constant(m['backend'][name],val,'backend '+name)
    keys(m['backend']['source_sha256'],['adapter','loader','wrapper','codec','validation','camera_contract','kernel_tree'],'sources')
    if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in m['backend']['source_sha256'].values()):raise ValueError('Exact source hashes required')
    if not isinstance(m['projections'],dict) or not m['projections']:raise ValueError('Recorded projection required')
    for name,pr in m['projections'].items():
        keys(pr,['radial_coefficients'],'projection');NativeCameraContractV2(name,m['input_domain'],m['coordinates']['pixels']).validate()
        if name=='pinhole' and pr['radial_coefficients'] is not None:raise ValueError('Pinhole cannot reinterpret fisheye')
        if name=='opencv_fisheye4':array(pr['radial_coefficients'],(4,),'KB4')
    if m['scope']=='person-reconstruction' and (m['input_domain']=='synthetic_known_camera' or len(m['projections'])!=1):raise ValueError('One captured input projection required')
    keys(m['model'],['asset','vertices','sh_degree'],'model');positive_int(m['model']['vertices'],'vertices')
    if type(m['model']['sh_degree'])is not int or m['model']['sh_degree'] not in range(4):raise ValueError('Complete SH0..3 required')
    modelpath=checked_asset(path.parent,m['model']['asset']);camerapath=checked_asset(path.parent,m['cameras'])
    records=read_json(camerapath)
    if not isinstance(records,list) or not records:raise ValueError('Nonempty camera list required')
    cameras={};names=set()
    for c in records:
        validate_camera(c)
        if c['id'] in cameras or c['image_name'] in names:raise ValueError('Duplicate frame identity')
        cameras[c['id']]=c;names.add(c['image_name'])
    p=read_standard_ply(modelpath)
    if len(p['means'])!=m['model']['vertices'] or p['sh_degree']!=m['model']['sh_degree']:raise ValueError('PLY disagrees with manifest')
    checked_asset(path.parent,m['model']['asset']);checked_asset(path.parent,m['cameras'])
    if sha(path)!=initial:raise ValueError('Manifest changed while loading')
    if check_runtime:validate_runtime(m['backend'])
    return NativeScene(path,m,cameras,p,initial)

def write_scene(path,parameters,cameras,*,contract,radial=None,scope):
    if not isinstance(contract,NativeCameraContractV2):raise ValueError('V2 contract required')
    contract.validate()
    if scope not in ('synthetic-control-only','person-reconstruction'):raise ValueError('Explicit scope required')
    for c in cameras:validate_camera(c)
    path=Path(path);path.mkdir(exist_ok=False)
    backend=runtime_record();validate_runtime(backend)
    write_standard_ply(path/'point_cloud.ply',**parameters);write_json(path/'cameras.json',cameras)
    degree=int(np.sqrt(parameters['sh'].shape[1])-1)
    manifest=dict(schema=SCHEMA,scope=scope,coordinates=COORDINATES,input_domain=contract.input_domain,
        model=dict(asset=dict(path='point_cloud.ply',sha256=sha(path/'point_cloud.ply')),vertices=len(parameters['means']),sh_degree=degree),
        cameras=dict(path='cameras.json',sha256=sha(path/'cameras.json')),backend=backend,renderer=RENDERER,
        projections={contract.projection:dict(radial_coefficients=radial)})
    write_json(path/'scene.json',manifest);load_scene(path/'scene.json')
    return path/'scene.json'

def render_recorded(scene,camera_id,projection_name):
    if scene.manifest['schema']!=SCHEMA:raise ValueError('V2 scene required')
    camera=scene.camera(camera_id);pr=scene.projection(projection_name);validate_runtime(scene.manifest['backend'])
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no screen-space fallback')
    t=lambda v:torch.as_tensor(np.asarray(v),dtype=torch.float32,device='cuda')
    p=scene.parameters
    with torch.no_grad():
        rgb,alpha,meta=render_native(t(p['means']),t(p['quats']),t(p['scales']),t(p['opacities']),t(p['sh']),t([camera['w2c']]),t([camera['K']]),
            camera['width'],camera['height'],contract=NativeCameraContractV2(projection_name,scene.manifest['input_domain']),sh_degree=p['sh_degree'],
            radial_coeffs=None if pr['radial_coefficients'] is None else t([pr['radial_coefficients']]))
    torch.cuda.synchronize()
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():raise RuntimeError('Nonfinite native response')
    return rgb[0].cpu().numpy(),alpha[0].cpu().numpy(),dict(scene_sha256=scene.manifest_sha256,camera_id=camera_id,projection=projection_name,
        adapter_revision=REVISION,visible_gaussians=int((meta['radii']>0).all(-1).sum()),native_interactive_UI=False,physical_calibration_validated=False)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--scene',required=True);a.add_argument('--camera',required=True)
    a.add_argument('--projection',required=True,choices=['pinhole','opencv_fisheye4']);a.add_argument('--output',required=True);args=a.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    scene=load_scene(args.scene);rgb,alpha,receipt=render_recorded(scene,args.camera,args.projection)
    print(json.dumps(save_render(args.output,rgb,alpha,receipt),indent=2))
