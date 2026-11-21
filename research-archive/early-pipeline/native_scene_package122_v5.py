"""Exact-array package binding each output-residual camera to its complete lens.

PLY is a covariance/SH preview asset. It does not encode nonlinear camera rays.
Old package schemas and viewers reject this new contract rather than ignore D.
"""
from pathlib import Path
import hashlib
import numpy as np
from native_scene_package122 import (COORDINATES,NativeScene,read_json,keys,checked_asset,
    assert_constant,sha,save_render)
from native_scene_package122_v2 import write_json,RENDERER as OLD_RENDERER
from native_gaussian_ply122 import write_standard_ply
from native_portrait_renderer122_v3 import REVISION,render_cameras
from native_residual_camera122 import ResidualRenderContract,validate_camera
from native_residual_domain122 import sampled_domain_check
import native_linear_archive122 as archive

SCHEMA='shuolee-native-scene-v5-output-residual'
RENDERER=dict(OLD_RENDERER,adapter_revision=REVISION,coefficient_source='each recorded camera residual_projection')

def runtime_record():
    import torch,gsplat
    from gsplat.cuda import _wrapper
    import native_portrait_renderer122_v3 as adapter
    import native_residual_camera122 as camera
    import native_residual_domain122 as domain
    import native_scene_package122 as validation
    import native_gaussian_ply122 as codec
    root=Path(gsplat.__file__).parent/'cuda'
    files=sorted(p for d in [root/'csrc',root/'include'] for p in d.rglob('*') if p.suffix in ('.cu','.cuh','.cpp','.h'))
    if not files: raise RuntimeError('Kernel source unavailable')
    tree=hashlib.sha256()
    for p in files: tree.update((p.relative_to(root).as_posix()+'\0'+sha(p)+'\n').encode())
    return dict(name='gsplat',version=gsplat.__version__,torch=torch.__version__,adapter_revision=REVISION,
        source_sha256=dict(adapter=sha(adapter.__file__),loader=sha(__file__),wrapper=sha(_wrapper.__file__),
            camera=sha(camera.__file__),domain=sha(domain.__file__),validation=sha(validation.__file__),codec=sha(codec.__file__),
            archive=sha(archive.__file__),kernel_tree=tree.hexdigest()))

def validate_runtime(record):
    if record != runtime_record() or record['version']!='1.5.3' or record['torch']!='2.7.1+cu128':
        raise ValueError('Exact output-residual runtime/source mismatch; no backend substitution')

def validate_cameras(cameras,contract):
    if not cameras or not isinstance(cameras,list): raise ValueError('Recorded camera list required')
    contract.validate();ids=set();names=set()
    for c in cameras:
        validate_camera(c)
        if c['id'] in ids or c['image_name'] in names: raise ValueError('Duplicate camera identity')
        if c['residual_projection']['input_domain']!=contract.input_domain: raise ValueError('Mixed output pixel domains')
        sampled_domain_check(c)
        ids.add(c['id']);names.add(c['image_name'])

def write_scene(path,parameters,cameras,*,contract,scope):
    if not isinstance(contract,ResidualRenderContract): raise ValueError('Residual contract required')
    if scope not in ('synthetic-control-only','unadmitted-candidate'): raise ValueError('This new component cannot certify a person')
    if scope=='synthetic-control-only' and contract.input_domain!='synthetic_known_camera': raise ValueError('Synthetic scope requires known synthetic camera')
    validate_cameras(cameras,contract)
    n,degree=archive.validate(parameters);backend=runtime_record();validate_runtime(backend)
    path=Path(path);path.mkdir(exist_ok=False)
    archive.write(path/'native-parameters.npz',parameters)
    write_standard_ply(path/'point_cloud.ply',**parameters)
    write_json(path/'cameras.json',cameras)
    asset=lambda n:dict(path=n,sha256=sha(path/n))
    m=dict(schema=SCHEMA,scope=scope,coordinates=COORDINATES,camera_contract=contract.manifest(),
        model=dict(native_asset=asset('native-parameters.npz'),preview_asset=asset('point_cloud.ply'),
                   vertices=n,sh_degree=degree,parameter_format=archive.FORMAT,
                   preview_equivalence='must-be-measured-separately'),
        cameras=asset('cameras.json'),backend=backend,renderer=RENDERER,
        preview_camera_compatibility='nonlinear projection requires native reader; never discard coefficients',
        physical_quality_accepted=False)
    write_json(path/'scene.json',m);load_scene(path/'scene.json')
    return path/'scene.json'

def load_scene(path,*,check_runtime=True):
    path=Path(path).resolve(strict=True);initial=sha(path);m=read_json(path)
    keys(m,['schema','scope','coordinates','camera_contract','model','cameras','backend','renderer',
            'preview_camera_compatibility','physical_quality_accepted'],'scene-v5')
    if m['schema']!=SCHEMA or m['scope'] not in ('synthetic-control-only','unadmitted-candidate'): raise ValueError('Explicit residual package schema and scope required')
    assert_constant(m['coordinates'],COORDINATES,'coordinates');assert_constant(m['renderer'],RENDERER,'renderer')
    if m['physical_quality_accepted'] is not False: raise ValueError('Package cannot self-admit physical quality')
    assert_constant(m['preview_camera_compatibility'],'nonlinear projection requires native reader; never discard coefficients','preview')
    contract=ResidualRenderContract(m['camera_contract']['input_domain'])
    assert_constant(m['camera_contract'],contract.manifest(),'camera contract')
    if m['scope']=='synthetic-control-only' and contract.input_domain!='synthetic_known_camera': raise ValueError('Synthetic input domain mismatch')
    model=m['model']
    keys(model,['native_asset','preview_asset','vertices','sh_degree','parameter_format','preview_equivalence'],'model')
    assert_constant(model['parameter_format'],archive.FORMAT,'native arrays')
    assert_constant(model['preview_equivalence'],'must-be-measured-separately','preview equivalence')
    native=checked_asset(path.parent,model['native_asset']);checked_asset(path.parent,model['preview_asset'])
    records=read_json(checked_asset(path.parent,m['cameras']));validate_cameras(records,contract)
    p=archive.read(native,vertices=model['vertices'],degree=model['sh_degree']);p['sh_degree']=model['sh_degree']
    if check_runtime: validate_runtime(m['backend'])
    for a in [model['native_asset'],model['preview_asset'],m['cameras']]: checked_asset(path.parent,a)
    if sha(path)!=initial: raise ValueError('Manifest changed while loading')
    return NativeScene(path,m,{c['id']:c for c in records},p,initial)

def render_recorded(scene,camera_id):
    if scene.manifest['schema']!=SCHEMA: raise ValueError('V5 output-residual scene required')
    validate_runtime(scene.manifest['backend'])
    import torch
    p={n:torch.as_tensor(v,dtype=torch.float32,device='cuda') for n,v in scene.parameters.items() if n!='sh_degree'}
    c=scene.camera(camera_id);contract=ResidualRenderContract(scene.manifest['camera_contract']['input_domain'])
    with torch.no_grad(): rgb,alpha,meta=render_cameras(p,[c],contract=contract,sh_degree=scene.parameters['sh_degree'])
    torch.cuda.synchronize()
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all(): raise RuntimeError('Nonfinite native residual response')
    return rgb[0].cpu().numpy(),alpha[0].cpu().numpy(),dict(scene_sha256=scene.manifest_sha256,
        camera_id=camera_id,adapter_revision=REVISION,physical_quality_accepted=False)
