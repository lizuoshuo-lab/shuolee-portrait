"""Execute one registered anisotropic/SH/PLY synthetic CUDA suite, never train."""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, time, traceback

R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O = R / 'covariance-sh-ply'
H = Path(__file__).parent
repo = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
sys.path.insert(0, str(repo))
plan = json.loads((O/'plan.json').read_text(encoding='utf-8'))
if (O/'progress.json').exists() or (O/'result.json').exists():
    raise RuntimeError('R1b has already been registered/executed; inspect PID/results, do not repeat')
for name, digest in plan['code_sha256'].items():
    if hashlib.sha256((H/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Registered source changed before execution: '+name)
t0 = time.monotonic()
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def progress(phase):
    save(O/'progress.json', {'pid': os.getpid(), 'phase': phase,
                            'updated': datetime.now(timezone.utc).isoformat(), 'seconds': time.monotonic()-t0})
progress('import')
result = {'passed': False, 'scope': plan['scope'], 'checks': [], 'training_started': False,
          'real_calibration_accepted': False, 'synthetic_plan_sha256': hashlib.sha256((O/'plan.json').read_bytes()).hexdigest()}
def check(name, passed, **metrics):
    result['checks'].append({'name': name, 'passed': bool(passed), **metrics})
try:
    import numpy as np
    import torch, gsplat
    from scipy.spatial.transform import Rotation
    from PIL import Image
    from plyfile import PlyData, PlyElement
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci, spherical_harmonics, UnscentedTransformParameters
    from gsplat.utils import save_ply
    from dataclasses import asdict
    from native_portrait_renderer122 import NativeCameraContract, render_native
    from native_gaussian_ply122 import write_standard_ply, read_standard_ply

    result['runtime'] = {'torch': torch.__version__, 'gsplat': gsplat.__version__, 'gpu': torch.cuda.get_device_name(),
                         'adapter_sha256': plan['code_sha256']['native_portrait_renderer122.py'],
                         'backend_rendering_sha256': hashlib.sha256((repo/'gsplat/rendering.py').read_bytes()).hexdigest()}
    T = lambda value: torch.as_tensor(value, dtype=torch.float32, device='cuda')
    rng = np.random.default_rng(plan['seed'])
    n = len(plan['means'])
    p = {key: np.asarray(plan[key], dtype=np.float32) for key in ['means','scales','opacities']}
    p['quats'] = Rotation.from_euler('xyz', plan['euler_xyz_radians']).as_quat()[:, [3,0,1,2]].astype(np.float32)
    p['sh'] = np.zeros((n,16,3), dtype=np.float32)
    p['sh'][:,0] = (np.asarray(plan['base_rgb'])-.5)/.28209479177387814
    p['sh'][:,1:] = rng.standard_normal((n,15,3))*.14
    def look_at(eye):
        eye = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(plan['camera_target'])-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward, plan['world_up']); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        v = np.eye(4); v[:3,:3] = np.stack([right,down,forward]); v[:3,3] = -v[:3,:3]@eye
        return v.astype(np.float32)
    views = np.stack([look_at(eye) for eye in plan['camera_eyes']])
    Ks = np.repeat(np.asarray(plan['K'], dtype=np.float32)[None], len(views), axis=0)
    cameras = []
    for i, (v,k) in enumerate(zip(views,Ks)):
        c2w = np.linalg.inv(v.astype(np.float64))
        cameras.append({'id':i, 'img_name':f'synthetic_{i}', 'width':plan['width'], 'height':plan['height'],
                        'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2]),
                        'K':k.tolist(),'position':c2w[:3,3].tolist(),'rotation':c2w[:3,:3].tolist(),
                        'w2c':v.tolist(),'projectionModel':'pinhole','pixelConvention':'pixel_corners'})
    save(O/'cameras.json', cameras)
    np.savez(O/'original-parameters.npz', **p, viewmats=views, Ks=Ks)
    progress('serialization_and_attribute_controls')
    write_info = write_standard_ply(O/'point_cloud.ply', **p)
    back = read_standard_ply(O/'point_cloud.ply')
    limits = plan['fixed_limits']
    parameter_errors = {k:float(np.max(np.abs(p[k]-back[k]))) for k in p}
    check('all_attributes_roundtrip', max(parameter_errors.values()) <= limits['parameter_max_abs'],
          errors=parameter_errors, standard=write_info)

    # Independently implemented upstream writer: attribute equality is the oracle,
    # not byte equality of different header comments or float logarithm rounding.
    splats = {'means':T(p['means']),'scales':torch.log(T(p['scales'])), 'quats':T(p['quats']),
              'opacities':torch.logit(T(p['opacities'])),'sh0':T(p['sh'][:,:1]),'shN':T(p['sh'][:,1:])}
    save_ply(splats, str(O/'upstream-reference.ply'))
    written = PlyData.read(O/'point_cloud.ply')['vertex'].data
    upstream = PlyData.read(O/'upstream-reference.ply')['vertex'].data
    field_errors = {name:float(np.max(np.abs(written[name]-upstream[name]))) for name in written.dtype.names}
    check('upstream_standard_layout', written.dtype.names == upstream.dtype.names and max(field_errors.values()) <= limits['parameter_max_abs'],
          maximum_field_difference=max(field_errors.values()), fields=len(field_errors))

    # scipy's independently implemented quaternion -> matrix conversion also
    # verifies that off-diagonal covariance is preserved, not just three scales.
    def covariance(param):
        rot = Rotation.from_quat(param['quats'][:,[1,2,3,0]].astype(np.float64)).as_matrix()
        return (rot*param['scales'].astype(np.float64)[:,None,:]**2)@rot.transpose(0,2,1)
    cov0, cov1 = covariance(p), covariance(back)
    cuda_cov, _ = quat_scale_to_covar_preci(T(p['quats']),T(p['scales']),compute_preci=False)
    covariance_errors = {'ply_readback':float(np.max(np.abs(cov0-cov1))),
                         'cuda_vs_scipy':float(np.max(np.abs(cuda_cov.cpu().numpy()-cov0)))}
    check('full_anisotropic_covariance', max(covariance_errors.values()) < limits['covariance_max_abs'],
          errors=covariance_errors, off_diagonal_max=float(np.max(np.abs(cov0[:,0,1]))),
          original_covariances=cov0.tolist(), readback_covariances=cov1.tolist())

    # Independent polynomial form of real SH0..3 (Graphdeco/Inria convention).
    # Backend uses a CUDA recurrence; all RGB channels and coefficients are active.
    def reference_sh(directions, coeff):
        d = directions.astype(np.float64); d /= np.linalg.norm(d,axis=-1,keepdims=True)
        x,y,z = d[...,0],d[...,1],d[...,2]; xx,yy,zz=x*x,y*y,z*z
        b = np.stack([np.ones_like(x)*.28209479177387814,
                      -.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x,
                      1.0925484305920792*x*y,-1.0925484305920792*y*z,
                      .31539156525252005*(2*zz-xx-yy),-1.0925484305920792*x*z,
                      .5462742152960396*(xx-yy),
                      -.5900435899266435*y*(3*xx-yy),2.890611442640554*x*y*z,
                      -.4570457994644658*y*(4*zz-xx-yy),.3731763325901154*z*(2*zz-3*xx-3*yy),
                      -.4570457994644658*x*(4*zz-xx-yy),1.445305721320277*z*(xx-yy),
                      -.5900435899266435*x*(xx-3*yy)],axis=-1)
        return np.einsum('cnk,nkd->cnd',b,coeff.astype(np.float64))
    centers = np.linalg.inv(views.astype(np.float64))[:,:3,3]
    dirs = p['means'][None]-centers[:,None]
    reference = reference_sh(dirs,p['sh'])
    cuda_sh = spherical_harmonics(3,T(dirs),T(p['sh'])[None].expand(len(views),-1,-1,-1).contiguous()).cpu().numpy()
    error = float(np.max(np.abs(reference-cuda_sh)))
    check('directional_SH_against_independent_polynomials',error < limits['sh_color_max_abs'],max_abs=error,
          pre_clamp_rgb_min=float((reference+.5).min()),pre_clamp_rgb_max=float((reference+.5).max()))

    # PLY intentionally has no camera/renderer metadata. Bind those sidecars by
    # SHA, without labeling this synthetic control as a newly reconstructed person.
    manifest = {'schema':'shuolee-native-fixture-v1','scope':'synthetic-control-only','worldUp':'+Y',
                'coordinate_transform_applied':False,'runtime':result['runtime'],
                'gaussians':write_info,'sh_degree':3,'camera_convention':'OpenCV world-to-camera, pixels from top-left corners',
                'renderer':{'with_ut':True,'with_eval3d':True,'packed':False,'eps2d':.3,'tile_size':16,
                            'near_plane':.01,'far_plane':1e10,'radius_clip':0,'render_mode':'RGB',
                            'rasterize_mode':'classic','background':[0,0,0],
                            'ut_parameters':asdict(UnscentedTransformParameters()),'sh_direction':'world point minus camera center'},
                'projections':[{'contract':NativeCameraContract(m,'synthetic_known_camera').manifest(),
                                'radial_coefficients':plan['fisheye_coeffs'] if m=='opencv_fisheye4' else None}
                               for m in plan['projections']],
                'files':{name:hashlib.sha256((O/name).read_bytes()).hexdigest() for name in ['point_cloud.ply','cameras.json','original-parameters.npz']},
                'preview_native_equivalent':False,'physical_calibration_validated':False,'gui_verified':False}
    save(O/'native-renderer.json', manifest)

    def render(param,model,gradient=False):
        args = {k:T(param[k]).detach().requires_grad_(gradient) for k in p}
        rgb,alpha,meta = render_native(args['means'],args['quats'],args['scales'],args['opacities'],args['sh'],
                                      T(views),T(Ks),plan['width'],plan['height'],
                                      contract=NativeCameraContract(model,'synthetic_known_camera'),
                                      radial_coeffs=T([plan['fisheye_coeffs']]*len(views)) if model=='opencv_fisheye4' else None,
                                      sh_degree=3)
        return rgb,alpha,meta,args
    def write_image(name, value):
        Image.fromarray((value.detach().cpu().clamp(0,1).numpy()*255).round().astype('uint8')).save(O/name)
    for model in plan['projections']:
        progress('native_roundtrip_'+model)
        original,alpha,meta,variables = render(p,model,gradient=True)
        reloaded,alpha1,meta1,_ = render(back,model)
        torch.cuda.synchronize()
        rgb_error = float((original.detach()-reloaded).abs().max())
        alpha_error = float((alpha.detach()-alpha1).abs().max())
        visible = bool((meta['radii']>0).all() and (meta1['radii']>0).all())
        finite = bool(torch.isfinite(original).all() and torch.isfinite(alpha).all() and torch.isfinite(reloaded).all() and torch.isfinite(alpha1).all())
        check('native_image_roundtrip_'+model, finite and visible and rgb_error<limits['native_rgb_max_abs'] and alpha_error<limits['native_alpha_max_abs'],
              rgb_max_abs=rgb_error,alpha_max_abs=alpha_error,all_18_observations_visible=visible,finite=finite)
        for c in range(len(views)):
            write_image(f'{model}_original_{c}.png',original[c])
            write_image(f'{model}_ply_readback_{c}.png',reloaded[c])
        yy,xx=torch.meshgrid(torch.arange(plan['height'],device='cuda'),torch.arange(plan['width'],device='cuda'),indexing='ij')
        weights=1+xx[None,:,:,None]/plan['width']+.7*yy[None,:,:,None]/plan['height']
        loss=(original*weights).sum()/100
        loss.backward()
        grads={k:{'finite':v.grad is not None and bool(torch.isfinite(v.grad).all()),
                  'max_abs':None if v.grad is None else float(v.grad.abs().max())} for k,v in variables.items()}
        check('geometry_opacity_SH_backward_'+model,all(g['finite'] and g['max_abs']>0 for g in grads.values()),gradients=grads,
              scope='Finite nonzero gradients, not a finite-difference accuracy claim')
        if model=='pinhole':
            negatives={}
            for kind in ['wrong_SH_layout','wrong_quaternion_order']:
                bad={k:v.copy() for k,v in p.items()}
                if kind=='wrong_SH_layout':
                    bad['sh'][:,1:]=p['sh'][:,1:].transpose(0,2,1).reshape(n,15,3)
                else:
                    bad['quats']=p['quats'][:,[1,2,3,0]]
                bad_rgb,_,_,_=render(bad,model)
                negatives[kind]=float((original.detach()-bad_rgb).abs().max())
                write_image(kind+'_0.png',bad_rgb[0])
            check('layout_and_rotation_negative_controls',all(v>limits['negative_rgb_min_abs'] for v in negatives.values()),rgb_max_differences=negatives)

    damaged=written[[k for k in written.dtype.names if k!='f_rest_44']].copy()
    PlyData([PlyElement.describe(damaged,'vertex')],text=False,byte_order='<').write(O/'deliberately_incomplete.ply')
    rejected=False
    try:
        read_standard_ply(O/'deliberately_incomplete.ply')
    except ValueError as e:
        rejected=True;message=str(e)
    check('incomplete_SH_rejected',rejected,error=message if rejected else None)
    result['passed']=all(c['passed'] for c in result['checks'])
except Exception as e:
    result.update(error=type(e).__name__,message=str(e),traceback=traceback.format_exc())
finally:
    result.update(finished=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-t0)
    save(O/'result.json',result)
    progress('completed' if result['passed'] else 'failed')
    print(json.dumps(result,ensure_ascii=False,indent=2))
