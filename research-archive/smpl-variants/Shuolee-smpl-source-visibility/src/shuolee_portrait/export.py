"""Rigid world-to-viewer export, including every covariance and SH coefficient."""
import html
import hashlib
import json
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile
from . import runtime as rt


def rigid_rotation64(raw):
    """Remove float32 roundoff only; reject reflections, scale and deformation."""
    R=np.asarray(raw,dtype=np.float64)
    if R.shape!=(3,3) or not np.isfinite(R).all():
        raise RuntimeError('Export orientation is not a finite 3x3 matrix.')
    U,s,Vt=np.linalg.svd(R)
    error=float(abs(R@R.T-np.eye(3)).max());det=float(np.linalg.det(R))
    if det<=0 or float(abs(s-1).max())>1e-6 or error>2e-6:
        raise RuntimeError('Export orientation contains non-roundoff scale/shear or reflection.')
    proper=U@Vt
    return proper,dict(input_orthogonality_max=error,input_determinant=det,
                       input_singular_values=s.tolist(),
                       rotation_adjustment_max=float(abs(proper-R).max()),
                       normalized_orthogonality_max=float(abs(proper@proper.T-np.eye(3)).max()),
                       reason='float32 orientation roundoff only; same proper rotation applied to positions, covariance, SH and cameras')


def reference_pose_index(fit, receipt):
    """Match export orientation to the named pose baked into the world PLY."""
    if 'frame_ids' not in fit:
        return 0, None
    ids = np.asarray(fit['frame_ids'])
    reference = np.asarray(fit['reference_frame_id'])
    if (ids.ndim != 1 or ids.dtype.kind not in 'US' or reference.shape != () or
            reference.dtype.kind not in 'US'):
        raise ValueError('Invalid named export reference.')
    ids = ids.astype(str).tolist(); reference = str(reference.item())
    if len(set(ids)) != len(ids) or reference not in ids:
        raise ValueError('Export reference must identify exactly one fitted pose.')
    slot = ids.index(reference)
    if receipt.get('reference_frame_id') != reference or receipt.get('reference_pose_index') != slot:
        raise ValueError('World PLY and body export reference pose differ.')
    rotations, translations = np.asarray(fit['root_rotation']), np.asarray(fit['world_translation'])
    if (rotations.shape != (len(ids), 3, 3) or translations.shape != (len(ids), 3) or
            not np.isfinite(rotations).all() or not np.isfinite(translations).all()):
        raise ValueError('Named export root transform arrays differ from the pose roster.')
    return slot, reference


def rotate_ply(source, target, R, origin, *, sh_degree=3):
    # Explicit buffers release file handles before Windows validates/renames a
    # candidate; a memory-mapped PLY may otherwise keep that file locked.
    data = PlyData.read(source, mmap=False); vertex = data['vertex'].data.copy(); names=vertex.dtype.names
    if sh_degree not in (0, 3):
        raise ValueError('Supported Gaussian PLY SH degrees are 0 and 3.')
    rest_names = {f'f_rest_{i}' for i in range(45)} if sh_degree == 3 else set()
    required = ['x','y','z','opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]+sorted(rest_names)
    if not len(vertex) or any(x not in names for x in required) or {x for x in names if x.startswith('f_rest_')} != rest_names:
        raise RuntimeError(f'Incomplete or unexpected fields in standard degree-{sh_degree} Gaussian PLY.')
    if not all(np.isfinite(vertex[x]).all() for x in required): raise RuntimeError('PLY contains nonfinite parameters.')
    if not np.allclose(R@R.T,np.eye(3),atol=1e-8) or np.linalg.det(R)<.999999:
        raise RuntimeError('Export requires a rigid proper rotation.')
    xyz=np.column_stack([vertex[x] for x in ['x','y','z']]).astype(float); mapped=(xyz-origin)@R.T
    q=np.column_stack([vertex[f'rot_{i}'] for i in range(4)]).astype(float)
    if (np.linalg.norm(q,axis=1)<1e-8).any():raise RuntimeError('Zero rotation quaternion.')
    old_rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();new_q=Rotation.from_matrix(R[None]@old_rot).as_quat()[:,[3,0,1,2]]
    for i,key in enumerate(['x','y','z']):vertex[key]=mapped[:,i]
    for i in range(4):vertex[f'rot_{i}']=new_q[:,i]
    basis_error = 0.
    if sh_degree == 3:
        from utils.sh_utils import eval_sh
        rng=np.random.default_rng(23033);dirs=rng.normal(size=(256,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
        basis=lambda d:eval_sh(3,np.eye(16),d)
        transform=np.linalg.lstsq(basis(dirs),basis(dirs@R),rcond=None)[0]
        test=rng.normal(size=(1024,3));test/=np.linalg.norm(test,axis=1)[:,None]
        basis_error=float(abs(basis(test)@transform-basis(test@R)).max())
        if basis_error>1e-10:raise RuntimeError('SH rotation failed independent direction check.')
        dc=np.column_stack([vertex[f'f_dc_{i}'] for i in range(3)])[:,None,:]
        rest=np.column_stack([vertex[f'f_rest_{i}'] for i in range(45)]).reshape(-1,3,15).transpose(0,2,1)
        features=np.einsum('ij,njc->nic',transform,np.concatenate([dc,rest],axis=1))
        for i in range(3):vertex[f'f_dc_{i}']=features[:,0,i]
        flat=features[:,1:,:].transpose(0,2,1).reshape(-1,45)
        for i in range(45):vertex[f'f_rest_{i}']=flat[:,i]
    # Normals are unused by standard 3DGS, but preserve their frame when present.
    if all(k in names for k in ['nx','ny','nz']):
        normals=np.column_stack([vertex[k] for k in ['nx','ny','nz']])@R.T
        for i,k in enumerate(['nx','ny','nz']):vertex[k]=normals[:,i]
    PlyData([PlyElement.describe(vertex,'vertex')],text=False,byte_order='<').write(target)
    loaded=PlyData.read(target, mmap=False)['vertex'].data
    if sh_degree == 0 and any(not np.array_equal(loaded[f'f_dc_{i}'], vertex[f'f_dc_{i}']) for i in range(3)):
        raise RuntimeError('Degree-zero color changed during coordinate export.')
    inv=np.column_stack([loaded[x] for x in ['x','y','z']])@R+origin
    position_error=float(abs(inv-xyz).max());scale=np.exp(np.column_stack([loaded[f'scale_{i}'] for i in range(3)]).astype(float))
    m=Rotation.from_quat(np.column_stack([loaded[f'rot_{i}'] for i in [1,2,3,0]])).as_matrix()
    covariance=lambda rot:np.einsum('nij,nj,nkj->nik',rot,scale**2,rot)
    expected=covariance(old_rot);actual=R.T@covariance(m)@R
    covariance_error=float(abs(actual-expected).max());extent=max(float(np.ptp(xyz,axis=0).max()),1.)
    if position_error>2e-6*extent or covariance_error>1e-5*max(float(abs(expected).max()),1e-8):
        raise RuntimeError('Export inverse geometry check failed.')
    return dict(points=len(vertex),position_inverse_max=position_error,covariance_inverse_max=covariance_error,
                sh_basis_rotation_max=basis_error,sh_degree=sh_degree,sha256=rt.sha(target),coordinate_space='world +Y; body-prior root orientation, not measured gravity',
                rotation=R.tolist(),origin_world=origin.tolist(),scale=1.,quality_accepted=False)


def _reference_mesh(out, snapshot, fit, source_row, receipt, report):
    path = snapshot/'reference-mesh.npz'
    if rt.sha(path) != receipt.get('mesh_sha256'):
        raise ValueError('Reference mesh does not match the native PLY receipt.')
    with np.load(path, allow_pickle=False) as mesh:
        if (mesh['reference_frame_id'].item() != receipt['reference_frame_id'] or
                mesh['reference_pose_index'].item() != receipt['reference_pose_index'] or
                not np.array_equal(mesh['faces'], fit['faces'])):
            raise ValueError('Reference mesh topology or frame differs from the fitted sequence.')
        world = mesh['vertices_world'].astype(np.float64)
        faces, offset = mesh['faces'], mesh['static_offset']
    R, origin = np.asarray(report['rotation']), np.asarray(report['origin_world'])
    vertices = (world-origin) @ R.T
    E, K = np.asarray(source_row['w2c']), np.asarray(source_row['K'])
    [camera] = rt.read(out/'cameras.json')
    if (not np.isfinite(vertices).all() or not np.isfinite(offset).all() or offset.shape != world.shape or
            world.shape != fit['vertices_world'].shape[1:] or
            [camera['width'], camera['height']] != source_row['wh'] or
            [camera['fx'], camera['fy'], camera['cx'], camera['cy']] != [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]):
        raise ValueError('Invalid reference geometry or serialized native intrinsics.')
    before = world @ E[:3, :3].T + E[:3, 3]
    after = (vertices-np.asarray(camera['position'])) @ np.asarray(camera['rotation'])
    camera_error = float(np.abs(before-after).max())
    if (before[:, 2] <= 0).any() or (after[:, 2] <= 0).any() or camera_error >= 2e-6:
        raise RuntimeError('Reference mesh and exported camera disagree in 3D.')
    a, b = before @ K.T, after @ K.T
    error = float(np.linalg.norm(a[:, :2]/a[:, 2:] - b[:, :2]/b[:, 2:], axis=1).max())
    if error >= 1e-5:
        raise RuntimeError(f'Reference mesh and exported camera differ by {error} px.')
    target = out/'reference-mesh.obj'
    with target.open('x', encoding='ascii', newline='\n') as stream:
        stream.write('# Trained SMPL-X surface in the paired Gaussian coordinates\n')
        np.savetxt(stream, vertices, fmt='v %.17g %.17g %.17g')
        np.savetxt(stream, faces.astype(np.int64)+1, fmt='f %d %d %d')
    readback = np.loadtxt(target, dtype=str)
    v, f = readback[:len(vertices)], readback[len(vertices):]
    if (not (v[:, 0] == 'v').all() or not (f[:, 0] == 'f').all() or
            not np.array_equal(v[:, 1:].astype(float), vertices) or
            not np.array_equal(f[:, 1:].astype(np.int64)-1, faces)):
        raise RuntimeError('Reference OBJ roundtrip failed.')
    provenance = dict(schema='shuolee-reference-mesh-v1', mesh_file=target.name, mesh_sha256=rt.sha(target),
        source_mesh_sha256=receipt['mesh_sha256'], ply_sha256=report['sha256'], camera_sha256=rt.sha(out/'cameras.json'),
        reference_frame=source_row['frame'], camera_img_name=camera['img_name'], coordinate_space=report['coordinate_space'],
        rotation=report['rotation'], origin_world=report['origin_world'], scale=1.,
        vertex_count=len(vertices), triangle_count=len(faces), shared_vertexoffset_applied=True,
        nonzero_offset_vertices=int(np.any(offset != 0, axis=1).sum()), reprojection_max_px=error, quality_accepted=False)
    rt.save(out/'mesh-provenance.json', provenance)
    return provenance


def _package_smpl_sh0(out, result, report, mesh):
    cloud, camera = out/'point_cloud.ply', out/'cameras.json'
    if rt.sha(cloud) != report['sha256'] or rt.sha(camera) != mesh['camera_sha256']:
        raise RuntimeError('PLY or camera changed before packaging.')
    model_path = f"point_cloud/iteration_{result['steps']}/point_cloud.ply"
    cameras = rt.read(camera)
    provenance = rt.read(out/'camera_provenance.json')
    provenance.update(schema='shuolee-cameras-v1', coordinateConvention='opencv-camera-to-world',
        pixelConvention='pixel_corners', cameras=cameras, model=dict(name=model_path, sha256=report['sha256']))
    rt.save(out/'camera_provenance.json', provenance)
    manifest = dict(schema='shuolee-scene-v1', steps=result['steps'], initialView=0,
        referenceFrameId=mesh['camera_img_name'], referenceFrameIsPoseSnapshot=True,
        model=dict(file=model_path, sha256=report['sha256']),
        cameras=dict(file='cameras.json', sha256=mesh['camera_sha256'], provenance='camera_provenance.json', count=1),
        referenceMesh=dict(file=mesh['mesh_file'], sha256=mesh['mesh_sha256'], provenance='mesh-provenance.json',
            provenanceSha256=rt.sha(out/'mesh-provenance.json'), referenceFrameId=mesh['camera_img_name'],
            vertexCount=mesh['vertex_count'], triangleCount=mesh['triangle_count']),
        rendererContract=dict(schema='shuolee-renderer-contract-v1', cameraProjection='pinhole',
            gaussianEvaluation='screen-space-2d', trainingRenderer='diff-gaussian-rasterization'),
        auxiliaryModels=['input.ply'], world=report['coordinate_space'], qualityValidated=False)
    rt.save(out/'scene.json', dict(manifest, model=dict(manifest['model'], file=cloud.name), auxiliaryModels=[]))
    files = {name: out/name for name in ('cameras.json', 'camera_provenance.json',
        'mesh-provenance.json', 'reference-mesh.obj', 'validation.json')}
    files[model_path] = cloud
    files.update({name: out/'colmap'/name for name in ('cameras.txt', 'images.txt', 'points3D.txt')})
    vertices = PlyData.read(cloud, mmap=False)['vertex'].data
    proxy = np.empty(len(vertices), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i, (axis, channel) in enumerate(zip('xyz', ('red', 'green', 'blue'))):
        proxy[axis] = vertices[axis]
        proxy[channel] = np.rint(np.clip(.5+.28209479177387814*vertices[f'f_dc_{i}'], 0, 1)*255).astype(np.uint8)
    from io import BytesIO
    buffer = BytesIO()
    PlyData([PlyElement.describe(proxy, 'vertex')], text=False, byte_order='<',
            comments=['Gaussian centers for SIBR preview only']).write(buffer)
    auxiliary = {'input.ply': buffer.getvalue(), 'cfg_args': (
        "Namespace(sh_degree=0, source_path='.', model_path='.', images='images', "
        "resolution=-1, white_background=False, data_device='cuda', eval=False)\n").encode('ascii')}
    auxiliary['scene.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    expected = {name: rt.sha(path) for name, path in files.items()}
    expected.update({name: hashlib.sha256(data).hexdigest() for name, data in auxiliary.items()})
    temporary = out/'scene.zip.tmp'
    with ZipFile(temporary, 'x', compression=ZIP_STORED) as bundle:
        for name, path in files.items():
            bundle.write(path, name)
        for name, data in auxiliary.items():
            bundle.writestr(name, data)
    with ZipFile(temporary) as bundle:
        if set(bundle.namelist()) != set(expected):
            raise RuntimeError('Scene ZIP entries differ from the delivery contract.')
        for name, digest in expected.items():
            with bundle.open(name) as stream:
                check = hashlib.sha256()
                for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                    check.update(chunk)
            if check.hexdigest() != digest:
                raise RuntimeError('Scene ZIP readback failed: ' + name)
    destination = out/'scene.zip'
    temporary.rename(destination)
    return rt.sha(destination)


def export(root,cfg):
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from .training import camera,image
    result=rt.read(root/'gaussian-run/result.json');snapshot=root/'gaussian-run'/result['final_snapshot']
    stable = result.get('gaussian_profile') == 'smpl_sh0'
    if stable and (result.get('state') != 'completed' or result.get('steps') != 10000 or result.get('sh_degree') != 0):
        raise ValueError('A terminal degree-zero training receipt is required.')
    receipt=rt.read(snapshot/'native-ply-parity.json')
    if not receipt['passed'] or receipt['sha256']!=rt.sha(snapshot/'world.ply'):
        raise RuntimeError('Native world export is not verified.')
    with np.load(root/'fit/fitted.npz',allow_pickle=False) as archive:fit=dict(archive)
    reference_slot,reference_frame=reference_pose_index(fit,receipt)
    rows=rt.read(root/'data/training-inputs.json')['records']
    if stable:
        from .training import verify_official_fit
        bound = verify_official_fit(root)
        if bound != rt.read(root/'gaussian-run/config.json')['official_fit_provenance']:
            raise ValueError('Official fitting inputs changed after training.')
        rows = [row for row in rows if row['frame'] == reference_frame]
        if len(rows) != 1:
            raise ValueError('SH0 export requires exactly one fitted reference camera.')
    standard_export=cfg.get('gaussian_representation')=='static_clothed_world' or reference_frame is not None
    out=root/'export';out.mkdir()
    R,rotation_roundoff=rigid_rotation64(np.asarray(fit['root_rotation'][reference_slot],float).T)
    origin=np.asarray(fit['world_translation'][reference_slot],float)
    frame_mode='reference_camera_up' if stable else cfg.get('export_frame_mode','body_root')
    if frame_mode in ('nearest_scene_axes','scene_up_keep_axis_order'):
        candidates=[]
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product([-1.,1.],repeat=3):
                m=np.eye(3)[list(permutation)]*np.asarray(signs)[:,None]
                if np.linalg.det(m)>0:candidates.append(m)
        R=min(candidates,key=lambda m:np.linalg.norm(m-R))
        if frame_mode=='scene_up_keep_axis_order':
            # Yaw is an export gauge. Keep the same chosen up axis, but avoid
            # unnecessary X/Z swaps that alter float32 projection summation.
            same_up=[m for m in candidates if np.array_equal(m[1],R[1])]
            R=min(same_up,key=lambda m:(np.count_nonzero(m-np.diag(np.diag(m))),-m[0,0],-m[2,2]))
        origin=np.zeros(3)
    elif stable:
        R,rotation_roundoff=rigid_rotation64(np.diag([1.,-1.,-1.])@np.asarray(rows[0]['w2c'])[:3,:3])
        origin=np.zeros(3)
    elif frame_mode!='body_root':
        raise RuntimeError('Unknown export coordinate-frame mode.')
    degree = 0 if stable else 3
    checking=out/'point_cloud.checking.ply';report=rotate_ply(snapshot/'world.ply',checking,R,origin,sh_degree=degree)
    report['orientation_roundoff']=rotation_roundoff
    report['export_frame_mode']=frame_mode
    report['reference_frame_id']=reference_frame
    report['reference_pose_index']=reference_slot
    if stable:
        report['coordinate_space']='world +Y using reference-camera axes; physical gravity and calibration unverified'
    if frame_mode in ('nearest_scene_axes','scene_up_keep_axis_order'):
        report['coordinate_space']='world +Y using nearest proper scene-axis permutation; original tilt and translation preserved, not measured gravity'
        report['body_up_residual_degrees']=float(np.degrees(np.arccos(np.clip((R@fit['root_rotation'][reference_slot][:,1])[1],-1,1))))
    targetrows=[]
    for row in rows:
        E=np.asarray(row['w2c']);mapped=E.copy();mapped[:3,:3]=E[:3,:3]@R.T;mapped[:3,3]=E[:3,:3]@origin+E[:3,3]
        targetrow=dict(frame=row['frame'],source_frame_index=row['source_frame_index'],time_seconds=row['time_seconds'],wh=row['wh'],K=row['K'],w2c=mapped.tolist())
        if standard_export:
            targetrow['name']=row.get('name',row['frame'])
        targetrows.append(targetrow)
    if standard_export:
        from .camera_exchange import write_camera_exchange
        write_camera_exchange(out,targetrows,world_to_viewer=report)
    else:
        # Preserve the historical formal-route camera manifest contract.
        rt.save(out/'cameras.json',dict(records=targetrows,convention='OpenCV world-to-camera; COLMAP pixel corners',world_to_viewer=report))
    source=GaussianModel(degree);source.load_ply(str(snapshot/'world.ply'));portable=GaussianModel(degree);portable.load_ply(str(checking))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda');checks=[]
    if not standard_export:
        review=out/'review';review.mkdir()
    panels=[]
    with torch.no_grad():
        for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
            height = None if stable else 768
            ca,_=camera(rows[i],int(i),height,pose_index=reference_slot);cb,_=camera(targetrows[i],int(i),height,pose_index=reference_slot)
            a=render(ca,source,pipe,bg)['render'];b=render(cb,portable,pipe,bg)['render'];d=abs(a-b)
            checks.append(dict(frame=rows[i]['frame'],mean_abs=float(d.mean()),max_abs=float(d.max())))
            if not standard_export:
                image(a).save(review/f'{rows[i]["frame"]}_native.png');image(b).save(review/f'{rows[i]["frame"]}_viewer.png')
                from PIL import Image
                with Image.open(rows[i]['image']) as im:im.convert('RGB').resize((ca.image_width,ca.image_height),Image.Resampling.LANCZOS).save(review/f'{rows[i]["frame"]}_input.png')
                panels.append('<figure><figcaption>'+html.escape(rows[i]['frame'])+'</figcaption>'+''.join(f'<img src="review/{rows[i]["frame"]}_{suffix}.png" alt="{suffix}">' for suffix in ['input','native','viewer'])+'</figure>')
        if not standard_export:
            xyz=portable.get_xyz.cpu().numpy();center=np.median(xyz,axis=0);height=float(np.quantile(xyz[:,1],.99)-np.quantile(xyz[:,1],.01));radius=max(height*1.4,.1)
            report['viewer_focus']=center.tolist()
            for i,angle in enumerate(np.linspace(0,2*np.pi,36,endpoint=False)):
                eye=center+radius*np.array([np.sin(angle),.12,np.cos(angle)]);forward=center-eye;forward/=np.linalg.norm(forward)
                right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
                E=np.eye(4);E[:3,:3]=np.stack([right,down,forward]);E[:3,3]=-E[:3,:3]@eye
                row=dict(frame=f'novel_{i:03d}',wh=[512,768],K=[[768,0,256],[0,768,384],[0,0,1]],w2c=E.tolist())
                cam,_=camera(row,i,pose_index=reference_slot);image(render(cam,portable,pipe,bg)['render']).save(review/f'novel_{i:03d}.png')
    report['native_render_checks']=checks;report['format_passed']=all(c['mean_abs']<2e-6 and c['max_abs']<.005 for c in checks)
    if stable and report['points'] != result.get('points'):
        raise RuntimeError('SH0 point count differs from the completed training receipt.')
    if not standard_export:report['viewer_ui_review']='pending'
    rt.save(out/'validation.json',report)
    if not report['format_passed']:raise RuntimeError('+Y PLY render parity failed; candidate retained as checking.ply, not published.')
    checking.rename(out/'point_cloud.ply')
    delivery = {}
    if stable:
        mesh = _reference_mesh(out, snapshot, fit, rows[0], receipt, report)
        package_hash = _package_smpl_sh0(out, result, report, mesh)
        delivery.update(gaussian_profile='smpl_sh0', sh_degree=0, points=report['points'], mesh='export/reference-mesh.obj',
            camera='export/cameras.json', scene='export/scene.zip', scene_sha256=package_hash)
    if not standard_export:
        htmltext='<!doctype html><meta charset="utf-8"><title>Shuolee reconstruction review</title><style>body{font:16px sans-serif;background:#eee;margin:24px}figure{margin:16px 0}img{max-width:30%;height:auto}</style><h1>新视频静态重建检查</h1><p>每行：原图 / 原生渲染 / +Y PLY。尚未通过重影与未知视角质量验收。图片保持比例，不叠字。</p>'+''.join(panels)+'<h2>连续未知视角（无实拍真值）</h2><input id="angle" type="range" min="0" max="35" value="0"><br><img id="orbit" src="review/novel_000.png"><script>angle.oninput=()=>orbit.src="review/novel_"+angle.value.padStart(3,"0")+".png"</script>'
        (out/'review.html').write_text(htmltext,encoding='utf-8')
    rt.save(root/'result.json',dict(state='completed' if standard_export else 'completed_pending_quality_review',ply='export/point_cloud.ply',sha256=report['sha256'],format_passed=True,quality_accepted=False,reference_frame_id=reference_frame,reference_pose_index=reference_slot,from_scratch=True,frames=len(result['per_frame_updates']) if stable else len(rows),steps=result['steps'],local_adapter=True,execution_kind=result.get('execution_kind','formal_training'),formal_training_admitted=result.get('formal_training_admitted',False),**delivery))
