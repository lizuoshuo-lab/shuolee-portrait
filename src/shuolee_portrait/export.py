"""Rigid world-to-viewer export, including every covariance and SH coefficient."""
import html
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
from types import SimpleNamespace
from . import runtime as rt


def rotate_ply(source, target, R, origin):
    from utils.sh_utils import eval_sh
    data = PlyData.read(source); vertex = data['vertex'].data.copy(); names=vertex.dtype.names
    required = ['x','y','z','opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]
    if any(x not in names for x in required): raise RuntimeError('Incomplete standard degree-3 Gaussian PLY.')
    if not all(np.isfinite(vertex[x]).all() for x in required): raise RuntimeError('PLY contains nonfinite parameters.')
    if not np.allclose(R@R.T,np.eye(3),atol=1e-8) or np.linalg.det(R)<.999999:
        raise RuntimeError('Export requires a rigid proper rotation.')
    xyz=np.column_stack([vertex[x] for x in ['x','y','z']]).astype(float); mapped=(xyz-origin)@R.T
    q=np.column_stack([vertex[f'rot_{i}'] for i in range(4)]).astype(float)
    if (np.linalg.norm(q,axis=1)<1e-8).any():raise RuntimeError('Zero rotation quaternion.')
    old_rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();new_q=Rotation.from_matrix(R[None]@old_rot).as_quat()[:,[3,0,1,2]]
    for i,key in enumerate(['x','y','z']):vertex[key]=mapped[:,i]
    for i in range(4):vertex[f'rot_{i}']=new_q[:,i]
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
    loaded=PlyData.read(target)['vertex'].data
    inv=np.column_stack([loaded[x] for x in ['x','y','z']])@R+origin
    position_error=float(abs(inv-xyz).max());scale=np.exp(np.column_stack([loaded[f'scale_{i}'] for i in range(3)]).astype(float))
    m=Rotation.from_quat(np.column_stack([loaded[f'rot_{i}'] for i in [1,2,3,0]])).as_matrix()
    covariance=lambda rot:np.einsum('nij,nj,nkj->nik',rot,scale**2,rot)
    expected=covariance(old_rot);actual=R.T@covariance(m)@R
    covariance_error=float(abs(actual-expected).max());extent=max(float(np.ptp(xyz,axis=0).max()),1.)
    if position_error>2e-6*extent or covariance_error>1e-5*max(float(abs(expected).max()),1e-8):
        raise RuntimeError('Export inverse geometry check failed.')
    return dict(points=len(vertex),position_inverse_max=position_error,covariance_inverse_max=covariance_error,
                sh_basis_rotation_max=basis_error,sha256=rt.sha(target),coordinate_space='world +Y; body-prior root orientation, not measured gravity',
                rotation=R.tolist(),origin_world=origin.tolist(),scale=1.,quality_accepted=False)


def export(root,cfg):
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from .training import camera,image
    result=rt.read(root/'gaussian-run/result.json');snapshot=root/'gaussian-run'/result['final_snapshot']
    receipt=rt.read(snapshot/'native-ply-parity.json')
    if not receipt['passed'] or receipt['sha256']!=rt.sha(snapshot/'world.ply'):
        raise RuntimeError('Native world export is not verified.')
    out=root/'export';out.mkdir();fit=dict(np.load(root/'fit/fitted.npz'))
    R=np.asarray(fit['root_rotation'][0],float).T;origin=np.asarray(fit['world_translation'][0],float)
    checking=out/'point_cloud.checking.ply';report=rotate_ply(snapshot/'world.ply',checking,R,origin)
    rows=rt.read(root/'data/training-inputs.json')['records'];targetrows=[]
    for row in rows:
        E=np.asarray(row['w2c']);mapped=E.copy();mapped[:3,:3]=E[:3,:3]@R.T;mapped[:3,3]=E[:3,:3]@origin+E[:3,3]
        targetrows.append(dict(frame=row['frame'],source_frame_index=row['source_frame_index'],time_seconds=row['time_seconds'],wh=row['wh'],K=row['K'],w2c=mapped.tolist()))
    rt.save(out/'cameras.json',dict(records=targetrows,convention='OpenCV world-to-camera; COLMAP pixel corners',world_to_viewer=report))
    source=GaussianModel(3);source.load_ply(str(snapshot/'world.ply'));portable=GaussianModel(3);portable.load_ply(str(checking))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda');checks=[]
    review=out/'review';review.mkdir();panels=[]
    with torch.no_grad():
        for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
            ca,_=camera(rows[i],int(i),768);cb,_=camera(targetrows[i],int(i),768)
            a=render(ca,source,pipe,bg)['render'];b=render(cb,portable,pipe,bg)['render'];d=abs(a-b)
            checks.append(dict(frame=rows[i]['frame'],mean_abs=float(d.mean()),max_abs=float(d.max())))
            image(a).save(review/f'{rows[i]["frame"]}_native.png');image(b).save(review/f'{rows[i]["frame"]}_viewer.png')
            from PIL import Image
            with Image.open(rows[i]['image']) as im:im.convert('RGB').resize((ca.image_width,ca.image_height),Image.Resampling.LANCZOS).save(review/f'{rows[i]["frame"]}_input.png')
            panels.append('<figure><figcaption>'+html.escape(rows[i]['frame'])+'</figcaption>'+''.join(f'<img src="review/{rows[i]["frame"]}_{suffix}.png" alt="{suffix}">' for suffix in ['input','native','viewer'])+'</figure>')
        xyz=portable.get_xyz.cpu().numpy();center=np.median(xyz,axis=0);height=float(np.quantile(xyz[:,1],.99)-np.quantile(xyz[:,1],.01));radius=max(height*1.4,.1)
        for i,angle in enumerate(np.linspace(0,2*np.pi,36,endpoint=False)):
            eye=center+radius*np.array([np.sin(angle),.12,np.cos(angle)]);forward=center-eye;forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
            E=np.eye(4);E[:3,:3]=np.stack([right,down,forward]);E[:3,3]=-E[:3,:3]@eye
            row=dict(frame=f'novel_{i:03d}',wh=[512,768],K=[[768,0,256],[0,768,384],[0,0,1]],w2c=E.tolist())
            cam,_=camera(row,i);image(render(cam,portable,pipe,bg)['render']).save(review/f'novel_{i:03d}.png')
    report['native_render_checks']=checks;report['format_passed']=all(c['mean_abs']<2e-6 and c['max_abs']<.005 for c in checks)
    report['viewer_ui_review']='pending';rt.save(out/'validation.json',report)
    if not report['format_passed']:raise RuntimeError('+Y PLY render parity failed; candidate retained as checking.ply, not published.')
    checking.rename(out/'point_cloud.ply')
    htmltext='<!doctype html><meta charset="utf-8"><title>Shuolee reconstruction review</title><style>body{font:16px sans-serif;background:#eee;margin:24px}figure{margin:16px 0}img{max-width:30%;height:auto}</style><h1>新视频静态重建检查</h1><p>每行：原图 / 原生渲染 / +Y PLY。尚未通过重影与未知视角质量验收。图片保持比例，不叠字。</p>'+''.join(panels)+'<h2>连续未知视角（无实拍真值）</h2><input id="angle" type="range" min="0" max="35" value="0"><br><img id="orbit" src="review/novel_000.png"><script>angle.oninput=()=>orbit.src="review/novel_"+angle.value.padStart(3,"0")+".png"</script>'
    (out/'review.html').write_text(htmltext,encoding='utf-8')
    rt.save(root/'result.json',dict(state='completed_pending_quality_review',ply='export/point_cloud.ply',sha256=report['sha256'],format_passed=True,quality_accepted=False,from_scratch=True,frames=len(rows),steps=cfg['iterations'],local_adapter=True))
