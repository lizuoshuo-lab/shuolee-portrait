"""Re-express a verified posed PLY in a +Y-up world frame, never refit it.

Reuses the SH basis rotation verified in audit_standalone_export.py. The up
direction comes from observed floor support, not an unaccepted body root.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
import shuolee_common as c
torch=c.runtime()
from scene.gaussian_model import GaussianModel
from scene.cameras import CameraProj2
from floor_reference_cameras import make_cameras
from utils.general_utils import build_rotation
from utils.sh_utils import eval_sh
from gaussian_renderer import render
def pil(t):
    return Image.fromarray((t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8))

PIPE=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
def camera(row,i):
    E=np.asarray(row['w2c']);K=np.asarray(row['K'])
    return CameraProj2(i,E[:3,:3].T,E[:3,3],K[0,0],K[1,1],'white',row['width'],None,row['height'],'',row['view'],i,cx=K[0,2],cy=K[1,2])
def basis(d):return eval_sh(3,np.eye(16),d)

@torch.no_grad()
def main(root,allow_depth_tie_roundoff=False):
    root=Path(root);run=root/'gaussian-run';out=run/'viewer-export';out.mkdir(exist_ok=True)
    receipt=c.read(run/'delivery.json');source=run/'point_cloud.ply';source_hash=c.sha(source)
    assert receipt['compatibility_passed'] and source_hash==receipt['sha256']
    original=GaussianModel(3);original.load_ply(str(source))
    cams,info,_=make_cameras()
    up=np.asarray(info['floor']['floor_normal_world']);up=up/np.linalg.norm(up)
    origin=np.asarray(info['target_world'])-.5*info['support_height_world_units']*up
    E=np.asarray(cams[0]['w2c']);front=(-E[:3,:3].T@E[:3,3])-np.asarray(info['target_world'])
    front-=front@up*up;front/=np.linalg.norm(front)
    right=np.cross(up,front);right/=np.linalg.norm(right);front=np.cross(right,up)
    R=np.stack([right,up,front]);assert np.max(np.abs(R@R.T-np.eye(3)))<1e-10 and np.linalg.det(R)>.999999
    Rt=torch.tensor(R,dtype=torch.float32,device='cuda');Ot=torch.tensor(origin,dtype=torch.float32,device='cuda')
    portable=GaussianModel(3);portable.active_sh_degree=3
    # Compose the translation and rotation in float64 before the single PLY
    # float32 rounding; avoid cancellation from two separate CUDA float32 ops.
    xyz64=(original._xyz.cpu().numpy().astype(np.float64)-origin)@R.T
    portable._xyz=torch.tensor(xyz64,dtype=torch.float32,device='cuda')
    portable._scaling=original._scaling.clone();portable._opacity=original._opacity.clone()
    # Compose rotations in float64 from the original quaternions, avoiding an
    # intermediate float32 matrix before the final portable quaternion rounding.
    source_q=original.get_rotation.cpu().numpy().astype(np.float64)
    matrices=R[None]@Rotation.from_quat(source_q[:,[1,2,3,0]]).as_matrix()
    q=Rotation.from_matrix(matrices).as_quat()[:,[3,0,1,2]]
    portable._rotation=torch.as_tensor(q,dtype=torch.float32,device='cuda')
    rng=np.random.default_rng(23033);dirs=rng.normal(size=(256,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    transform=np.linalg.lstsq(basis(dirs),basis(dirs@R),rcond=None)[0]
    check=rng.normal(size=(1024,3));check/=np.linalg.norm(check,axis=1)[:,None]
    sh_error=float(np.abs(basis(check)@transform-basis(check@R)).max());assert sh_error<1e-10
    features=torch.einsum('ij,njc->nic',torch.tensor(transform,dtype=torch.float32,device='cuda'),original.get_features)
    portable._features_dc=features[:,:1,:].contiguous();portable._features_rest=features[:,1:,:].contiguous()
    path=out/'point_cloud.checking.ply';portable.save_ply(str(path));reload=GaussianModel(3);reload.load_ply(str(path))
    inv_pos=float((reload._xyz@Rt+Ot-original._xyz).abs().max())
    def cov(m):
        rot=build_rotation(m.get_rotation)
        return rot@torch.diag_embed(m.get_scaling.square())@rot.transpose(1,2)
    inv_cov=float((Rt.T@cov(reload)@Rt-cov(original)).abs().max())
    assert inv_pos<2e-5 and inv_cov<1e-6
    row=c.read(root/'data/inputs.json')['records'][receipt['pose_index']]
    K=np.array(row['K'])*.35;K[2,2]=1
    cams=[dict(view='exported_pose_source_camera',w2c=row['w2c'],K=K.tolist(),width=round(row['wh'][0]*.35),height=round(row['wh'][1]*.35))]+cams
    bg=torch.ones(3,device='cuda');checks=[];tiles=[]
    for i,row in enumerate(cams):
        source_camera=camera(row,i)
        # Transform the matrix actually consumed by CUDA, rather than repeating
        # camera-constructor inverse operations on a nominal input extrinsic.
        E=source_camera.world_view_transform.cpu().numpy().T.astype(np.float64)
        mapped=E.copy();mapped[:3,:3]=E[:3,:3]@R.T;mapped[:3,3]=E[:3,:3]@origin+E[:3,3]
        target_camera=camera({**row,'w2c':mapped},i)
        target_camera.world_view_transform=torch.tensor(mapped.T,dtype=torch.float32)
        target_camera.full_proj_transform=target_camera.world_view_transform@source_camera.projection_matrix
        target_camera.camera_center=torch.tensor(np.linalg.inv(mapped)[:3,3],dtype=torch.float32)
        a=render(source_camera,original,PIPE,bg)['render']
        b=render(target_camera,reload,PIPE,bg)['render']
        diff=(a-b).abs();codes=((a.clamp(0,1)*255+.5).to(torch.int16)-(b.clamp(0,1)*255+.5).to(torch.int16)).abs()
        index=diff.argmax();flat_a=a.flatten();flat_b=b.flatten()
        checks.append(dict(view=row['view'],max_abs=float(diff.max()),mean_abs=float(diff.mean()),max_8bit=int(codes.max()),channels_over_one_level=int((codes>1).sum()),
            raw_values_at_largest_error=[float(flat_a[index]),float(flat_b[index])],clamped_max_abs=float((a.clamp(0,1)-b.clamp(0,1)).abs().max())))
        for suffix,im in [('source world',pil(a)),('+Y-up PLY',pil(b))]:
            im.thumbnail((330,500));tile=Image.new('RGB',(340,535),'white');tile.paste(im,((340-im.width)//2,30));ImageDraw.Draw(tile).text((6,7),row['view']+' | '+suffix,fill='black');tiles.append(tile)
    strict_passed=all(x['mean_abs']<1e-6 and x['max_abs']<1/255 and x['channels_over_one_level']==0 for x in checks)
    passed=strict_passed;roundoff=None
    if not strict_passed and allow_depth_tie_roundoff:
        # A global rigid transform preserves the shape but does not guarantee
        # invariant float32 depth sorting at coplanar ties. Accept only with a
        # separate exhaustive audit bound to these exact PLY bytes. Keep the
        # original strict-pixel failure visible in the delivery receipt.
        diagnostic=out/'order-diagnostic.json';roundoff=c.read(diagnostic)
        assert roundoff['source_sha256']==source_hash and roundoff['candidate_sha256']==c.sha(path)
        assert roundoff['all_exceptions_explained'] and len(roundoff['views'])==len(checks)
        assert sum(x['pixels_reviewed'] for x in roundoff['views'])==len(roundoff['pixels'])
        assert all(x['pixels_reviewed']<=max(1,int(x['pixel_count']*1e-4)) for x in roundoff['views'])
        assert all(x['mean_abs']<1e-6 for x in checks)
        passed=True
    sheet=Image.new('RGB',(1360,2140),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*340,(i//4)*535))
    sheet.save(out/'upright-parity.jpg',quality=94)
    result={**receipt,'created':c.stamp(),'coordinate_space':('shared static world; +Y up; floor-referenced global rigid transform' if receipt.get('representation')=='shared_static_world' else 'world; selected pose baked; +Y up; floor-referenced global rigid transform'),
        'source_world_ply_sha256':source_hash,'sha256':c.sha(path),'local_path':str(out/'point_cloud.ply'),
        'world_to_viewer':dict(rotation=R.tolist(),origin_world=origin.tolist(),uniform_scale=1.,up='+Y',front='+Z',floor_sha256=info['floor_sha256']),
        'inverse_position_max':inv_pos,'inverse_covariance_max':inv_cov,'sh_basis_rotation_max_error':sh_error,
        'source_world_parameter_checks':receipt['parameter_checks'],
        'parameter_checks':dict(inverse_position_max=inv_pos,inverse_covariance_max=inv_cov,sh_basis_rotation_max_error=sh_error),
        'export_code_sha256':c.sha(__file__),
        'upright_render_checks':checks,'source_unchanged':c.sha(source)==source_hash,
        'compatibility_passed':passed,'strict_pixel_parity_passed':strict_passed,
        'roundoff_exception_audit':dict(path=str(out/'order-diagnostic.json'),sha256=c.sha(out/'order-diagnostic.json'),
            reviewed_pixels=len(roundoff['pixels']),basis='Rigid geometry and all SH preserved; every pixel exception explained by float32 near-depth ordering or contribution cutoff; not bit-identical rendering') if roundoff is not None else None,
        'local_geometry_changed':False,'new_optimizer_updates':0}
    c.save(out/'delivery.json',result)
    assert passed,'Upright PLY parity failed; do not publish checking file'
    path.replace(out/'point_cloud.ply');print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
