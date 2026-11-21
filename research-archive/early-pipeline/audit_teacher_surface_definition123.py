"""Read-only CPU audit: bound Gaussian centers, covariance and mesh winding.

Executes the saved model's original property methods on CPU tensors. It does
not reimplement a renderer or equate Gaussian centers with physical surfaces.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import sys
import time
import numpy as np
import torch
import roma
from plyfile import PlyData
from PIL import Image,ImageDraw
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,SNAP,OUT as AUDIT,verify
from gaussian_surface_guidance123 import closest_bound_barycentrics

OUT=AUDIT/'teacher-surface-definition'
CONTRACT=AUDIT/'frozen-skeleton-contract'
MODEL=io.PROJECT/'src/avatar_core/scene/gaussian_model.py'
GRAPHICS=io.PROJECT/'src/avatar_core/utils/graphics_utils.py'


def prepare():
    verify();OUT.mkdir(exist_ok=False)
    files=[Path(__file__),MODEL,GRAPHICS,Path(__file__).with_name('gaussian_surface_guidance123.py'),
        CONTRACT/'frozen-lbs-cache.npz',SNAP/'current-meshes.npz',io.ASSETS/'smplx/SMPLX_NEUTRAL.npz',
        SNAP/'bound-local.ply',SNAP/'world.ply',SNAP/'binding.npz',
        AUDIT/'teacher-surface-laplacian-100/result.json',ROOT/'data/training-inputs.json']
    io.save(OUT/'plan.json',dict(created=io.now(),cpu_only=True,max_seconds=180,frames=59,source_step=120000,
        files={str(p):io.sha(p) for p in files},optimizer_updates=0,
        hypotheses=['Stored binding normal direction has a reproducible source; never infer its sign from a rendered thumbnail.',
            'A splatted image surface need not pass through Gaussian centers; measure normal-direction covariance extent separately.',
            'Earlier center-in-mask/frustum support is not actual occlusion or per-pixel contribution.'],
        controls=['Run exact original get_xyz/get_scaling/get_rotation property code on CPU.',
            'Compare reference33 to original standard worldPLY.',
            'Check neutral mesh topology/winding and all59 posed signed-volume signs without changing winding.',
            'Check signed normal distance equals original bound local-Y times original triangle scale.',
            'Replay immutable input hashes and center-in-mask proxy; report any difference from prior GPU support summary.'],
        no_residual_based_selection=True,no_training_or_new_PLY=True,quality_accepted=False,
        limitation='1/2/3 sigma ellipsoid-plane reach is a covariance diagnostic, not a rendered surface or isovalue truth. No actual occlusion or contribution claim without original rasterizer evidence.',
        figures='All59 same-frame raw / saved120k Gaussian render / center projection (red negative normal, green positive normal). Integer crop, no resizing, no image text. Center display is explicitly a projection proxy.'))
    print('TEACHER_SURFACE_DEFINITION_REGISTERED',flush=True)


def check():
    verify();p=io.read(OUT/'plan.json')
    for f,h in p['files'].items():assert io.sha(f)==h,f
    return p


def methods():
    tree=ast.parse(MODEL.read_text());scope={'torch':torch,**vars(roma)}
    functions={}
    for name in ('get_xyz','get_scaling','get_rotation'):
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
        assert len(nodes)==1
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(MODEL),'exec'),scope)
        functions[name]=scope[name].fget
    return functions


def fields(ply,names):return torch.from_numpy(np.stack([np.asarray(ply[n]) for n in names],axis=-1).copy()).float()


def quantile(x,w):
    x=x.detach().numpy();w=w.detach().numpy();sel=w>0;x,w=x[sel],w[sel]
    order=np.argsort(x);c=np.cumsum(w[order]);c=c/c[-1]
    return dict(zip(('p10','median','p90'),map(float,np.interp([.1,.5,.9],c,x[order]))))


def volume(v,f):
    t=v.double()[f]-v.double().mean(0)
    return float((t[:,0]*torch.cross(t[:,1],t[:,2],dim=-1)).sum()/6)


def run():
    p=check();assert not (OUT/'result.json').exists();started=time.monotonic();torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    sys.path.insert(0,str(io.PROJECT/'src/avatar_core'))
    from utils.graphics_utils import compute_face_orientation
    with np.load(CONTRACT/'frozen-lbs-cache.npz',allow_pickle=False) as z:cache=dict(z)
    with np.load(io.ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True) as z:
        template=torch.tensor(z['v_template'],dtype=torch.float32);neutral_faces=np.asarray(z['f']);dominant=np.asarray(z['weights']).argmax(1)
    assert np.array_equal(neutral_faces,cache['faces'])
    # Use literal checkpoint vertices, not a numerically close CPU LBS replay.
    # Tiny input rounding can be magnified by narrow bound triangles.
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:saved=dict(z)
    assert np.array_equal(saved['faces'],cache['faces'])
    f=torch.tensor(saved['faces'].astype('int64'));v=torch.tensor(saved['vertices_camera'])
    rows=io.read(ROOT/'data/training-inputs.json')['records'];slots={str(x):i for i,x in enumerate(saved['frame_ids'])}
    ply=PlyData.read(str(SNAP/'bound-local.ply'))['vertex'];world=PlyData.read(str(SNAP/'world.ply'))['vertex']
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:binding=torch.tensor(z['triangle_ids'],dtype=torch.long)
    bnames=[n for n in ply.data.dtype.names if n.startswith('binding')]
    assert len(bnames)==1 and np.array_equal(np.asarray(ply[bnames[0]]).reshape(-1),binding.numpy())
    obj=SimpleNamespace(binding=binding,_xyz=fields(ply,['x','y','z']),_scaling=fields(ply,[f'scale_{i}' for i in range(3)]),
        _rotation=fields(ply,[f'rot_{i}' for i in range(4)]),scaling_activation=torch.exp,
        rotation_activation=lambda x:torch.nn.functional.normalize(x))
    opacity=torch.sigmoid(fields(ply,['opacity']))[:,0];fn=methods()
    learn=torch.tensor(np.isin(dominant,[0,1,2,3,4,5,6,9,13,14,16,17,18,19]));facelearn=learn[f].all(-1)
    edges=torch.cat((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));undirected=torch.sort(edges,1).values
    unique,inverse,counts=torch.unique(undirected,return_inverse=True,return_counts=True,dim=0)
    sign=torch.where(edges[:,0]<edges[:,1],1,-1);winding=torch.zeros(len(unique),dtype=torch.long);winding.scatter_add_(0,inverse,sign)
    topology=dict(vertices=len(template),faces=len(f),edges=len(unique),boundary_edges=int((counts==1).sum()),
        nonmanifold_edges=int((counts>2).sum()),inconsistent_two_face_winding_edges=int(((counts==2)&(winding!=0)).sum()),
        template_signed_volume_m3=volume(template,f),neutral_topology_matches=True)
    prior=io.read(AUDIT/'teacher-surface-laplacian-100/result.json')
    old={x['frame']:x for x in prior['teacher_support']};oldsigned={x['frame']:x for x in prior['teacher_normal_support']}
    result=[];images=[];weights_cache=[];signed_cache=[];sigma_cache=[];parity=None
    folder=OUT/'review';folder.mkdir(exist_ok=True);assert not list(folder.iterdir())
    record=Path('/mnt/d/00/记录/demo');record.mkdir(exist_ok=True)
    with torch.no_grad():
        for i,row in enumerate(rows):
            assert time.monotonic()-started<p['max_seconds'];assert io.sha(row['image'])==row['source_sha256'] and io.sha(row['mask'])==row['mask_sha256']
            verts=v[slots[row['frame']]];R,scale=compute_face_orientation(verts,f,return_scale=True)
            obj.face_orien_mat=R;obj.face_scaling=scale.clamp_min(1e-8);obj.face_center=verts[f].mean(1)
            obj.face_orien_quat=roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(R))
            xyz=fn['get_xyz'](obj);s=fn['get_scaling'](obj);q=fn['get_rotation'](obj)
            rotation=roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(q))
            if row['frame']=='frame_000033':
                worldxyz=fields(world,['x','y','z']);worldscale=fields(world,[f'scale_{j}' for j in range(3)]).exp()
                worldR=roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(torch.nn.functional.normalize(fields(world,[f'rot_{j}' for j in range(4)]))))
                covariance=lambda rot,sc: (rot*sc[:,None,:].square())@rot.transpose(1,2)
                parity=dict(centers_max_abs_m=float((xyz-worldxyz).abs().max()),
                    scaling_max_abs_m=float((s-worldscale).abs().max()),
                    covariance_max_abs_m2=float((covariance(rotation,s)-covariance(worldR,worldscale)).abs().max()))
                assert parity['centers_max_abs_m']<3e-6 and parity['scaling_max_abs_m']<3e-6 and parity['covariance_max_abs_m2']<1e-7,parity
            tri=verts[f[binding]];bary,valid,inside=closest_bound_barycentrics(tri,xyz)
            n=torch.nn.functional.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
            signed=((xyz-(bary[...,None]*tri).sum(1))*n).sum(-1)
            # Covariance support radius along the mesh normal; not a visibility test.
            normal_local=(rotation.transpose(1,2)@n[:,:,None]).squeeze(-1)
            sigma=(normal_local.square()*s.square()).sum(-1).sqrt()
            E=torch.tensor(row['w2c'],dtype=torch.float32);K=torch.tensor(row['K'],dtype=torch.float32)
            camera_xyz=xyz@E[:3,:3].T+E[:3,3];projected=camera_xyz@K.T;uv=projected[:,:2]/projected[:,2:].clamp_min(1e-9)
            ix,iy=uv[:,0].floor().long(),uv[:,1].floor().long();w,h=row['wh']
            raw=np.asarray(Image.open(row['image']).convert('RGB'));mask=np.asarray(Image.open(row['mask']).convert('L')).copy()
            assert raw.shape[:2]==mask.shape==(h,w)
            inimage=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(camera_xyz[:,2]>.2)
            weight=valid*inside*inimage*facelearn[binding]*opacity*torch.tensor(mask/255,dtype=torch.float32)[iy.clamp(0,h-1),ix.clamp(0,w-1)]
            sums=torch.zeros(len(f));sums.scatter_add_(0,binding,weight);weight/=sums[binding].clamp_min(1e-20);weight/=weight.sum()
            support=weight>0;pos=float(weight[signed>0].sum());normal_id_err=float((signed-obj._xyz[:,1]*obj.face_scaling[binding,0]).abs().max())
            # Points outside the triangle use an edge footpoint; its displacement
            # stays in the same plane, so the signed normal identity still holds.
            assert normal_id_err<3e-6,normal_id_err
            stats=dict(frame=row['frame'],points=int(support.sum()),triangles=int((sums>0).sum()),
                signed_normal_mm=quantile(signed*1000,weight),normal_sigma_mm=quantile(sigma*1000,weight),
                positive_normal_weight=pos,negative_normal_weight=float(weight[signed<0].sum()),
                center_plane_distance_less_than_sigma_weight={str(k):float(weight[signed.abs()<=k*sigma].sum()) for k in (1,2,3)},
                weighted_anisotropy_ratio=quantile(s.max(1).values/s.min(1).values.clamp_min(1e-20),weight),
                prior_point_count_difference=int(support.sum())-old[row['frame']]['points'],
                prior_positive_normal_weight_difference=pos-oldsigned[row['frame']]['positive_normal_weight'],
                bound_local_Y_identity_max_abs_m=normal_id_err,
                pose_world_rotation_determinant=float(torch.det(E[:3,:3])),posed_mesh_signed_volume_m3=volume(verts,f),
                minimum_depth_of_supported_centers_m=float(camera_xyz[support,2].min()))
            result.append(stats);weights_cache.append(weight.numpy());signed_cache.append(signed.numpy());sigma_cache.append(sigma.numpy())
            # Existing saved native Gaussian image; no new render claimed here.
            gauss=np.asarray(Image.open(SNAP/(row['frame']+'_gaussian.png')).convert('RGB'));assert gauss.shape==raw.shape
            points=Image.fromarray(raw.copy());draw=ImageDraw.Draw(points)
            for j in torch.where(support)[0].tolist():
                x,y=int(ix[j]),int(iy[j]);colour=(0,220,80) if signed[j]>0 else (240,60,60)
                draw.ellipse((x-1,y-1,x+1,y+1),fill=colour)
            yy,xx=np.where(mask>127);box=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21)]
            panels=[Image.fromarray(raw).crop(box),Image.fromarray(gauss).crop(box),points.crop(box)]
            sheet=Image.new('RGB',(panels[0].width*3,panels[0].height))
            for j,part in enumerate(panels):sheet.paste(part,(j*part.width,0))
            name=f'分析_高斯中心与表面_{row["frame"]}_原图_120k渲染_中心投影红负绿正.png'
            path=folder/name;sheet.save(path);dest=record/name;assert not dest.exists();dest.write_bytes(path.read_bytes())
            images.append(dict(frame=row['frame'],path=str(path),record_path=str(dest),sha256=io.sha(path),crop_integer_xyxy=box,no_resampling=True,no_text_added=True,
                center_projection_is_occlusion_unchecked=True))
            io.save(OUT/'progress.json',dict(state='cpu_zero_update_audit',frames=i+1,elapsed_seconds=time.monotonic()-started))
            print('TEACHER_SURFACE_CPU_FRAME',i+1,row['frame'],flush=True)
    check();assert not torch.cuda.is_initialized()
    np.savez_compressed(OUT/'center-support-analysis.npz',frame_ids=np.array([x['frame'] for x in rows]),
        weights=np.array(weights_cache),normal_distance_m=np.array(signed_cache),normal_sigma_m=np.array(sigma_cache),binding=binding.numpy())
    summary=dict(state='cpu_geometry_audit_complete_visibility_still_unverified',updated=io.now(),source_step=120000,
        topology=topology,original_world_PLY_parity=parity,frames=59,points=len(binding),frames_result=result,images=images,
        maximum_support_count_difference=max(abs(x['prior_point_count_difference']) for x in result),
        maximum_positive_weight_difference=max(abs(x['prior_positive_normal_weight_difference']) for x in result),
        normal_sign_source='Exact bound local-Y times original positive face scale. No normal flip or relabeling was performed.',
        covariance_scope='1/2/3sigma ellipsoid-plane reach only; the training renderer is screen-space alpha splatting, not an isosurface renderer.',
        center_support_scope='Positive depth>.2m, center in native mask, valid original bound triangle and existing anatomical groups. Prior GPU radius>0 summary comparison is not actual occlusion equivalence.',
        actual_visibility_contribution_verified=False,optimizer_updates=0,cuda_initialized=False,new_mesh_or_PLY=False,
        raw_pixels_and_geometry_unchanged=True,quality_accepted=False,actual_visual_review=False,
        elapsed_cpu_seconds=time.monotonic()-started)
    io.save(OUT/'result.json',summary);io.save(OUT/'progress.json',dict(state=summary['state'],frames=59,pid=None))
    print('TEACHER_SURFACE_CPU_COMPLETE',topology,'PLY parity',parity,'support differences',summary['maximum_support_count_difference'],summary['maximum_positive_weight_difference'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args();globals()[args.mode]()
