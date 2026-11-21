"""Frozen output-camera projections and continuous mesh self-visibility.

The mesh is a coarse anatomical prior. These functions cannot establish actual
material identity, external occlusion, calibration accuracy or staticness.
"""
import copy
import numpy as np
from native_residual_camera122 import validate_camera, project_world, rays_world
from native_residual_domain122 import sampled_domain_check


def cameras_from_fit_rows(rows):
    cameras=[]
    for row in rows:
        c=row.get('camera')
        if c is None:raise ValueError('Fit row requires its complete recorded camera')
        validate_camera(c);sampled_domain_check(c)
        if str(row['frame'])!=c['id'] or row['wh']!=[c['width'],c['height']]:raise ValueError('Fit row/camera identity differs')
        if not np.array_equal(row['K'],c['K']) or not np.array_equal(row['w2c'],c['w2c']):raise ValueError('Legacy prior copies differ from authoritative complete camera')
        cameras.append(copy.deepcopy(c))
    if len({c['id'] for c in cameras})!=len(cameras):raise ValueError('Duplicate fit camera')
    return cameras


class FixedOutputProjector:
    """Autograd only through world points; camera records remain immutable."""
    def __init__(self, cameras, *, device, dtype):
        import torch
        if not cameras: raise ValueError('Explicit per-view cameras required')
        if len({c['id'] for c in cameras}) != len(cameras): raise ValueError('Unique camera IDs required')
        self.cameras=copy.deepcopy(cameras)
        for c in self.cameras: validate_camera(c); sampled_domain_check(c)
        self.E=torch.tensor([c['w2c'] for c in cameras],device=device,dtype=dtype)
        self.K=torch.tensor([c['K'] for c in cameras],device=device,dtype=dtype)
        self.D=torch.tensor([sum([c['residual_projection'][k] for k in ['radial','tangential','thin_prism']],[]) for c in cameras],device=device,dtype=dtype)

    def project(self, xyz):
        import torch
        if xyz.ndim!=3 or xyz.shape[0]!=len(self.cameras) or xyz.shape[-1]!=3 or not torch.isfinite(xyz).all():
            raise ValueError('Finite [camera,point,3] world positions required')
        if xyz.dtype!=self.E.dtype or xyz.device!=self.E.device: raise ValueError('Projector dtype/device mismatch')
        pc=xyz@self.E[:,:3,:3].transpose(1,2)+self.E[:,:3,3][:,None,:]
        if (pc[...,2]<=1e-8).any(): raise ValueError('Fit candidate has nonpositive depth; no clamped pinhole fallback')
        x,y=(pc[...,:2]/pc[...,2:]).unbind(-1);r2=x*x+y*y
        k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4=[d[:,None] for d in self.D.unbind(-1)]
        num=1+r2*(k1+r2*(k2+r2*k3));den=1+r2*(k4+r2*(k5+r2*k6))
        if (den<=1e-12).any() or (num<=0).any():raise ValueError('Invalid residual mapping at fit points')
        radial=num/den
        u=x*radial+2*p1*x*y+p2*(r2+2*x*x)+s1*r2+s2*r2*r2
        v=y*radial+p1*(r2+2*y*y)+2*p2*x*y+s3*r2+s4*r2*r2
        uv=torch.stack([self.K[:,0,0,None]*u+self.K[:,0,2,None]-.5,
                        self.K[:,1,1,None]*v+self.K[:,1,2,None]-.5],-1)
        if not torch.isfinite(uv).all():raise ValueError('Nonfinite residual fit projection')
        return uv,pc


class TriangleBVH:
    """Median-split BVH, double precision rays, deterministic two-sided hits.

    Mesh coordinates are normalized by their bbox only for numerical stability.
    Ray origins/directions/hit distances are converted consistently. This does
    not alter the published world frame. No interpolated screen-space triangles.
    """
    def __init__(self, vertices, faces, *, leaf_size=12):
        v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces)
        if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or not len(v):raise ValueError('Finite mesh vertices required')
        if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer) or not len(f) or f.min()<0 or f.max()>=len(v):raise ValueError('Valid mesh topology required')
        if leaf_size!=12:raise ValueError('Registered BVH leaf size is12')
        self.center=(v.min(0)+v.max(0))*.5;self.scale=float(np.max(np.ptp(v,axis=0)))
        if self.scale<=0:raise ValueError('Degenerate mesh extent')
        self.vertices=v.copy();self.faces=f.astype(np.int64,copy=True)
        self.tri=((v-self.center)/self.scale)[self.faces]
        self.e1=self.tri[:,1]-self.tri[:,0];self.e2=self.tri[:,2]-self.tri[:,0]
        area=np.linalg.norm(np.cross(self.e1,self.e2),axis=1)
        if (area<=1e-15).any():raise ValueError('Degenerate mesh triangle; no implicit removal')
        lo=self.tri.min(1);hi=self.tri.max(1);centroid=self.tri.mean(1)
        self.nodes=[]
        def build(ids):
            index=len(self.nodes);self.nodes.append(None)
            node=dict(lo=lo[ids].min(0)-1e-10,hi=hi[ids].max(0)+1e-10)
            if len(ids)<=leaf_size:node['ids']=ids
            else:
                axis=int(np.argmax(np.ptp(centroid[ids],axis=0)))
                ordered=ids[np.argsort(centroid[ids,axis],kind='stable')];mid=len(ids)//2
                node['left']=build(ordered[:mid]);node['right']=build(ordered[mid:])
            self.nodes[index]=node
            return index
        build(np.arange(len(f)))

    def trace(self, origins, directions):
        d=np.asarray(directions,dtype=np.float64);o=np.asarray(origins,dtype=np.float64)
        if d.ndim!=2 or d.shape[1]!=3 or o.shape not in [(3,),d.shape] or not np.isfinite(d).all() or not np.isfinite(o).all():raise ValueError('Finite matching ray origins and directions required')
        norms=np.linalg.norm(d,axis=1)
        if (norms<=1e-15).any():raise ValueError('Nonzero ray direction required')
        d=d/norms[:,None];o=np.broadcast_to((o-self.center)/self.scale,d.shape)
        best=np.full(len(d),np.inf);triangle=np.full(len(d),-1,dtype=np.int64)
        def box(node, ids):
            od=o[ids];dd=d[ids];parallel=np.abs(dd)<1e-15
            outside=np.any(parallel&((od<node['lo'])|(od>node['hi'])),axis=1)
            safe=np.where(parallel,1.,dd)
            a=(node['lo']-od)/safe;b=(node['hi']-od)/safe
            near=np.where(parallel,-np.inf,np.minimum(a,b)).max(1)
            far=np.where(parallel,np.inf,np.maximum(a,b)).min(1)
            return ids[(~outside)&(far>=np.maximum(near,0))&(near<=best[ids])]
        stack=[(0,np.arange(len(d)))]
        while stack:
            index,ids=stack.pop();node=self.nodes[index];ids=box(node,ids)
            if not len(ids):continue
            if 'ids' not in node:
                stack.append((node['right'],ids));stack.append((node['left'],ids));continue
            ts=node['ids'];e1=self.e1[ts];e2=self.e2[ts];a=self.tri[ts,0]
            for start in range(0,len(ids),512):
                ii=ids[start:start+512]
                p=np.cross(d[ii,None,:],e2[None]);det=np.einsum('rti,ti->rt',p,e1)
                ok=np.abs(det)>1e-14;inv=np.divide(1.,det,out=np.zeros_like(det),where=ok)
                tv=o[ii,None,:]-a[None];u=np.sum(tv*p,-1)*inv
                q=np.cross(tv,e1[None]);w=np.sum(d[ii,None,:]*q,-1)*inv
                t=np.einsum('rti,ti->rt',q,e2)*inv
                ok&=(u>=-2e-10)&(w>=-2e-10)&(u+w<=1+2e-10)&(t>1e-10)
                t=np.where(ok,t,np.inf);j=np.argmin(t,axis=1);nearest=t[np.arange(len(ii)),j]
                update=nearest<best[ii]
                best[ii[update]]=nearest[update];triangle[ii[update]]=ts[j[update]]
        return best*self.scale,triangle


def point_visibility(bvh, camera, points_world, normals_world, *, tolerance):
    """Exact geometric ray at each surface point, residual sensor membership.

    Tolerance is a camera-z world-unit tolerance, matching legacy coarse-body
    surface support semantics. It is not a physical or subpixel truth bound.
    """
    validate_camera(camera)
    p=np.asarray(points_world,dtype=np.float64);n=np.asarray(normals_world,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(n).all():raise ValueError('Finite matched surface points/normals required')
    if not np.isfinite(tolerance) or tolerance<0:raise ValueError('Nonnegative explicit model tolerance required')
    e=np.asarray(camera['w2c']);pc=p@e[:3,:3].T+e[:3,3]
    center=-e[:3,:3].T@e[:3,3];delta=p-center;distance=np.linalg.norm(delta,axis=1)
    positive=(pc[:,2]>1e-8)&(distance>1e-10)
    uv=np.full((len(p),2),np.nan)
    if positive.any():uv[positive]=project_world(p[positive],camera)-.5
    in_image=positive&(uv[:,0]>=0)&(uv[:,0]<=camera['width']-1)&(uv[:,1]>=0)&(uv[:,1]<=camera['height']-1)
    nd=np.linalg.norm(n,axis=1);facing=np.zeros(len(p),bool)
    good=positive&(nd>1e-15)
    facing[good]=np.sum(n[good]/nd[good,None]*(-delta[good]/distance[good,None]),axis=1)>.05
    nearest=np.full(len(p),np.inf);tri=np.full(len(p),-1,dtype=np.int64)
    query=np.flatnonzero(in_image)
    if len(query):nearest[query],tri[query]=bvh.trace(center,delta[query])
    hit=tri>=0;z=np.full(len(p),np.nan)
    z[hit]=nearest[hit]*pc[hit,2]/distance[hit]
    agreement=hit&(np.abs(z-pc[:,2])<=tolerance)
    visible=in_image&facing&agreement
    return visible,dict(in_image=in_image,facing=facing,depth_supported=hit,depth_agreement=agreement,
        projected_uv=uv,nearest_camera_z=z,triangle=tri,external_occluders_modeled=False)


def mesh_normals(vertices, faces):
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces,dtype=np.int64);tri=v[f]
    fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);vn=np.zeros_like(v)
    for j in range(3):np.add.at(vn,f[:,j],fn)
    return fn,vn


def shared_anatomical_visibility(vertices_world, faces, cameras, regional_weights,
                                face_vertex_ids, face_bary, face_triangle_ids, feet_vertex_ids):
    """One shared static mesh, full per-view sensor and exact point-ray queries."""
    v=np.asarray(vertices_world,dtype=np.float64);faces=np.asarray(faces,dtype=np.int64)
    if v.ndim!=3 or not np.array_equal(v,np.broadcast_to(v[0],v.shape)):raise ValueError('One identical static mesh in every camera is required')
    if len(cameras)!=len(v):raise ValueError('Camera/mesh count mismatch')
    rw=np.asarray(regional_weights,dtype=np.float64)
    if rw.ndim!=2 or rw.shape[0]!=len(v[0]) or (rw<0).any():raise ValueError('Fixed nonnegative anatomical region weights required')
    bvh=TriangleBVH(v[0],faces);fn,vn=mesh_normals(v[0],faces)
    face_points=(v[0][face_vertex_ids]*np.asarray(face_bary)[:,:,None]).sum(1)
    points=np.concatenate([v[0],face_points,v[0][feet_vertex_ids]],axis=0)
    normals=np.concatenate([vn,fn[face_triangle_ids],vn[feet_vertex_ids]],axis=0)
    tolerance=.01*float(np.linalg.norm(np.ptp(v[0],axis=0)))
    regions=[];facelist=[];feetlist=[];nv=len(v[0]);nf=len(face_points)
    for camera in cameras:
        visible,_=point_visibility(bvh,camera,points,normals,tolerance=tolerance)
        fraction=(visible[:nv,None]*rw).sum(0)/np.maximum(rw.sum(0),1)
        regions.append(np.clip(fraction/.20,.05,1))
        facelist.append(visible[nv:nv+nf]);feetlist.append(visible[nv+nf:])
    return np.asarray(regions),np.asarray(facelist),np.asarray(feetlist)


def diagnostic_depth(bvh,camera):
    """Observed-pixel raycast for a small registered diagnostic; no image warp."""
    if camera['width']*camera['height']>100000:raise ValueError('Diagnostic raster bounded to100k pixels')
    x,y=np.meshgrid(np.arange(camera['width'])+.5,np.arange(camera['height'])+.5)
    uv=np.column_stack([x.ravel(),y.ravel()]);origin,dirs=rays_world(uv,camera)
    distance,triangle=bvh.trace(origin,dirs)
    raycam=dirs@np.asarray(camera['w2c'])[:3,:3].T
    depth=np.where(triangle>=0,distance*raycam[:,2],0)
    return depth.reshape(camera['height'],camera['width']),triangle.reshape(camera['height'],camera['width'])
