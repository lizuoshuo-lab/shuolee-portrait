"""Read back a joint run and compare mesh, splats and source matte on identical native pixels."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from fuse_scene import render, read_gaussians, to_bgr
from msv2_scene_engine import make_rasterizer
from train_joint import prepare_frames, combine, visible_head
from train_msv2_vhap import MeshBoundHead, protected_triangles
from vhap_mesh_offset import VhapMeshOffset
from probe_vhap_profile_translation import visible_points
# Same left/right FLAME proxy vertices as fit_vhap_ear_offset, without loading the VHAP fitting runtime.
EAR_IDS = (6, 616)


def array(x):
    return x.detach().cpu().numpy()


def summary(x):
    x = np.asarray(x)
    if not x.size:
        return None
    return dict(count=x.size, median=float(np.median(x)), p95=float(np.quantile(x,.95)), max=float(x.max()))


def project(xyz, calibration):
    K, _, r, c = calibration
    uvz = (xyz-c) @ r @ K.T
    return uvz[:,:2] / uvz[:,2:]


def source_map(scene, frames, K, size):
    names={f.name for f in frames}
    paths=[p for p in (scene/'colmap_bg/sparse').glob('*/cameras.bin')
           if {r.name for r in read_extrinsics_binary(p.with_name('images.bin')).values()}==names]
    if len(paths)!=1:
        raise ValueError('Expected one original COLMAP reconstruction with the same cameras')
    cam,=read_intrinsics_binary(paths[0]).values()
    if cam.model!='OPENCV':
        raise ValueError(cam.model)
    fx,fy,cx,cy,*dist=cam.params
    raw=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]])
    return cv2.initUndistortRectifyMap(raw,np.asarray(dist),np.eye(3),K,size,cv2.CV_32FC1)


def mesh_mask(vertices, faces, frame, shape):
    mask=np.zeros(shape,np.uint8)
    xy=np.rint(project(vertices,frame.rasterizer.calibration)*16).astype(np.int32)
    for triangle in xy[faces]:
        cv2.fillConvexPoly(mask,triangle,1,shift=4)
    return mask.astype(bool)


def binding_distances(v, faces, head, actor, scale, frame):
    centers,axes,quat=[array(head[i]) for i in (0,3,4)]
    binding=array(actor.binding)
    triangles=v[faces[binding//4]]
    ctr=triangles.mean(1)
    slot=binding%4
    anchor=ctr.copy()
    peripheral=slot>0
    anchor[peripheral]+=.5*(triangles[peripheral,slot[peripheral]-1]-ctr[peripheral])
    a,b,c=triangles.transpose(1,0,2)
    normal=np.cross(b-a,c-a)
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    signed=((centers-a)*normal).sum(1)
    plane=centers-signed[:,None]*normal
    inside=np.ones(len(centers),bool)
    edges=[]
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start
        inside &= (np.cross(edge,plane-start)*normal).sum(1)>=0
        t=((centers-start)*edge).sum(1)/np.square(edge).sum(1)
        edges.append(np.linalg.norm(centers-(start+np.clip(t,0,1)[:,None]*edge),axis=1))
    distance=np.where(inside,np.abs(signed),np.min(edges,axis=0))
    alpha=array(head[2]).ravel()
    rot=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
    normal_sigma=np.linalg.norm(np.einsum('ni,nij->nj',normal,rot)*axes,axis=1)
    uv_motion=np.linalg.norm(project(centers,frame.rasterizer.calibration)-project(anchor,frame.rasterizer.calibration),axis=1)
    protected=array(actor.protected)[binding]
    groups={"all":np.ones(len(alpha),bool),"alpha_ge_01":alpha>=.1,"alpha_ge_01_free":(alpha>=.1)&~protected,
            "alpha_ge_01_protected":(alpha>=.1)&protected}
    return {k:dict(own_triangle_distance_tracker_mm=summary(distance[s]*1000/scale),
                   above_3mm=int((distance[s]*1000/scale>3).sum()),
                   center_to_site_px=summary(uv_motion[s]), normal_sigma_tracker_mm=summary(normal_sigma[s]*1000/scale))
            for k,s in groups.items()}


def top_curve(mask, xs):
    m=mask[:,xs]
    return np.where(m.any(0),m.argmax(0),np.nan)


def contour_panel(bgr, masks):
    panel=bgr.copy()
    for m,color in masks:
        contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel,contours,-1,color,1,cv2.LINE_AA)
    return panel


@torch.enable_grad()
def l1_gradient_split(actor, frame, photo, background):
    head=actor.head(frame)
    rgb=render(frame.rasterizer,head)
    alpha=render(frame.rasterizer,head,alpha_only=True)
    l1=(rgb/alpha.clamp_min(1e-4)-frame.reference).abs().mean(0)
    params=(actor.mesh.offset,actor.head_offset,actor.head_dc)
    masks=[torch.as_tensor(m,device=rgb.device) for m in (background,photo&~background)]
    parts=[torch.autograd.grad(l1[m].sum()/photo.sum(),params,retain_graph=i==0) for i,m in enumerate(masks)]
    result={}
    for key,a,b in zip(('shared_mesh','local_centers','sh0'),*parts):
        na,nb=float(a.norm()),float(b.norm())
        result[key]=dict(background_norm=na,other_norm=nb,norm_share=na/(na+nb) if na+nb else None,
                         cosine=float(torch.sum(a*b)/(na*nb)) if na*nb else None)
    return result


@torch.no_grad()
def run(root, out):
    out.mkdir(parents=True,exist_ok=False)
    read=lambda p: json.loads(p.read_bytes())
    state=read(root/'run.json')
    stats=read(root/'joint/metrics.json')
    command=state['stages']['gaussian_training']['command']
    path=lambda key: Path(command[command.index('--'+key)+1])
    args=argparse.Namespace(scene=path('scene'),background=path('background'),bound=path('bound'),
            tracker=path('tracker'),landmarks=path('landmarks'),registered_vhap=path('registered-vhap'),
            background_ply=path('background')/'point_cloud.ply',holdout=stats['holdout_frames'],export_frame=stats['reference'])
    K,size,scale,_,_,frames=prepare_frames(args,make_rasterizer)
    byname={f.name:f for f in frames}
    target=byname[args.export_frame]
    mesh=VhapMeshOffset(args.registered_vhap,frames,path('flame-mask'))
    actor=MeshBoundHead(target,protected_triangles(args.registered_vhap,path('flame-mask')),mesh,surface_samples=4)
    actor.fit_footprints=True
    actor.head_offset.requires_grad_(True)
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
        checkpoint=torch.load(root/'joint/training-state.pt',map_location='cpu',weights_only=True)
    if checkpoint['signature']!=stats['source_signature'] or not torch.equal(checkpoint['head']['binding'],actor.binding.cpu()):
        raise ValueError('Expected the unpruned four-site reference')
    actor.load_state_dict(checkpoint['head'])
    del checkpoint
    ref_head=actor.head(target)
    packed=read_gaussians(root/'joint/point_cloud.ply')
    scene_count=stats['scene_and_head_point_counts'][0]
    scene=tuple(x[:scene_count] for x in packed)
    xyz_delta=float((packed[0][scene_count:]-ref_head[0]).abs().max())
    if xyz_delta>1e-5:
        raise ValueError(f'PLY and checkpoint centers differ by {xyz_delta}')
    faces=array(mesh.faces)
    ref_v=array(mesh.vertices_world(target))
    report=dict(reference=target.name,ply_center_parity=xyz_delta,frames={},distances=binding_distances(
            ref_v,faces,ref_head,actor,scale,target))
    mx,my=source_map(args.scene,frames,K,size)
    det=np.load(args.scene.parent/'body-observations.npz',allow_pickle=False)
    ear_ix=[det['joint_names'].tolist().index(n) for n in ('left_ear','right_ear')]
    if det['names'].tolist()!=list(byname):
        raise ValueError('Observation frame order mismatch')
    selected=list(dict.fromkeys([target.name,'00300.jpg','00850.jpg',*args.holdout]))
    for name in selected:
        f=byname[name]
        read_gray=lambda folder:cv2.imread(str(args.scene/folder/name),0)
        remap=lambda img:cv2.remap(img,mx,my,cv2.INTER_LINEAR)
        native=read_gray('mask_undist/images')
        mapped=remap(read_gray('person_masks'))
        alpha=remap(read_gray('alpha_maps'))/255.
        box=f.crop
        iou=lambda x,y:float(np.count_nonzero(x&y)/np.count_nonzero(x|y))
        match=iou(mapped[box]>=224,native[box]>=224)
        if match<.98:
            raise ValueError(f'{name} matte mapping failed: {match}')
        own=actor.head(f)
        v=array(mesh.vertices_world(f))
        zero=array(mesh.vertices_world(f,deformed=False))
        rgb=to_bgr(f.reference)
        own_rgb=render(f.rasterizer,combine(scene,own)).clamp(0,1)
        fixed_rgb=render(f.rasterizer,combine(scene,ref_head)).clamp(0,1)
        head_rgb=render(f.rasterizer,own)
        opaque=array(render(f.rasterizer,own,alpha_only=True))
        visible=array(visible_head(f.rasterizer,combine(scene,own),scene_count))
        allowed=array(f.masks['head_allowed'])
        train_photo=allowed&(native>252.45)
        clean=allowed&(alpha>.9)
        background=train_photo&(alpha<.05)
        head_color=array(head_rgb)/np.maximum(opaque,1e-4)[None]
        l1=np.abs(head_color-array(f.reference)).mean(0)
        target_m=alpha>=.5
        masks={k:mesh_mask(verts,faces,f,native.shape) for k,verts in [('seed',zero),('mesh',v),('fixed_mesh',ref_v)]}
        masks.update(matte=target_m,gauss=opaque>=.5,train=train_photo)
        crops={k:m[box] for k,m in masks.items()}
        ys,xs=np.where(crops['matte'])
        # Restrict to the upper dome. Identical central columns for every contour.
        ys_top,xs_top=np.where(crops['matte'] & (np.indices(crops['matte'].shape)[0]<=ys.min()+.27*np.ptp(np.where(crops['mesh'])[0])))
        cols=np.arange(int(np.quantile(xs_top,.15)),int(np.quantile(xs_top,.85))+1)
        truth=top_curve(crops['matte'],cols)
        errors={}
        for k,m in crops.items():
            diff=top_curve(m,cols)-truth
            live=np.isfinite(diff)
            errors[k]=dict(signed_median=float(np.median(diff[live])),abs=summary(np.abs(diff[live])),missing=int((~live).sum()))
        pose_delta=np.linalg.norm(project(v,f.rasterizer.calibration)-project(ref_v,f.rasterizer.calibration),axis=1)
        ear_id=np.array(EAR_IDS)
        ix=list(byname).index(name)
        _,_,r,c=f.rasterizer.calibration
        ears={}
        ear_visibility={}
        for kind,geom in [('own',v),('fixed',ref_v)]:
            pts=geom[ear_id]
            valid=visible_points((geom-c)@r,(pts-c)@r,faces,K) & (det['scores'][ix,ear_ix]>0) & (det['flip_disagreement'][ix,ear_ix]<=3)
            errs=np.linalg.norm(project(pts,f.rasterizer.calibration)-det['keypoints'][ix,ear_ix],axis=1)
            ears[kind]={n:float(e) for n,e,ok in zip(['left','right'],errs,valid) if ok}
            if kind=='own':
                yy,xx=np.ogrid[:size[1],:size[0]]
                radius=max(4,round(.055*np.ptp(project(v,f.rasterizer.calibration)[:,1])))
                for label,(x,y),ok in zip(['left','right'],det['keypoints'][ix,ear_ix],valid):
                    if not ok:
                        continue
                    region=clean & ((xx-x)**2+(yy-y)**2<=radius**2)
                    if region.any() and opaque[region].sum()>0:
                        ear_visibility[label]=dict(radius_px=radius,n=int(region.sum()),own_alpha=float(opaque[region].mean()),
                            blocked_fraction=float(1-visible[region].sum()/opaque[region].sum()))
        def mae(im):
            return float(np.abs(array(im-f.reference)[:,clean]).mean()*255)
        report['frames'][name]=dict(split='holdout' if name in args.holdout else 'train',dilated_mask_warp_iou=match,
               train_photo_pixels=int(train_photo.sum()),train_photo_raw_alpha_lt_005=float((alpha[train_photo]<.05).mean()),
               background_band_l1_share=float(l1[background].sum()/l1[train_photo].sum()),
               head_l1_gradient_split=l1_gradient_split(actor,f,train_photo,background),
               background_band_with_head_alpha_gt_001=float((opaque[background]>.01).mean()),
               ear_proxy_visibility=ear_visibility,
               clean_head_scene_occlusion=float(1-visible[clean].sum()/opaque[clean].sum()),
               clean_head_rgb_mae255=dict(own=mae(own_rgb),fixed=mae(fixed_rgb)),
               top_edge_px=errors,ear_proxy_px=ears,mesh_change_from_fixed_projection_px=summary(pose_delta),
               raw_matte_025_to_075_edge_px=summary(np.abs(top_curve(alpha[box]>.25,cols)-top_curve(alpha[box]>.75,cols))))
        images=[rgb[box],contour_panel(rgb[box],[(crops['train'],(180,50,235)),(crops['matte'],(0,231,100))]),
                contour_panel(rgb[box],[(crops['matte'],(0,231,100)),(crops['mesh'],(0,180,255))]),
                contour_panel(rgb[box],[(crops['matte'],(0,231,100)),(crops['gauss'],(255,195,55))]),
                to_bgr(own_rgb)[box],to_bgr(fixed_rgb)[box]]
        labels=[name,'Matte / RGB support','Matte / mesh','Matte / head alpha','Own pose','Fixed '+target.name[:5]]
        row=[]
        for im,label in zip(images,labels):
            tile=cv2.resize(im,None,fx=1.4,fy=1.4,interpolation=cv2.INTER_LINEAR)
            tile=cv2.copyMakeBorder(tile,24,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,label,(4,17),cv2.FONT_HERSHEY_SIMPLEX,.47,(250,250,250),1,cv2.LINE_AA)
            row.append(tile)
        if not cv2.imwrite(str(out/(name[:-4]+'.png')),np.hstack(row)):
            raise OSError(name)
    (out/'surface-fit.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    run(args.run.resolve(),args.output.resolve())
