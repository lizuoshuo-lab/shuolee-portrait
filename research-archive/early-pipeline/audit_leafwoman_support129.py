"""Read-only CPU support/pruning audit of the actual129 final snapshot.

Projected centers are explicitly a proxy, not visibility, ray contribution or
an approved removal mask. No model, source mask or training code is written.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

import numpy as np
from PIL import Image,ImageFilter
import cv2

BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/leafwoman-half12-paired18k129')
SNAP=BASE/'gaussian-run/step-018000'
MODEL=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
H=Path(__file__).parent


def local(path):
    if path.startswith('/mnt/c/'):return Path('C:/'+path[7:])
    if path.startswith('/mnt/d/'):return Path('D:/'+path[7:])
    if path.startswith('/opt/'):return Path('//wsl.localhost/Ubuntu-20.04'+path)
    return Path(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8'))


def ply(path):
    types={'float':'<f4','float32':'<f4','double':'<f8','int':'<i4','uint':'<u4','uchar':'u1'}
    with Path(path).open('rb') as f:
        assert f.readline()==b'ply\n';properties=[];count=None;vertex=False
        for raw in f:
            line=raw.decode('ascii').strip();parts=line.split()
            if line.startswith('format '):assert line=='format binary_little_endian 1.0'
            elif line.startswith('element '):
                vertex=parts[1]=='vertex'
                if vertex:count=int(parts[2])
            elif line.startswith('property ') and vertex:
                assert parts[1] in types;properties.append((parts[2],types[parts[1]]))
            elif line=='end_header':break
        assert count is not None
        data=np.fromfile(f,dtype=np.dtype(properties),count=count)
        assert len(data)==count
    return data


def xyz(data):return np.stack([data[k] for k in ('x','y','z')],1).astype(np.float64)
def normalize(x):return x/np.sqrt(np.maximum((x*x).sum(1,keepdims=True),1e-20))
def q(x):return np.quantile(x,[0,.05,.5,.95,1]).tolist() if len(x) else None


def geometry(vertices,faces):
    tri=vertices[faces];edge=tri[:,1]-tri[:,0]
    a0=normalize(edge);a1=normalize(np.cross(a0,tri[:,2]-tri[:,0]));a2=-normalize(np.cross(a1,a0))
    rotation=np.stack((a0,a1,a2),2)
    scale=(np.linalg.norm(edge,axis=1)+np.abs((a2*(tri[:,2]-tri[:,0])).sum(1)))/2
    return tri.mean(1),rotation,np.maximum(scale,1e-8)


def classify(points,K,mask,validity,crop):
    uvh=points@K.T;uv=uvh[:,:2]/uvh[:,2:]
    ix=np.floor(uv[:,0]+.5).astype(np.int64);iy=np.floor(uv[:,1]+.5).astype(np.int64)
    native=(points[:,2]>0)&np.isfinite(uv).all(1)&(ix>=0)&(ix<mask.shape[1])&(iy>=0)&(iy<mask.shape[0])
    roi=native&(ix>=crop[0])&(ix<crop[2])&(iy>=crop[1])&(iy<crop[3])
    foreground=np.zeros(len(points),bool);background=foreground.copy()
    indices=np.flatnonzero(roi)
    known=validity[iy[indices],ix[indices]]>0
    foreground[indices]=known&(mask[iy[indices],ix[indices]]>0)
    background[indices]=known&(mask[iy[indices],ix[indices]]==0)
    return native,roi,foreground,background


def main():
    sources=read(BASE/'source-native-inputs.json')['records'];rows=read(BASE/'data/training-inputs.json')['records']
    mappings=read(BASE/'pixel-mapping.json')['records'];review=read(SNAP/'review/result.json')
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:binding=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:mesh=dict(z)
    world,bound=ply(SNAP/'world.ply'),ply(SNAP/'bound-local.ply')
    ids=binding['triangle_ids'];faces=mesh['faces'].astype(np.int64);local_xyz=xyz(bound);opacity=1/(1+np.exp(-world['opacity'].astype(np.float64)))
    assert len(ids)==len(world)==len(bound)==221600
    assert np.array_equal(bound['binding_0'].astype(np.int64),ids)
    assert np.array_equal(world['opacity'],bound['opacity'])
    assert binding['frame_ids'].astype(str).tolist()==mesh['frame_ids'].astype(str).tolist()==[r['frame'] for r in sources]
    counts=np.bincount(ids,minlength=len(faces));assert len(counts)==20908 and (counts>=1).all()
    low=opacity<.005;low_count=np.bincount(ids[low],minlength=len(faces))
    veto_faces=(counts-low_count)==0;vetoed=low&veto_faces[ids]
    with np.load(MODEL,allow_pickle=False) as z:weights=z['weights']
    joint=weights[faces].mean(1).argmax(1)
    lower_face=np.isin(joint,[4,5,7,8,10,11]);lower_points=lower_face[ids]
    ever=[np.zeros(len(ids),bool) for _ in range(4)];face_ever=[np.zeros(len(faces),bool) for _ in range(4)]
    per=[];parity=None
    immutable=[SNAP/'world.ply',SNAP/'bound-local.ply',SNAP/'binding.npz',SNAP/'current-meshes.npz',BASE/'pixel-mapping.json',BASE/'data/training-inputs.json',BASE/'source-native-inputs.json',BASE/'adapter.py',MODEL]
    before={str(p):sha(p) for p in immutable}
    for i,(source,row,mapping) in enumerate(zip(sources,rows,mappings)):
        assert source['frame']==row['frame']==mapping['frame'] and np.array_equal(source['w2c'],np.eye(4))
        mask=np.asarray(Image.open(local(source['mask'])).convert('L'));valid=np.asarray(Image.open(local(source['validity'])).convert('L'))
        dm=Image.open(local(row['mask'])).convert('L');dv=np.asarray(Image.open(local(row['validity'])).convert('L'))
        assert np.array_equal(np.asarray(dm)[::2,::2],mask[mapping['crop_xyxy'][1]:mapping['crop_xyxy'][3],mapping['crop_xyxy'][0]:mapping['crop_xyxy'][2]])
        assert np.all(valid==255) and np.all(dv==255)
        center,rotation,scale=geometry(mesh['vertices_camera'][i],faces)
        point=center[ids]+np.einsum('nij,nj->ni',rotation[ids],local_xyz)*scale[ids,None]
        if i==0:
            err=np.abs(point-xyz(world));parity=dict(max_abs_m=float(err.max()),mean_abs_m=float(err.mean()),existing_bridge_tolerance_m=2e-6)
            assert err.max()<2e-6,'CPU point transport oracle failed; do not interpret support proxy.'
        point_support=classify(point,np.asarray(source['K_integer_centers']),mask,valid,mapping['crop_xyxy'])
        face_support=classify(center,np.asarray(source['K_integer_centers']),mask,valid,mapping['crop_xyxy'])
        for a,b in zip(ever,point_support):a|=b
        for a,b in zip(face_ever,face_support):a|=b
        alpha_path=SNAP/(row['frame']+'_cumulative-alpha.npy');alpha=np.load(alpha_path,allow_pickle=False)
        assert sha(alpha_path)==read(SNAP/(row['frame']+'_cumulative-alpha.json'))['alpha_npy_sha256']
        foreground=np.asarray(dm)>0;interior=np.asarray(dm.filter(ImageFilter.MinFilter(11)))>0
        record=next(x for x in review['records'] if x['frame']==row['frame'])
        arms=next(x for x in record['images'] if x['region']=='arms')['crop_original_xyxy'];x0,y0,x1,y1=mapping['crop_xyxy']
        ax0,ay0,ax1,ay1=[2*(v-o) for v,o in zip(arms,[x0,y0,x0,y0])]
        arm_region=np.zeros(foreground.shape,bool);arm_region[max(0,ay0):min(alpha.shape[0],ay1),max(0,ax0):min(alpha.shape[1],ax1)]=True
        holes=interior&(alpha<.5);components=[]
        num,labels,stats,_=cv2.connectedComponentsWithStats(holes.astype(np.uint8),8)
        for n in np.argsort(stats[1:,4])[::-1][:3]+1:
            xx,yy,ww,hh,area=stats[n].tolist();components.append(dict(derived_pixels=area,native_bbox_xyxy=[x0+xx/2,y0+yy/2,x0+(xx+ww)/2,y0+(yy+hh)/2]))
        per.append(dict(frame=row['frame'],source_wh=list(mask.shape[::-1]),derived_wh=row['wh'],source_validity_all255=True,derived_validity_all255=True,all_foreground_retained_by_crop=True,
            point_center_in_ROI=int(point_support[1].sum()),point_center_on_known_foreground=int(point_support[2].sum()),
            source_mask_pixels=int(foreground.sum()),eroded5_derived_foreground_pixels=int(interior.sum()),
            eroded_foreground_alpha_lt_0_5_pixels=int(holes.sum()),
            arms_ROI_eroded_foreground_pixels=int((interior&arm_region).sum()),arms_ROI_eroded_alpha_lt_0_5_pixels=int((holes&arm_region).sum()),
            largest_internal_alpha_components=components))
    def group(mask):
        return dict(points=int(mask.sum()),opacity_quantiles=q(opacity[mask]),low_opacity_lt_0_005=int((low&mask).sum()),
            opacity_only_prune_vetoed_by_nonempty_face_rule=int((vetoed&mask).sum()),
            center_never_in_training_ROI=int((mask&~ever[1]).sum()),center_never_on_known_foreground=int((mask&~ever[2]).sum()))
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='read_only_CPU_support_audit',step=18000,optimizer_updates=17999,
        inputs_sha256=before,helper_sha256=sha(__file__),CPU_point_transport_oracle=parity,
        actual_loss='Full derived ROI RGB:0.8 L1+0.2 DSSIM; paired Uniform RGB prediction/target backgrounds. All validity pixels255. Source joint-support flags do not disable GS photometric pixels.',
        supervision_scope='Mask foreground is positive RGB supervision and ROI mask-background is paired-background negative RGB supervision. Outside derived ROI/source raster has no image loss. Projected hidden body/leg triangles may fall onto photographed dress and therefore receive dress RGB, not true anatomical supervision.',
        no_direct_alpha_or_mesh_silhouette_or_observed3D_surface_loss=True,
        binding=dict(points=len(ids),triangles=len(counts),covered_triangles=int((counts>0).sum()),minimum_points_per_triangle=int(counts.min()),single_point_faces=int((counts==1).sum()),
            all_below_0_005_faces_current=int(veto_faces.sum()),opacity_only_prune_candidates=int(low.sum()),opacity_only_candidates_vetoed_by_nonempty_face_rule=int(vetoed.sum()),
            caveat='Exact current-state simulation of original opacity-only prune mask, not historical proof each point was vetoed at its last pruning step. Original rule can retain every proposed point on a face if all would be removed.'),
        point_groups=dict(all=group(np.ones(len(ids),bool)),lower_limb_binding_proxy=group(lower_points)),
        source_center_proxy=dict(point_centers_never_in_any_training_ROI=int((~ever[1]).sum()),face_centers_never_in_any_training_ROI=int((~face_ever[1]).sum()),
            face_centers_never_on_any_known_foreground=int((~face_ever[2]).sum()),
            limitation='Nearest projected centers only; not ellipsoid footprints, occlusion, rays or physical surface support. No point is removed or labeled physically wrong solely from this proxy.'),
        per_frame=per,
        minimal_next_diagnostic='Keep immutable model; reproducibly annotate Gaussian/triangle IDs with any-view known-foreground/ROI support using all12 articulated frames and fullK. For a later single native isolation render, compare low-support lower-binding group versus complement on same cameras and frozen RGB/alpha. Do not auto-prune from center masks, unknown feet/back labels or per-point opacity.',
        minimal_filter_candidate='Only after footprint/ray support verification, a separate reversible preview/export confidence filter could suppress points with zero contribution to every known valid training foreground and confirmed unsupported extent. Preserve full model and same-coordinate cameras, compare all12 and unknown views; do not change training/prune floor or force alpha1 in this audit.',
        filter_risks='Can erase valid unobserved back/occluded clothing and silhouette splats; support is not visibility. It cannot fill missing positive-mask skirt pixels or correct face/hand ghosting.',
        quality_accepted=False,GPU_used=False,training_or_model_mutated=False,filter_implemented=False)
    assert before=={str(p):sha(p) for p in immutable}
    dest=BASE/'diagnostics/source-support-audit129.json';dest.parent.mkdir(exist_ok=True);assert not dest.exists();dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(dest),sha256=sha(dest),binding=result['binding'],point_groups=result['point_groups'],source_center_proxy=result['source_center_proxy']),ensure_ascii=False))


if __name__=='__main__':main()
