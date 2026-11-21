"""Read-only review of the stopped mask pilot and inherited triangle geometry."""
from datetime import datetime,timezone
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a

BASE=a.REC/'demo-vhap-initialization'
OUT=BASE/'body-gaussian-upstream-offset/gaussian-mesh-guidance-audit/mask-clothing-surface-590'


def stats(x):
    x=np.asarray(x);return dict(min=float(x.min()),median=float(np.median(x)),mean=float(x.mean()),max=float(x.max()))


def measures(t):
    n=np.cross(t[...,1,:]-t[...,0,:],t[...,2,:]-t[...,0,:])
    edges=np.linalg.norm(np.roll(t,1,axis=-2)-t,axis=-1)
    twice_area=np.linalg.norm(n,axis=-1)
    q=2*np.sqrt(3)*twice_area/np.square(edges).sum(-1).clip(1e-30)
    return n,edges,twice_area,q


def main():
    now=datetime.now(timezone.utc).isoformat();result=a.read(OUT/'result.json')
    assert result['steps']==80 and result['state']=='stopped_by_geometry_guard'
    with np.load(a.lp(OUT/'candidate-diagnostic-only.npz')) as z:c=dict(z)
    before,after=c['vertices_before'],c['vertices_after'];faces=c['faces']
    bn,be,ba,bq=measures(before[:,faces]);an,ae,aa,aq=measures(after[:,faces])
    dot=np.sum(bn*an,-1);changed=np.argwhere((dot<=0)&(ba>1e-12));flips=[]
    for frame,face in changed:
        vid=faces[face];tri=before[frame,vid];center=tri.mean(0)
        p=c['K'][frame]@(c['w2c'][frame,:3,:3]@center+c['w2c'][frame,:3,3]);p=p[:2]/p[2]
        flips.append(dict(frame=str(c['frame_ids'][frame]),triangle=int(face),vertices=vid.tolist(),
            initial_quality=float(bq[frame,face]),candidate_quality=float(aq[frame,face]),
            initial_edge_lengths_mm=(be[frame,face]*1000).tolist(),
            initial_area_mm2=float(ba[frame,face]/2*1e6),candidate_area_mm2=float(aa[frame,face]/2*1e6),
            shortest_altitude_mm=float(ba[frame,face]/be[frame,face].max()*1000),
            normal_cos=float(dot[frame,face]/(ba[frame,face]*aa[frame,face])),
            vertex_movement_mm=(np.linalg.norm(after[frame,vid]-tri,axis=-1)*1000).tolist(),native_center_xy=p.tolist()))
    trace=[]
    for label in ('initial','epoch-30','epoch-60','epoch-90','epoch-120'):
        f=BASE/'smplify-vhap-pose-prior-long'/label/'tracked-smplx-params.npz'
        with np.load(a.lp(f)) as z:ids=z['frame_ids'].astype(str).tolist();verts=z['vertices_camera'];topology=z['faces']
        assert np.array_equal(topology,faces)
        n,e,area,q=measures(verts[:,faces]);i=ids.index('frame_000036');j=1508
        trace.append(dict(checkpoint=label,path=str(f),sha256=a.sha(f),
            all_frame_triangle_quality_min=float(q.min()),very_thin_frame_faces_q_lt_001=int((q<.01).sum()),
            triangle1508_frame36=dict(quality=float(q[i,j]),area_mm2=float(area[i,j]*.5e6),
                edges_mm=(e[i,j]*1000).tolist(),altitude_mm=float(area[i,j]/e[i,j].max()*1000))))
    iou=[x['iou_change'] for x in result['mask_evaluation']];outside=[x['outside_change'] for x in result['mask_evaluation']]
    review=dict(updated=now,state='stopped_candidate_readonly_review',steps=80,frames=59,
        mask_loss_reduction_fraction=result['mask_loss_reduction_fraction'],all59_iou_change=stats(iou),
        iou_improved_frames=int(np.sum(np.array(iou)>0)),outside_change=stats(outside),
        face_motion_m=result['geometry']['face_displacement_max_abs_m'],relative_normal_reversals=flips,
        checkpoint_geometry_trace=trace,source_result_sha256=a.sha(OUT/'result.json'),
        scope=['Relative face-normal reversal is not a tested global self-intersection.',
            'Checkpoint geometry changes include pose/shape/offset changes, so this trace alone does not isolate which parameter caused them.',
            'All59 mask improvement establishes silhouette change, not physical depth or face/hand fit.',
            'The new user-requested SMPLify-X direct Gaussian route supersedes further optimization of this VHAP-derived pilot.'],
        quality_accepted=False,model_promoted=False,new_optimization=False)
    a.save(OUT/'cpu-review.json',review)
    overview=[];ims=sorted(result['images'],key=lambda x:x['frame'])
    for start in range(0,len(ims),10):
        tiles=[]
        for row in ims[start:start+10]:
            f=row['record_path'].replace('/mnt/d/','D:/');im=Image.open(f).convert('RGB')
            im.thumbnail((700,440));tile=Image.new('RGB',(700,440),'white');tile.paste(im,(0,0));tiles.append(tile)
        page=Image.new('RGB',(1400,440*((len(tiles)+1)//2)),'white')
        for i,tile in enumerate(tiles):page.paste(tile,((i%2)*700,(i//2)*440))
        path=OUT/f'overview-{start+1:02d}-{start+len(tiles):02d}.jpg';page.save(a.lp(path),quality=92)
        overview.append(dict(path=str(path),frames=[x['frame'] for x in ims[start:start+10]],resized_for_overview_only=True))
    a.save(OUT/'overview-manifest.json',dict(images=overview,native_record_images_unchanged=True,text_on_images=False))
    print(dict(iou_improved=review['iou_improved_frames'],flips=flips,
        trace=[dict(checkpoint=x['checkpoint'],triangle=x['triangle1508_frame36']) for x in trace]))


if __name__=='__main__':main()
