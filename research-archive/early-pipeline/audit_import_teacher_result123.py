"""Read-only CPU interpretation of the stopped original-import teacher pilot."""
from datetime import datetime,timezone
import numpy as np
import audit_gaussianavatars_params123 as a
from publish_import_teacher123 import OUT,PILOT

def summary(x):
    x=np.asarray(x)
    return dict(min=float(x.min()),median=float(np.median(x)),mean=float(x.mean()),max=float(x.max()))

def main():
    r=a.read(PILOT/'result.json');assert r['state']=='stopped_by_geometry_guard' and r['steps']==520
    with np.load(a.lp(PILOT/'candidate-diagnostic-only.npz'),allow_pickle=False) as z:c=dict(z)
    with np.load(a.lp(OUT/'teacher-support-diagnostic-only.npz'),allow_pickle=False) as z:t=dict(z)
    b,cand=c['vertices_before'],c['vertices_after'];faces=c['faces']
    btri=b[:,faces];ctri=cand[:,faces]
    bn=np.cross(btri[:,:,1]-btri[:,:,0],btri[:,:,2]-btri[:,:,0])
    cn=np.cross(ctri[:,:,1]-ctri[:,:,0],ctri[:,:,2]-ctri[:,:,0])
    ba=np.linalg.norm(bn,axis=-1);ca=np.linalg.norm(cn,axis=-1)
    dot=(bn*cn).sum(-1)
    def q(tri,normal):
        edges=np.roll(tri,1,axis=-2)-tri
        return 2*np.sqrt(3)*np.linalg.norm(normal,axis=-1)/np.square(edges).sum((-2,-1)).clip(1e-30)
    bq,cq=q(btri,bn),q(ctri,cn)
    flips=[]
    for fi,face in np.argwhere((dot<=0)&(ba>1e-12)):
        world=(btri[fi,face].mean(0));E=c['w2c'][fi];K=c['K'][fi]
        cam=E[:3,:3]@world+E[:3,3];pixel=K@cam;pixel=pixel[:2]/pixel[2]
        flips.append(dict(frame=str(c['frame_ids'][fi]),triangle=int(face),vertices=faces[face].tolist(),
            base_quality=float(bq[fi,face]),candidate_quality=float(cq[fi,face]),area_ratio=float(ca[fi,face]/ba[fi,face]),
            normal_cos=float(dot[fi,face]/(ba[fi,face]*ca[fi,face])),original_center_pixel=pixel.tolist(),
            limitation='Normal change relative to this initial frame; this is not independently tested global self-intersection.'))
    bary=t['barycentric_per_frame'];ids=faces[t['binding']]
    start=(b[:,ids]*bary[:,:,:,None]).sum(2);end=(cand[:,ids]*bary[:,:,:,None]).sum(2)
    weights=t['weights'];normal=bn[:,t['binding']]/ba[:,t['binding'],None].clip(1e-30)
    goal_delta=t['targets']-start;move=end-start
    goal_sign=(goal_delta*normal).sum(-1);move_sign=(move*normal).sum(-1)
    per_frame=[]
    for i,fid in enumerate(c['frame_ids']):
        weight=weights[i];assert np.isclose(weight.sum(),1)
        per_frame.append(dict(frame=str(fid),positive_normal_teacher_weight=float(weight[goal_sign[i]>0].sum()),
            negative_normal_teacher_weight=float(weight[goal_sign[i]<0].sum()),
            guide_distance_before_mm=float((np.linalg.norm(goal_delta[i],axis=-1)*weight).sum()*1000),
            guide_distance_after_mm=float((np.linalg.norm(t['targets'][i]-end[i],axis=-1)*weight).sum()*1000),
            weighted_normal_movement_mm=float((move_sign[i]*weight).sum()*1000)))
    changes=np.array([x['iou_change'] for x in r['mask_evaluation']])
    outside=np.array([x['outside_change'] for x in r['mask_evaluation']])
    magnitude=np.linalg.norm(c['additional_canonical_delta'],axis=-1)*1000
    value=dict(updated=datetime.now(timezone.utc).isoformat(),state='stopped_candidate_cpu_review_complete',
        frames=59,steps=r['steps'],guide_reduction_fraction=r['guide_reduction_fraction'],flipped_frame_faces=flips,
        clothing_vertex_movement_mm=summary(magnitude[c['learnable_vertices']]),
        mask_iou_change=summary(changes),mask_iou_improved_frames=int((changes>0).sum()),
        mask_iou_worsened_frames=int((changes<0).sum()),
        outside_ratio_change=summary(outside),outside_ratio_improved_frames=int((outside<0).sum()),
        negative_normal_teacher_weight=summary([x['negative_normal_teacher_weight'] for x in per_frame]),
        weighted_normal_movement_mm=summary([x['weighted_normal_movement_mm'] for x in per_frame]),
        per_frame=per_frame,source_result_sha256=a.sha(PILOT/'result.json'),
        candidate_sha256=a.sha(PILOT/'candidate-diagnostic-only.npz'),
        evidence_boundary=['Teacher targets are learned Gaussian centers, not measured clothing-surface samples.',
            'Normal sign follows original triangle winding; no independent outward-normal or material-surface acceptance.',
            'Gaussian finite extent, covariance and opacity mean apparent rendered clothing need not coincide with the Gaussian mean.',
            'A smaller center-to-mesh loss cannot override the unchanged normal and silhouette protection gates.'],
        no_new_optimizer_updates=True,quality_accepted=False,model_promoted=False)
    a.save(PILOT/'cpu-result-review.json',value)
    print({k:v for k,v in value.items() if k not in ['per_frame','evidence_boundary']})

if __name__=='__main__':main()
