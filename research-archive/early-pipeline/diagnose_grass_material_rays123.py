"""One read-only fixed-camera diagnostic of already saved appearance tracks."""
import argparse,time,json
import numpy as np
from scipy.optimize import least_squares
from trial_grass_surface123 import ROOT,OUT,read,save,sha,now

O=OUT/'provenance-audit'
NOTES=[
('hair_neck_boundary','Hair/neck boundary follows a similar region; individual hair material and rigidity unverified.'),
('nose_region','Nose/nostril region stays visually coherent in six patches; no calibrated subpixel identity claim.'),
('nose_region','Another nose feature stays near the same nasal region; may be related to sample01, not independent truth.'),
('hair_shoulder_boundary','Hair/shoulder boundary; silhouette and hair displacement make exact physical identity uncertain.'),
('hair_tip','Hair curl tip is followed, with changing curl shape and occlusion.'),
('hand_hair_boundary','Hand gripping hair moves relative to clothes; cross lies at a hand/hair boundary.'),
('hair_tip','Lower hair curl remains recognizable; rigidity and exact fiber are unverified.'),
('lace_boundary','Repeated lace/neckline pattern near hand; physical patch correspondence ambiguous.'),
('toe_strap_region','A toe/strap junction appears coherent across side/front views; visibility changes.'),
('shoe_strap_corner','Upper strap corner appears coherent; visual inspection does not prove static 3D position.'),
('toe_region','Toe/nail region appears coherent over six views; no obvious cross-shoe switch observed.'),
('shoe_inner_edge','Inner shoe/ankle boundary; silhouette point is not necessarily a fixed material point.'),
('nose_contour','Nasal side/contour remains close; contour motion and exact material identity uncertain.'),
('shoulder_fold','Shoulder fabric/fold with no unique texture; material identity uncertain.'),
('hair_shoulder_boundary','Hair over shoulder; deformable occlusion boundary.'),
('hair_neck_boundary','Hair/neck shadow boundary; material identity uncertain.'),
('lace_hair_occlusion','Repeated lace patch partly covered by hair; exact physical patch uncertain.'),
('hair_strand','Long dark hair region, no unique fiber identification.'),
('hand_cuff','Hand/cuff region changes occlusion by dress; not a rigid reference.'),
('repeated_lace','Similar neckline lace elements; possible repeated-pattern ambiguity.'),
('toe_shoe_contour','Outer toe/shoe silhouette; view-dependent edge.'),
('shoe_strap_corner','Shoe strap corner appears coherent over the six displayed views.'),
('toe_strap_region','Toe/strap junction appears coherent; nearby features could belong to same physical family.'),
('toe_region','Toe region appears coherent, close to sample22; independence unproven.'),
('distant_tree_crown','Distant tree crown outline; leaves can move and texture repeats.'),
('tree_crown','Tree crown pattern visually similar; individual leaf identity unverified.'),
('tree_gap','Small gap in foliage, with foreground occlusion in one view; not a rigid fiducial.'),
('tree_crown_edge','Tree crown boundary; viewpoint and foliage movement can alter its locus.'),
('tree_branch','Thin tree branches/leaves, changing overlap; correspondence uncertain.'),
('branch_fork','Branch fork appears coherent, but tree rigidity not established.'),
('antenna_structure','Repeated antenna hardware appears coherent; exact edge and repeated elements require care.'),
('branch_fork','Near sample29 branch region; independence not established.'),
]

def stats(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    return dict(count=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(count=0)

def register():
    assert not (O/'ray-plan.json').exists()
    source=read(O/'result.json');assert len(source['samples'])==len(NOTES)==32
    save(O/'visual-review.json',dict(updated=now(),reviewer='Codex actual view_image inspection',
        all32_panels_opened=True,native_crop_size=[176,176],sample_nodes_max=6,all_observations_reviewed=False,
        samples=[dict(id=s['id'],file=s['file'],sha256=s['sha256'],held=s['held'],category=a,note=b)
                 for s,(a,b) in zip(source['samples'],NOTES)],
        conclusion='Some nose and shoe features appear coherent; hair, fabric, contour and foliage tracks are ambiguous. No universal physical identity or subpixel/static accuracy claim.',
        split_changed=False,point_exclusion=False,quality_accepted=False))
    save(O/'ray-plan.json',dict(registered=now(),hypothesis='Do original person tracks fail the saved cameras even without SMPL-X? Separate camera/track static-ray inconsistency from body-model constraints.',
        inputs=[dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in ['camera/solver-output.npz','camera/appearance-tracks.json']],
        selection='All tracks with any person-mask observation; original32 visual samples; first100 purely outsideFIT and first100 purely outsideHELD by SHA ID. No residual selection.',
        operation='Saved cameras andK stay fixed. Native adjacent-observation Sampson errors; DLT on even-indexed observations, optional point-only linear-loss refinement, then evaluate odd-indexed observations. No observations are dropped.',
        interpretation='Development diagnostic, not independent calibration truth. Mask regions and human visual checks do not establish rigidity or subpixel identity.',
        budget=dict(CPU_seconds=120,point_max_nfev=80,camera_updates=0,feature_passes=0,training_runs=0),
        stop='One completed run or budget/failure. Never use these points as an initializer. Preserve originalHELD split.'))
    print('Registered fixed-camera material-ray diagnostic.')

def run():
    plan=read(O/'ray-plan.json');assert not (O/'ray-result.json').exists();t=time.monotonic()
    for f in plan['inputs']:assert sha(f['path'])==f['sha256']
    labels=read(O/'classifications.json')['records'];sample_ids={r['id'] for r in read(O/'result.json')['samples']}
    chosen={r['id'] for r in labels if r['any_inside']}|sample_ids
    for held in [False,True]:chosen.update(r['id'] for r in sorted([r for r in labels if r['held']==held and not r['any_inside']],key=lambda r:r['id'])[:100])
    byid={r['id']:r for r in labels};tracks=read(ROOT/'camera/appearance-tracks.json')['records']
    saved=dict(np.load(ROOT/'camera/solver-output.npz'));K=saved['K'];E=saved['w2c'];P=K@E[:,:3];invK=np.linalg.inv(K)
    xy=[np.load(ROOT/f'camera/features-{i:03d}.npz')['xy'] for i in range(len(E))];records=[]
    for tr in tracks:
        if tr['id'] not in chosen:continue
        assert time.monotonic()-t<120,'CPU budget exceeded'
        ids=np.array([i for i,j in tr['nodes']]);uv=np.array([xy[i][j] for i,j in tr['nodes']]);Ps=P[ids]
        ep=[]
        for z in range(len(ids)-1):
            i,j=ids[z:z+2];R=E[j,:3,:3]@E[i,:3,:3].T;v=E[j,:3,3]-R@E[i,:3,3]
            tx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            F=invK.T@tx@R@invK;a=np.r_[uv[z],1];b=np.r_[uv[z+1],1];Fa=F@a;Fb=F.T@b
            den=np.sqrt(Fa[:2]@Fa[:2]+Fb[:2]@Fb[:2]);ep.append(float(abs(b@F@a)/max(den,1e-15)))
        anchors=np.arange(0,len(ids),2);checks=np.arange(1,len(ids),2)
        A=[]
        for z in anchors:A.extend([uv[z,0]*Ps[z,2]-Ps[z,0],uv[z,1]*Ps[z,2]-Ps[z,1]])
        _,_,vh=np.linalg.svd(A);point=vh[-1,:3]/vh[-1,3]
        def projection(x):
            q=np.einsum('nij,j->ni',Ps[:,:,:3],x)+Ps[:,:,3]
            z=q[:,2];safe=np.where(abs(z)<1e-9,np.where(z<0,-1e-9,1e-9),z)
            return q[:,:2]/safe[:,None],z
        point_refined=False
        if np.isfinite(point).all() and (projection(point)[1][anchors]>0).all():
            fit=least_squares(lambda x:(projection(x)[0][anchors]-uv[anchors]).ravel(),point,max_nfev=80,loss='linear')
            point=fit.x;point_refined=True
        pred,depth=projection(point);errors=np.linalg.norm(pred-uv,axis=1)
        meta=byid[tr['id']];records.append(dict(id=tr['id'],held=tr['held'],group='inside' if meta['all_inside'] else 'mixed' if meta['any_inside'] else 'outside',
            region=meta['region'],length=len(ids),camera_fit_selected=meta['camera_fit_selected'],sampson_adjacent_px=stats(ep),sampson_values_px=ep,
            anchors=ids[anchors].tolist(),checks=ids[checks].tolist(),point_refined=point_refined,positive_depth_all=bool((depth>0).all()),
            check_errors_px=errors[checks].tolist(),anchor_errors_px=errors[anchors].tolist(),check_summary=stats(errors[checks]),
            point_world_diagnostic_only=point.tolist()))
    aggregate=[]
    for held in [False,True]:
        for group in ['inside','mixed','outside']:
            rr=[r for r in records if r['held']==held and r['group']==group]
            aggregate.append(dict(split='HELD' if held else 'FIT',group=group,tracks=len(rr),positive_depth_tracks=sum(r['positive_depth_all'] for r in rr),
                sampson_adjacent_px=stats([v for r in rr for v in r['sampson_values_px']]),check_projection_px=stats([v for r in rr for v in r['check_errors_px']]),
                positive_depth_check_projection_px=stats([v for r in rr if r['positive_depth_all'] for v in r['check_errors_px']])))
    for f in plan['inputs']:assert sha(f['path'])==f['sha256']
    result=dict(updated=now(),runtime_seconds=time.monotonic()-t,aggregate=aggregate,records=records,
        original_camera_and_K_unchanged=True,no_training=True,no_source_changes=True,points_for_initialization=False,quality_accepted=False)
    save(O/'ray-result.json',result);print(json.dumps(dict(seconds=result['runtime_seconds'],aggregate=aggregate),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.register:register()
    elif a.run:run()
    else:p.error('Choose --register or --run.')
