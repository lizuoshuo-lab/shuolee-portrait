"""Read existing heldout residuals; never fit/modify cameras, points or images."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, math, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ROOT=Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/raw-domain-121')

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(v):
    a=np.asarray(v,float)
    if not len(a): return {'n':0}
    return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'max':float(a.max())}

plan_path=R/'fisheye-residual-review-plan.json'
plan={
    'registered':datetime.now(timezone.utc).isoformat(),
    'authorization':'User asks to continue fisheye analysis and confirms no stabilization was added.',
    'input_confirmation':'All four MP4s already undistorted, downsampled and cropped; no stabilization added per user.',
    'scope':'Read only previous heldout predictions. No new feature extraction, triangulation, fitting, threshold selection or training.',
    'tests':[
        'Compare heldout face ROI and retained static-support image-position/radius coverage using all prior predicted observations.',
        'For every face observation find nearest retained static-support observation in the SAME frame, without spatial filtering.',
        'Resolve residual into principal-point radial and tangential components; quantify radial energy descriptively, not a new distortion fit.',
        'Summarize per-frame mean vector agreement across >=3 face observations as descriptive evidence only; points/frames are correlated.',
        'Compute analytic sensitivity to focal scale, principal point, angle and translation; these are one-source examples, not a joint accuracy budget.'
    ],
    'stop':'One pass, existing sofa and portrait diagnostic results only. No cameras for grass/indoor means not testable; do not rerun their workers.',
    'causal_limits':[
        'Face ROI includes headwear and occasional background and is not physical ground truth.',
        'The existing cameras/triangulated points can absorb lens error, so nonradial LOO residuals cannot disprove fisheye error.',
        'Separated scene regions/depths cannot distinguish a camera model error from object motion.',
        'No stabilization per user removes EIS as an assumed processing cause, not sensor rolling shutter or actual camera/subject motion.'
    ],
    'training_started':False
}
if plan_path.exists():
    raise RuntimeError('One-pass analysis already registered; inspect the existing outputs instead of repeating it.')
save(plan_path,plan)

summaries=[]
for subject in ['sofa','portrait']:
    pp=ROOT/subject/'global-fixedk/heldout-predictions.json'
    cp=ROOT/subject/'global-fixedk/cameras-diagnostic.json'
    op=ROOT/subject/'roi-supplement/observations.json'
    hashes={str(p):sha(p) for p in [pp,cp,op]}
    pred=read(pp);cams=read(cp);obs=read(op)
    K=np.array(cams[0]['K_integer_centers']);wh=np.array(obs['records'][0]['wh']);ppoint=K[:2,2]
    groups={}
    for name in ['face_roi','body_roi','background_static_support_candidate','background_excluded_material']:
        q=[p for p in pred if p['state']=='predicted' and p['region']==name]
        if not q: continue
        uv=np.array([p['observed_uv'] for p in q]);projected=np.array([p['predicted_uv'] for p in q]);d=projected-uv
        radius=np.linalg.norm(uv-ppoint,axis=1);rd=(uv-ppoint)/np.maximum(radius[:,None],1e-12)
        tang=np.column_stack([-rd[:,1],rd[:,0]]);radial=np.sum(d*rd,axis=1);tangent=np.sum(d*tang,axis=1)
        perframe=[]
        for frame in sorted({p['frame'] for p in q}):
            ix=np.array([p['frame']==frame for p in q]);ds=d[ix];norm=np.linalg.norm(ds,axis=1)
            perframe.append({'frame':frame,'n':int(ix.sum()),'mean_dx_dy':ds.mean(axis=0).tolist(),
                             'direction_agreement_norm_mean_over_mean_norm':float(np.linalg.norm(ds.mean(axis=0))/max(norm.mean(),1e-12)),
                             'error_px':stats(norm)})
        groups[name]={
            'observations':len(q),'tracks':len({p['track_id'] for p in q}),
            'u':stats(uv[:,0]),'v':stats(uv[:,1]),'normalized_u':stats(uv[:,0]/wh[0]),'normalized_v':stats(uv[:,1]/wh[1]),
            'radius_from_conditional_principal_point_px':stats(radius),
            'error_px':stats(np.linalg.norm(d,axis=1)),
            'abs_radial_component_px':stats(abs(radial)),'abs_tangential_component_px':stats(abs(tangent)),
            'radial_fraction_of_squared_vector_energy':float(np.sum(radial**2)/max(np.sum(d*d),1e-12)),
            'positive_radial_fraction':float(np.mean(radial>0)),
            'per_frame':perframe,
            'interpretation':'Descriptive on conditional cameras; not a fitted lens model or accepted physical material.'
        }
    face=[p for p in pred if p['state']=='predicted' and p['region']=='face_roi']
    bg=[p for p in pred if p['state']=='predicted' and p['region']=='background_static_support_candidate']
    distances=[];pairs=[]
    for p in face:
        q=[b for b in bg if b['frame']==p['frame']]
        if not q: continue
        uv=np.array(p['observed_uv']);ds=np.linalg.norm(np.array([b['observed_uv'] for b in q])-uv,axis=1);j=int(ds.argmin())
        distances.append(float(ds[j]));pairs.append({'frame':p['frame'],'face_track':p['track_id'],'background_track':q[j]['track_id'],'distance_px':float(ds[j])})
    samples=[x['direction_agreement_norm_mean_over_mean_norm'] for x in groups['face_roi']['per_frame'] if x['n']>=3]
    face_r=groups['face_roi']['radius_from_conditional_principal_point_px']
    fx=float(K[0,0]);fy=float(K[1,1]);rmax=face_r['max']
    summary={
        'subject':subject,'source_sha256':hashes,'image_wh':wh.tolist(),'K_integer_centers':K.tolist(),'groups':groups,
        'face_to_static_support_same_frame':{'observations_with_control':len(distances),'nearest_distance_px':stats(distances),'pairs':pairs},
        'face_direction_agreement_for_frames_with_at_least_3_observations':stats(samples),
        'analytic_sensitivity_not_estimation':{
            'same_ray_1percent_focal_scale_shift_at_face_radius_px':{'median':face_r['median']*.01,'max':rmax*.01},
            'isolated_uniform_focal_relative_error_for_1px_at_largest_observed_face_radius':1/rmax,
            'optical_axis_small_angle_degrees_for_1px_horizontal':math.degrees(1/fx),
            'lateral_translation_millimeters_for_1px_at_example_depth_2m':2000/fx,
            'depth_2m_is_example_not_measured':True,
            'principal_point_vector_shift_1px_produces_1px_same_ray_shift':True,
            'do_not_sum_or_assign_full_1px_to_each_source':True
        },
        'accepted':False,'training_started':False
    }
    assert all(sha(Path(p))==h for p,h in hashes.items())
    summaries.append(summary)

save(R/'fisheye-residual-review.json',{
    'updated':datetime.now(timezone.utc).isoformat(),'plan':plan_path.name,'subjects':summaries,
    'stabilization_added':False,'stabilization_source':'Direct user confirmation, 2026-10-27',
    'cameras_changed':False,'images_changed':False,'new_features':False,'training_started':False,
    'physical_root_cause_unique':False
})
for s in summaries:
    print(json.dumps({
        'subject':s['subject'],'face_uv_range':{k:[s['groups']['face_roi'][k]['min'],s['groups']['face_roi'][k]['max']] for k in ['u','v']},
        'static_uv_range':{k:[s['groups']['background_static_support_candidate'][k]['min'],s['groups']['background_static_support_candidate'][k]['max']] for k in ['u','v']},
        'face_radius':s['groups']['face_roi']['radius_from_conditional_principal_point_px'],
        'control_distance':s['face_to_static_support_same_frame']['nearest_distance_px'],
        'face_radial_energy_fraction':s['groups']['face_roi']['radial_fraction_of_squared_vector_energy'],
        'face_tangential_error':s['groups']['face_roi']['abs_tangential_component_px'],
        'sensitivity':s['analytic_sensitivity_not_estimation']
    },ensure_ascii=False))
