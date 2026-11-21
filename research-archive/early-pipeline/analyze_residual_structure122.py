"""N7: one read-only spatial/depth/coherence audit of the frozen N5/N6 results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,traceback,sys
import numpy as np
from PIL import Image,ImageDraw

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
B=R/'video-update-20261027/background-geometry';G=B/'shared-path/solver-repair';N=B/'residual-radial/serialization-repair'
P=B/'residual-structure-plan.json';O=B/'residual-structure'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def register():
    assert not P.exists() and not O.exists()
    inputs=[G/'result.json',G/'fit-diagnostic.npz',N/'result.json',B/'observations.json',B/'visual-review.json',B/'comparison-review.json']
    save(P,dict(stage='N7_readonly_spatial_depth_residual',registered=now(),source_sha256=sha(Path(__file__)),
        preceding_goal_turn='progress: N5/N6 were actually reviewed/published; new camera-team/PDF evidence and4000basis persisted. No verified live fitting/training process at intake.',
        hypotheses=['A time-invariant outputwarp should leave a spatially reproducible signature, but radial rejection alone is insufficient.',
                    'Near-planar background support or large image-space distance fromface may permit off-plane pose error despite small background residuals.',
                    'Coherent per-view face residual can be caused by pose/K, head motion or trackingbias; coherence alone does not uniquely identify motion.'],
        actions=['PCA ofALL935savedFIT3D points, allvaliddepth/parallax summaries; nooutlierremoval.',
                 'All681background and57face target-excluded residual directions, spatialbins andperviewcentering asDESCRIPTIVEstats only.',
                 'For eachface target recompute existing target-excludedDLT and its fixed-camera linearized1px iid support-measurement sensitivity; this isnot empiricaluncertainty.',
                 'Native unscaled patches for5nearestheldoutbackground observations tofacecenter ineachview, chosen bydistance beforeerrorinspection; nofit/holdoutchanges.'],
        unchanged=['K4000basis','Allimages','Cameras','Shared3Dparameters','Allobservations andfamilysplits','No directrawfisheye or2D imagewarp'],
        budget=dict(runs=1,wall_seconds=120,GPU=0),
        outputs='JSON evidence andnativeinspectionpatches only. No point/camera export totraining; no new BA/featureextraction.',
        stop='Singulargeometry retained as missing evidence, not deleted. A model change requires a separate evidence-backed registration; no automatic higher-orderfit.',
        input_hashes={str(p):sha(p) for p in inputs}))
    print('N7 read-only audit registered',flush=True)

def summarize(x):
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a),min=float(a.min()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                p90=float(np.percentile(a,90)) if len(a) else None,max=float(a.max()) if len(a) else None)

def project(K,C,X):
    q=C[:3,:3]@X+C[:3,3];uv=K[:2,:2]@(q[:2]/q[2])+K[:2,2]
    jq=np.array([[K[0,0]/q[2],0.,-K[0,0]*q[0]/q[2]**2],[0.,K[1,1]/q[2],-K[1,1]*q[1]/q[2]**2]])
    return uv,q[2],jq@C[:3,:3]

def dlt(K,cams,obs):
    rows=[]
    for f,uv in obs:
        m=K@cams[f][:3];rows.extend([uv[0]*m[2]-m[0],uv[1]*m[2]-m[1]])
    _,_,v=np.linalg.svd(np.array(rows));return v[-1,:3]/v[-1,3]

def controls():
    K=np.array([[800.,0.,319.3],[0.,820.,240.8],[0.,0.,1.]])
    cams={0:np.eye(4),1:np.eye(4),2:np.eye(4)};cams[1][0,3]=-.5;cams[2][1,3]=.35
    X=np.array([.12,-.2,3.]);obs=[(i,project(K,c,X)[0]) for i,c in cams.items()]
    a=dlt(K,cams,obs);err=float(np.max(abs(a-X)));assert err<1e-10
    _,_,j=project(K,cams[1],X);eye=np.eye(3);h=1e-6
    num=np.stack([(project(K,cams[1],X+h*e)[0]-project(K,cams[1],X-h*e)[0])/(2*h) for e in eye],axis=1)
    jerr=float(np.max(abs(j-num)));assert jerr<1e-6
    return dict(known3D_DLT_error=err,analytic_pointJac_finite_difference=jerr)

def run():
    assert not O.exists();O.mkdir();start=time.monotonic();plan=read(P)
    for f,h in plan['input_hashes'].items():assert sha(Path(f))==h
    checks=controls();base=read(G/'result.json');rad=read(N/'result.json');alltracks=read(B/'observations.json')['tracks']
    K=np.array(base['K_integer_centers']);frames=base['actual_registered_frames'];cams={c['frame']:np.array(c['w2c']) for c in base['cameras']}
    fit=np.load(G/'fit-diagnostic.npz');X=fit['points'];ids=fit['track_ids'];lookup={int(t):i for i,t in enumerate(ids)}
    assert len(X)==935 and np.isfinite(X).all()
    centroid=X.mean(axis=0);_,singular,axes=np.linalg.svd(X-centroid,full_matrices=False)
    plane=(X-centroid)@axes[2];rms_extent=np.sqrt(np.mean((X-centroid)**2,axis=0).sum())
    tracks=[t for t in alltracks if t['split']=='fit'];assert len(tracks)==935
    depthrows=[]
    for t in tracks:
        point=X[lookup[t['track']]];rays=[];depths=[];uvs=[]
        for ob in t['observations']:
            c=cams[ob['frame']];center=-c[:3,:3].T@c[:3,3];rays.append((point-center)/np.linalg.norm(point-center));depths.append(project(K,c,point)[1]);uvs.append(ob['uv'])
        angles=[float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))) for k,a in enumerate(rays) for b in rays[k+1:]]
        depthrows.append(dict(track=t['track'],family=t['family'],observations=len(t['observations']),
            max_parallax_deg=max(angles),min_parallax_deg=min(angles),median_camera_depth=float(np.median(depths)),
            plane_distance=float(abs(plane[lookup[t['track']]])),image_y_mean=float(np.mean(uvs,axis=0)[1])))
    rows=[];nearest=[];sensitivity=[];face_points=[]
    total_face_energy=0.;center_energy=0.
    for fi in frames:
        face=[o for o in base['validation']['face_candidates']['observations'] if o['target']==fi]
        bg=[o for o in base['validation']['background']['observations'] if o['target']==fi]
        assert all(o['valid_projection'] for o in face+bg)
        fd=np.array([o['delta'] for o in face]);bd=np.array([o['delta'] for o in bg]);fc=np.array([o['observed'] for o in face]).mean(axis=0)
        fm=fd.mean(axis=0);bm=bd.mean(axis=0);fe=float(np.sum(fd**2));ce=float(len(fd)*np.sum(fm**2));total_face_energy+=fe;center_energy+=ce
        bgxy=np.array([o['observed'] for o in bg]);dist=np.linalg.norm(bgxy-fc,axis=1);ix=np.argsort(dist,kind='stable')[:5]
        fit_uv=np.array([o['uv'] for t in tracks for o in t['observations'] if o['frame']==fi]);fit_dist=np.linalg.norm(fit_uv-fc,axis=1)
        radial_basis=(np.array([o['observed'] for o in face])-K[:2,2]);radial_basis/=np.linalg.norm(radial_basis,axis=1,keepdims=True)
        radial=np.sum(fd*radial_basis,axis=1);tangent=np.sum(fd*np.stack([-radial_basis[:,1],radial_basis[:,0]],axis=1),axis=1)
        row=dict(frame=fi,face_n=len(face),background_n=len(bg),face_center=fc.tolist(),
            face_mean_vector_px=fm.tolist(),background_mean_vector_px=bm.tolist(),
            face_RMS_px=float(np.sqrt(fe/len(fd))),face_centered_descriptive_RMS_px=float(np.sqrt(np.mean(np.sum((fd-fm)**2,axis=1)))),
            face_common_vector_energy_fraction=ce/fe,face_radial_RMS=float(np.sqrt(np.mean(radial**2))),face_tangential_RMS=float(np.sqrt(np.mean(tangent**2))),
            nearest_FIT_distance_px=float(min(fit_dist)),nearest_heldout_distance_px=float(min(dist)),
            nearest5_heldout_residual_RMS_px=float(np.sqrt(np.mean(np.sum(bd[ix]**2,axis=1)))),
            nearest5_heldout_mean_vector_px=bd[ix].mean(axis=0).tolist())
        rows.append(row)
        im=Image.open(B/f'frame_{fi:06d}.png').convert('RGB');sheet=Image.new('RGB',(96*5,96*2),'white');locations=[]
        for col,j in enumerate(ix):
            o=bg[j];u,v=o['observed'];x,y=int(round(u))-48,int(round(v))-48;crop=im.crop((x,y,x+96,y+96));sheet.paste(crop,(col*96,0))
            marked=crop.copy();draw=ImageDraw.Draw(marked);draw.ellipse((u-x-2,v-y-2,u-x+2,v-y+2),outline=(0,255,80));px,py=o['predicted'];draw.line((px-x-3,py-y,px-x+3,py-y),fill=(255,40,180));draw.line((px-x,py-y-3,px-x,py-y+3),fill=(255,40,180));sheet.paste(marked,(col*96,96))
            loc=dict(frame=fi,track=o['track'],family=o['family'],distance_to_face_center_px=float(dist[j]),
                observed=o['observed'],predicted=o['predicted'],error_px=o['error_px'],crop=[x,y,x+96,y+96],crop_native_pixels=True)
            locations.append(loc)
        sheet.save(O/f'nearest_background_{fi:06d}.png');nearest.append(dict(frame=fi,patches=locations))
        for o in face:
            support=[r for r in base['validation']['face_candidates']['observations'] if r['track']==o['track'] and r['target'] in o['support_frames']]
            obs=[(r['target'],r['observed']) for r in support];point=dlt(K,cams,obs);uv,z,jt=project(K,cams[fi],point)
            assert np.max(abs(uv-np.array(o['predicted'])))<1e-6
            jac=np.vstack([project(K,cams[f],point)[2] for f,_ in obs]);_,s,vt=np.linalg.svd(jac,full_matrices=False)
            cov=(vt.T*(1/s**2))@vt;pc=jt@cov@jt.T;ev=np.linalg.eigvalsh(pc)
            sensitivity.append(dict(frame=fi,track=o['track'],support=len(obs),support_jacobian_condition=float(s[0]/s[-1]),
                prediction_sigma_max_px_at1px_support_iid=float(np.sqrt(ev[-1])),
                observed_error_px=o['error_px'],plane_distance_fitworld=float(abs((point-centroid)@axes[2])),camera_depth=float(z)))
            face_points.append(point)
    # Spatial bins describe retained observations, never refit a distortion field.
    bins=[]
    for group in ['background','face_candidates']:
        observations=base['validation'][group]['observations']
        for iy in range(4):
            for ix in range(4):
                a=[o for o in observations if min(3,int(o['observed'][0]/1520*4))==ix and min(3,int(o['observed'][1]/2704*4))==iy]
                if a:
                    delta=np.array([o['delta'] for o in a]);bins.append(dict(group=group,bin=[ix,iy],n=len(a),mean_vector_px=delta.mean(axis=0).tolist(),norm=summarize(np.linalg.norm(delta,axis=1))))
    geom=dict(points=len(X),first_baseline_units=True,centroid=centroid.tolist(),PCA_singular_values=singular.tolist(),
        smallest_to_largest_singular_ratio=float(singular[-1]/singular[0]),plane_normal=axes[2].tolist(),
        signed_plane_distance_summary=summarize(plane),absolute_plane_distance_summary=summarize(abs(plane)),
        rms_extent=float(rms_extent),plane_RMS_to_extent=float(np.sqrt(np.mean(plane**2))/rms_extent),
        max_parallax_degrees=summarize([r['max_parallax_deg'] for r in depthrows]),median_depth=summarize([r['median_camera_depth'] for r in depthrows]),
        face_target_excluded_plane_distance=summarize([r['plane_distance_fitworld'] for r in sensitivity]))
    result=dict(stage='N7',updated=now(),state='numeric_complete_pending_native_material_review',elapsed_seconds=time.monotonic()-start,
        controls=checks,input_hashes=plan['input_hashes'],geometry=geom,per_view=rows,spatial_bins=bins,
        face_common_vector_energy_fraction_allviews=center_energy/total_face_energy,
        fixed_camera_support_sensitivity=summarize([r['prediction_sigma_max_px_at1px_support_iid'] for r in sensitivity]),
        assumptions='PCAgeometry isconditionalonalreadyfittedcameras/K, notmetrictruth. Sensitivityassumesfixedexactcameras andiid1pxsupportnoise, excludescameraerror/bias/motion. Per-viewcentering isdescriptive only; noimage/observationschanged.',
        unique_cause_identified=False,training_started=False,cameras_changed=False,physical_face_certified=False)
    assert time.monotonic()-start<120
    save(O/'result.json',result);save(O/'fit-point-support.json',depthrows);save(O/'face-support-sensitivity.json',sensitivity);save(O/'nearest-patch-index.json',nearest)
    for f,h in plan['input_hashes'].items():assert sha(Path(f))==h
    save(O/'progress.json',dict(state='completed_pending_review',pid=None,updated=now()))
    print(json.dumps(dict(geometry=geom,per_view=rows,common_vector_energy_fraction=result['face_common_vector_energy_fraction_allviews'],sensitivity=result['fixed_camera_support_sensitivity'])),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception as e:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
