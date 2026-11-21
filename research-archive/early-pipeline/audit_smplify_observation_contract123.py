"""Frozen observation contract and cross-view consistency; no optimizer/GPU."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os, sys, time
import numpy as np
import cv2

H=Path(__file__).parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'; REF=REC/'smplifyx-reference-control'
OUT=REF/'observation-contract'; FIT=REF/'conditioned-reference'
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
SITE=Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/lib/python3.10/site-packages')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.observation_contract import inspect_coco133_targets

read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
stats=lambda a:dict(n=int(np.size(a)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(np.max(a)))

def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def fundamental(a,b):
    A=np.asarray(a['w2c']);B=np.asarray(b['w2c'])
    R=B[:3,:3]@A[:3,:3].T
    t=B[:3,3]-R@A[:3,3]
    if np.linalg.norm(t)<1e-10:raise ValueError('Zero baseline has no usable epipolar constraint.')
    return np.linalg.inv(b['K_integer_centers']).T@skew(t)@R@np.linalg.inv(a['K_integer_centers'])

def epipolar(F,a,b):
    aa=np.c_[a,np.ones(len(a))];bb=np.c_[b,np.ones(len(b))]
    lb=aa@F.T;la=bb@F
    denom_a=np.linalg.norm(la[:,:2],axis=1);denom_b=np.linalg.norm(lb[:,:2],axis=1)
    if min(denom_a.min(),denom_b.min())<1e-12:raise ValueError('Degenerate epipolar line.')
    val=np.abs(np.sum(bb*lb,axis=1))
    return np.maximum(val/denom_a,val/denom_b)

def controls():
    K=np.array([[721.,0,319.25],[0,733.,240.75],[0,0,1.]])
    A=np.eye(4);B=np.eye(4);B[0,3]=-.7
    a={'K_integer_centers':K,'w2c':A};b={'K_integer_centers':K,'w2c':B}
    xyz=np.array([[-.3,.2,3.],[.8,-.3,4.5],[0,0,2.3]])
    project=lambda X,E:(X@E[:3,:3].T+E[:3,3])@K.T
    qa=project(xyz,A);qb=project(xyz,B);uv_a=qa[:,:2]/qa[:,2:];uv_b=qb[:,:2]/qb[:,2:]
    F=fundamental(a,b);exact=epipolar(F,uv_a,uv_b)
    moved=uv_b.copy();moved[:,1]+=3
    value=epipolar(F,uv_a,moved)
    assert exact.max()<1e-10 and np.allclose(value,3,atol=1e-10)
    B2=B.copy();B2[:3,3]*=7
    assert np.allclose(epipolar(fundamental(a,dict(b,w2c=B2)),uv_a,moved),value,atol=1e-10)
    return dict(known_3D_projection_max_px=float(exact.max()),known_vertical_offset_px=value.tolist(),
                translation_scale_invariant=True,maximum_of_two_point_to_line_distances=True)

def main():
    plan=read(REF/'observation-contract-plan.json')
    assert plan['state']=='registered_not_executed'
    assert not (OUT/'result.json').exists() and not (OUT/'progress.json').exists()
    OUT.mkdir(exist_ok=True)
    official=REC/'grass-exploratory/provenance-audit/mapping-audit/official/smplifyx/utils.py'
    srcs=[ROOT/'data/observations.npz',ROOT/'data/inputs.json',ROOT/'camera/candidate-cameras.json',
          FIT/'result.json',official,H/'observe_from_coverage123.py',H/'smplify_score_contract123.py',
          PROJECT/'src/shuolee_portrait/observation_contract.py',PROJECT/'src/shuolee_portrait/alignment_inputs.py',
          SITE/'rtmlib/tools/pose_estimation/rtmpose.py',SITE/'rtmlib/tools/pose_estimation/post_processings.py',
          SITE/'rtmlib/tools/pose_estimation/pre_processings.py',SITE/'rtmlib/visualization/skeleton/coco133.py']
    srcs += [FIT/f'slot{s:03d}-fit.npz' for s in [0,31,62]]
    hashes={str(p):sha(p) for p in srcs}
    save(OUT/'execution-plan.json',dict(registered=now(),source_hashes=hashes,original_plan=plan,
        read_only_cross_view='Use frozen fullK/E and saved per-image projections. Epipolar residual is scale-invariant; world root-orientation and centred body shapes use rotations only. No absolute SfM metric-scale assumption, PnP, triangulation, rigid correction or new BA.',
        visibility='Unknown stays unknown. No score or current model is used to assert observed.',
        budget=dict(cpu_seconds=120,views=3,pairs=3,detector_passes=0,fit_steps=0),source_sha256=sha(Path(__file__))))
    save(OUT/'progress.json',dict(state='running',pid=os.getpid(),started=now()))
    started=time.monotonic()
    obs=dict(np.load(ROOT/'data/observations.npz')); names=obs['joint_names'].astype(str).tolist()
    data=read(ROOT/'data/inputs.json')['records'];cams=read(ROOT/'camera/candidate-cameras.json')['records']
    assert [r['frame'] for r in data]==[r['frame'] for r in cams]==obs['frame_ids'].astype(str).tolist()
    for a,b in zip(data,cams):
        for key in ['w2c','K_integer_centers']:assert np.array_equal(a[key],b[key])
    report=inspect_coco133_targets(obs['keypoints'],names,obs['frame_ids'],detector_scores=obs['scores'])
    tree=ast.parse(official.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='smpl_to_openpose'];scope={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(official),'exec'),scope)
    mapping=scope['smpl_to_openpose'](model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    coco=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    slots=np.r_[coco,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    oracle=dict(zip(src.tolist(),mapping[slots].tolist()))
    assert all(oracle[x['source_channel']]==x['model_joint_index'] for x in report['bindings'])
    meta={};exec(compile((SITE/'rtmlib/visualization/skeleton/coco133.py').read_text(),'installed-coco133','exec'),{},meta)
    assert names==[meta['coco133']['keypoint_info'][i]['name'] for i in range(133)]
    save(OUT/'native-target-report.json',report)
    save(OUT/'source-contract.json',dict(updated=now(),source_hashes=hashes,
        actual_stored_names_equal_installed_metadata=True,all116_bindings_equal_pinned_official_map=True,
        same_crop_for_body_and_hand_channels=True,
        coordinate_restore='Same zero-rotation bbox affine for all133 channels; argmax/2 then /[288,384]*scale+center-scale/2. No separate hand inverse transform in this RTMW output.',
        frozen_extraction='Native forward plus horizontally flipped pass, semantics swapped by installed metadata; keypoints=(a+b)/2. New extraction not performed.',
        score='Installed 2D RTMLib combines X/Y maxima by mean; frozen flip aggregation is min(forward,flip). Relative q90/flip confidence is not calibrated visibility.',
        official_annotation_reference='COCO-WholeBody body fields retained separately from added hand fields; validity and annotation visibility are distinct from inference response maxima.',
        exact_body_vs_hand_anatomical_placement_equivalence='not specified to subpixel precision by inspected documentation; shared SMPLX anchor is exact but target annotation equivalence remains uncertain',
        fixed_face51_material_truth='not established by matching landmark index',
        known_decoder_audit_reused=str(REC/'grass-preflight/generic-input/decoder-audit.json'),
        training_started=False))
    checks=controls(); fits={s:dict(np.load(FIT/f'slot{s:03d}-fit.npz')) for s in [0,31,62]}
    groups={'body17':coco,'feet6':np.arange(19,25),'left_hand21':np.arange(25,46),'right_hand21':np.arange(46,67),'face51':np.arange(67,118)}
    pair_results=[]
    for i,j in [(0,31),(31,62),(0,62)]:
        F=fundamental(cams[i],cams[j]);a,b=fits[i],fits[j]
        pred=epipolar(F,a['projected'][0],b['projected'][0]);target=epipolar(F,a['targets'][0],b['targets'][0])
        Rwi=np.asarray(cams[i]['w2c'])[:3,:3].T@cv2.Rodrigues(a['global_orient'].reshape(3))[0]
        Rwj=np.asarray(cams[j]['w2c'])[:3,:3].T@cv2.Rodrigues(b['global_orient'].reshape(3))[0]
        angle=float(np.linalg.norm(cv2.Rodrigues(Rwi.T@Rwj)[0])*180/np.pi)
        world=[]
        for s,f in [(i,a),(j,b)]:
            xyz=f['joints_camera'][0];centre=xyz[[9,12]].mean(0)
            world.append((xyz-centre)@np.asarray(cams[s]['w2c'])[:3,:3])
        torso=float(np.mean([np.linalg.norm(w[2]-w[9])+np.linalg.norm(w[5]-w[12]) for w in world])/2)
        normalized=np.linalg.norm(world[0]-world[1],axis=-1)/torso
        r=dict(slots=[i,j],source_frames=[cams[i]['source_frame_index'],cams[j]['source_frame_index']],fundamental=F.tolist(),
            projected_epipolar_max_of_two_distances_px={g:stats(pred[idx]) for g,idx in groups.items()},
            detector_epipolar_max_of_two_distances_px={g:stats(target[idx]) for g,idx in groups.items()},
            root_orientation_world_difference_deg=angle,
            centred_world_joint_difference_in_torso_lengths={g:stats(normalized[idx]) for g,idx in groups.items()},
            per_target_projected_epipolar_px=pred.tolist(),per_target_detector_epipolar_px=target.tolist(),
            visibility_not_filtered_or_certified=True)
        pair_results.append(r)
    for path,digest in hashes.items():assert sha(Path(path))==digest
    assert time.monotonic()-started<120
    result=dict(updated=now(),state='readonly_contract_and_cross_view_complete',frames=94,views=3,
        checks=checks,source_contract='source-contract.json',native_report='native-target-report.json',pairs=pair_results,
        all_input_hashes_unchanged=True,all116_mapping_bindings_verified=True,source_coordinates_changed=False,
        interpretation='Same-image residual measures per-image fit. Cross-view epipolar and world-rotation inconsistency can come from camera/K error, observation bias or pose change; this diagnostic does not identify a unique cause. No per-frame correction is fitted.',
        independent_physical_ground_truth=False,physical_quality_passed=False,new_gaussian_steps=0,seconds=time.monotonic()-started)
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(state='complete',pid=None,finished=now()))
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=False))
    for r in pair_results:print(json.dumps({k:r[k] for k in ['slots','projected_epipolar_max_of_two_distances_px','detector_epipolar_max_of_two_distances_px','root_orientation_world_difference_deg']},ensure_ascii=False))

if __name__=='__main__':main()
