"""Read-only replay of the two completed company00 face controls. No fitting/GPU."""
from pathlib import Path
from datetime import datetime,timezone
import os,time,json,hashlib
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=R/'grayT-face-observation-cross-check'
A=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
B=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
D=Path('D:/Projects/Shuolee-analysis-20261028')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def stats(x):
    a=np.asarray(x);return dict(n=int(a.size),minimum=float(a.min()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()),ge1=int(np.sum(a>=1)))
def pose_stats(p,keep):
    mats=Rotation.from_rotvec(p[keep]).as_matrix();rel=np.einsum('aij,bkj->abik',mats,mats)
    angle=np.rad2deg(np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1)))
    return dict(maximum_pairwise_difference_deg=float(angle.max()),maximum_parameter_norm_deg=float(np.rad2deg(np.linalg.norm(p[keep],axis=-1)).max()))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');args=ap.parse_args()
    if args.register:
        O.mkdir(exist_ok=True);assert not (O/'plan.json').exists()
        plan=dict(updated=datetime.now(timezone.utc).isoformat(),previous_goal_turn='progress',
          question='Does the reported 2.42px result replay in native pixels, and which conditions prevent using it to admit a static person?',
          mode='Read-only existing RTMW versus face-crop FAN controls. No optimization, extraction, detector inference, camera change or training.',
          inputs=[str(B/'data/training-inputs.json'),str(B/'fit/fitted.npz'),str(B/'fit/observation-weights.npz'),str(B/'data/observations.npz'),str(D/'face-local-detector/observations.npz'),str(D/'head-motion-control/report.json'),str(R/'grayT-facial-articulation-review/cross-run-result.json')],
          variants=['dynamic-face-control','dynamic-face-fan-control'],
          evidence_contract='Replay saved support vertices, weights and canonical material point through saved deformation, then unchanged K/w2c; compare all 6 anchor/35 held observations without fitting. Reused M0 is secondary validation of one physical mark, not new held material or complete eyes/nose/mouth coverage.',
          tests=['Identity/frame/K scale and half-pixel parity','Saved projection replay all41','All35 native HELD errors including failures','Semantic observations and active weights by face region','Difference from static body pose and temporal rotation range','One existing candidate per detector, no reoptimization'],
          native_review_frames=[41,49,57,65,73,79],crop_pixels=320,image_policy='Unscaled integer native crops, separate raw/detector columns and preserved source hashes; no burned labels.',
          budget=dict(cpu_seconds=120,runs=1),stop_if='Any changed frozen input, identity mismatch, projection replay discrepancy >0.0001px or nonfinite data. Save evidence; do not rerun fit or alter controls.',
          acceptance='This audit cannot admit cameras or static training; all existing face thresholds remain unchanged.',training_started=False)
        save(O/'plan.json',plan);print(str(O/'plan.json'));raise SystemExit
    plan=read(O/'plan.json');assert not (O/'result.json').exists() and not (O/'started.json').exists()
    started=time.monotonic();stamp=datetime.now(timezone.utc).isoformat();save(O/'started.json',dict(updated=stamp,pid=os.getpid(),source_sha256=sha(Path(__file__))))
    sources=[Path(p) for p in plan['inputs']]+[D/'fit_dynamic_face.py',D/'evaluate_dynamic_face.py',D/'check_face_local_detector.py']
    for name in plan['variants']:
        sources += [D/name/f for f in ['motion.npz','fitted.npz','report.json','skin-validation.json']]
        if (D/name/'fit-script.py').exists():sources.append(D/name/'fit-script.py')
    sources += [A/f'raw/frame_{n:06d}.png' for n in plan['native_review_frames']]
    source_hashes={str(p):sha(p) for p in sources}
    rows=read(B/'data/training-inputs.json')['records'];arows=read(A/'data/training-inputs.json')['records'];n=len(rows)
    assert n==95 and [r['frame'] for r in rows]==[r['frame'] for r in arows]
    assert [r['source_frame_index'] for r in rows]==[r['source_frame_index'] for r in arows]
    ratio=np.array([1520,2704])/np.array(rows[0]['wh']);E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows])
    K_native=K.copy();K_native[:,0]*=ratio[0];K_native[:,1]*=ratio[1]
    kdiff=float(abs(K_native-np.array([r['K'] for r in arows])).max());assert kdiff<1e-8
    old=dict(np.load(B/'fit/fitted.npz'));weights=dict(np.load(B/'fit/observation-weights.npz'))['face'];obs=dict(np.load(B/'data/observations.npz'))
    fan=dict(np.load(D/'face-local-detector/observations.npz'));assert np.array_equal(fan['frame_ids'],old['frame_ids'])
    assert np.array_equal(obs['frame_ids'],old['frame_ids'])
    fanparity=float(abs((fan['decoded_face68']+.5)*ratio-.5-fan['native_face68']).max());assert fanparity<.001
    cross=read(R/'grayT-facial-articulation-review/cross-run-result.json')['company00_M0']
    base=read(D/'head-motion-control/report.json')['independent_skin_track_before'];skin=base['observations']
    ids=np.array([r['frame_number']-1 for r in skin]);anchors=np.array([r['role']=='ANCHOR' for r in skin]);targets=np.array([r['target'] for r in skin])
    assert np.array_equal(ids+1,[r['frame_number'] for r in cross['observations']])
    assert np.array_equal(targets,[r['xy'] for r in cross['observations']])
    assert anchors.sum()==6 and len(ids)==41 and np.array_equal(ids,np.arange(39,80))
    region={'brows':range(17,27),'nose':range(27,36),'eyes':range(36,48),'mouth':range(48,68)}
    active=weights>0;regioncounts={name:dict(joint_ids=[j for j in js if active[:,j-17].any()],active_observations=int(active[:,[j-17 for j in js]].sum())) for name,js in region.items()}
    runs={};replays={};source_script_digests={}
    for name in plan['variants']:
        folder=D/name;motion=dict(np.load(folder/'motion.npz'));fitted=dict(np.load(folder/'fitted.npz'));r=read(folder/'skin-validation.json');fitreport=read(folder/'report.json')
        assert np.array_equal(fitted['frame_ids'],old['frame_ids'])
        assert all(x['frame']==int(i+1) and x['role']==('ANCHOR' if a else 'HELD') and np.array_equal(x['target'],t) for x,i,a,t in zip(r['observations'],ids,anchors,targets))
        nearest=np.array(r['support_vertices']);w=np.array(r['support_weights']);assert abs(w.sum()-1)<1e-8
        p=np.array(r['point']);ref=motion['vertices_world'][0,nearest];aff=motion['vertex_affine'][1:][ids][:,nearest];seq=motion['vertices_world'][1:][ids][:,nearest]
        transformed=np.einsum('nkij,kj->nki',aff,p[None]-ref)+seq
        world=(transformed*w[None,:,None]).sum(1)
        cam=np.einsum('nij,nj->ni',E[ids,:3,:3],world)+E[ids,:3,3]
        homogeneous=np.einsum('nij,nj->ni',K_native[ids],cam);uv=homogeneous[:,:2]/homogeneous[:,2:]-.5
        recorded=np.array([x['projected'] for x in r['observations']]);delta=float(abs(uv-recorded).max());assert delta<.0001,delta
        err=np.linalg.norm(uv-targets,axis=-1);assert abs(np.median(err[~anchors])-r['held']['median'])<.0001
        params=fitted['parameters'];assert params.shape==(95,19)
        bodydiff=np.max(abs(fitted['body_pose'].reshape(n,21,3)-old['body_pose'].reshape(n,21,3)),axis=(0,2))
        native_semantic=(fitted['projected_face51']+.5)*ratio-.5
        true_target=(obs['keypoints'][:,40:91]+.5)*ratio-.5 if name=='dynamic-face-control' else fan['native_face68'][:,17:68]
        semerr=np.linalg.norm(native_semantic-true_target,axis=-1)
        src=folder/'fit-script.py';source_script_digests[name]=sha(src) if src.exists() else None
        runs[name]=dict(replay_max_abs_px=delta,held=stats(err[~anchors]),anchors=stats(err[anchors]),all_depths_positive=bool((cam[:,2]>0).all()),
          semantic_residual_native_px=stats(semerr[active]),semantic_report_coordinate_domain='Stored fit report uses899x1600 working pixels, not1520x2704 native pixels',
          neck_parameters_skin_window=pose_stats(params[:,:3],ids),head_parameters_skin_window=pose_stats(params[:,3:6],ids),
          expression_component_span_max=float(np.ptp(params[ids,9:],axis=0).max()),jaw_component_span_max=float(np.ptp(params[ids,6:9],axis=0).max()),
          changed_body_pose_joint_indices=np.flatnonzero(bodydiff>1e-5).tolist(),unchanged_body_pose_max_abs_except_neck_head=float(bodydiff[[j for j in range(21) if j not in [11,14]]].max()),
          per_frame_parameter_count=int(params.size),frozen_fit_script_available=src.exists(),fit_report_source_sha256=fitreport.get('source_sha256'),
          observations=[dict(frame=int(i+1),role='ANCHOR' if a else 'HELD',target=t.tolist(),replayed=pr.tolist(),error_px=float(e)) for i,a,t,pr,e in zip(ids,anchors,targets,uv,err)])
        replays[name]=uv
        if src.exists():assert sha(src)==fitreport['source_sha256']
        if time.monotonic()-started>120:raise RuntimeError('CPU audit budget exceeded')
    gallery=[];idlookup={int(i+1):j for j,i in enumerate(ids)}
    for f in plan['native_review_frames']:
        i=f-1;im=Image.open(A/f'raw/frame_{f:06d}.png').convert('RGB');assert im.size==(1520,2704)
        box=np.array(arows[i]['face_box_xyxy']).reshape(2,2);center=box.mean(0);left,top=np.floor(center-160).astype(int);cropbox=[int(left),int(top),int(left+320),int(top+320)]
        raw=im.crop(cropbox);panels=[raw.copy() for _ in range(4)]
        previous=(obs['keypoints'][i,40:91]+.5)*ratio-.5;new=fan['native_face68'][i,17:68]
        for panel,pts,color in [(panels[1],previous,'#00d3b0'),(panels[2],new,'#338dff')]:
            draw=ImageDraw.Draw(panel)
            for x,y in pts[active[i]]-np.array([left,top]):draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
        j=idlookup[f];draw=ImageDraw.Draw(panels[3]);xy=targets[j]-[left,top];draw.ellipse((* (xy-3),*(xy+3)),outline='#00d3b0',width=1)
        for name,color in [('dynamic-face-control','#ed42ce'),('dynamic-face-fan-control','#ffae24')]:
            x,y=replays[name][j]-[left,top];draw.line((x-3,y,x+3,y),fill=color,width=1);draw.line((x,y-3,x,y+3),fill=color,width=1)
        joined=Image.new('RGB',(1280,320))
        for col,panel in enumerate(panels):joined.paste(panel,(col*320,0))
        path=O/f'frame_{f:03d}_native_face_and_skin_replay.png';joined.save(path);raw.save(O/f'frame_{f:03d}_raw.png')
        gallery.append(dict(frame=f,file=path.name,crop=cropbox,caption='原图／全身RTMW有效点／脸部FAN有效点／同一皮肤点：绿圈观测、紫叉RTMW逐帧拟合、橙叉FAN逐帧拟合；原生整数裁片，无缩放。',source=str(A/f'raw/frame_{f:06d}.png'),source_sha256=source_hashes[str(A/f'raw/frame_{f:06d}.png')]))
    unchanged=all(sha(Path(p))==v for p,v in source_hashes.items());assert unchanged
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='read_only_replay_complete',seconds=time.monotonic()-started,same95_frame_ids=True,
       native_K_parity_max_abs=kdiff,FAN_native_decode_parity_max_abs_px=fanparity,same_reused_M0_41_targets_and_split=True,
       baseline_skin_refit=base['held'],earlier_original_M0=cross['held'],baseline_comparability='Company control refitted the6anchor material point with Huber f_scale1 working pixel. It is not numerically identical to the earlier original company00_M0 baseline.',
       active_regions=regioncounts,variants=runs,frozen_source_scripts=source_script_digests,source_hashes=source_hashes,source_files_unchanged=unchanged,
       limitations=['One reused physical skin mark, no heldout eye/nose/mouth material coverage','All95frames receive19 per-frame neck/head/jaw/expression degrees of freedom plus temporal priors; this is not a static fit','Both methods are learned semantic detectors; FAN input uses native image but its face network still resamples its crop','Same original face weights are reused; no new per-view anatomical occlusion proof','The reported improvement cannot distinguish better localization from model prior and camera-error absorption','No K, extrinsic, mesh, observations or training state changed in this audit'],
       training_started=False,static_training_admitted=False,quality_accepted=False)
    save(O/'result.json',result);save(O/'index.json',dict(title='00新面部观测对照的只读回放',gallery=gallery,quality_accepted=False))
    print(json.dumps(dict(state=result['state'],seconds=result['seconds'],active_regions=regioncounts,runs={k:{x:v[x] for x in ['replay_max_abs_px','held','semantic_residual_native_px','head_parameters_skin_window','neck_parameters_skin_window','changed_body_pose_joint_indices']} for k,v in runs.items()},source_files_unchanged=unchanged),ensure_ascii=False))
