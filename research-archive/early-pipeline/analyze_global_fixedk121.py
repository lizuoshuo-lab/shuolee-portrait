"""Read-only causal review of a finished bounded camera diagnostic, no refitting."""
from global_fixedk121 import *
from review_global_fixedk121 import crop

plan=read(R/'global-fixedk-plan.json'); summaries=[]
dest=R/'visual-review/global-fixedk'
for s in SUBJECTS:
    result=read(R/f'{s}-global-fixedk.json');obs,tracks,hs=input_data(s);apply_material_gate(s,obs,tracks,plan)
    byid={t['track_id']:t for t in tracks};perframe={r['frame']:i for i,r in enumerate(obs['records'])}
    review=dict(subject=s,calibration_accepted=False,training_admitted=False,source_result_sha256=sha(R/f'{s}-global-fixedk.json'),state=result['state'])
    if result.get('registered_count',0):
        d=np.load(ROOT/s/'global-fixedk/background-structure.npz');xyz=d['xyz'];cloud_ids=set(d['track_ids'].tolist())
        evals=np.linalg.svd(xyz-np.median(xyz,axis=0),compute_uv=False)
        review['background_structure']=dict(singular_values=evals.tolist(),smallest_to_largest=float(evals[-1]/evals[0]),note='Computed candidate structure, not measured ground truth. Ground-only K observability and multi-depth support remain unproven.')
        known=set(result['registered_frames'])
        review['unregistered_material_overlap']=[dict(frame=r['frame'],known_3d_tracks=sum(t['track_id'] in cloud_ids and i in t['by_frame'] for t in tracks)) for i,r in enumerate(obs['records']) if r['frame'] not in known]
        preds=read(ROOT/s/'global-fixedk/heldout-predictions.json');rois={r['frame']:r for r in obs['rois']}
        bands={k:[] for k in ['head_rectangle','torso_hands_rectangle','lower_body_rectangle','feet_rectangle']}
        for p in preds:
            if p['state']!='predicted' or p['region'] not in ['face_roi','body_roi']: continue
            bb=rois[p['frame']]['body_box'];v=(p['observed_uv'][1]-bb[1])/(bb[3]-bb[1])
            band='head_rectangle' if v<.25 else 'torso_hands_rectangle' if v<.65 else 'lower_body_rectangle' if v<.85 else 'feet_rectangle'
            bands[band].append(p)
        review['independent_rectangular_body_bands']={k:dict(observations=len(p),tracks=len({x['track_id'] for x in p}),error_px=stats([x['error_px'] for x in p])) for k,p in bands.items()}
        # Failure diagnosis only: show full track patches for worst UNIQUE heldout tracks, not camera optimization.
        for reg in ['face_roi','body_roi']:
            worst=sorted([p for p in preds if p['region']==reg and p['state']=='predicted'],key=lambda p:-p['error_px'])
            seen=set();tiles=[];receipt=[]
            for p in worst:
                if p['track_id'] in seen: continue
                seen.add(p['track_id']);t=byid[p['track_id']];frames=sorted(t['by_frame']);keys=[frames[0],frames[len(frames)//2],frames[-1]]
                tiles.append(np.concatenate([crop(cv2.imread(obs['records'][i]['image']),t['by_frame'][i]['uv']) for i in keys],axis=1))
                receipt.append(dict(track_id=t['track_id'],max_test_error_px=p['error_px'],max_test_frame=p['frame'],columns=[dict(frame=obs['records'][i]['frame'],uv=t['by_frame'][i]['uv']) for i in keys]))
                if len(tiles)==6: break
            if tiles: cv2.imwrite(str(dest/f'{s}_{reg}_高误差真实对应三视角.png'),np.concatenate(tiles,axis=0))
            save(dest/f'{s}_{reg}_worst_material_tracks.json',dict(selection='Failure review only; no changes to fit/holdout. first/middle/last observation of worst unique heldout tracks.',tracks=receipt))
        ev=result['heldout_leave_one_view_out'];bg=ev['background_static_support_candidate'];face=ev['face_roi'];body=ev['body_roi']
        review['numerical_gates']=dict(all24_registered=result['registered_count']==24,background_every_frame10=min(x['n'] for x in bg['per_frame'].values())>=10,background_median_le1=bg['error_px'].get('median',float('inf'))<=1,background_p90_le3=bg['error_px'].get('p90',float('inf'))<=3,face_tracks_ge5=face['independent_track_count']>=5,face_median_le1p5=face['error_px'].get('median',float('inf'))<=1.5,face_p90_le3=face['error_px'].get('p90',float('inf'))<=3,body_tracks_ge10=body['independent_track_count']>=10,body_median_le1p5=body['error_px'].get('median',float('inf'))<=1.5,body_p90_le3=body['error_px'].get('p90',float('inf'))<=3,BA_converged=result['bundle_adjustment']['optimizer_success'])
    if s=='grass':review['conclusion']='No supported near-ground static control in existing native cycle tracks; foliage-heavy candidates do not establish static-camera evidence. Stop before a misleading camera fit.'
    elif s=='indoor':review['conclusion']='Static material support insufficient after removing the unsafe assumption that bright chandelier/foliage are reliable background truth. No pose candidate or Gaussian training.'
    elif s=='sofa':review['conclusion']='Pair connectivity does not establish a common 3D reconstruction: only4/24 frames registered before PnP overlap fell below20. Local face heldout also fails. Do not interpolate or train from fragment.'
    else:review['conclusion']='24/24 conditional-K ground-based poses established, but face/body heldout fail despite low ground error. This isolates a static-person consistency problem without uniquely separating K/depth extrapolation, material identity, and actual person motion. BA also reaches80evaluation budget without convergence; do not rerun or train.'
    summaries.append(review)
save(R/'global-fixedk-analysis.json',dict(updated=now(),subjects=summaries,new_gaussian_training=False,causal_attribution='Not unique. Low ground residual is not evidence of full physical calibration or proof of micro-motion.',no_more_candidate_or_threshold_scan=True))
print(json.dumps(summaries,ensure_ascii=False,indent=2))
