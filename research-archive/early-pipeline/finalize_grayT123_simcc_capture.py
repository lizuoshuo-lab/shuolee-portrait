"""Recover image index JSON after completed inference; do not change result."""
from probe_grayT123_simcc import *

original_result=read(OUT/'result.json');original_result_hash=sha(OUT/'result.json')
if (OUT/'index.json').exists():raise RuntimeError('Image index already exists; do not overwrite.')
plan=read(OUT/'plan.json');saved=np.load(OUT/'raw-responses.npz');obs=np.load(ROOT/'data/observations.npz')
rows=read(ROOT/'data/training-inputs.json')['records']
assert sha(ROOT/'data/observations.npz')==plan['observations_sha256']
assert sha(ROOT/'data/training-inputs.json')==plan['training_inputs_sha256']
from rtmlib.visualization.skeleton.coco133 import coco133
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs,get_warp_matrix
from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
names=[coco133['keypoint_info'][i]['name'] for i in range(133)]
flip=np.array([names.index(coco133['keypoint_info'][i]['swap']) if coco133['keypoint_info'][i]['swap'] else i for i in range(133)])
records=[];images=[];maxbase=0.;maxscore=0.;maxaffine=0.;recompute=0.
for frame in plan['frames']:
    slot=frame-1;row=rows[slot];W=row['wh'][0];box=obs['boxes'][slot]
    fbox=np.array([W-1-box[2],box[1],W-1-box[0],box[3]])
    legacy=[];candidates=[];scores=[];valids=[];details=[]
    for branch,b in enumerate([box,fbox]):
        center,scale=bbox_xyxy2cs(b,padding=1.25)
        scale=np.array([scale[0],scale[0]/.75]) if scale[0]>scale[1]*.75 else np.array([scale[1]*.75,scale[1]])
        raw=[saved[f'f{frame:03d}_b{branch}_axis{axis}'] for axis in range(2)]
        loc,score=get_simcc_maximum(*raw)
        prev=loc/2/SIZE*scale+center-scale/2
        axs=[];flags=[]
        for a,k in zip(raw,KERNELS):
            d,ff=refine_axis(a,k);axs.append(d);flags.append(ff)
        candidate=np.stack(axs,axis=-1)/2/SIZE*scale+center-scale/2
        valid=np.array([f['valid'] for f in flags[0]])&np.array([f['valid'] for f in flags[1]])
        inverse=cv2.invertAffineTransform(get_warp_matrix(center,scale,0,tuple(SIZE)))
        independent=cv2.transform((loc/2).astype(np.float64),inverse)
        maxaffine=max(maxaffine,float(np.abs(independent-prev).max()))
        if branch:
            prev=prev[:,flip];candidate=candidate[:,flip];score=score[:,flip];valid=valid[flip]
            prev[:,:,0]=W-1-prev[:,:,0];candidate[:,:,0]=W-1-candidate[:,:,0]
        legacy.append(prev[0]);candidates.append(candidate[0]);scores.append(score[0]);valids.append(valid)
        details.append(dict(branch=branch,center=center.tolist(),scale=scale.tolist(),flags=flags))
    old=np.mean(legacy,axis=0);candidate=np.mean(candidates,axis=0);score=np.minimum(*scores);good=valids[0]&valids[1]
    assert np.array_equal(old,saved[f'f{frame:03d}_argmax_native'])
    assert np.array_equal(candidate,saved[f'f{frame:03d}_continuous_native'],equal_nan=True)
    assert np.array_equal(good,saved[f'f{frame:03d}_candidate_valid'])
    parity=np.linalg.norm(old-obs['keypoints'][slot],axis=-1);scoreerr=float(np.abs(score-obs['scores'][slot]).max())
    maxbase=max(maxbase,float(parity.max()));maxscore=max(maxscore,scoreerr)
    records.append(dict(frame=frame,image_sha256=row['source_sha256'],box=box.tolist(),old_observation_max_parity_px=float(parity.max()),
        old_score_max_difference=scoreerr,all_keypoints_continuous_valid=int(good.sum()),nose_valid=good[50:59].tolist(),
        nose_old=old[50:59].tolist(),nose_candidate=candidate[50:59].tolist(),
        nose_shift_native_px=stats(np.linalg.norm(candidate[50:59]-old[50:59],axis=-1)),details=details))
    c=np.rint(old[50:59].mean(0)).astype(int)-80
    images.append(dict(frame=row['frame'],file=f'frame_{frame:03d}_raw_argmax_continuous_native.png',
        region='原图／原argmax青点／连续解码候选橙点；均为原生160px裁片',columns=['raw','argmax','candidate'],crop=[int(c[0]),int(c[1]),int(c[0]+160),int(c[1]+160)]))
assert maxbase<.001 and maxscore<1e-5 and maxaffine<.001
shifts=np.concatenate([np.linalg.norm(np.array(r['nose_candidate'])-np.array(r['nose_old']),axis=-1) for r in records])
result=dict(created=now(),frames=plan['frames'],inferences=12,additional_recovery_inferences=0,
    original_decoder_max_parity_px=maxbase,original_score_max_difference=maxscore,independent_inverse_affine_max_px=maxaffine,
    nose_candidate_shift_native_px=stats(shifts),records=records,raw_responses_sha256=sha(OUT/'raw-responses.npz'),
    decoder_control_passed=True,physical_accuracy_validated=False,anatomical_visibility_validated=False,
    new_observations_installed=False,geometry_or_training_executed=False,quality_accepted=False,
    serialization_recovery='Initial inference completed all12 calls and saved result.json/raw NPZ/images; image index JSON failed on NumPy crop scalars. This recovery recomputes only decoding from saved responses, checks exact parity, and writes the image index; no model inference repeated.')
# Discover/record exact serialization issue rather than silently broad string conversion.
found=[]
def scalarize(x,path='$'):
    if isinstance(x,np.generic):found.append(dict(path=path,type=type(x).__name__));return x.item()
    if isinstance(x,dict):return {k:scalarize(v,path+'.'+k) for k,v in x.items()}
    if isinstance(x,list):return [scalarize(v,path+f'[{i}]') for i,v in enumerate(x)]
    return x
clean=scalarize(result)
assert original_result['nose_candidate_shift_native_px']==clean['nose_candidate_shift_native_px']
assert original_result['original_decoder_max_parity_px']==maxbase
assert original_result['raw_responses_sha256']==sha(OUT/'raw-responses.npz')
save(OUT/'serialization-recovery.json',dict(created=now(),issue='index.json crop contained NumPy int64 values; result.json and all inference outputs had already saved successfully.',
    original_source_sha256=plan['source_sha256'],recovery_source_sha256=sha(Path(__file__)),numpy_scalars=found,
    additional_inferences=0,raw_responses_preserved=True,exact_recomputed_coordinates=True,result_json_preserved=True,
    result_json_sha256=original_result_hash,fix='Image crop values converted explicitly with int().'))
save(OUT/'index.json',dict(images=images,image_labels_burned_in=False))
assert sha(OUT/'result.json')==original_result_hash
save(OUT/'progress.json',dict(updated=now(),stage='completed_pending_actual_review',done=6,total=6,inference_process_completed=True))
print(json.dumps({k:v for k,v in clean.items() if k!='records'},ensure_ascii=False))
print('SERIALIZATION_SCALARS',found[:8])
