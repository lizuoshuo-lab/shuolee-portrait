"""Bounded read-only provenance, projection and track-attrition audit.

No feature extraction, camera solve, threshold alteration or training. Samples
are selected by image region, observed length and span, never by holdout error.
"""
from pathlib import Path
from datetime import datetime,timezone
from functools import lru_cache
import argparse,json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
import cv2
from trial_grass_surface123 import ROOT,OUT,read,save,sha

O=OUT/'provenance-audit'
def now():return datetime.now(timezone.utc).isoformat()
def summarize(v):
    x=np.asarray(v,float)
    return dict(count=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(np.max(x))) if len(x) else dict(count=0)
def register():
    assert not O.exists();O.mkdir()
    save(O/'plan.json',dict(registered=now(),previous_goal_turn='progress: actual fresh94camera/prior/body execution and negative clothing-volume evidence',
        hypothesis=['Hidentity: frame/image/camera/observation order or pixel conventions differ.',
                    'Hsupport: a numerically small camera fitting residual is dominated by unrelated image regions or rejected person tracks.',
                    'Hmaterial: appearance components connect different physical features.'],
        original_readonly_inputs=[dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in ['decoded.json','data/inputs.json','data/training-inputs.json',
            'data/observations.npz','camera/solver-output.npz','camera/appearance-tracks.json','fit/fitted.npz']],
        budget=dict(CPU_seconds=180,new_camera_candidates=0,new_feature_passes=0,new_training_runs=0,max_review_tracks=32,patch_nodes_per_track=6),
        sample_policy='All-person tracks: first4 by each head/body/feet region forFIT thenHELD. OutsideFIT: first4 in image-near-person band, first4 elsewhere. Descending raw appearance length, then span, then ID. This is development evidence, no reallocation ofHELD or physical truth.',
        stop='Finish once, preserve every mismatch and all source files. Do not continue optimization from this audit.',
        frame_domain='Original2704x1520 decodedMP4, integer centers; storedrendererK has+0.5 principal. No second undistortion.',
        intervention='None. This isolates software transport before attributing error to calibration or motion.'))
    print(O/'plan.json')

def audit():
    plan=read(O/'plan.json');assert not (O/'result.json').exists();started=time.monotonic()
    for f in plan['original_readonly_inputs']:assert sha(f['path'])==f['sha256'],f['path']
    decoded=read(ROOT/'decoded.json')['records'];rows=read(ROOT/'data/training-inputs.json')['records']
    raw=read(ROOT/'data/inputs.json')['records'];d=dict(np.load(ROOT/'camera/solver-output.npz'))
    body=dict(np.load(ROOT/'data/observations.npz'));fit=dict(np.load(ROOT/'fit/fitted.npz'))
    tracks=read(ROOT/'camera/appearance-tracks.json')['records'];byid={r['id']:r for r in tracks}
    assert len(rows)==len(decoded)==len(raw)==94
    K=d['K'];E=d['w2c'];features=[];inside=[];interior=[];stagecounts=[];identity=[]
    for i,(row,rawrow,dec) in enumerate(zip(rows,raw,decoded)):
        for key in ['frame','source_frame_index','time_seconds','name','source_sha256','wh']:
            assert row[key]==rawrow[key]==dec[key],(i,key)
        assert row['official_prediction_slot']==i
        assert body['frame_ids'][i]==fit['frame_ids'][i]==row['frame']
        assert np.array_equal(np.array(row['w2c']),E[i])
        assert np.array_equal(np.array(row['K_integer_centers']),K)
        assert np.max(abs(np.array(row['K'])[:2,2]-K[:2,2]-.5))<1e-10
        assert sha(row['image'])==dec['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        xy=np.load(ROOT/f'camera/features-{i:03d}.npz')['xy'];features.append(xy)
        mask=np.array(Image.open(row['mask']).convert('L'))>127
        p=np.rint(xy).astype(int);assert (p>=0).all() and (p[:,0]<mask.shape[1]).all() and (p[:,1]<mask.shape[0]).all()
        hits=mask[p[:,1],p[:,0]];ins=cv2.erode(mask.astype(np.uint8),np.ones((9,9),np.uint8))[p[:,1],p[:,0]]>0
        inside.append(hits);interior.append(ins)
        stagecounts.append(dict(frame=row['frame'],raw_features=len(xy),raw_inside=int(hits.sum()),raw_interior=int(ins.sum())))
        identity.append(dict(frame=row['frame'],source_index=dec['source_frame_index'],image_sha256=dec['source_sha256'],mask_sha256=row['mask_sha256']))
    ci=d['obs_cam'];pi=d['obs_point'];uv=d['observations'];ids=d['track_ids']
    node_maps={str(x):dict(byid[str(x)]['nodes']) for x in ids}
    feature_error=[]
    for idx in range(len(ci)):
        tid=str(ids[pi[idx]]);node=node_maps[tid][int(ci[idx])]
        feature_error.append(float(np.max(abs(features[ci[idx]][node]-uv[idx]))))
    assert max(feature_error)==0
    matrix_pred=[];opencv_pred=[]
    for i in range(94):
        selected=np.flatnonzero(ci==i);X=d['points'][pi[selected]]
        q=X@E[i,:3,:3].T+E[i,:3,3];pq=q@K.T
        matrix_pred.append(pq[:,:2]/pq[:,2:])
        pc,_=cv2.projectPoints(X,cv2.Rodrigues(E[i,:3,:3])[0],E[i,:3,3],K,None)
        opencv_pred.append(pc[:,0])
    projection_agreement=float(np.max(abs(np.concatenate(matrix_pred)-np.concatenate(opencv_pred))))
    co=fit['coco17_world'].astype(float)
    q=np.einsum('nij,nkj->nki',E[:,:3,:3],co)+E[:,:3,3][:,None,:]
    q=np.einsum('ij,nkj->nki',K,q);independent=q[:,:,:2]/q[:,:,2:]
    fit_projection_agreement=float(abs(independent-fit['projected_coco17']).max())
    total_raw=sum(len(x) for x in features);total_raw_inside=sum(int(x.sum()) for x in inside)
    selected_ids={str(x) for x in ids};used={str(t):set(ci[pi==j].tolist()) for j,t in enumerate(ids)}
    classifications=[]
    for tr in tracks:
        flags=[bool(inside[i][j]) for i,j in tr['nodes']]
        tr2=dict(id=tr['id'],held=tr['held'],length=len(flags),span=tr['nodes'][-1][0]-tr['nodes'][0][0],
                 foreground_observations=sum(flags),all_inside=all(flags),any_inside=any(flags),
                 camera_fit_selected=tr['id'] in selected_ids,selected_observations=len(used.get(tr['id'],[])))
        ys=[];near=[]
        for i,j in tr['nodes']:
            box=np.array(rows[i]['initial_box_xyxy']);h=box[3]-box[1];p=features[i][j]
            ys.append(float((p[1]-box[1])/h))
            near.append(bool(box[0]-h*.5<=p[0]<=box[2]+h*.5 and box[1]<=p[1]<=box[3]))
        y=float(np.median(ys));tr2['region']='head' if y<.25 else 'feet' if y>.85 else 'body'
        tr2['near_image_band']=sum(near)>=len(near)/2
        classifications.append(tr2)
    summaries=[]
    for split in [False,True]:
        for group,fn in [('all_inside',lambda t:t['all_inside']),('mixed',lambda t:t['any_inside'] and not t['all_inside']),('outside',lambda t:not t['any_inside'])]:
            arr=[t for t in classifications if t['held']==split and fn(t)]
            summaries.append(dict(split='HELD' if split else 'FIT',group=group,tracks=len(arr),observations=sum(t['length'] for t in arr),
                foreground_observations=sum(t['foreground_observations'] for t in arr),
                selected_tracks=sum(t['camera_fit_selected'] for t in arr),selected_observations=sum(t['selected_observations'] for t in arr)))
    sample=[]
    rank=lambda t:(-t['length'],-t['span'],t['id'])
    for held in [False,True]:
        for region in ['head','body','feet']:
            sample.extend(sorted([t for t in classifications if t['held']==held and t['all_inside'] and t['region']==region],key=rank)[:4])
    for near in [True,False]:sample.extend(sorted([t for t in classifications if not t['held'] and not t['any_inside'] and t['near_image_band']==near],key=rank)[:4])
    assert len(sample)<=32
    @lru_cache(maxsize=12)
    def source(i):return Image.open(rows[i]['image']).convert('RGB')
    for index,tr in enumerate(sample):
        nodes=byid[tr['id']]['nodes'];pick=np.unique(np.linspace(0,len(nodes)-1,min(6,len(nodes))).astype(int));tiles=[];views=[]
        panel=Image.new('RGB',(176*len(pick),352),'white')
        for k,jj in enumerate(pick):
            i,j=nodes[jj];p=features[i][j];c=np.rint(p).astype(int);box=[int(c[0]-88),int(c[1]-88),int(c[0]+88),int(c[1]+88)]
            tile=source(i).crop(box);marked=tile.copy();draw=ImageDraw.Draw(marked)
            x,y=p[0]-box[0],p[1]-box[1];draw.line((x-4,y,x+4,y),fill=(255,30,210),width=1);draw.line((x,y-4,x,y+4),fill=(255,30,210),width=1)
            panel.paste(tile,(k*176,0));panel.paste(marked,(k*176,176))
            views.append(dict(frame=rows[i]['frame'],source_frame=rows[i]['source_frame_index'],xy=p.tolist(),crop=box))
        name=f'{index:02d}_{"HELD" if tr["held"] else "FIT"}_{tr["region"] if tr["all_inside"] else "outside"}_{tr["id"][:10]}.png'
        panel.save(O/name);tr.update(file=name,views=views,sha256=sha(O/name),sample_selection='region, length, span, ID only; no residual-based selection')
    assert time.monotonic()-started<180
    result=dict(updated=now(),runtime_seconds=time.monotonic()-started,source_count=94,
        identity_checks=dict(all94_images_masks_camera_order_frame_ids_slots_agree=True,solver_observation_to_saved_feature_max_abs_px=max(feature_error)),
        numeric_projection=dict(independent_OpenCV_vs_matrix_max_abs_px=projection_agreement,
            independent_saved_SMPLX_vs_saved_projection_max_abs_px=fit_projection_agreement,acceptance='Software consistency only; no physical calibration claim'),
        raw_feature_count=total_raw,raw_foreground_features=total_raw_inside,raw_foreground_fraction=total_raw_inside/total_raw,
        mask_mean_fraction=float(np.mean([x['mask_fraction'] for x in read(ROOT/'mask-audit.json')['records']])),
        track_stages=summaries,per_frame=stagecounts,samples=sample,
        strict_pixel_accuracy_verified=False,material_review_pending=True,no_source_changes=True,no_new_optimization=True)
    for f in plan['original_readonly_inputs']:assert sha(f['path'])==f['sha256']
    save(O/'result.json',result);save(O/'classifications.json',dict(records=classifications))
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_frame','samples']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.register:register()
    elif a.audit:audit()
    else:p.error('Choose register or audit.')
