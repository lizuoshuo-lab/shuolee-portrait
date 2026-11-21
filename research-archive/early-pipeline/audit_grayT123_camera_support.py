"""One bounded read-only audit of grayT's frozen camera observations.

Regions are diagnostic labels, never static/material truth or a new fit split.
No features, cameras, geometry, observations, masks or training are changed.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, time
import numpy as np
from PIL import Image

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = T/'shuolee/current/local-clothed-surface-123'
ROOT = Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
OUT = REC/'grayT-camera-support-audit'
REVIEW = REC/'grayT-camera-support-review'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, d): p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def native(p):
    return Path('//wsl.localhost/Ubuntu-20.04'+p) if p.startswith('/opt/') else Path(p[5].upper()+':/'+p[7:])
def stats(x):
    x=np.asarray(x, float)
    return dict(n=len(x), median=float(np.median(x)), p90=float(np.quantile(x,.9)), maximum=float(x.max()),
        lt1px=int((x<1).sum()), ge1px=int((x>=1).sum()), ge10px=int((x>=10).sum())) if len(x) else dict(n=0)
def inside(xy, box, margin=0):
    if box is None: return np.zeros(len(xy),bool)
    a,b,c,d=box
    return (xy[:,0]>=a-margin)&(xy[:,0]<=c+margin)&(xy[:,1]>=b-margin)&(xy[:,1]<=d+margin)
def spaced(a, n):
    a=sorted(set(a))
    return [a[i] for i in np.unique(np.linspace(0,len(a)-1,min(n,len(a))).astype(int))] if a else []

def main():
    started=time.monotonic()
    if OUT.exists() or REVIEW.exists(): raise RuntimeError('Audit exists: inspect it; do not rerun.')
    OUT.mkdir(); REVIEW.mkdir()
    inputs=[ROOT/'camera/appearance-tracks.json',ROOT/'camera/solver-output.npz',ROOT/'camera/held-evaluation.json',
        ROOT/'data/training-inputs.json',ROOT/'camera/result.json',ROOT/'decoded.json']
    plan=dict(created=datetime.now(timezone.utc).isoformat(),subject='grayT',budget_seconds=120,
        question='Does low full-image camera error reflect static-scene support while the person remains inconsistent?',
        operation='Classify frozen FIT and HELD observations spatially; recompute existing projections only; inspect preselected native crops.',
        changed_causal_parameter=None,no_optimization=True,no_feature_extraction=True,no_error_based_selection=True,
        regions=['all','person_mask','outside_person_mask','outside_all_person_boxes_plus20px','face_ROI_only'],
        static_proxy='Whole original appearance track outside every frozen person box plus20px in all its observed views; NOT proof of static material.',
        sample_selection='Per FIT/HELD and whole-track outside-box/face-ROI category, 3 evenly spaced lexicographic IDs. Earliest/middle/latest native256px crops. No use of error.',
        image_labels_burned_in=False,observations_removed=0,held_never_promoted=True,
        caveats=['FIT observations were already geometry-filtered before saved BA; report this selection bias.',
                 'Mask/ROI membership and apparent repeated texture cannot certify physical identity.',
                 'No source-relative scale, K, pose or dynamic change may be inferred from these group summaries alone.'],
        sources=[dict(path=str(p),sha256=sha(p)) for p in inputs])
    save(OUT/'plan.json',plan)
    rows=read(ROOT/'data/training-inputs.json')['records']; n=len(rows)
    records=read(ROOT/'camera/appearance-tracks.json')['records']; trmap={r['id']:r for r in records}
    held=read(ROOT/'camera/held-evaluation.json')['records']
    z=dict(np.load(ROOT/'camera/solver-output.npz'))
    assert len(rows)==95 and np.array_equal(z['w2c'],np.array([r['w2c'] for r in rows]))
    assert np.max(np.abs(np.array([r['K_integer_centers'] for r in rows])-z['K']))==0
    assert [r['official_prediction_slot'] for r in rows]==list(range(n))
    features=[]; flags=[];feature_hashes=[];mask_hashes=[]
    for i,row in enumerate(rows):
        fp=ROOT/f'camera/features-{i:03d}.npz'; xy=np.load(fp)['xy'];features.append(xy)
        feature_hashes.append(sha(fp));mp=native(row['mask']);mask_hashes.append(sha(mp))
        assert mask_hashes[-1]==row['mask_sha256']
        with Image.open(mp) as im: mask=np.asarray(im.convert('L'))
        w,h=row['wh'];assert mask.shape==(h,w)
        ij=np.rint(xy).astype(int);assert ((ij>=0)&(ij<[w,h])).all()
        inmask=mask[ij[:,1],ij[:,0]]>127
        inany=inside(xy,row['initial_box_xyxy'],20)
        for b in row.get('other_person_boxes_xyxy',[]):inany|=inside(xy,b,20)
        flags.append(np.stack([np.ones(len(xy),bool),inmask,~inmask,~inany,inside(xy,row.get('face_box_xyxy'))],axis=1))
        if time.monotonic()-started>120:raise RuntimeError('Audit CPU budget exceeded')
    original_categories={}
    for tr in records:
        q=np.array([flags[i][j] for i,j in tr['nodes']])
        original_categories[tr['id']]=dict(all_outside_boxes=bool(q[:,3].all()),any_face=bool(q[:,4].any()),
            any_person=bool(q[:,1].any()),all_person=bool(q[:,1].all()),span=tr['nodes'][-1][0]-tr['nodes'][0][0],nodes=len(q))
    track_ids=[str(x) for x in z['track_ids']]
    assert all(not trmap[x]['held'] for x in track_ids)
    assert all(trmap[r['id']]['held'] for r in held)
    assert not set(track_ids)&{r['id'] for r in held}
    ci=z['obs_cam'];pi=z['obs_point'];uv=z['observations'];E=z['w2c'];K=z['K']
    q=np.einsum('nij,nj->ni',E[ci,:3,:3],z['points'][pi])+E[ci,:3,3]
    pred=q@K.T;pred=pred[:,:2]/pred[:,2,None];fiterr=np.linalg.norm(pred-uv,axis=1)
    fflags=[];maxidentity=0.
    for obs_index,(i,p) in enumerate(zip(ci,pi)):
        ident=track_ids[p]; node=dict(trmap[ident]['nodes'])[int(i)]
        maxidentity=max(maxidentity,float(np.max(np.abs(features[i][node]-uv[obs_index]))));fflags.append(flags[i][node])
    assert maxidentity<1e-9
    ff=np.asarray(fflags);fids=np.array(track_ids)[pi]
    hci=[];hids=[];herr=[];hf=[];hxy=[];hdepth=[];maxheld=0.
    for tr in held:
        assert tr['test_frames']==[i for i,j in tr['nodes'][1::2]]
        for (i,j),err in zip(tr['nodes'][1::2],tr['errors']):
            p=E[i,:3,:3]@np.asarray(tr['point'])+E[i,:3,3]; pp=K@p; pp=pp[:2]/pp[2]
            recomputed=float(np.linalg.norm(pp-features[i][j]));maxheld=max(maxheld,abs(recomputed-err))
            hci.append(i);hids.append(tr['id']);herr.append(err);hf.append(flags[i][j]);hxy.append(features[i][j]);hdepth.append(p[2])
    assert maxheld<1e-8
    datasets={'FIT':dict(ci=ci,ids=np.asarray(fids),errors=fiterr,flags=ff,xy=uv,depth=q[:,2]),
        'HELD':dict(ci=np.array(hci),ids=np.array(hids),errors=np.array(herr),flags=np.asarray(hf),xy=np.asarray(hxy),depth=np.array(hdepth))}
    result=dict(subject='grayT',frames=n,fit_tracks=len(track_ids),held_tracks=len(held),source_identity_max_abs_px=maxidentity,
        held_projection_reproduction_max_abs_px=maxheld,groups={},whole_track_groups={},per_frame={},sample_selection={},
        fit_observations_preselected_by_prior_geometry=True,held_observations_removed_for_error=0,
        physical_identity_verified=False,static_camera_accuracy_proven=False,quality_accepted=False)
    selected=[]
    for split,d in datasets.items():
        result['groups'][split]={}
        for k,region in enumerate(plan['regions']):
            ok=d['flags'][:,k]
            result['groups'][split][region]=dict(**stats(d['errors'][ok]),tracks=len(set(d['ids'][ok])),views=len(set(d['ci'][ok])),
                nonpositive_depth=int((d['depth'][ok]<=0).sum()))
        result['per_frame'][split]=[dict(frame=rows[i]['frame'],source_frame_index=rows[i]['source_frame_index'],
            **{region:stats(d['errors'][(d['ci']==i)&d['flags'][:,k]]) for k,region in enumerate(plan['regions'])}) for i in range(n)]
        outside=np.array([original_categories[t]['all_outside_boxes'] for t in d['ids']])
        result['whole_track_groups'][split]={}
        for label,ok in [('all_nodes_outside_person_boxes',outside),('at_least_one_node_near_person_box',~outside)]:
            unique=sorted(set(d['ids'][ok]));spans=[original_categories[t]['span'] for t in unique]
            result['whole_track_groups'][split][label]=dict(**stats(d['errors'][ok]),tracks=len(unique),views=len(set(d['ci'][ok])),
                temporal_span_frames=stats(spans),nonpositive_depth=int((d['depth'][ok]<=0).sum()))
        for category,eligible in [('all_nodes_outside_person_boxes',d['ids'][outside]),('face_ROI_candidates',d['ids'][d['flags'][:,4]])]:
            ids=spaced(eligible,3);result['sample_selection'][split+'_'+category]=dict(eligible=len(set(eligible)),selected=ids)
            selected.extend((split,category,x) for x in ids)
    np.savez_compressed(OUT/'diagnostic-observations.npz',fit_cam=ci,fit_ids=fids,fit_errors=fiterr,fit_xy=uv,fit_region_flags=ff,
        held_cam=np.array(hci),held_ids=np.array(hids),held_errors=np.array(herr),held_xy=np.array(hxy),held_region_flags=np.array(hf))
    save(OUT/'selected-samples.json',dict(selection_before_visual_review=True,selection_uses_no_error=True,
        samples=[dict(split=a,category=b,track_id=c) for a,b,c in selected]))
    images=[]
    for a,b,ident in selected:
        nodes=trmap[ident]['nodes'];chosen=[nodes[i] for i in np.unique(np.linspace(0,len(nodes)-1,3).astype(int))]
        parts=[];sources=[]
        for i,j in chosen:
            xy=features[i][j];cx,cy=map(int,np.rint(xy));crop=(cx-128,cy-128,cx+128,cy+128)
            with Image.open(native(rows[i]['image'])) as im:parts.append(im.convert('RGB').crop(crop))
            sources.append(dict(frame=rows[i]['frame'],source_frame_index=rows[i]['source_frame_index'],xy=xy.tolist(),crop=crop))
        sheet=Image.new('RGB',(256*len(parts),256))
        for k,im in enumerate(parts):sheet.paste(im,(256*k,0))
        name=f'{a}_{b}_{ident[:12]}.png';sheet.save(REVIEW/name)
        images.append(dict(frame=ident,region=a+'_'+b,file=name,columns=[r['frame'] for r in sources],
            sources=sources,sha256=sha(REVIEW/name),native_pixels_no_warp=True,labels_in_pixels=False,selected_without_error=True))
    save(REVIEW/'index.json',dict(subject='grayT',images=images,actual_review_pending=True))
    save(OUT/'input-hashes.json',dict(feature_sha256=feature_hashes,mask_sha256=mask_hashes))
    assert all(sha(Path(s['path']))==s['sha256'] for s in plan['sources'])
    result['seconds']=time.monotonic()-started
    if result['seconds']>120:raise RuntimeError('Audit CPU budget exceeded')
    save(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_frame'},ensure_ascii=False))

if __name__=='__main__':main()
