"""Locate face support lost between frozen appearance tracks and saved BA.
No reconstruction, geometry filtering, feature extraction or training.
"""
from pathlib import Path
from datetime import datetime,timezone
import time,json
import numpy as np
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native,inside,spaced

def main():
    start=time.monotonic();out=REC/'grayT-track-attrition';review=REC/'grayT-track-attrition-review'
    if out.exists() or review.exists():raise RuntimeError('Read existing result; do not repeat.')
    out.mkdir();review.mkdir()
    sources=[ROOT/'camera/appearance-tracks.json',ROOT/'camera/solver-output.npz',ROOT/'data/training-inputs.json']
    plan=dict(created=datetime.now(timezone.utc).isoformat(),reason='All nine saved BA face-ROI observations visually lie on background; locate whether actual face-region tracks existed before the final BA selection.',
        operation='Count exact frozen feature identities through raw SIFT, unambiguous appearance components, final saved BA. No new observations.',
        strong_face_region='Every original component node lies in that frame face box AND frozen person mask. Region only, not guaranteed material identity.',
        selection='At most3 evenly spaced lexicographic FIT whole-face-component IDs; all original observations retained. Earliest/middle/latest, no error use.',
        budget_seconds=120,no_geometry_recomputation=True,no_parameters_changed=True,no_fit_holdout_changes=True,
        limits='The failed run did not save per-track rejection reasons; absence from final BA cannot distinguish geometry rejection from point budget truncation.',
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources])
    save(out/'plan.json',plan)
    rows=read(sources[2])['records'];tracks=read(sources[0])['records'];z=dict(np.load(sources[1]));features=[];region=[]
    raw_counts=[]
    for i,row in enumerate(rows):
        xy=np.load(ROOT/f'camera/features-{i:03d}.npz')['xy'];features.append(xy)
        with Image.open(native(row['mask'])) as im:mask=np.asarray(im.convert('L'))
        ij=np.rint(xy).astype(int);r=inside(xy,row.get('face_box_xyxy'))&(mask[ij[:,1],ij[:,0]]>127);region.append(r)
        raw_counts.append(dict(frame=row['frame'],all=len(xy),face_mask_intersection=int(r.sum())))
        if time.monotonic()-start>120:raise RuntimeError('120second budget')
    ids=[str(t) for t in z['track_ids']];inba=set(ids);summaries={};whole=[];node_sets={'FIT':set(),'HELD':set()}
    for split,isheld in [('FIT',False),('HELD',True)]:
        recs=[]
        for tr in tracks:
            if tr['held']!=isheld:continue
            flags=[region[i][j] for i,j in tr['nodes']]
            for (i,j),b in zip(tr['nodes'],flags):
                if b:node_sets[split].add((i,j))
            if any(flags):recs.append(dict(track_id=tr['id'],face_nodes=sum(map(int,flags)),all_nodes=len(flags),all_nodes_face=bool(all(flags)),in_final_BA=tr['id'] in inba))
            if all(flags):whole.append((split,tr))
        summaries[split]=dict(components_with_any_face_mask_node=len(recs),unique_face_mask_nodes=len(node_sets[split]),
            whole_components_in_face_mask=sum(x['all_nodes_face'] for x in recs),face_components_retained_in_final_BA=sum(x['in_final_BA'] for x in recs),
            whole_face_components_retained_in_final_BA=sum(x['all_nodes_face'] and x['in_final_BA'] for x in recs))
        save(out/(split+'-face-components.json'),recs)
    saved_face_nodes=0
    lookup={r['id']:dict(r['nodes']) for r in tracks}
    for i,p in zip(z['obs_cam'],z['obs_point']):saved_face_nodes+=int(region[i][lookup[ids[p]][int(i)]])
    chosen=spaced([tr['id'] for split,tr in whole if split=='FIT'],3)
    save(out/'selected-samples.json',dict(selected=chosen,selection_uses_no_error=True))
    images=[]
    trmap={tr['id']:tr for tr in tracks}
    for ident in chosen:
        tr=trmap[ident];nodes=tr['nodes'];nodes=[nodes[i] for i in np.unique(np.linspace(0,len(nodes)-1,3).astype(int))]
        im=Image.new('RGB',(256*len(nodes),256));ss=[]
        for j,(i,k) in enumerate(nodes):
            xy=features[i][k];cx,cy=map(int,np.rint(xy));crop=(cx-128,cy-128,cx+128,cy+128)
            with Image.open(native(rows[i]['image'])) as raw:im.paste(raw.convert('RGB').crop(crop),(j*256,0))
            ss.append(dict(frame=rows[i]['frame'],source_frame_index=rows[i]['source_frame_index'],xy=xy.tolist(),crop=crop))
        name=f'FIT_whole_face_component_{ident[:12]}.png';im.save(review/name);mark=im.copy();draw=ImageDraw.Draw(mark)
        for j,s in enumerate(ss):
            x=j*256+s['xy'][0]-s['crop'][0];y=s['xy'][1]-s['crop'][1]
            draw.ellipse((x-10,y-10,x+10,y+10),outline=(0,255,255),width=1)
        loc=name.replace('.png','_locator.png');mark.save(review/loc)
        images.append(dict(frame=ident,region='FIT_whole_face_component_before_BA',file=name,locator=loc,columns=[s['frame'] for s in ss],sources=ss,
            in_final_BA=ident in inba,native_pixels_no_warp=True,labels_in_pixels=False))
    save(review/'index.json',dict(subject='grayT',images=images,actual_review_pending=True))
    result=dict(subject='grayT',raw_feature_total=sum(x['all'] for x in raw_counts),raw_face_mask_nodes=sum(x['face_mask_intersection'] for x in raw_counts),
        stages=summaries,final_BA_face_mask_observations=saved_face_nodes,selected_samples=chosen,per_frame=raw_counts,
        exact_rejection_reasons_unavailable=True,physical_identity_verified=False,training_allowed=False,seconds=time.monotonic()-start)
    assert all(sha(Path(s['path']))==s['sha256'] for s in plan['sources'])
    assert result['seconds']<=120
    save(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='per_frame'},ensure_ascii=False))

if __name__=='__main__':main()
