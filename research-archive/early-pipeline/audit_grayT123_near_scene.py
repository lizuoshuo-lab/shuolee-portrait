"""One bounded selection+native review preparation for same-frame near scene controls.
Selection never uses projection errors. No camera/point fitting or new features.
"""
from datetime import datetime,timezone
import time,json
import numpy as np
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native,inside

OUT=REC/'grayT-near-scene-support-review'

def main():
    started=time.monotonic()
    if OUT.exists():raise RuntimeError('Bounded audit exists; do not rerun.')
    p=read(REC/'grayT-near-scene-support-plan.json');assert p['status']=='planned_not_executed'
    OUT.mkdir()
    files=[REC/'grayT-near-scene-support-plan.json',ROOT/'camera/held-evaluation.json',ROOT/'camera/appearance-tracks.json',
        ROOT/'camera/solver-output.npz',ROOT/'data/training-inputs.json',REC/'grayT-skin-texture-probe/geometry-result.json']
    plan=dict(created=datetime.now(timezone.utc).isoformat(),source_plan=str(files[0]),
        sources=[dict(path=str(f),sha256=sha(f)) for f in files],
        frozen_selection='HELDcomponents wholly outside every original personbox+20px. Rank by count of original HELDcheck IDs among58/60/62/65 descending, then mean absolute log(saved scene depth / diagnostic skin depth) ascending, then lexicalID.',
        frame_numbers=[58,60,62,65],max_components=6,cpu_budget_seconds=120,
        depth_is_saved_estimate_not_ground_truth=True,whole_track_visible_material_review_required=True,
        selection_does_not_access_errors=True,all_candidate_eligibility_saved=True,
        no_optimization=True,no_new_features=True,no_training=True)
    save(OUT/'plan.json',plan)
    # Immediately drop residual fields; the selection stage never accesses them.
    held=[{k:r[k] for k in ['id','nodes','point','anchor_frames','test_frames']} for r in read(files[1])['records']]
    appearance={r['id']:r for r in read(files[2])['records']};z=np.load(files[3]);E=z['w2c'];K=z['K'];rows=read(files[4])['records']
    skin=read(files[5]);depths={r['frame_number']-1:r['depth'] for r in skin['observations'] if r['frame_number'] in plan['frame_numbers']}
    targets=set(depths);features=[];outside=[];hashes=[]
    for i,row in enumerate(rows):
        path=ROOT/f'camera/features-{i:03d}.npz';xy=np.load(path)['xy'];features.append(xy);hashes.append(sha(path))
        mask=inside(xy,row['initial_box_xyxy'],20)
        for box in row.get('other_person_boxes_xyxy',[]):mask|=inside(xy,box,20)
        outside.append(~mask)
    candidates=[];eligible=[]
    for tr in held:
        assert appearance[tr['id']]['held'] and appearance[tr['id']]['nodes']==tr['nodes']
        common=sorted(targets&set(tr['test_frames']))
        if not common:continue
        alloutside=all(outside[i][j] for i,j in tr['nodes'])
        ds={i:float((E[i,:3,:3]@np.asarray(tr['point'])+E[i,:3,3])[2]) for i in common}
        positive=all(v>0 and np.isfinite(v) for v in ds.values())
        ratio={i:ds[i]/depths[i] for i in common}
        score=float(np.mean([abs(np.log(v)) for v in ratio.values()])) if positive else None
        c=dict(id=tr['id'],coverage=len(common),frames=[i+1 for i in common],nodes=len(tr['nodes']),
            all_nodes_outside_person_boxes=bool(alloutside),positive_in_selected_views=positive,depth_by_frame={i+1:v for i,v in ds.items()},
            depth_ratio_by_frame={i+1:v for i,v in ratio.items()},closeness_score=score,eligible=bool(alloutside and positive))
        candidates.append(c)
        if c['eligible']:eligible.append(c)
    eligible.sort(key=lambda c:(-c['coverage'],c['closeness_score'],c['id']))
    selected=eligible[:6];assert len(selected)<=6
    save(OUT/'selection.json',dict(created=datetime.now(timezone.utc).isoformat(),selected=selected,candidates=candidates,
        skin_depth_reference={i+1:v for i,v in depths.items()},selection_uses_errors=False,
        positive_eligible_count=len(eligible),candidate_count=len(candidates),all_features_sha256=hashes,
        coverage_counts={str(n):sum(c['coverage']==n for c in eligible) for n in range(1,5)},
        within20percent_depth_count=sum(max(abs(v-1) for v in c['depth_ratio_by_frame'].values())<=.2 for c in eligible)))
    # Selection is now on disk before any residual fields are read by a subsequent stage.
    trmap={t['id']:t for t in held};requests={};sheets={};images=[]
    for k,c in enumerate(selected,1):
        tr=trmap[c['id']];nodes=tr['nodes'];base=f'near_scene_{k:02d}_{tr["id"][:12]}'
        w=128*min(8,len(nodes));h=128*int(np.ceil(len(nodes)/8));sheets[tr['id']]=[Image.new('RGB',(w,h),(245,245,245)),Image.new('RGB',(w,h),(245,245,245))]
        ss=[]
        for n,(i,j) in enumerate(nodes):
            requests.setdefault(i,[]).append((tr['id'],n,j));ss.append(dict(frame=rows[i]['frame'],frame_number=i+1,
                source_frame_index=rows[i]['source_frame_index'],role='ANCHOR' if n%2==0 else 'HELD',
                same_frame_target=i in common if False else i in targets and n%2==1))
        images.append(dict(frame=tr['id'],region='原始背景HELD组件全部节点，含锚点与留出',file=base+'_native.png',locator=base+'_locator.png',
            columns=[s['frame'] for s in ss],sources=ss,all_original_nodes=len(nodes)))
    imap={im['frame']:im for im in images}
    for i,reqs in sorted(requests.items()):
        with Image.open(native(rows[i]['image'])) as raw:
            rgb=raw.convert('RGB')
            for ident,n,j in reqs:
                uv=features[i][j].astype(float);cx,cy=map(int,np.round(uv));crop=[cx-64,cy-64,cx+64,cy+64];tile=rgb.crop(crop)
                original,marked=sheets[ident];x=(n%8)*128;y=(n//8)*128;original.paste(tile,(x,y));marked.paste(tile,(x,y));d=ImageDraw.Draw(marked)
                a=uv-crop[:2]+[x,y];d.ellipse((a[0]-5,a[1]-5,a[0]+5,a[1]+5),outline=(0,255,255),width=1)
                imap[ident]['sources'][n].update(xy=uv.tolist(),crop=crop,source_sha256=rows[i]['source_sha256'])
        if time.monotonic()-started>120:raise RuntimeError('Bounded CPU budget exceeded; no retry.')
    for im in images:
        for v,key in zip(sheets[im['frame']],['file','locator']):v.save(OUT/im[key])
        im['sha256']=sha(OUT/im['file']);im['locator_sha256']=sha(OUT/im['locator'])
    save(OUT/'index.json',dict(subject='grayT',images=images,actual_review_pending=True,read_order='chronological,left to right,top to bottom'))
    assert all(sha(f)==s['sha256'] for f,s in zip(files,plan['sources']))
    save(OUT/'preparation-result.json',dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        selected_components=len(images),observations=sum(im['all_original_nodes'] for im in images),sources_unchanged=True,
        no_camera_update=True,no_optimization=True,no_new_features=True,quality_accepted=False))
    print(json.dumps(dict(selected=selected,eligible=len(eligible),components=len(images),observations=sum(im['all_original_nodes'] for im in images),seconds=time.monotonic()-started),ensure_ascii=False))

if __name__=='__main__':main()
