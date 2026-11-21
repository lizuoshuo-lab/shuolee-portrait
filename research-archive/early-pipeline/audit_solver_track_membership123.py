"""Once-only, read-only audit of the three registered pairs and all13 scene holdouts."""
from pathlib import Path
from collections import Counter,defaultdict
import json,time,os,traceback
import numpy as np
from PIL import Image,ImageDraw
import audit_smplify_observation_contract123 as a

O=a.REF/'solver-track-membership'
RECORD=Path('/mnt/d/00/记录/005/分析/静止场景支持')
read,save,sha=a.read,a.save,a.sha

def stats(x):return a.stats(x) if len(x) else {'n':0}

def main():
    plan=read(a.REF/'solver-track-membership-plan.json')
    assert plan['state']=='registered_not_executed'
    assert not O.exists(),'Existing audit; inspect rather than rerun.'
    O.mkdir();RECORD.mkdir(parents=True,exist_ok=True)
    source=a.H/'local_surface123_camera.py'
    assert sha(source)==read(a.ROOT/'camera/plan.json')['source_sha256']
    negative=a.OUT/'pair-negative-control.json'
    inputs=[source,negative,a.ROOT/'camera/solver-output.npz',a.ROOT/'camera/appearance-tracks.json',
            a.ROOT/'data/inputs.json',a.ROOT/'data/training-inputs.json']
    hashes={str(p):sha(p) for p in inputs}
    save(O/'execution-plan.json',dict(registered=a.now(),plan=plan,input_hashes=hashes,
        budget_seconds=120,no_optimizer_or_detector=True,source_sha256=sha(Path(__file__))))
    save(O/'progress.json',dict(state='running',pid=os.getpid(),started=a.now()))
    begin=time.monotonic()
    rows=read(a.ROOT/'data/training-inputs.json')['records']
    tracks={r['id']:r for r in read(a.ROOT/'camera/appearance-tracks.json')['records']}
    data=dict(np.load(a.ROOT/'camera/solver-output.npz',allow_pickle=False))
    tids=data['track_ids'].astype(str).tolist();ids={t:i for i,t in enumerate(tids)}
    assert len(ids)==len(tids)
    assert all(not tracks[t]['held'] for t in ids)
    used=defaultdict(dict)
    for oi,(ci,pi) in enumerate(zip(data['obs_cam'],data['obs_point'])):
        t=tids[int(pi)];ci=int(ci)
        assert ci not in used[t]
        used[t][ci]=oi
    allpairs=read(negative)['pairs'];slots=sorted({s for p in allpairs for s in p['slots']})
    xy={s:np.load(a.ROOT/f'camera/features-{s:03d}.npz')['xy'] for s in slots}
    for s in slots:
        assert np.array_equal(data['w2c'][s],rows[s]['w2c'])
        assert np.array_equal(data['K'],rows[s]['K_integer_centers'])
    source_images={s:Image.open(rows[s]['image']).convert('RGB') for s in slots}
    for s in slots:assert source_images[s].size==(2704,1520)
    records=[];pair_summaries=[]
    for pp in allpairs:
        ss=pp['slots'];F=a.fundamental(rows[ss[0]],rows[ss[1]])
        pair_rows=[]
        for r in pp['records']:
            tid=r['id'];tr=tracks[tid];nodes=dict(tr['nodes'])
            assert tr['held']==(r['split']=='HELD')
            points=np.array([xy[s][nodes[s]] for s in ss])
            assert np.array_equal(points,r['uv'])
            err=float(a.epipolar(F,points[:1],points[1:])[0])
            assert abs(err-r['error_px'])<1e-8
            present=[s in used.get(tid,{}) for s in ss]
            if tr['held']:membership='reserved_never_in_BA'
            elif all(present):membership='both_observations_in_BA'
            elif any(present):membership='one_observation_in_BA'
            elif tid in ids:membership='track_in_BA_but_neither_pair_observation'
            else:membership='track_not_in_BA'
            projections=[];relative=[]
            for s,uv,in_ba in zip(ss,points,present):
                if in_ba:assert np.array_equal(data['observations'][used[tid][s]],uv)
                if tid in ids:
                    X=data['points'][ids[tid]];E=data['w2c'][s]
                    q=E[:3,:3]@X+E[:3,3];v=data['K']@q;pred=v[:2]/v[2]
                    projections.append(dict(slot=s,positive_depth=bool(q[2]>0),
                        solver_depth=float(q[2]),error_px=float(np.linalg.norm(pred-uv)),
                        actually_in_BA=in_ba))
                b=np.array(rows[s]['initial_box_xyxy']);height=b[3]-b[1]
                within=bool(b[0]-.5*height<=uv[0]<=b[2]+.5*height and b[1]<=uv[1]<=b[3])
                relative.append(dict(slot=s,relative_xy=[float((uv[0]-(b[0]+b[2])/2)/height),float((uv[1]-b[1])/height)],
                                     in_image_near_person_band=within))
            item=dict(r,slots=ss,source_frames=[rows[s]['source_frame_index'] for s in ss],
                      solver_membership=membership,observations_in_BA=present,
                      full_track_frames=[i for i,j in tr['nodes']],solver_track_selected=tid in ids,
                      actual_solver_projection=projections,image_band=relative,
                      physical_material_identity='not_yet_reviewed')
            pair_rows.append(item);records.append(item)
        groups=defaultdict(list)
        for x in pair_rows:groups[x['solver_membership']+'__'+x['region']].append(x['error_px'])
        pair_summaries.append(dict(slots=ss,source_frames=[rows[s]['source_frame_index'] for s in ss],
            count=len(pair_rows),counts=dict(Counter(x['solver_membership'] for x in pair_rows)),
            groups={k:stats(v) for k,v in groups.items()}))
    candidates=[r for r in records if r['split']=='HELD' and r['region']=='both_outside_person_mask']
    assert len({r['id'] for r in candidates})<=13 and len(candidates)<=13
    figures=[]
    for number,r in enumerate(sorted(candidates,key=lambda r:(r['slots'],r['id']))):
        tiles=[];patches=[]
        for s,uv in zip(r['slots'],r['uv']):
            uv=np.array(uv);center=np.rint(uv).astype(int)
            box=[int(center[0]-96),int(center[1]-96),int(center[0]+96),int(center[1]+96)]
            raw=source_images[s].crop(tuple(box));marked=raw.copy();draw=ImageDraw.Draw(marked)
            x,y=uv-np.array(box[:2]);draw.line((x-5,y,x+5,y),fill=(255,30,210),width=1);draw.line((x,y-5,x,y+5),fill=(255,30,210),width=1)
            tile=Image.new('RGB',(192,384));tile.paste(raw);tile.paste(marked,(0,192));tiles.append(tile)
            patches.append(dict(slot=s,source_frame=rows[s]['source_frame_index'],uv=uv.tolist(),crop=box))
        sheet=Image.new('RGB',(384,384));sheet.paste(tiles[0]);sheet.paste(tiles[1],(192,0))
        name=f'20261029_原帧{r["source_frames"][0]:06d}与{r["source_frames"][1]:06d}_留出轨迹{r["id"][:12]}_上原图下紫色中心.png'
        path=RECORD/name;assert not path.exists();sheet.save(path)
        with Image.open(path) as check:check.load();assert check.size==(384,384)
        figures.append(dict(path=str(path),sha256=sha(path),track_id=r['id'],slots=r['slots'],patches=patches,
                            physical_identity='pending_actual_review',selection='every outside-mask HELD track in registered pairs, no residual selection'))
    for s in slots:
        im=source_images[s].copy();draw=ImageDraw.Draw(im);seen=set()
        for r in candidates:
            if s not in r['slots'] or r['id'] in seen:continue
            seen.add(r['id']);p=np.array(r['uv'][r['slots'].index(s)])
            draw.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6),outline=(255,30,210),width=2)
        path=RECORD/f'20261029_原帧{rows[s]["source_frame_index"]:06d}_原生全图_本轮全部场景留出位置.png'
        assert not path.exists();im.save(path)
        figures.append(dict(path=str(path),sha256=sha(path),kind='native_scene_context',slot=s,unique_targets=len(seen)))
    for path,digest in hashes.items():assert sha(Path(path))==digest
    assert time.monotonic()-begin<120
    save(O/'result.json',dict(updated=a.now(),source_implementation_hash_matches_original_plan=True,
        source_hashes=hashes,pairs=pair_summaries,records=records,
        source_logic='Two-view triangulation and per-observation reprojection/depth checks precede selection of up to5000 tracks for BA. The exact final selected track IDs and observation camera/point indices are saved.',
        no_new_geometry_fit=True,source_coordinates_unchanged=True,original_splits_unchanged=True,
        new_gaussian_steps=0,physical_quality_passed=False,seconds=time.monotonic()-begin))
    save(O/'rendered-evidence.json',dict(updated=a.now(),figures=figures,no_text_in_images=True,no_warp=True,
                                       record_root=str(RECORD),new_work_log_images=False))
    save(O/'progress.json',dict(state='complete',pid=None,finished=a.now()))
    print(json.dumps(dict(pairs=pair_summaries,holdout_track_images=len(candidates),images=len(figures)),ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if O.exists():save(O/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),auto_retry=False))
        raise
