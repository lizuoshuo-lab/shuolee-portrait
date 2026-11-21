"""Reproduce existing HELD projections after the frozen material visual review.

No triangulation, optimization, feature extraction or observation selection.
"""
from datetime import datetime, timezone
import time, json
import numpy as np
from PIL import Image, ImageDraw
from audit_grayT123_camera_support import ROOT, REC, read, save, sha, native, stats

OUT = REC/'grayT-material-static-check'
REVIEW = REC/'grayT-material-identity-review'
BG = ['0006b9fe08c4', '8036d2b7f840', 'ffef1195aa16']

def main():
    started = time.monotonic()
    if OUT.exists(): raise RuntimeError('Saved check exists; do not repeat.')
    review = read(REVIEW/'actual-review.json')
    assert review['all_nodes_actually_reviewed'] and review['observations_removed'] == 0
    face_ids = [r['track_id'] for r in review['records'] if r['split'] == 'HELD']
    assert len(face_ids) == 3
    sources = [REVIEW/'actual-review.json', ROOT/'camera/held-evaluation.json', ROOT/'camera/solver-output.npz',
               ROOT/'camera/appearance-tracks.json', ROOT/'data/training-inputs.json']
    OUT.mkdir()
    plan = dict(created=datetime.now(timezone.utc).isoformat(), budget_seconds=120,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in sources],
        question='What do existing saved static-point projections show for the three predetermined whole-face HELD tracks?',
        face_track_ids=face_ids, background_proxy_prefixes=BG,
        background_proxy_limit='Three previously reviewed diagnostic samples only; not an independent certified static-scene calibration.',
        review_before_error_read=True, keep_all_existing_test_observations=True, use_original_anchor_check_split=True,
        no_triangulation=True, no_K_E_change=True, no_feature_extraction=True, no_training=True,
        stopping_rule='Unsupported physical identity/staticity makes <1px acceptance unknown or failed, never true from these residuals alone.')
    save(OUT/'plan.json', plan)
    held = read(sources[1])['records']
    z = np.load(sources[2]); K=z['K']; E=z['w2c']
    tracks = {r['id']:r for r in read(sources[3])['records']}
    rows=read(sources[4])['records']
    assert np.array_equal(E,np.array([r['w2c'] for r in rows]))
    assert np.max(np.abs(np.array([r['K_integer_centers'] for r in rows])-K)) == 0
    picked = [r for r in held if r['id'] in face_ids or any(r['id'].startswith(s) for s in BG)]
    assert len(picked)==6
    feature_cache={}; results=[]; images=[]; allface=[]
    for tr in sorted(picked,key=lambda r:(r['id'] not in face_ids,r['id'])):
        ident=tr['id']; assert tracks[ident]['held']
        assert tr['nodes']==tracks[ident]['nodes']
        assert tr['anchor_frames']==[i for i,j in tr['nodes'][::2]]
        assert tr['test_frames']==[i for i,j in tr['nodes'][1::2]]
        records=[]
        for (i,j), expected in zip(tr['nodes'][1::2],tr['errors']):
            if i not in feature_cache:feature_cache[i]=np.load(ROOT/f'camera/features-{i:03d}.npz')['xy']
            uv=feature_cache[i][j].astype(float); p=E[i,:3,:3]@np.array(tr['point'])+E[i,:3,3]
            q=K@p; projected=q[:2]/q[2]; err=float(np.linalg.norm(projected-uv))
            assert abs(err-expected)<1e-7
            records.append(dict(frame=rows[i]['frame'], source_frame_index=rows[i]['source_frame_index'],
                xy=uv.tolist(),projected=projected.tolist(), residual=(projected-uv).tolist(),
                euclidean_error_px=err, depth=float(p[2]), positive_depth=bool(p[2]>0)))
            if ident in face_ids:
                allface.append(records[-1]);cx,cy=map(int,np.round(uv));crop=[cx-128,cy-128,cx+128,cy+128]
                with Image.open(native(rows[i]['image'])) as raw:im=raw.convert('RGB').crop(crop)
                marked=im.copy();d=ImageDraw.Draw(marked)
                a=uv-np.array(crop[:2]);b=projected-np.array(crop[:2])
                d.line((*a,*b),fill=(255,176,0),width=1)
                d.ellipse((a[0]-5,a[1]-5,a[0]+5,a[1]+5),outline=(0,255,255),width=1)
                d.line((b[0]-5,b[1],b[0]+5,b[1]),fill=(255,176,0),width=1)
                d.line((b[0],b[1]-5,b[0],b[1]+5),fill=(255,176,0),width=1)
                num=len(images)+1;name=f'HELD_check_{num:02d}_{ident[:12]}_{rows[i]["frame"]}'
                im.save(OUT/(name+'_native.png'));marked.save(OUT/(name+'_projection.png'))
                images.append(dict(frame=rows[i]['frame'],region='原始留出检查：青圈观测／橙十字冻结三维点投影',
                    file=name+'_projection.png',raw=name+'_native.png',columns=['native 256x256 pixels'],
                    track_id=ident,source_frame_index=rows[i]['source_frame_index'],crop=crop,
                    source_sha256=rows[i]['source_sha256'],projection_inside_crop=bool(np.all(b>=0) and np.all(b<256))))
        results.append(dict(track_id=ident, group='face_region_HELD' if ident in face_ids else 'background_proxy_HELD',
            anchors=tr['anchor_frames'],checks=records,stats=stats([r['euclidean_error_px'] for r in records]),
            physical_identity_staticity_certified=False, used_for_fit=False))
    canvas=Image.new('RGB',(4*256,3*256),(245,245,245));rawcanvas=canvas.copy()
    assert len(images)==12
    for n,im in enumerate(images):
        with Image.open(OUT/im['file']) as v:canvas.paste(v,((n%4)*256,(n//4)*256))
        with Image.open(OUT/im['raw']) as v:rawcanvas.paste(v,((n%4)*256,(n//4)*256))
    canvas.save(OUT/'all_twelve_held_checks_projection.png');rawcanvas.save(OUT/'all_twelve_held_checks_native.png')
    save(OUT/'index.json',dict(subject='grayT',images=images,actual_review_pending=True,
        legend='Separate diagnostic overlay: cyan circle=saved material candidate observation, orange cross=frozen existing3Dpoint projection. No text in images.'))
    assert all(sha(p)==s['sha256'] for p,s in zip(sources,plan['source_files']))
    result=dict(completed=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        face_stats=stats([r['euclidean_error_px'] for r in allface]), face_test_views=len(set(r['frame'] for r in allface)),
        face_all_component_views=len(set(i for tr in picked if tr['id'] in face_ids for i,j in tr['nodes'])),
        selected_test_views=95,positive_face_depth=sum(r['positive_depth'] for r in allface),
        tracks=results,existing_saved_errors_reproduced=True,source_hashes_unchanged=True,
        physical_identity_staticity_certified=False, cameras_modified=False, geometry_modified=False,
        independent_subpixel_acceptance=False,quality_accepted=False)
    assert result['seconds']<120
    save(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='tracks'},ensure_ascii=False))
    print(json.dumps([dict(id=t['track_id'][:12],group=t['group'],stats=t['stats']) for t in results]))

if __name__=='__main__':main()
