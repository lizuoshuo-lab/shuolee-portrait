"""Read-only replay of saved appearance tracks through the completed mesh sequence.

This does not fit points, cameras or people and never modifies a training input.
Appearance matches retain their historical uncertainty and FIT/HELD identities.
"""
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import argparse
import hashlib
import json
import os
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from review_full_sequence123 import relocate, read, sha


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                maximum=float(np.max(a))) if len(a) else dict(count=0)


def ray_hit(triangles, direction):
    """Closest positive Moller-Trumbore intersection, camera origin at zero."""
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1]-v0, triangles[:, 2]-v0
    p = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum('ij,ij->i', e1, p)
    inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det)>1e-14)
    q = np.cross(-v0, e1)
    u = np.einsum('ij,ij->i', -v0, p)*inv
    v = (q@direction)*inv
    t = np.einsum('ij,ij->i', e2, q)*inv
    valid = (np.abs(det)>1e-14) & (u>=-1e-9) & (v>=-1e-9) & (u+v<=1+1e-9) & (t>0)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return None
    i = int(ids[np.argmin(t[ids])])
    return i, np.array([1-u[i]-v[i], u[i], v[i]]), float(t[i])


def face_frame(tri):
    def unit(a):
        return a/np.sqrt(max(float(a@a), 1e-20))
    a0 = unit(tri[1]-tri[0])
    a1 = unit(np.cross(a0, tri[2]-tri[0]))
    a2 = -unit(np.cross(a1, a0))
    scale = (np.sqrt(max(float((tri[1]-tri[0])@(tri[1]-tri[0])), 1e-20))+
             abs(float(a2@(tri[2]-tri[0]))))/2
    return tri.mean(0), np.stack([a0, a1, a2], axis=-1), max(scale, 1e-8)


def project(point, E, K):
    camera = E[:3, :3]@point+E[:3, 3]
    if camera[2] <= 0:
        return None, camera
    q = K@camera
    return q[:2]/q[2], camera


def register(root):
    out = root/'reviews/material-transport'
    out.mkdir(exist_ok=False)
    source = root.parents[2]/'grass-exploratory/provenance-audit'
    raw = Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
    report = read(source/'result.json')
    samples = [s for s in report['samples'] if s['any_inside']]
    assert len(samples) == 24
    paths = [root/'fit/fitted.npz', root/'data/training-inputs.json', source/'result.json',
             source/'visual-review.json', source/'plan.json', raw/'decoded.json', raw/'camera/appearance-tracks.json',
             Path(__file__), relocate('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee/utils/graphics_utils.py')]
    plan = dict(registered=now(),hypothesis='Do saved official per-frame SMPL-X surfaces carry the same existing appearance observations across frames?',
        operation='Anchor at first chronological observation using nearest positive mesh ray hit. Keep its face and barycentric location, and separately replay the existing face-similarity Gaussian transport. Evaluate every remaining saved observation; no point fitting or residual filtering.',
        selection='All24 previously reviewed person track samples, unchanged identities/splits. No new matching. Anchor without residual selection; missing mesh hit is reported, never snapped to another joint.',
        sample_ids=[s['id'] for s in samples], inputs=[dict(path=str(p),sha256=sha(p)) for p in paths],
        original_tracks_root=str(raw), prior_review_root=str(source),
        budget=dict(seconds=180,CPU_only=True,track_samples=24,source_updates=0,optimizer_updates=0,training_runs=0),
        limits=['Development appearance tracks are not independently certified physical truth.',
                'A mesh hit is not anatomical correctness or visibility through hair/clothes.',
                'Mesh self-visibility is reported separately; no observations are excluded from all-observation statistics.',
                'Transport residual combines fitting, correspondence, cloth/hair motion and conditional calibration. It cannot isolate a unique cause.',
                'The fixed surface test is not a replay of the learned free Gaussian offsets.'],
        image_contract='Each row: raw native crop / existing2D green / transported3D purple. Six historical sample views. Integer176px crops, no scaling or text.',
        stop='One completed read-only replay or failure. Preserve all outcomes; no parameter scan.')
    save(out/'plan.json',plan)
    print(json.dumps(dict(state='registered',tracks=len(samples),output=str(out))))


def run(root):
    out = root/'reviews/material-transport'
    assert not (out/'result.json').exists() and not (out/'progress.json').exists()
    plan = read(out/'plan.json')
    started = time.monotonic()
    save(out/'progress.json',dict(state='running',pid=os.getpid(),started=now(),command=__import__('sys').argv))
    def bound():
        if time.monotonic()-started>plan['budget']['seconds']:
            raise RuntimeError('Registered CPU replay budget exceeded')
    try:
        for item in plan['inputs']:
            assert sha(relocate(item['path']))==item['sha256'], item['path']
        # Independent numerical checks: known triangle hit and an outside ray.
        test = np.array([[[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]]])
        hit = ray_hit(test,np.array([0.,0.,1.]))
        assert hit[0]==0 and np.max(np.abs(hit[1]-[.25,.25,.5]))<1e-14 and hit[2]==2
        assert ray_hit(test,np.array([4.,0.,1.])) is None
        with np.load(root/'fit/fitted.npz',allow_pickle=False) as z:
            fit={k:z[k] for k in ['vertices_world','vertices_camera','faces','frame_ids','source_frames','image_sha256','K_integer_centers','w2c']}
        rows=read(root/'data/training-inputs.json')['records']
        ids=fit['frame_ids'].astype(str).tolist()
        assert ids==[r['frame'] for r in rows] and len(rows)==94
        assert np.array_equal(fit['source_frames'],[r['source_frame_index'] for r in rows])
        assert fit['image_sha256'].tolist()==[r['source_sha256'] for r in rows]
        assert np.array_equal(fit['K_integer_centers'],[r['K_integer_centers'] for r in rows])
        assert np.array_equal(fit['w2c'],[r['w2c'] for r in rows])
        E,K=fit['w2c'],fit['K_integer_centers']
        world=fit['vertices_world'];faces=fit['faces'].astype(int)
        camera=np.einsum('nij,nvj->nvi',E[:,:3,:3],world)+E[:,:3,3][:,None]
        replay=float(np.max(np.abs(camera-fit['vertices_camera'])))
        assert replay<1e-5
        raw=relocate(plan['original_tracks_root']);prior=relocate(plan['prior_review_root'])
        for original in read(prior/'plan.json')['original_readonly_inputs']:
            if original['path'].endswith(('/decoded.json','/camera/appearance-tracks.json')):
                assert sha(relocate(original['path']))==original['sha256']
        decoded=read(raw/'decoded.json')['records']
        assert [d['source_sha256'] for d in decoded]==[r['source_sha256'] for r in rows]
        selected={s['id']:s for s in read(prior/'result.json')['samples']}
        notes={s['id']:s for s in read(prior/'visual-review.json')['samples']}
        tracks={t['id']:t for t in read(raw/'camera/appearance-tracks.json')['records'] if t['id'] in plan['sample_ids']}
        slots=sorted({i for tr in tracks.values() for i,_ in tr['nodes']})
        xy={};feature_hashes=[]
        for i in slots:
            p=raw/f'camera/features-{i:03d}.npz'
            with np.load(p,allow_pickle=False) as z:xy[i]=z['xy'].copy()
            feature_hashes.append(dict(path=str(p),sha256=sha(p)))
            bound()
        image_hashes={}
        @lru_cache(maxsize=6)
        def image(i):
            p=relocate(rows[i]['image'])
            digest=sha(p);assert digest==rows[i]['source_sha256']
            image_hashes[str(p)]=digest
            with Image.open(p) as im:
                assert im.size==tuple(rows[i]['wh'])
                return im.convert('RGB')
        @lru_cache(maxsize=8)
        def triangles_camera(i):return camera[i][faces]
        records=[];images=[]
        (out/'images').mkdir()
        record_dir=relocate('D:/00/记录/005/特征点')
        for index,tid in enumerate(plan['sample_ids']):
            bound();tr=tracks[tid];sample=selected[tid];note=notes[tid]
            assert tr['held']==sample['held']==note['held']
            nodes=sorted(tr['nodes']);anchor_slot,anchor_feature=nodes[0]
            obs={i:xy[i][j].astype(float) for i,j in nodes}
            for view in sample['views']:
                j=ids.index(view['frame'])
                assert np.array_equal(obs[j],view['xy'])
            hit=ray_hit(triangles_camera(anchor_slot),np.linalg.inv(K[anchor_slot])@np.r_[obs[anchor_slot],1.])
            item=dict(id=tid,sample_index=index,held=tr['held'],category=note['category'],physical_identity_note=note['note'],
                      anchor_frame=ids[anchor_slot],anchor_uv=obs[anchor_slot].tolist(),observations=len(nodes),mesh_ray_hit=hit is not None,rows=[])
            predictions={}
            if hit is not None:
                face,bary,depth=hit;tri=world[anchor_slot,faces[face]]
                c,A,s=face_frame(tri);point=bary@tri;local=A.T@(point-c)/s
                item.update(face_id=face,anchor_barycentric=bary.tolist(),anchor_local=local.tolist())
                for i,j in nodes:
                    bound();tri_i=world[i,faces[face]];ci,Ai,si=face_frame(tri_i)
                    q=ci+Ai@(local*si);qb=bary@tri_i
                    uv,pc=project(q,E[i],K[i]);uvb,_=project(qb,E[i],K[i])
                    row=dict(frame=ids[i],source_frame=rows[i]['source_frame_index'],observed_uv=obs[i].tolist(),anchor=(i==anchor_slot),positive_depth=uv is not None and uvb is not None)
                    if uv is not None and uvb is not None:
                        nearest=ray_hit(triangles_camera(i),pc/pc[2])
                        visible=nearest is not None and abs(nearest[2]-pc[2])<=max(1e-5,abs(pc[2])*1e-5)
                        row.update(transported_uv=uv.tolist(),barycentric_uv=uvb.tolist(),body_mesh_self_visible=bool(visible),
                                   error_px=float(np.linalg.norm(uv-obs[i])),barycentric_error_px=float(np.linalg.norm(uvb-obs[i])),
                                   transport_vs_barycentric_px=float(np.linalg.norm(uv-uvb)))
                        predictions[i]=uv
                    item['rows'].append(row)
                check=[x for x in item['rows'] if not x['anchor'] and x['positive_depth']]
                item['transport_error_px']=stats([x['error_px'] for x in check])
                item['self_visible_error_px']=stats([x['error_px'] for x in check if x['body_mesh_self_visible']])
                item['transport_vs_barycentric_px']=stats([x['transport_vs_barycentric_px'] for x in check])
            records.append(item)
            # Historical six views are preserved. Missing hit yields no purple point.
            sheet=Image.new('RGB',(528,len(sample['views'])*176))
            for row_idx,view in enumerate(sample['views']):
                i=ids.index(view['frame']);box=view['crop'];assert box[2]-box[0]==box[3]-box[1]==176
                native=image(i).crop(box);det=native.copy();pred=native.copy()
                for panel,uv,color in [(det,obs[i],(0,255,0)),(pred,predictions.get(i),(200,0,255))]:
                    if uv is None:continue
                    u,v=uv-np.array(box[:2]);draw=ImageDraw.Draw(panel)
                    draw.line([(u-4,v),(u+4,v)],fill=color,width=1)
                    draw.line([(u,v-4),(u,v+4)],fill=color,width=1)
                for col,panel in enumerate([native,det,pred]):sheet.paste(panel,(col*176,row_idx*176))
            name=f'20261030_官方表面跨帧_{index:02d}_{tid[:10]}_原图_已有纹理点绿_固定表面点紫.png'
            dst=record_dir/name;assert not dst.exists();sheet.save(dst)
            copy=out/'images'/f'track-{index:02d}.png';sheet.save(copy)
            images.append(dict(sample_index=index,id=tid,category=note['category'],path=str(dst),review_copy=str(copy),sha256=sha(copy),no_text=True,native_crops_unchanged=True,actual_viewed=False))
        aggregates=[]
        for held in [False,True]:
            rr=[x for x in records if x['held']==held]
            vv=[v for x in rr for v in x['rows'] if not v['anchor'] and v['positive_depth']]
            aggregates.append(dict(split='HELD' if held else 'FIT',tracks=len(rr),mesh_hit_tracks=sum(x['mesh_ray_hit'] for x in rr),
                transport_error_px=stats([v['error_px'] for v in vv]),
                self_visible_error_px=stats([v['error_px'] for v in vv if v['body_mesh_self_visible']]),
                transport_vs_barycentric_px=stats([v['transport_vs_barycentric_px'] for v in vv])))
        for item in plan['inputs']+feature_hashes:
            assert sha(relocate(item['path']))==item['sha256']
        result=dict(state='completed_pending_visual_review',updated=now(),seconds=time.monotonic()-started,
            world_camera_vertex_replay_max=replay,known_triangle_ray_checks=True,plan_sha256=sha(out/'plan.json'),
            records=records,aggregate=aggregates,images=images,feature_hashes=feature_hashes,image_hashes=image_hashes,
            original_sources_unchanged=True,new_features=0,optimized_points=0,camera_updates=0,training_steps=0,
            quality_accepted=False,independent_physical_correspondence_accepted=False,limits=plan['limits'])
        save(out/'result.json',result);save(out/'progress.json',dict(state='complete',pid=None,seconds=result['seconds']))
        print(json.dumps(dict(state=result['state'],aggregate=aggregates,seconds=result['seconds'])))
    except BaseException as exc:
        save(out/'failure.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),updated=now()))
        save(out/'progress.json',dict(state='failed',pid=None))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--register',action='store_true');a=p.parse_args()
    (register if a.register else run)(relocate(a.root))
