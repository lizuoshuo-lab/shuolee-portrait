"""Bind source-verified 021 shoe depth to visible SMPL-X foot surfaces."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fit_video21_seated149 as seated
from extract_video21_shoe_depth162 import OUT as MATERIAL
from update_leafwoman129_state import atomic, read, sha


BASE = seated.base.OUT
CAMERA = BASE/'fixedk-dense159-reference/seated021/global-fixedk/cameras-diagnostic.json'
OUT = BASE/'shoe-mesh-binding163'
SIDES = {
    'image_left_shoe_welt':'right', 'image_right_shoe_welt':'left',
    'near_shoe_eyelet':'right', 'far_shoe_lace':'left',
}
FOOT_BONES = {'left': [7,10], 'right': [8,11]}


def visible_hit(uv, triangles, depth, p):
    """Pixel-center ray through the nearest intersected triangle."""
    a,b,c = (uv[triangles[:,j]] for j in range(3))
    ab,ac = b-a,c-a
    det = ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    valid = np.abs(det)>1e-8
    det = np.where(valid,det,1.)
    d = np.asarray(p)-a
    wb = (d[:,0]*ac[:,1]-d[:,1]*ac[:,0])/det
    wc = (ab[:,0]*d[:,1]-ab[:,1]*d[:,0])/det
    w = np.column_stack([1.-wb-wc,wb,wc])
    valid &= (w>=-1e-8).all(1) & (depth[triangles]>.05).all(1)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return None
    bary_over_z = w[ids]/depth[triangles[ids]]
    inverse_z = bary_over_z.sum(1)
    k = int(np.argmax(inverse_z))
    return int(ids[k]),bary_over_z[k]/inverse_z[k],float(1./inverse_z[k])


def bind_one(verts, faces, tags, K, p, side):
    q=verts@K.T
    uv=q[:,:2]/q[:,2,None]
    offsets=sorted((dx*dx+dy*dy,dx,dy) for dx in range(-2,3) for dy in range(-2,3)
                   if dx*dx+dy*dy<=4)
    first_hit=None
    for sq,dx,dy in offsets:
        xy=np.asarray(p)+[dx,dy]
        hit=visible_hit(uv,faces,verts[:,2],xy)
        if hit is None:
            continue
        face,w,z=hit
        if first_hit is None:
            first_hit=tags[face]
        if tags[face]!=side:
            continue
        ray_point=(verts[faces[face]]*w[:,None]).sum(0)
        projected=ray_point@K.T
        assert abs(ray_point[2]-z)<1e-8 and np.max(np.abs(projected[:2]/projected[2]-xy))<1e-6
        return dict(bound=True,face=face,vertices=faces[face].tolist(),barycentric=w.tolist(),
                    bound_pixel=xy.tolist(),pixel_distance=float(np.sqrt(sq)),seed_z_model=z)
    return dict(bound=False,first_intersected_region=first_hit)


def main():
    assert not OUT.exists()
    source=read(MATERIAL/'measurements.json')
    assert source['replay_result_sha256']==sha(MATERIAL/'result.json')
    plan=read(seated.OUT/'plan.json')
    seed=read(seated.OUT/'result.json')
    assert seed['plan_sha256']==sha(seated.OUT/'plan.json')
    camera=read(CAMERA)
    cams={r['frame']:r for r in camera}
    plan159=read(CAMERA.parents[2]/'global-fixedk-plan.json')
    assert plan159['candidate_frames']==plan['source_frames']
    native={f:n for f,n in zip(plan['source_frames'],plan['native_indices'],strict=True)}
    camera_native={r['source_index']:cams[r['frame']]
                   for r in read(BASE/'dense-camera-tracks158/result.json')['camera_frame_receipts']}
    seeds={row['frame']:row for row in seed['frames']}
    assert set(seeds)==set(plan['source_frames'])
    K=np.asarray(plan['K_integer_centers'])
    model=seated.base.original.ASSET
    with np.load(model,allow_pickle=False) as z:
        weights=z['weights']
    output={}
    for window,measurements in source['reviewed_materials'].items():
        assert len(measurements)==2
        chosen=[f for f,n in native.items() if all(n in m['fit_native_indices'] for m in measurements)]
        for f in chosen:
            assert f not in plan['retained_holdouts']
            row=seeds[f]; assert sha(row['output'])==row['sha256']
            with np.load(row['output'],allow_pickle=False) as z:
                verts=z['vertices_camera'][0].astype(float)
                faces=z['faces']
                assert np.array_equal(K,z['K_integer_centers'])
            tags=np.full(len(faces),'other',dtype='<U5')
            for side,indices in FOOT_BONES.items():
                foot=weights[:,indices].sum(1)>.5
                tags[np.all(foot[faces],axis=1)]=side
            C=np.asarray(cams[f]['C_world'])
            R=np.asarray(cams[f]['R_w2c'])
            items=[]
            for m in measurements:
                side=SIDES[m['material']]
                uv=m['observed_uv'][str(native[f])]
                X=np.asarray(m['xyz_arbitrary_scale'])
                z=float((R@(X-C))[2])
                assert z>0
                rays=[X-np.asarray(camera_native[i]['C_world']) for i in m['triangulated_from']]
                dirs=[ray/np.linalg.norm(ray) for ray in rays]
                angle=float(np.arccos(np.clip(dirs[0]@dirs[1],-1.,1.)))
                hit=bind_one(verts,faces,tags,K,uv,side)
                items.append(dict(material=m['material'],side=side,source_uv=uv,measured_z_gauge=z,
                                  triangulation_angle_deg=float(np.degrees(angle)),**hit))
            items.sort(key=lambda m:m['side'])
            assert [m['side'] for m in items]==['left','right']
            measured=float(np.log(items[0]['measured_z_gauge']/items[1]['measured_z_gauge']))
            usable=all(m['bound'] and m['triangulation_angle_deg']>=.5 for m in items)
            result=dict(frame=f,window=window,native_index=native[f],seed=row,materials=items,
                        measured_log_left_over_right=measured,usable=usable)
            if usable:
                result['seed_log_left_over_right']=float(np.log(items[0]['seed_z_model']/items[1]['seed_z_model']))
            output[f]=result
            print('SHOE_BINDING',f,[(m['side'],m['bound'],m.get('pixel_distance'),m.get('first_intersected_region')) for m in items],
                  'target',round(measured,4),'seed',result.get('seed_log_left_over_right'),flush=True)
    atomic(OUT/'result.json',dict(updated=datetime.now(timezone.utc).isoformat(),
        worker_sha256=sha(__file__),measurements_sha256=sha(MATERIAL/'measurements.json'),
        camera_sha256=sha(CAMERA),model_sha256=sha(model),seated_fit_sha256=sha(seated.OUT/'result.json'),
        association_rule='Nearest visible triangle at observed ray or within 2 px, both feet must match official topology. At least .5 degree source triangulation. No alternate seed selected.',
        retained_holdouts=plan['retained_holdouts'],rows=output,
        absolute_metric_scale_known=False,gaussian_started=False))


if __name__=='__main__':
    main()
