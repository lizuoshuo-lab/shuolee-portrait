"""Independent CPU audit; reads derived inputs and never imports a trainer/CUDA."""
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import random
import struct
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image

ROOT = Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = Path(__file__).resolve().parent
DERIVED = ROOT / 'outputs/large-scene-gs-001/official-inputs/truck-four-chunks-001'
BASELINE = ROOT / 'outputs/large-scene-gs-001/training/truck-baseline-001'
PREP = ROOT / 'outputs/large-scene-gs-001/data-preparation'
SCENE = ROOT / 'data/large-scene-gs/tandt/truck'
REPO = ROOT / 'repos/hierarchical-3d-gaussians'

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def readjson(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def unpack(f, fmt):
    n = struct.calcsize('<'+fmt)
    b = f.read(n)
    require(len(b) == n, 'Truncated COLMAP binary')
    return struct.unpack('<'+fmt, b)

def quat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def readmodel(path):
    cameras, images, points = {}, {}, {}
    with (path/'cameras.bin').open('rb') as f:
        n, = unpack(f,'Q')
        for _ in range(n):
            cid, model, width, height = unpack(f,'iiQQ')
            require(model == 1, 'This audit expects actual PINHOLE model')
            params = np.array(unpack(f,'4d'))
            require(cid not in cameras, 'Duplicate camera ID')
            cameras[cid] = SimpleNamespace(id=cid,model='PINHOLE',width=width,height=height,params=params)
        require(not f.read(1), 'Camera trailing bytes')
    with (path/'images.bin').open('rb') as f:
        n, = unpack(f,'Q')
        for _ in range(n):
            iid,*v = unpack(f,'i7di')
            name = bytearray()
            while True:
                b = f.read(1)
                require(bool(b), 'Image name truncated')
                if b == b'\0': break
                name.extend(b)
            count, = unpack(f,'Q')
            raw = f.read(count*24)
            require(len(raw)==count*24 and iid not in images, 'Image record truncated/duplicate')
            obs = np.frombuffer(raw, dtype=[('x','<f8'),('y','<f8'),('point_id','<i8')]).copy()
            q = np.array(v[:4]); t = np.array(v[4:7]); R = quat(q)
            images[iid] = SimpleNamespace(id=iid,qvec=q,tvec=t,camera_id=v[7],name=name.decode(),
                                         obs=obs,R=R,center=-R.T@t)
        require(not f.read(1), 'Image trailing bytes')
    with (path/'points3D.bin').open('rb') as f:
        n, = unpack(f,'Q')
        for _ in range(n):
            v = unpack(f,'Q3d3BdQ')
            count = v[8]; raw=f.read(count*8)
            require(len(raw)==count*8 and v[0] not in points, 'Point track truncated/duplicate')
            tracks = np.frombuffer(raw,dtype=[('image_id','<i4'),('point2D_idx','<i4')]).copy()
            points[v[0]] = SimpleNamespace(id=v[0],xyz=np.array(v[1:4]),rgb=np.array(v[4:7],dtype=np.uint8),
                                          error=v[7],tracks=tracks)
        require(not f.read(1), 'Point trailing bytes')
    return cameras,images,points

def K(cam):
    fx,fy,cx,cy = cam.params
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])

def c2w(im):
    out=np.eye(4);out[:3,:3]=im.R.T;out[:3,3]=im.center
    return out

def summaries(a):
    a=np.asarray(a)
    return dict(min=float(a.min()),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),max=float(a.max()))

def official_split_cpu(path, cams, images, eval_flag, train_test_exp):
    """Execute official reader AST with independent metadata and stubbed PCD I/O.

    No scene module import, PLY creation, Camera construction, or CUDA call.
    The official test-file and train/test boolean branches execute unchanged.
    """
    source = (REPO/'scene/dataset_readers.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    wanted={'CameraInfo','SceneInfo','readColmapCameras','readColmapSceneInfo'}
    nodes=[n for n in tree.body if getattr(n,'name',None) in wanted]
    require(len(nodes)==4,'Official reader AST functions missing')
    scope={'NamedTuple':NamedTuple,'np':np,'os':os,'sys':sys,'BasicPointCloud':object,
           'focal2fov':lambda focal,pixels:2*math.atan(pixels/(2*focal)),
           'qvec2rotmat':quat,'read_extrinsics_binary':lambda p:images,'read_intrinsics_binary':lambda p:cams,
           'fetchPt':lambda xyz,rgb:None,'getNerfppNorm':lambda c:dict(radius=0.,translate=[0.,0.,0.])}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REPO/'scene/dataset_readers.py'),'exec'),scope)
    with contextlib.redirect_stdout(io.StringIO()):
        result=scope['readColmapSceneInfo'](str(path),str(SCENE/'images_4'),'','',eval_flag,train_test_exp)
    return dict(train=sorted(c.image_name for c in result.train_cameras),test=sorted(c.image_name for c in result.test_cameras))

def main():
    started=datetime.now(timezone.utc).isoformat()
    manifest=readjson(DERIVED/'manifest.json'); source_manifest=readjson(PREP/'input-manifest.json')
    frozen=readjson(PREP/'split.json'); baseline_split=readjson(BASELINE/'split.json')
    require(baseline_split['train_names']==frozen['train_names'] and baseline_split['test_names']==frozen['test_names'],'Baseline frozen split mismatch')
    require(digest(PREP/'split.json')==source_manifest['split_sha256'],'Frozen split SHA mismatch')
    configuration=readjson(BASELINE/'configuration.json')
    with np.load(BASELINE/'parser_geometry.npz',allow_pickle=False) as f:
        g={k:f[k] for k in f.files}
    require(digest(BASELINE/'parser_geometry.npz')==manifest['baseline_geometry_sha256'],'Geometry SHA mismatch')
    names=g['image_names'].tolist(); name_index={n:i for i,n in enumerate(names)}
    mapped={n:str(Path(n).with_suffix('.png')) for n in names}
    train={mapped[n] for n in frozen['train_names']}; test={mapped[n] for n in frozen['test_names']}
    require(mapped==manifest['original_to_png'] and len(set(mapped.values()))==251,'PNG mapping mismatch/collision')
    require(sorted(train)==sorted(manifest['train_names']) and sorted(test)==sorted(manifest['test_names']),'Derived split mismatch')
    require(set(train).isdisjoint(test) and len(train)==219 and len(test)==32,'Frozen split leakage')
    file_manifest=readjson(PREP/'data-files.json')
    require(digest(PREP/'data-files.json')==source_manifest['file_manifest_sha256'],'Source file manifest identity mismatch')
    bypath={x['path'].replace('\\','/'):x for x in file_manifest}
    verified=[]
    for name in mapped.values():
        path=SCENE/'images_4'/name; relative='images_4/'+name
        require(digest(path)==bypath[relative]['sha256'],'Derived image hash mismatch')
        with Image.open(path) as im:
            im.load(); require(im.mode=='RGB' and im.size==(245,136),'RGB image dimensions mismatch')
        verified.append(dict(name=name,sha256=digest(path),width=245,height=136))
    for filename in ('cameras.bin','images.bin','points3D.bin'):
        require(digest(SCENE/'sparse/0'/filename)==bypath['sparse/0/'+filename]['sha256'],'Original model file changed')
    oc,oi,op=readmodel(SCENE/'sparse/0'); ac,ai,ap=readmodel(DERIVED/'aligned/sparse/0')
    require(set(oc)==set(ac) and set(oi)==set(ai) and set(op)==set(ap),'Aligned IDs differ from source')
    require(len(oi)==251 and len(op)==136029,'Unexpected original source counts')
    require((DERIVED/'aligned/sparse/0/test.txt').read_text().splitlines()==[mapped[n] for n in frozen['test_names']],'Aligned test file not frozen')
    T=g['global_transform']; A=T[:3,:3]; shift=T[:3,3]
    sv=np.linalg.svd(A,compute_uv=False); scale=float(sv.mean()); Q=A/scale
    require(np.max(np.abs(sv-scale))<1e-12 and np.linalg.det(Q)>0 and np.max(np.abs(Q.T@Q-np.eye(3)))<1e-12,'T not positive isotropic Sim3')
    point_errors=[]; point_ids=np.array(sorted(op),dtype=np.uint64)
    point_row={int(p):i for i,p in enumerate(point_ids)}
    source_xyz=np.array([op[int(p)].xyz for p in point_ids]); aligned_xyz=np.array([ap[int(p)].xyz for p in point_ids])
    point_max=float(np.max(np.abs(aligned_xyz-(source_xyz.astype(np.float32).astype(np.float64)@A.T+shift))))
    require(point_max<1e-12,'Aligned point Sim3 mismatch')
    for pid,p in op.items():
        a=ap[pid]
        require(np.array_equal(p.rgb,a.rgb) and p.error==a.error and np.array_equal(p.tracks,a.tracks),'Aligned point RGB/error/track changed')
        for iid in np.unique(p.tracks['image_id']):
            obs_indices=p.tracks['point2D_idx'][p.tracks['image_id']==iid]
            require(int(iid) in oi and np.all(obs_indices>=0) and np.all(obs_indices<len(oi[int(iid)].obs)),'Track index invalid')
            require(np.all(oi[int(iid)].obs['point_id'][obs_indices]==pid),'Original track reciprocity broken')
    cam_max=xy_max=proj_max=depth_max=rot_max=0.; positive=observations=0
    for iid,im in oi.items():
        a=ai[iid]; index=name_index[im.name]
        require(a.name==mapped[im.name] and a.camera_id==im.camera_id,'Aligned image name/camera ID differs')
        require(np.array_equal(im.obs['point_id'],a.obs['point_id']),'Aligned image point IDs changed')
        cam_max=max(cam_max,float(np.max(np.abs(c2w(a)-g['camtoworlds'][index]))))
        expected_c=c2w(im).copy(); expected_c[:3,:3]=Q@expected_c[:3,:3];expected_c[:3,3]=A@im.center+shift
        require(np.max(np.abs(c2w(a)-expected_c))<1e-12,'Camera trajectory Sim3 mismatch')
        rot_max=max(rot_max,float(np.max(np.abs(a.R@a.R.T-np.eye(3)))))
        oldK,newK=K(oc[im.camera_id]),K(ac[a.camera_id])
        require(ac[a.camera_id].width==245 and ac[a.camera_id].height==136,'Aligned camera wrong dimensions')
        require(np.array_equal(newK,g['K'][index]),'Aligned intrinsics are not baseline saved K')
        xy=np.column_stack([im.obs['x'],im.obs['y']]); targetxy=np.column_stack([a.obs['x'],a.obs['y']])
        expectedxy=(xy-oldK[:2,2])*(np.array([newK[0,0]/oldK[0,0],newK[1,1]/oldK[1,1]]))+newK[:2,2]
        xy_max=max(xy_max,float(np.max(np.abs(targetxy-expectedxy))))
        mask=im.obs['point_id']>=0; pids=im.obs['point_id'][mask]
        rows=np.array([point_row[int(p)] for p in pids])
        pc_old=source_xyz[rows].astype(np.float32).astype(np.float64)@im.R.T+im.tvec
        pc_new=aligned_xyz[rows]@a.R.T+a.tvec
        olduv=pc_old[:,:2]/pc_old[:,2:3]*np.array([newK[0,0],newK[1,1]])+newK[:2,2]
        newuv=pc_new[:,:2]/pc_new[:,2:3]*np.array([newK[0,0],newK[1,1]])+newK[:2,2]
        proj_max=max(proj_max,float(np.max(np.abs(olduv-newuv))))
        depth_max=max(depth_max,float(np.max(np.abs(pc_new[:,2]-scale*pc_old[:,2]))))
        positive+=int(np.count_nonzero(pc_new[:,2]>0)); observations+=len(pids)
    require(cam_max<1e-12 and xy_max<1e-9 and proj_max<1e-7 and positive==observations,'Global projection/positive-depth check failed')
    camera_centers=np.array([ai[iid].center for iid in sorted(ai,key=lambda i:ai[i].name)])
    np.savez(OUT/'global-cameras-points-audit.npz',image_names=np.array(sorted(mapped.values())),camera_centers=camera_centers,
             point_ids=point_ids,points_xyz=aligned_xyz,global_transform=T)
    geometry_audit=dict(status='passed_global_frame_and_track_identity',image_count=len(ai),point_count=len(ap),
        original_model_dimensions=[1957,1091],original_image_dimensions=[979,546],derived_dimensions=[245,136],
        camera_model='PINHOLE',K=g['K'][0].tolist(),sim3_scale=scale,scale_unit='arbitrary normalized SfM units, not metres',
        camera_c2w_max_abs_error=cam_max,point_xyz_max_abs_error=point_max,rotation_orthonormal_max_error=rot_max,
        transformed_observation_xy_max_error_px=xy_max,projection_pair_count=observations,projection_max_error_px=proj_max,
        depth_scale_max_error=depth_max,positive_depth_count=positive,original_point_tracks_retained_in_aligned=True,
        point_error_metadata='Original COLMAP model-pixel error retained; not recomputed for resized derived K',
        trajectory_step_lengths_normalized_units=summaries(np.linalg.norm(np.diff(camera_centers,axis=0),axis=1)))
    save('geometry-audit.json',geometry_audit)
    # Independently reproduce official seed-0 selection using parsed bytes.
    aligned_order=list(ai.values())
    centers32=np.array([-x.R.astype(np.float32).T@x.tvec.astype(np.float32) for x in aligned_order])
    xyz32=aligned_xyz.astype(np.float32); errors32=np.array([ap[int(p)].error for p in point_ids],dtype=np.float32)
    eligible=errors32<10.; bbox=np.stack([centers32.min(axis=0),centers32.max(axis=0)])
    size=float(manifest['chunk_size_normalized_scene_units']); extent=bbox[1]-bbox[0]
    padding=np.array([size-extent[0]%size,size-extent[1]%size]);bbox[0,:2]-=padding/2;bbox[1,:2]+=padding/2
    bbox[0,2]=-1e12;bbox[1,2]=1e12
    nw,nh=round((bbox[1,0]-bbox[0,0])/size),round((bbox[1,1]-bbox[0,1])/size)
    require((nw,nh)==(2,2),'Unexpected grid dimensions')
    points_ordered=np.zeros((int(point_ids.max())+1,3),dtype=np.float64);points_ordered[point_ids[eligible]]=xyz32[eligible]
    images_points={}
    for image in aligned_order:
        p=image.obs['point_id'];p=p[(p>=0)&(p<len(points_ordered))];xyz=points_ordered[p]
        images_points[image.id]=xyz[(xyz!=0).sum(axis=-1)>0]
    rng=random.Random(0);chunks=[];membership=np.zeros((len(point_ids),4),dtype=bool)
    all_train=set();all_test=set();view_sets={};point_sets={};future_flags=[];source_visibility=[]
    for i in range(nw):
        for j in range(nh):
            name=f'{i}_{j}';croot=DERIVED/'chunks'/name;cc,ci,cp=readmodel(croot/'sparse/0')
            lo=bbox[0]+np.array([i*size,j*size,0]);hi=bbox[0]+np.array([(i+1)*size,(j+1)*size,1e12]);lo[2]=-1e12;hi[2]=1e12
            plo,phi=lo.copy(),hi.copy()
            if i==0:plo[0]=-1e12
            if j==0:plo[1]=-1e12
            if i==nw-1:phi[0]=1e12
            if j==nh-1:phi[1]=1e12
            owned=eligible&np.all(xyz32>plo,axis=1)&np.all(xyz32<phi,axis=1)
            expected_ids=set(map(int,point_ids[owned]));require(set(cp)==expected_ids,'Chunk point ownership differs from official spatial predicate')
            membership[:,i*nh+j]=owned
            valid=np.all(centers32>lo,axis=1)&np.all(centers32<hi,axis=1)
            center=(lo+hi)/2;halves=(hi-lo)/2
            for n,im in enumerate(aligned_order):
                v=images_points[im.id];npoints=int((np.all(v>plo,axis=1)&np.all(v<phi,axis=1)).sum()) if len(v) else 0
                if np.all(centers32[n]>lo) and np.all(centers32[n]<hi):valid[n]=npoints>50
                elif np.all(centers32[n]>center-2*halves) and np.all(centers32[n]<center+2*halves):valid[n]=npoints>50 and rng.uniform(0,1)>0.5
                if not valid[n] and npoints>10:valid[n]=rng.uniform(0,.5)<float(npoints)/len(v)
            selected_ids={im.id for n,im in enumerate(aligned_order) if valid[n]}
            require(set(ci)==selected_ids,'Seed-0 chunk view selection differs')
            require(np.max(np.abs(np.loadtxt(croot/'center.txt')-center))<1e-12,'Chunk center mismatch')
            require(np.max(np.abs(np.loadtxt(croot/'extent.txt')-(hi-lo)))<1e-12,'Chunk extent mismatch')
            require((croot/'sparse/0/test.txt').read_text().splitlines()==[mapped[n] for n in frozen['test_names']],'Chunk test file not global frozen file')
            for cid,cam in cc.items():
                require(cid in ac and np.array_equal(cam.params,ac[cid].params) and (cam.width,cam.height)==(245,136),'Chunk K differs from common K')
            for iid,im in ci.items():
                require(im.name==ai[iid].name and np.array_equal(im.qvec,ai[iid].qvec) and np.array_equal(im.tvec,ai[iid].tvec),'Chunk camera not in common global gauge')
                require(len(im.obs)==0,'Official chunk observations unexpectedly retained')
            for pid,p in cp.items():
                a=ap[pid]
                require(np.array_equal(p.xyz,a.xyz.astype(np.float32).astype(np.float64)) and np.array_equal(p.rgb,a.rgb),'Chunk point position/RGB changed beyond official float32 storage')
                require(p.error==float(np.float32(a.error)) and len(p.tracks)==0,'Chunk point error/tracks differ from official storage')
            present={im.name for im in ci.values()};ct=present-test;ce=present&test
            require(ct<=train,'Non-frozen train view entered chunk')
            all_train|=ct;all_test|=ce;view_sets[name]=ct;point_sets[name]=set(cp)
            m=next(x for x in manifest['chunks'] if x['name']==name)
            require(sorted(ct)==m['train_names'] and sorted(ce)==m['test_names'] and len(cp)==m['initial_points'],'Derived manifest chunk differs from binary')
            split_ok=official_split_cpu(croot,cc,ci,True,False)
            split_noeval=official_split_cpu(croot,cc,ci,False,False)
            split_exp=official_split_cpu(croot,cc,ci,True,True)
            require(split_ok==dict(train=sorted(ct),test=sorted(ce)),'Official eval true split mismatch')
            require(set(split_noeval['train'])==present and not split_noeval['test'],'Official no-eval behavior changed')
            require(set(split_exp['train'])==present and set(split_exp['test'])==ce,'Official exposure-half behavior changed')
            train_iids={iid for iid,x in ci.items() if x.name in ct}
            support=np.array([np.count_nonzero(np.isin(ap[pid].tracks['image_id'],list(train_iids))) for pid in sorted(cp)],dtype=np.int32)
            source_visibility.append(dict(name=name,point_ids=np.array(sorted(cp),dtype=np.uint64),train_observation_counts=support))
            chunk=dict(name=name,total_images=len(ci),train_images=len(ct),test_images=len(ce),initial_points=len(cp),
                center=center.tolist(),extent=(hi-lo).tolist(),point_bound_min=plo.tolist(),point_bound_max=phi.tolist(),
                cleared_image_observations=True,cleared_point_tracks=True,source_point_ids_retained=True,
                selected_points_with_zero_selected_train_observations=int(np.count_nonzero(support==0)),
                source_track_train_support_summary=summaries(support),train_names=sorted(ct),test_names=sorted(ce))
            chunks.append(chunk)
            future_flags.append(dict(name=name,required_flags=dict(eval=True,train_test_exp=False,resolution=1),
                                     correct=dict(train=len(ct),test=len(ce)),without_eval=dict(train=len(present),test=0),
                                     with_train_test_exp=dict(train=len(present),test=len(ce))))
    require(all_train==train and all_test==test,'Some frozen cameras not covered by chunk union')
    multiplicity=membership.sum(axis=1)
    require(np.all(multiplicity[eligible]==1) and np.all(multiplicity[~eligible]==0),'Point ownership coverage/duplicates failure')
    pairs=[]
    for n,a in enumerate(chunks):
        for b in chunks[n+1:]:
            pairs.append(dict(a=a['name'],b=b['name'],train_view_overlap=len(view_sets[a['name']]&view_sets[b['name']]),
                              point_overlap=len(point_sets[a['name']]&point_sets[b['name']])))
    require([dict(a=p['a'],b=p['b'],views=p['train_view_overlap']) for p in pairs]==manifest['train_pair_overlaps'],'Manifest overlap differs')
    np.savez(OUT/'point-ownership-audit.npz',point_ids=point_ids,chunk_names=np.array([x['name'] for x in chunks]),membership=membership)
    for row in source_visibility:
        np.savez(OUT/f"{row['name']}-source-track-support.npz",point_ids=row['point_ids'],train_observation_counts=row['train_observation_counts'])
    aligned_flags=official_split_cpu(DERIVED/'aligned',ac,ai,True,False)
    require(aligned_flags==dict(train=sorted(train),test=sorted(test)),'Official scaffold split mismatch')
    split_audit=dict(status='passed_official_reader_metadata_split_cpu',global_train_count=len(train),global_test_count=len(test),
        chunks=future_flags,required_future_training_flags='--eval; omit --train_test_exp (False); --resolution 1 for already-downscaled images',
        scope='Executed unchanged official reader AST; independent binary metadata and inert point-cloud loading stubs; no Camera/GPU/trainer import',
        heldout_boundary='Only GS photometric image optimization is held out. Public SfM, sparse initialization and visibility partition may include all cameras/tracks, including test.')
    save('split-audit.json',split_audit)
    coverage=dict(status='passed_complete_nonoverlapping_point_ownership_and_overlapping_view_coverage',chunks=chunks,pairs=pairs,
        global_sparse_point_count=len(ap),eligible_points=int(eligible.sum()),owned_points=int(np.count_nonzero(multiplicity)),
        unowned_eligible_points=int(np.count_nonzero(eligible&(multiplicity==0))),multiply_owned_points=int(np.count_nonzero(multiplicity>1)),
        train_union=len(all_train),test_union=len(all_test),total_train_view_occurrences=sum(len(x) for x in view_sets.values()),
        point_partition='Strict open X/Y spatial cuts, unlimited outer edge bounds, error<10 official filter; no point overlap in these inputs',
        limitation='View overlap is input context, not evidence of trained Gaussian continuity or duplicate removal. Chunk COLMAP associations are intentionally empty; support audit uses retained aligned global tracks.')
    save('coverage-audit.json',coverage)
    # Pixel-centre conventions: gsplat evaluates at (j+.5,i+.5); GraphDeco
    # projects K means to (u-.5,v-.5) and evaluates at (j,i), equivalent offsets.
    convention=dict(status='source_convention_offsets_equivalent',graphdeco_mean_shift_relative_to_K_pixels=[-.5,-.5],
        graphdeco_pixel_samples='integer pixel indices',gsplat_pixel_samples='pixel indices + 0.5',
        gaussian_offset_difference_px=[0.,0.],principal_point_retained=True,
        limitation='Source mathematical check only; does not claim numerical identity of distinct rasterizers, covariance clamps, or near-plane handling')
    save('renderer-coordinate-conventions.json',convention)
    pinned=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    official_paths=['preprocess/make_chunk.py','scene/dataset_readers.py','scene/cameras.py','utils/graphics_utils.py']
    sources=[]
    for relative in official_paths:
        p=REPO/relative;committed=subprocess.check_output(['git','-C',str(REPO),'show',f'HEAD:{relative}'])
        # Git working tree CRLF can differ from committed LF; compare normalized bytes.
        require(p.read_bytes().replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'),'Official source modified')
        sources.append(dict(path=str(p),sha256=digest(p),unmodified_from_git_head=True))
    paths=[DERIVED/'manifest.json',DERIVED/'partition.log',BASELINE/'parser_geometry.npz',BASELINE/'split.json',PREP/'split.json',PREP/'input-manifest.json',ROOT/'scripts/large_scene_gs/prepare_official_inputs.py']
    paths.extend(sorted(p for p in DERIVED.rglob('*') if p.is_file() and p.name in {'cameras.bin','images.bin','points3D.bin','test.txt','center.txt','extent.txt','blending_dict.json'}))
    provenance=dict(official_hierarchical_commit=pinned,sources=sources,inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in paths],
                    source_images_verified=verified,audit_script_sha256=digest(__file__),read_only_inputs=True,cuda_used=False)
    save('provenance.json',provenance)
    report='''# 四块官方输入独立 CPU 审计

共同全局 Sim(3)、相机轨迹、基线 245×136 内参与 JPG→PNG 对应通过。251 个相机与 136,029 个点的 ID 未重排；aligned 全局模型完整保持原始观测点 ID 和双向 tracks，1,083,072 对投影均为正深度。

四块的实际 train/test/point 数为 137/20/33,666、149/25/45,109、96/16/21,001、137/21/36,253。219 个 train、32 个 test 的并集均完整覆盖；点所有权覆盖全部 136,029 点，各点只属于一块。六对训练视图重叠数为 92、59、76、54、85、62。视图重叠不能代表已训练的块边界连续性。

官方 make_chunk.py 明确清空块内 images.bin 的二维观测与 points3D.bin 的 tracks；这一步保留原点 ID，但不能称各块仍保存原 tracks。全局 aligned 模型保持关联，点的选中训练相机可见性另用该全局轨迹审计并保存在逐块 source-track-support.npz。输入稀疏点与分块可见性来自完整场景 SfM，因此这不是对未知相机进行的端到端重建留出。

CPU 执行未改动的官方 reader 元数据分支确认：未来 scaffold 与 chunk 均必须 --eval，且 train_test_exp=False（不要传 --train_test_exp）；已缩小图像使用 --resolution 1。没有 --eval 时测试图全部进入训练；启用 train_test_exp 时测试图左半幅可参加训练，均不符合本次完整留出协议。

块边长 1.4 使用归一化 SfM 单位，没有米制标定，不可写成 1.4 米。相机和点保持共同全局框架，未按块另行归一化。原 point.error 被作为历史 COLMAP 像素误差保留，没有重算为缩小后的重投影误差。

官方相机保留离中心主点。源码检查表明 GraphDeco 投影均值的 −0.5 像素约定与 gsplat 的 +0.5 像素采样位置对应相同高斯偏移；这不证明两种光栅器数值完全一致。此次没有加载 trainer、Camera 或 CUDA，也没有修改输入文件。

所有数值、逐块支持统计、冻结 test-file 检查与 SHA256 身份见 geometry-audit.json、coverage-audit.json、split-audit.json、renderer-coordinate-conventions.json 和 provenance.json。
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    outcome=dict(status='passed_independent_cpu_four_chunk_input_audit',started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),
        cuda_used=False,global_images=251,global_points=136029,chunk_count=4,train=219,test=32,
        all_original_tracks_retained_in_aligned=True,chunk_tracks_intentionally_cleared=True)
    save('outcome.json',outcome)
    print(json.dumps(outcome,ensure_ascii=False))

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        save('outcome.json',dict(status='failed_independent_cpu_audit',error_type=type(e).__name__,error=str(e),cuda_used=False))
        raise
