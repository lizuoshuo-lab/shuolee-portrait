"""Fresh video frontend for route 119; no historical frames, cameras or masks.

Camera candidates use fresh SIFT tracks, whole-track holdouts reserved before
geometry, OpenCV Essential/PnP and sparse fixed-K BA. The deployed camera profile
maps the supplied rectified camera K through resize and centered crop for either
video orientation. This does not establish independent physical accuracy.

The numerical camera procedure is migrated from the audited
capture118_all_subjects.py research procedure; paths, old caches and per-person
state are deliberately absent. Official YOLO/SAM2/RTMW produce observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

DEFAULTS = {
    'frame_stride': 5, 'max_image_size': 1600, 'min_frames': 12,
    'max_frames': 300, 'camera_seconds': 1800, 'max_ba_points': 5000,
    'max_ba_nfev': 60,
}


class VideoPreparationNotReady(RuntimeError):
    """Preserved fresh inputs are not admitted for the requested next stage."""


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def settings_for(cfg):
    settings = {**DEFAULTS, **{k: cfg[k] for k in DEFAULTS if k in cfg}, **cfg.get('video119', {})}
    if 'intrinsics' not in settings:
        settings['intrinsics'] = cfg.get('intrinsics', _read(Path(__file__).resolve().parents[2] / 'configs/video119.json')['video119']['intrinsics'])
    for name in ['frame_stride', 'max_image_size', 'min_frames', 'max_frames', 'camera_seconds', 'max_ba_points', 'max_ba_nfev']:
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f'video119.{name} must be a positive integer.')
    if settings['min_frames'] < 3 or settings['min_frames'] > settings['max_frames'] or settings['max_frames'] > 300:
        raise ValueError('Video119 requires 3..300 frames without silent truncation.')
    if settings['max_ba_points'] > 5000 or settings['max_ba_nfev'] > 60 or settings['camera_seconds'] > 1800:
        raise ValueError('Video119 camera budget cannot exceed the registered 5000-point / 60-evaluation / 1800-second procedure.')
    intrinsics = settings['intrinsics']
    if not isinstance(intrinsics, dict) or intrinsics.get('mode') not in ['estimate_focal', 'fixed_pinhole', 'rectified_center_crop']:
        raise ValueError('video119.intrinsics.mode must be estimate_focal, fixed_pinhole or rectified_center_crop.')
    for name in ['quality_accepted', 'physical_calibration_accepted', 'physical_calibration_validated', 'formal_training_admitted']:
        if cfg.get(name, False) is not False:
            raise ValueError(f'An input configuration cannot preapprove {name}.')
    return settings


def validate_video_stream(stream):
    rotations = [float(stream.get('tags', {}).get('rotate', 0))]
    rotations.extend(float(row.get('rotation', 0)) for row in stream.get('side_data_list', []))
    if any(not np.isfinite(rotation) or abs(rotation) > .01 for rotation in rotations):
        raise VideoPreparationNotReady('Video rotation metadata needs an explicit reviewed camera mapping; no implicit autorotation is accepted.')
    if stream.get('color_transfer') in ['smpte2084', 'arib-std-b67']:
        raise VideoPreparationNotReady('HDR requires a reviewed SDR conversion; PQ/HLG is not silently treated as sRGB.')
    if min(int(stream.get('width', 0)), int(stream.get('height', 0))) <= 0:
        raise VideoPreparationNotReady('Video dimensions are missing or invalid.')


def fixed_intrinsics(profile, source_wh, decoded_wh):
    """Map a declared VIDEO-pixel K through resize only, never an assumed crop."""
    if profile.get('mode') != 'fixed_pinhole':
        raise ValueError('Expected fixed_pinhole profile.')
    if profile.get('image_wh') != list(source_wh):
        raise VideoPreparationNotReady('Fixed-K image_wh must match this video stream exactly; no sensor crop is guessed.')
    if not isinstance(profile.get('provenance'), str) or not profile['provenance'].strip():
        raise VideoPreparationNotReady('Fixed K needs a human-readable provenance for the exported-video pixel domain.')
    if profile.get('pixel_convention') not in ['integer_centers', 'pixel_corners']:
        raise VideoPreparationNotReady('Fixed K must declare integer_centers or pixel_corners.')
    if any(float(value) != 0 for value in profile.get('distortion', [])):
        raise VideoPreparationNotReady('This frontend accepts already rectified pinhole videos; nonzero distortion requires a reviewed remap.')
    K = np.asarray(profile.get('K'), dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise VideoPreparationNotReady('Fixed K must be a finite positive-focal 3x3 matrix.')
    if not np.allclose(K[2], [0, 0, 1], atol=1e-12) or abs(K[0, 1]) + abs(K[1, 0]) > 1e-12:
        raise VideoPreparationNotReady('Only zero-skew pinhole K is supported.')
    K = K.copy()
    if profile['pixel_convention'] == 'pixel_corners':
        K[:2, 2] -= .5
    sx, sy = np.asarray(decoded_wh, dtype=float) / np.asarray(source_wh, dtype=float)
    A = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1.]])
    return A @ K, A


def rectified_center_crop_intrinsics(profile, source_wh, decoded_wh):
    """Map the capture profile into the actual landscape or portrait pixel domain.

    Orientation changes the centered crop, not the sensor axes. Fractional crop
    edges preserve the original workflow's continuous resize/crop hypothesis;
    the precise device integer ROI is not independently measured here.
    """
    if profile.get('mode') != 'rectified_center_crop':
        raise ValueError('Expected rectified_center_crop profile.')
    if profile.get('input_domain') != 'rectified_pinhole':
        raise VideoPreparationNotReady('Center-crop input must already be rectified; raw fisheye is not remapped here.')
    reference_wh = profile.get('reference_wh')
    for name, wh in [('reference_wh', reference_wh), ('source_wh', source_wh), ('decoded_wh', decoded_wh)]:
        if not isinstance(wh, (list, tuple)) or len(wh) != 2 or any(type(v) is not int or v <= 0 for v in wh):
            raise VideoPreparationNotReady(f'{name} must contain two positive integer dimensions.')
    reference, _ = fixed_intrinsics({**profile, 'mode': 'fixed_pinhole', 'image_wh': list(reference_wh)},
                                    reference_wh, reference_wh)
    scale = max(source_wh[0] / reference_wh[0], source_wh[1] / reference_wh[1])
    resized_wh = np.asarray(reference_wh, dtype=float) * scale
    crop_xy = (resized_wh - np.asarray(source_wh)) / 2
    source_map = np.array([[scale, 0, .5 * scale - .5 - crop_xy[0]],
                           [0, scale, .5 * scale - .5 - crop_xy[1]], [0, 0, 1.]])
    source_K = source_map @ reference
    K, decoded_map = fixed_intrinsics({**profile, 'mode': 'fixed_pinhole', 'image_wh': list(source_wh),
        'K': source_K.tolist(), 'pixel_convention': 'integer_centers'}, source_wh, decoded_wh)
    return K, dict(mode='rectified_center_crop', profile=profile,
        orientation='landscape' if source_wh[0] > source_wh[1] else 'portrait' if source_wh[0] < source_wh[1] else 'square',
        reference_wh=list(reference_wh), reference_K_integer=reference.tolist(),
        resized_reference_wh=resized_wh.tolist(), centered_crop_xywh=[*crop_xy.tolist(), *source_wh],
        source_K_integer=source_K.tolist(), reference_to_video=source_map.tolist(),
        video_to_decoded=decoded_map.tolist(), pixel_map=(decoded_map @ source_map).tolist(),
        processing_assumption='Supplied rectified K, uniform resize, centered crop; no image-axis rotation.',
        exact_device_integer_ROI_verified=False, physical_calibration_accepted=False,
        profile_is_not_independent_accuracy_evidence=True)


def split_track_records(components):
    """Freeze whole appearance components before any pose, focal or point fit."""
    records = []
    for nodes in sorted((sorted(tuple(node) for node in group) for group in components), key=lambda v: tuple(v)):
        if len(nodes) < 3 or len({i for i, _ in nodes}) != len(nodes):
            continue
        ident = hashlib.sha256(json.dumps(nodes, separators=(',', ':')).encode()).hexdigest()
        records.append(dict(id=ident, held=int(ident[:8], 16) % 5 == 0, nodes=nodes))
    return records


def _summary(errors):
    values = np.asarray(errors)
    if not len(values):
        return {'count': 0}
    if not np.isfinite(values).all():
        raise VideoPreparationNotReady('Nonfinite diagnostic errors; failed observations cannot be hidden.')
    return dict(count=len(values), median=float(np.median(values)), p90=float(np.quantile(values, .9)), max=float(values.max()))


def _fresh_tracks(root, decoded, out, status):
    import cv2
    n = len(decoded)
    cv2.setNumThreads(6);cv2.setRNGSeed(119)
    sift=cv2.SIFT_create(nfeatures=7000);xy=[];ds=[];offset=[0]
    for i,r in enumerate(decoded):
        image=cv2.imread(str(root/'raw'/r['name']),cv2.IMREAD_GRAYSCALE)
        if image is None:raise VideoPreparationNotReady(f'Missing or unreadable new frame {r["name"]}.')
        key,desc=sift.detectAndCompute(image,None)
        if desc is None:raise VideoPreparationNotReady(f'No fresh texture in {r["name"]}.')
        order=sorted(range(len(key)),key=lambda j:(-key[j].response,key[j].pt[0],key[j].pt[1]))
        # Keep the strongest descriptor per small pixel neighborhood, including scale duplicates.
        cells={};kept=[]
        for j in order:
            p=np.asarray(key[j].pt);cell=tuple(np.floor(p/1.5).astype(int))
            neighbours=[v for dx in [-1,0,1] for dy in [-1,0,1] for v in cells.get((cell[0]+dx,cell[1]+dy),[])]
            if any(np.linalg.norm(p-v)<1.5 for v in neighbours):continue
            kept.append(j);cells.setdefault(cell,[]).append(p)
            if len(kept)==5000:break
        if len(kept)<100:raise RuntimeError(f'Insufficient fresh raw texture at {r["frame"]}.')
        xy.append(np.asarray([key[j].pt for j in kept],np.float64));ds.append(np.ascontiguousarray(desc[kept],np.float32));offset.append(offset[-1]+len(kept))
        np.savez_compressed(out/f'features-{i:03d}.npz',xy=xy[-1],descriptors=ds[-1])
        if i%10==0:status('fresh_SIFT',done=i+1)
    parent=np.arange(offset[-1]);sz=np.ones(offset[-1],int)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if sz[a]<sz[b]:a,b=b,a
        parent[b]=a;sz[a]+=sz[b]
    pairs=[];matcher=cv2.BFMatcher(cv2.NORM_L2)
    for i in range(n):
        js=sorted(set([i+d for d in [1,2,4,8,16] if i+d<n]+(list(range(max(i+1,n-8),n)) if i<8 else [])))
        for j in js:
            fw=matcher.knnMatch(ds[i],ds[j],k=2);bw=matcher.knnMatch(ds[j],ds[i],k=2)
            back={a.queryIdx:a.trainIdx for a,b in bw if a.distance<.75*b.distance}
            m=[(a.queryIdx,a.trainIdx) for a,b in fw if a.distance<.75*b.distance and back.get(a.trainIdx)==a.queryIdx]
            for a,b in m:union(offset[i]+a,offset[j]+b)
            pairs.append(dict(i=i,j=j,count=len(m)))
        if i%10==0:status('mutual_material_candidates',done=i+1)
    groups={}
    for i in range(n):
        for j in range(len(xy[i])):
            node=offset[i]+j;p=find(node)
            if sz[p]>=3:groups.setdefault(p,[]).append((i,j))
    components=sorted((v for v in groups.values() if len({i for i,j in v})==len(v)),key=lambda v:tuple(v))
    records=split_track_records(components)
    _save(out/'appearance-tracks.json',dict(records=records,pairs=pairs,rejected_ambiguous_components=len(groups)-len(components),physical_identity_verified=False))
    return xy, records


def estimate_focal_intrinsics(xy, records, wh):
    """One bounded self-calibration hypothesis; holds out entire tracks first."""
    import cv2
    from scipy.optimize import minimize_scalar
    train = [dict(record['nodes']) for record in records if not record['held']]
    fundamentals = []
    evidence = []
    for j in [1, 2, 4, 8, 16]:
        if j >= len(xy):
            continue
        shared = [track for track in train if 0 in track and j in track]
        if len(shared) < 160:
            continue
        a = np.asarray([xy[0][track[0]] for track in shared])
        b = np.asarray([xy[j][track[j]] for track in shared])
        F, inliers = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, .999, 10000)
        if F is None or F.shape != (3, 3) or inliers is None or int(inliers.sum()) < 80:
            continue
        fundamentals.append(F)
        evidence.append({'pair': [0, j], 'matches': len(shared), 'inliers': int(inliers.sum())})
    if len(fundamentals) < 2:
        raise VideoPreparationNotReady('Fresh focal estimation requires two supported view pairs; supply a reviewed fixed_pinhole profile or improve capture.')
    width, height = wh
    def matrix(f):
        return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.]])
    def residual(log_f):
        K = matrix(np.exp(log_f))
        values = []
        for F in fundamentals:
            singular = np.linalg.svd(K.T @ F @ K, compute_uv=False)
            values.append(((singular[0] - singular[1]) / max(singular[0] + singular[1], 1e-15)) ** 2)
        return float(np.mean(values))
    low, high = np.log([.35 * max(wh), 5 * max(wh)])
    result = minimize_scalar(residual, bounds=(low, high), method='bounded', options={'maxiter': 80, 'xatol': 1e-6})
    if not result.success or not np.isfinite(result.x) or min(result.x - low, high - result.x) < .01:
        raise VideoPreparationNotReady('Fresh focal solve failed or reached its declared bounds; no retry or calibration acceptance.')
    score = float(result.fun)
    # A flat objective indicates the focal parameter is not observable here.
    if min(residual(result.x - .1), residual(result.x + .1)) - score < 1e-8:
        raise VideoPreparationNotReady('Fresh focal is locally unobservable; a reviewed K profile or a better camera path is required.')
    return matrix(np.exp(result.x)), dict(mode='estimate_focal', evidence=evidence,
        essential_singular_value_cost=score, focal_bounds=[float(np.exp(low)), float(np.exp(high))],
        assumption='One constant focal, square pixels, zero skew, centered principal point and rectified static scene.',
        held_tracks_used=False, physical_calibration_accepted=False, independent_face_accuracy_proven=False)


def solve_camera_geometry(xy, records, K, out, settings=None, status=None):
    """Pure new-track geometry. Public for meaningful synthetic CPU verification."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.sparse import lil_matrix
    settings = settings_for({}) if settings is None else settings
    status = status or (lambda *args, **kwargs: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(xy)
    start = time.monotonic()
    K = np.asarray(K, dtype=np.float64)
    def project(E, X):
        q = X @ E[:3, :3].T + E[:3, 3]
        projected = q @ K.T
        return projected[:, :2] / projected[:, 2, None], q[:, 2]
    summary = _summary
    train=[r for r in records if not r['held']];held=[r for r in records if r['held']]
    tracks=[dict(r['nodes']) for r in train]
    if len(held)<30:raise RuntimeError('Fewer than30 whole-track holdouts; no geometry solve.')
    seed_pairs=[]
    for j in [1,2,4,8,16]:
        count=sum(0 in t and j in t for t in tracks)
        if j<n and count>=160:seed_pairs.append((j,count))
    if not seed_pairs:raise RuntimeError('No sufficiently supported fresh seed pair.')
    seed_j,seed_matches=max(seed_pairs)
    seed=[k for k,t in enumerate(tracks) if 0 in t and seed_j in t]
    a=np.array([xy[0][tracks[k][0]] for k in seed]);b=np.array([xy[seed_j][tracks[k][seed_j]] for k in seed])
    norm=lambda v:cv2.undistortPoints(v.reshape(-1,1,2),K,None).reshape(-1,2)
    essential,mask=cv2.findEssentialMat(norm(a),norm(b),np.eye(3),method=cv2.RANSAC,prob=.999,threshold=1.5/np.mean([K[0,0],K[1,1]]),maxIters=10000)
    if essential is None or essential.shape!=(3,3):raise RuntimeError('One essential matrix not found.')
    count,R,t,mask=cv2.recoverPose(essential,norm(a),norm(b),np.eye(3),mask=mask)
    _save(out/'seed.json',dict(pair=[0,seed_j],appearance_matches=seed_matches,cheirality=int(count)))
    if count<80:raise RuntimeError(f'Seed cheirality {count}<80; no threshold retry.')
    E=np.eye(4);E[:3,:3]=R;E[:3,3]=t[:,0];matrices={0:np.eye(4),seed_j:E};points={}
    def triangulate():
        centers={i:-e[:3,:3].T@e[:3,3] for i,e in matrices.items()};groups={}
        for ti,tr in enumerate(tracks):
            if ti in points:continue
            ids=sorted(set(tr)&matrices.keys())
            if len(ids)<2:continue
            _,i,j=max((np.linalg.norm(centers[i]-centers[j]),i,j) for z,i in enumerate(ids) for j in ids[z+1:])
            groups.setdefault((i,j),[]).append(ti)
        for (i,j),ids in groups.items():
            u=np.array([xy[i][tracks[k][i]] for k in ids]);v=np.array([xy[j][tracks[k][j]] for k in ids])
            X=cv2.triangulatePoints(K@matrices[i][:3],K@matrices[j][:3],u.T,v.T);X=(X[:3]/X[3]).T
            p,zi=project(matrices[i],X);q,zj=project(matrices[j],X)
            ra=X-centers[i];rb=X-centers[j];angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(ra,rb),axis=1),np.einsum('ij,ij->i',ra,rb)))
            good=np.isfinite(X).all(1)&(zi>0)&(zj>0)&(np.linalg.norm(p-u,axis=1)<3)&(np.linalg.norm(q-v,axis=1)<3)&(angle>1)
            for ti,x in zip(np.asarray(ids)[good],X[good]):points[int(ti)]=x
    triangulate();status('fresh_camera_seed',registered=2,points=len(points))
    while len(matrices)<n:
        candidates=[]
        for i in range(n):
            if i not in matrices:
                ids=[k for k in points if i in tracks[k]];candidates.append((len(ids),i,ids))
        done=0
        for _,i,ids in sorted(candidates,reverse=True):
            if len(ids)<50:continue
            X=np.array([points[k] for k in ids]);u=np.array([xy[i][tracks[k][i]] for k in ids])
            ok,r,t,ii=cv2.solvePnPRansac(X,u,K,None,iterationsCount=2000,reprojectionError=3,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if not ok or ii is None or len(ii)<50:continue
            ii=ii[:,0];r,t=cv2.solvePnPRefineLM(X[ii],u[ii],K,None,r,t)
            e=np.eye(4);e[:3,:3]=cv2.Rodrigues(r)[0];e[:3,3]=t[:,0];matrices[i]=e;done+=1
        triangulate();status('fresh_PnP_registration',registered=len(matrices),points=len(points))
        if not done:raise RuntimeError('Missing real camera poses: '+str(sorted(set(range(n))-matrices.keys())))
    verified={}
    for k,X in points.items():
        kept=[]
        for i,j in tracks[k].items():
            p,z=project(matrices[i],X[None])
            if np.isfinite(p).all() and z[0]>0 and np.linalg.norm(p[0]-xy[i][j])<3:kept.append((i,j))
        if len(kept)>=3:verified[k]=kept
    selected=sorted(verified,key=lambda k:(-len(verified[k]),k))[:settings["max_ba_points"]]
    ci=[];pi=[];uv=[]
    for p,k in enumerate(selected):
        for i,j in verified[k]:ci.append(i);pi.append(p);uv.append(xy[i][j])
    ci,pi,uv=np.array(ci,dtype=np.int64),np.array(pi,dtype=np.int64),np.array(uv,dtype=np.float64)
    if len(uv)==0 or np.bincount(ci,minlength=n).min()<50:raise RuntimeError('Insufficient verified fitting observations per real camera.')
    cp=np.array([np.r_[Rotation.from_matrix(matrices[i][:3,:3]).as_rotvec(),matrices[i][:3,3]] for i in range(n)])
    free=np.ones((n,6),bool);free[0]=False;free[seed_j,3+np.argmax(abs(cp[seed_j,3:]))]=False
    cols=np.full((n,6),-1,int);cols[free]=np.arange(free.sum());nc=int(free.sum())
    initial=np.r_[cp[free],np.array([points[k] for k in selected]).ravel()]
    sparsity=lil_matrix((2*len(uv),len(initial)),dtype=np.int8);rr=np.arange(len(uv))
    for k in range(6):
        valid=cols[ci,k]>=0;sparsity[2*rr[valid],cols[ci[valid],k]]=1;sparsity[2*rr[valid]+1,cols[ci[valid],k]]=1
    for k in range(3):sparsity[2*rr,nc+3*pi+k]=1;sparsity[2*rr+1,nc+3*pi+k]=1
    calls=0
    def residual(x):
        nonlocal calls
        calls+=1
        if time.monotonic()-start>settings["camera_seconds"]:raise RuntimeError('Camera BA time budget exceeded.')
        p=cp.copy();p[free]=x[:nc];R=Rotation.from_rotvec(p[:,:3]).as_matrix();X=x[nc:].reshape(-1,3)
        q=np.einsum('nij,nj->ni',R[ci],X[pi])+p[ci,3:];z=np.maximum(q[:,2],1e-6)
        pred=np.c_[K[0,0]*q[:,0]/z+K[0,2],K[1,1]*q[:,1]/z+K[1,2]]
        if calls%200==0:status('fixed_K_BA',residual_calls=calls)
        return (pred-uv).ravel()
    before=summary(np.linalg.norm(residual(initial).reshape(-1,2),axis=1))
    sol=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=settings["max_ba_nfev"],ftol=1e-5)
    p=cp.copy();p[free]=sol.x[:nc];Es=np.tile(np.eye(4),(n,1,1));Es[:,:3,:3]=Rotation.from_rotvec(p[:,:3]).as_matrix();Es[:,:3,3]=p[:,3:]
    if not np.isfinite(Es).all():raise RuntimeError('Nonfinite camera candidate.')
    hold_results=[]
    # Fit 3D on alternate reserved observations, evaluate remaining observations.
    # No camera or K optimization; no held observations excluded for error.
    for tr in held:
        nodes=tr['nodes'];anchors=nodes[::2];checks=nodes[1::2]
        if len(anchors)<2:continue
        A=[]
        for i,j in anchors:
            P=K@Es[i,:3];u,v=xy[i][j];A.extend([u*P[2]-P[0],v*P[2]-P[1]])
        _,_,v=np.linalg.svd(A);X=v[-1,:3]/v[-1,3]
        errors=[];depths=[]
        for i,j in checks:
            q,z=project(Es[i],X[None]);errors.append(float(np.linalg.norm(q[0]-xy[i][j])));depths.append(float(z[0]))
        hold_results.append(dict(id=tr['id'],nodes=nodes,anchor_frames=[i for i,j in anchors],test_frames=[i for i,j in checks],
            errors=errors,positive_depth=all(z>0 for z in depths),point=X.tolist()))
    _save(out/'held-evaluation.json',dict(records=hold_results,all_errors=summary([e for t in hold_results for e in t['errors']]),
        whole_tracks_reserved=True,landmarks_are_not_material_truth=True,physical_identity_reviewed=False,formal_calibration_accepted=False))
    errors = np.linalg.norm(residual(sol.x).reshape(-1, 2), axis=1)
    result = dict(state='camera_estimation_complete', frames=n, before=before,
        fitting_errors=summary(errors), optimizer_success=bool(sol.success), nfev=sol.nfev,
        held_track_count=len(held), held_test_count=len(hold_results),
        independent_physical_accuracy_proven=False, formal_training_admitted=False,
        physical_calibration_accepted=False, training_started=False)
    _save(out / 'result.json', result)
    np.savez_compressed(out / 'solver-output.npz', K=K, w2c=Es,
        points=sol.x[nc:].reshape(-1, 3), obs_cam=ci, obs_point=pi, observations=uv,
        track_ids=np.asarray([train[k]['id'] for k in selected]))
    return Es, result


def prepare_video(video, root, cfg, assets, *, camera_only=False):
    """Fill a fresh route119 run from this video; never import old prepared data.

    The caller creates root/data/logs/config.json/run.json. camera_only=True
    deliberately stops before masks/landmarks and is not a prepared training run.
    """
    from .video_sfm import extract
    video, root = Path(video).resolve(), Path(root).resolve()
    settings = settings_for(cfg)
    if not video.is_file():
        raise FileNotFoundError(video)
    for executable in ['ffprobe', 'ffmpeg']:
        if not shutil.which(executable):
            raise VideoPreparationNotReady(f'Missing {executable}; install FFmpeg and ffprobe on the target machine.')
    for name in ['raw', 'camera', 'decoded.json', 'preparation.json', 'intrinsic-contract.json', 'data/inputs.json', 'data/training-inputs.json', 'data/observations.npz', 'data/masks', 'fit', 'inference', 'gaussian-run']:
        if (root / name).exists():
            raise FileExistsError(f'Fresh video preparation only; preserve existing {root / name}.')
    if not (root / 'data').is_dir() or not (root / 'logs').is_dir():
        raise ValueError('Caller must create a fresh run with data and logs directories.')
    report = dict(video=str(video), video_sha256=_sha(video), from_this_video=True,
        old_subject_state_loaded=False, old_frames_loaded=False, old_cameras_loaded=False,
        old_masks_loaded=False, old_observations_loaded=False, from_scratch=True,
        training_started=False, quality_accepted=False, physical_calibration_accepted=False,
        independent_native_face_subpixel_admitted=False)
    start = time.monotonic()
    def status(stage, **kw):
        if time.monotonic() - start > settings['camera_seconds']:
            raise VideoPreparationNotReady('Fresh video camera preparation exceeded the declared time budget.')
        _save(root / 'video119-progress.json', dict(stage=stage, seconds=time.monotonic() - start,
            training_started=False, quality_accepted=False, **kw))
        print(json.dumps(dict(stage=stage, **kw)), flush=True)
    try:
        report['state'] = 'probing_new_video'
        _save(root / 'preparation.json', report)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(video)],
            capture_output=True, text=True, check=True)
        streams = json.loads(probe.stdout).get('streams', [])
        if len(streams) != 1:
            raise VideoPreparationNotReady('Expected exactly one selected video stream.')
        stream = streams[0]
        validate_video_stream(stream)
        effective = {**cfg, **{key: settings[key] for key in ['frame_stride', 'max_image_size', 'min_frames', 'max_frames']}}
        extract(video, root, effective)
        decoded_manifest = _read(root / 'decoded.json')
        if decoded_manifest['video_sha256'] != report['video_sha256']:
            raise VideoPreparationNotReady('Source video changed during decoding; preserve the partial run and use a fresh immutable input.')
        decoded = decoded_manifest['records']
        source_wh = [int(stream['width']), int(stream['height'])]
        wh = decoded[0]['wh']
        out = root / 'camera'
        out.mkdir()
        _save(out / 'plan.json', dict(solver='OpenCV SIFT/Essential/PnP + SciPy fixed-K sparse BA',
            seed=119, raw_features_new=True, old_poses_loaded=False, old_structure_loaded=False,
            held='one fifth of complete appearance tracks frozen before focal/pose/point estimation',
            physical_material_identity_verified=False, max_ba_points=settings['max_ba_points'],
            max_ba_nfev=settings['max_ba_nfev'], budget_seconds=settings['camera_seconds']))
        xy, tracks = _fresh_tracks(root, decoded, out, status)
        status('fresh_tracks_frozen', tracks=len(tracks), held_tracks=sum(row['held'] for row in tracks))
        profile = settings['intrinsics']
        if profile['mode'] == 'fixed_pinhole':
            K, mapping = fixed_intrinsics(profile, source_wh, wh)
            intrinsic = dict(mode='fixed_pinhole', profile=profile, pixel_map=mapping.tolist(),
                physical_calibration_accepted=False, profile_is_not_independent_accuracy_evidence=True)
        elif profile['mode'] == 'rectified_center_crop':
            K, intrinsic = rectified_center_crop_intrinsics(profile, source_wh, wh)
        else:
            K, intrinsic = estimate_focal_intrinsics(xy, tracks, wh)
        intrinsic.update(raw_K_integer=K.tolist(), wh=wh, original_video_wh=source_wh,
            images_warped=False, distortion_applied=False, principal_center_convention='integer; renderer receives +0.5',
            calibration_applicability_confirmed=False)
        _save(root / 'intrinsic-contract.json', intrinsic)
        Es, camera_report = solve_camera_geometry(xy, tracks, K, out, settings, status)
        Kc = K.copy()
        Kc[:2, 2] += .5
        rows = [dict(**decoded[i], image=str(root / 'raw' / decoded[i]['name']), K=Kc.tolist(),
            K_integer_centers=K.tolist(), w2c=Es[i].tolist(), mask=None, mask_sha256=None,
            official_prediction_slot=i, camera_kind='fresh_fixed_K_opencv_research',
            camera_registration=dict(interpolation_used=False), observation_weight=1.) for i in range(len(decoded))]
        manifest = dict(records=rows, from_scratch=True,
            video_sha256=report['video_sha256'],
            pixel_domain='fresh decoded exported video; no lens remap',
            coordinate_contract='K is pixel corners; detector observations use integer centers; renderer subtracts 0.5',
            camera_source='Fresh OpenCV/SciPy fixed-K candidate', formal_calibration_accepted=False,
            physical_calibration_accepted=False, formal_training_admitted=False)
        _save(root / 'data/inputs.json', manifest)
        _save(out / 'candidate-cameras.json', manifest)
        report.update(state='camera_candidate_complete_observations_pending', frames=len(rows),
            camera_result='camera/result.json', intrinsic_contract='intrinsic-contract.json',
            camera_candidate_complete=True, observations_complete=False)
        _save(root / 'preparation.json', report)
        if camera_only:
            return report
        if assets is None:
            raise VideoPreparationNotReady('Fresh camera candidate saved. Official detector/SAM2/RTMW asset directory is required for new masks/observations.')
        from . import runtime as rt
        rt.configure(assets, root)
        rt.verify_assets()
        from .observations import observe
        observe(root, cfg)
        report.update(state='observations_complete', observations_complete=True,
            masks_source='fresh_official_SAM2', observations_source='fresh_official_RTMw_COCO133',
            formal_training_admitted=False)
        _save(root / 'preparation.json', report)
        return report
    except BaseException as exc:
        report.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        _save(root / 'preparation.json', report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'configs/video119.json')
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--camera-only', action='store_true', help='CPU camera candidate only; no masks, observations, fit or training.')
    args = parser.parse_args(argv)
    cfg = _read(args.config)
    settings_for(cfg)
    if args.output.exists():
        raise FileExistsError(f'Fresh run only: {args.output}')
    args.output.mkdir(parents=True)
    for directory in ['data', 'logs']:
        (args.output / directory).mkdir()
    _save(args.output / 'config.json', cfg)
    _save(args.output / 'run.json', dict(video=str(args.video.resolve()), video_sha256=_sha(args.video), from_scratch=True, training_started=False))
    result = prepare_video(args.video, args.output, cfg, args.assets, camera_only=args.camera_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
