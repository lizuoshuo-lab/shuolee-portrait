"""Run the existing fixed-K SfM gate on the frozen 022 background families."""
import ast
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from extend_leafwoman_bark142 import families
from official_smplifyx_entry123 import sha
from prepare_leafwoman_bark142 import BASE, OUT as BARK, atomic, read
from probe_video22_background135 import K, W, H, norm


OUT=BASE/'global-multiview143'
LEGACY=Path(__file__).with_name('global_fixedk121.py')


class OpenCV:
    """Map the existing Essential/PnP calls to the Windows OpenCV runtime."""
    @staticmethod
    def estimate_relative_pose(a,b,cam1,cam2,options,_):
        assert cam1==cam2 and cam1['params']==[K[0,0],K[1,1],K[0,2],K[1,2]]
        cv2.setRNGSeed(options['seed'])
        x,y=norm(a),norm(b)
        E,mask=cv2.findEssentialMat(x,y,np.eye(3),method=cv2.RANSAC,prob=.999,
            threshold=options['max_epipolar_error']/np.mean(K[[0,1],[0,1]]),maxIters=options['max_iterations'])
        if E is None or E.shape!=(3,3) or mask is None:
            return None,dict(num_inliers=0)
        n,R,t,valid=cv2.recoverPose(E,x,y,np.eye(3),mask=mask)
        return SimpleNamespace(R=R,t=t[:,0]),dict(num_inliers=int(n),inliers=valid.ravel().astype(bool))

    @staticmethod
    def estimate_absolute_pose(uv,xyz,cam,options,_):
        M=np.array([[cam['params'][0],0,cam['params'][2]],[0,cam['params'][1],cam['params'][3]],[0,0,1.]])
        cv2.setRNGSeed(options['seed'])
        ok,r,t,inliers=cv2.solvePnPRansac(np.ascontiguousarray(xyz),np.ascontiguousarray(uv),M,None,
            iterationsCount=options.get('max_iterations',10000),reprojectionError=options['max_reproj_error'],
            confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<4:
            return None,dict(num_inliers=0)
        take=inliers[:,0]
        r,t=cv2.solvePnPRefineLM(xyz[take],uv[take],M,None,r,t)
        return SimpleNamespace(R=cv2.Rodrigues(r)[0],t=t[:,0]),dict(num_inliers=len(take))


def load_worker(input_data):
    # Import only the registered numerical routines, without Linux entry or I/O roots.
    names={'now','sha','save','stats','project','triangulate','components','bundle','evaluate','one','selftest'}
    tree=ast.parse(LEGACY.read_text(encoding='utf-8'))
    funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {n.name for n in funcs}==names
    import json
    import hashlib
    from datetime import datetime,timezone
    scope=dict(np=np,cv2=cv2,Rotation=Rotation,least_squares=least_squares,lil_matrix=lil_matrix,
        __file__=str(LEGACY),Path=Path,ROOT=OUT,R=OUT,time=time,hashlib=hashlib,json=json,datetime=datetime,
        timezone=timezone,poselib=OpenCV,input_data=input_data,apply_material_gate=lambda *args:None)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(LEGACY),'exec'),scope)
    return scope


def optimistic_registration(candidates, tracks):
    """Upper bound if every FIT point seen twice were triangulable."""
    families=[set(t['by_frame']) for t in tracks if t['camera_fit_eligible']]
    report=[]
    for p in candidates:
        if p['a']!=0: continue
        registered={0,p['b']}
        while True:
            points=[f for f in families if len(f&registered)>=2]
            add={i for i in range(24) if i not in registered and sum(i in f for f in points)>=20}
            if not add: break
            registered|=add
        report.append(dict(seed=p['frames'],reachable_slots=sorted(registered),reachable_count=len(registered)))
    return report


def run():
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    assert not OUT.exists()
    bark=read(BARK/'result.json'); bg=BASE/'background-motion135'; old=read(bg/'result.json')
    frames=bark['frames']; assert frames==old['frames'] and len(frames)==24
    sources=[('outer',bg/'matched_pixels.npz',old['matched_pixels_sha256']),
             ('bark',BARK/'bark-matched-pixels.npz',bark['pixels_sha256'])]
    tracks=[];counts={};xy=[{'fit':[], 'heldout':[]} for _ in frames]
    for label,path,hash_value in sources:
        assert sha(path)==hash_value
        with np.load(path,allow_pickle=False) as z: material=families(z)
        counts[label]=dict(total=len(material),held=sum(h for _,h in material))
        for n,(group,held) in enumerate(material):
            split='heldout' if held else 'fit'; assert len(group)>=3
            by_frame={i:{'uv':[x,y]} for i,x,y in group}
            for i,x,y in group: xy[i][split].append((x,y))
            tracks.append(dict(track_id=f'{label}-{n:04d}',camera_fit_eligible=not held,split=split,by_frame=by_frame,
                               regions={i:label for i in by_frame}))
    separation=min(float(np.linalg.norm(np.array(r['fit'])[:,None]-np.array(r['heldout'])[None,:],axis=2).min()) for r in xy)
    assert separation>=8
    records=[dict(frame=f,wh=[W,H]) for f in frames]
    # Co-observed FIT families supply additional non-adjacent pairs without redetection.
    counts_pair={(i,j):sum(i in t['by_frame'] and j in t['by_frame'] for t in tracks if t['camera_fit_eligible'])
                 for i in range(24) for j in range(i+1,24)}
    candidate_pairs=[dict(a=i,b=j,frames=[frames[i],frames[j]]) for (i,j),n in counts_pair.items() if n>=20]
    assert candidate_pairs and any(p['a']==0 for p in candidate_pairs)
    path_hashes={label:hash_value for label,path,hash_value in sources}
    input_data=lambda _:(dict(records=records,pairs=candidate_pairs),tracks,path_hashes)
    worker=load_worker(input_data)
    plan=dict(subjects={'022':{'K_integer_centers':K.tolist()}},material_preflight={'022':{'decision':'diagnostic_only'}},
        cpu_seconds_per_subject=120,workers_sha256={'adapter':sha(__file__),'original':sha(LEGACY)},source_hashes=path_hashes,
        counts=counts,fit_and_held_min_distance_px=separation,registered_pairs=candidate_pairs,
        max_nonlocal_pair_fit_families=max(n for (i,j),n in counts_pair.items() if j-i>=3),
        registered_nonlocal_pair_count=sum(p['b']-p['a']>=3 for p in candidate_pairs),
        change='Reuse whole FIT material families in existing global_fixedk121 geometry. Include every co-observed FIT pair with ≥20 points. '
               'Replace Linux PoseLib entry with the existing OpenCV Essential/PnP method. Keep fixed K, original counts, geometry gates, bundle and target-excluded evaluation.',
        no_held_data_in_geometry=True,no_prior_camera_or_geometry=True,diagnostic_only=True,
        conditional_K=True,actual_rear_views=0,training_admitted=False)
    atomic(OUT/'global-fixedk-plan.json',plan)
    # An existing synthetic projection/triangulation/PnP/gauge control also exercises the adapter.
    worker['selftest']()
    gate=optimistic_registration(candidate_pairs,tracks)
    atomic(OUT/'optimistic-reachability.json',dict(first_camera_fixed=True,seeds=gate,
        registered_pairs=len(candidate_pairs),maximum=max(g['reachable_count'] for g in gate)))
    if max(g['reachable_count'] for g in gate)<24:
        atomic(OUT/'execution.json',dict(plan_sha256=sha(OUT/'global-fixedk-plan.json'),
            synthetic_control_passed=True,state='stopped_combinatorial_3d_pnp_support',estimated_pair_count=0,
            usable_pair_count=0,registered_count=0,training_admitted=False,quality_accepted=False))
        print('VIDEO22_MULTIVIEW_REACHABILITY',gate,flush=True)
        return
    worker['one']('022',plan)
    result_path=OUT/'022/global-fixedk/result.json'; result=read(result_path)
    assert not result['training_admitted'] and not result['training_started']
    rows=read(OUT/'022/global-fixedk/relative-pairs.json')
    available=[x for x in rows if x['state']=='usable_background_pair']
    stats_by_source={}
    pred=OUT/'022/global-fixedk/heldout-predictions.json'
    if pred.exists():
        evidence=read(pred)
        for label,_,_ in sources:
            all_source=[x for x in evidence if x['region']==label]
            good=[x for x in all_source if x['state']=='predicted']
            stats_by_source[label]=dict(attempted=len(all_source),predicted=len(good),
                errors=worker['stats']([x['error_px'] for x in good]),
                per_frame={f:worker['stats']([x['error_px'] for x in good if x['frame']==f]) for f in frames})
    atomic(OUT/'execution.json',dict(result_sha256=sha(result_path),plan_sha256=sha(OUT/'global-fixedk-plan.json'),
        synthetic_control_passed=True,estimated_pair_count=len(rows),usable_pair_count=len(available),
        registered_count=result['registered_count'],heldout_by_source=stats_by_source,training_admitted=False,quality_accepted=False))


if __name__=='__main__':
    run()
