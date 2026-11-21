"""Bounded original SMPLify-X CPU test of two source-occluded BODY25 targets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image, ImageDraw

from audit_008_head_joint import PROTECTED_METERS, PROTECTED_PIXELS, winner
from audit_008_head_ownership import ASSET, LAB, PROJECT, RUN, need, read, save, semantics, sha, verify

sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait import official_sequence as seq, stable_selection as screen


OUT = LAB/'shuolee-smpl-008-rear-face-input'
PROOF = LAB/'shuolee-smpl-neck-head-bound-008/result.json'
FACE_HIDDEN = (0, 15)
LEFT_EAR = 18
COHORT = {
    'frame_000059': (465, '549cd5f3b1b1e68a6b86b128a84e4357e5abdb74097b0eb4acd7319617838fa9',
        '3420718bb428f79ff47b94703143b805a72e5f6c042a131d276cef8ed0acdc04',
        ((1279,532),(1285,532),(1293,544),(1297,563),(1296,583),(1288,595),(1281,591),(1278,572))),
    'frame_000060': (473, '55365fbe69386b6f66c0ebdf96def95d1ea4f86e5cb6fafef69a061a467b420e',
        '1bb98203e8ab75f222f211986e878f9ed661290b38d377e993afa12884bb9fa8',
        ((1280,537),(1287,537),(1295,549),(1299,568),(1298,589),(1291,601),(1283,596),(1279,576))),
}
CROP = (1210,445,1450,688)
PINS = {
    PROOF: '21fdcd80ddf5a0107285d54273a2eada66681fdbb942c1e10bb6a538376ee504',
    RUN/'config.json': '6ae1e685be00858eb0083b4dbff005a7b7a62ed3ca29ee2fa13011b706d397b8',
    RUN/'fit/official-fit-result.json': 'ead9039816741cb6c780ecc888fc139c6f826c7580a851344e12bfca45925ef0',
    RUN/'fit/selection.json': '44e85e52a01642cda7c294f9afbb8ef4520f3e31b53cf218fadca0dcf5c0950d',
    RUN/'fit/shared-body.json': '72672b67af8b3fcd6865df2b5343cdd900b6ddc228bc157c8522cd17abbeffc9',
    ASSET: '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992',
}
TARGET = '2da253e55a6fded5601e722098e85bae5f2f1d753941f160c629d28e184b992d'


def array_sha(array):
    from hashlib import sha256
    return sha256(array.tobytes()).hexdigest()


def preflight():
    pins = dict(PINS)
    verify(pins)
    cfg = read(RUN/'config.json')
    old_fit = read(RUN/'fit/official-fit-result.json')
    sel = read(RUN/'fit/selection.json')
    shared = read(RUN/'fit/shared-body.json')
    need(old_fit['selection'] == sel and set(COHORT).isdisjoint(sel['retained_holdouts'])
         and set(COHORT).isdisjoint(sel['selected_frames']), 'Unexpected training or heldout roster.')
    beta = np.asarray(shared['betas'], dtype=np.float32).reshape(1,10)
    latent = np.asarray(shared['pose_latent'], dtype=np.float32).reshape(1,32)
    checked = seq.checked_image
    with patch.object(seq, 'checked_image', lambda row: checked(row) if row['frame'] in COHORT else None):
        data = seq.load_inputs(RUN/'data/inputs.json', RUN/'data/observations.npz',
            camera_length_contract=cfg['camera_length_contract'], reference_frame=sel['reference_frame'],
            capture_layout=cfg['capture_layout'])
    need(data['confidence'] == old_fit['source']['confidence'] and
         data['bound'] == old_fit['source']['input_files'] and set(COHORT)<=set(data['scope']['fit_frame_ids']),
         'Archived input or confidence provenance changed.')
    official = Path(cfg['official_smplifyx']['source_root'])
    assets = ASSET.parent.parent
    pins.update({Path(p): v for p,v in data['bound'].items()})
    pins.update({official/k: v for k,v in seq.official.PINNED.items()})
    pins.update({assets/r['path']: r['sha256'] for r in read(PROJECT/'configs/server-assets.json')['files']
                 if r['path'].startswith(('smplx/','vposer/'))})
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    rows = []
    for frame,(index,wsha,rsha,ear) in COHORT.items():
        slot = data['ids'].index(frame)
        row, K = data['rows'][slot], data['K'][slot]
        old_path, report_path = winner(frame), winner(frame).with_name('result.json')
        pins.update({old_path:wsha,report_path:rsha,Path(row['image']):row['source_sha256'],
                     Path(row['mask']):row['mask_sha256']})
        verify({old_path:wsha,report_path:rsha})
        with np.load(old_path,allow_pickle=False) as archive:
            old = dict(archive)
        report = read(report_path)
        seq.check_result(report,row)
        a = data['targets'][slot:slot+1].copy()
        b = a.copy()
        b[:,FACE_HIDDEN,2] = 0.
        need(np.array_equal(np.argwhere(a!=b),[[0,0,2],[0,15,2]]) and a[0,LEFT_EAR,2]>0
             and np.array_equal(old['betas'],beta) and row['source_frame_index']==index
             and np.array_equal(row['w2c'],np.eye(4))
             and report['shared_body_target_sha256']==TARGET, 'Unintended observation, shape or camera difference.')
        reason = sel['reasons'][frame]
        scores,flags,_,_ = screen._candidate(row,old,a[0],K,beta,None,report,True)
        need(scores==reason['scores'] and flags==reason['screen_flags'] and flags,
             'The stock input and screen do not reproduce the archived rejection.')
        rows.append(dict(row=row,K=K,A=a,B=b,old=old,report=report,ear=ear))
    verify(pins)
    return pins,rows,beta,latent,official,assets


def plan(state):
    pins,rows,beta,latent,official,assets = state
    return dict(schema='008-rear-face-unsupported-body25-slots-v1', stock_fitter=True,
        pose_optimizer_edits=0, gpu_calls=0, maximum_fits=3, max_wall_seconds=180,
        cpu_workers=2, threads_per_worker=1, gaussian_updates=0, holds_used=False,
        unsupported_slots=list(FACE_HIDDEN), physically_visible_ear_slot=LEFT_EAR,
        source_rule='Back of head and thin left side profile. Nose and far right eye fully occluded in native RGB.',
        intent='Confidence only, zero slots 0 and 15. Keep all retained observations and active priors unchanged.',
        nonhead_protection=dict(max_m=PROTECTED_METERS,max_native_px=PROTECTED_PIXELS,
            retained_nonhead_landmark_max_move_px=PROTECTED_PIXELS),
        visible_ear_region_tolerance_px=2., source_crop=list(CROP), beta=beta.tolist(),
        shared_latent=latent.tolist(), shared_pose_sha256=TARGET,
        pins={str(p):v for p,v in pins.items()},
        rows=[dict(frame=r['row']['frame'],source_index=r['row']['source_frame_index'],
                   source_sha256=r['row']['source_sha256'], ear_polygon=[list(p) for p in r['ear']],
                   target_A_sha256=array_sha(r['A']),target_B_sha256=array_sha(r['B'])) for r in rows])


def source_panel(item):
    with Image.open(item['row']['image']) as source:
        im = source.convert('RGB').crop(CROP)
    outline = [(x-CROP[0],y-CROP[1]) for x,y in item['ear']]
    ImageDraw.Draw(im).line([*outline,outline[0]],fill='#00eaff',width=1)
    return im


def prepare(state):
    need(not OUT.exists(), 'Preserve any existing preregistration.')
    OUT.mkdir()
    pins = {}
    for item in state[1]:
        f=OUT/(item['row']['frame']+'-source-only.png')
        source_panel(item).save(f)
        pins[str(f)]=sha(f)
    save(OUT/'preregistration.json',dict(plan(state),source_only_panel_sha256=pins))
    print(OUT/'preregistration.json',flush=True)


def polygon_distance(point, polygon):
    q = np.asarray(point,dtype=np.float64)
    a = np.asarray(polygon,dtype=np.float64)
    b = np.roll(a,-1,axis=0)
    direction = b-a
    t = np.clip(np.sum((q-a)*direction,axis=1)/np.sum(direction*direction,axis=1),0,1)
    d = float(np.linalg.norm(a+t[:,None]*direction-q,axis=1).min())
    cross = (a[:,1]>q[1])!=(b[:,1]>q[1])
    cutting_x = a[cross,0]+(q[1]-a[cross,1])*direction[cross,0]/direction[cross,1]
    return 0. if np.count_nonzero(cutting_x>q[0])%2 else d


def project(xyz, K):
    q=xyz.astype(np.float64)@K.T
    return q[...,:2]/q[...,2:]


def assess(item, arrays, report, beta):
    row,K,A,B,old = (item[k] for k in ('row','K','A','B','old'))
    seq.check_result(report,row)
    need(all(report[k]==item['report'][k] for k in ('shared_body_target_sha256','shared_root_weight',
         'shared_body_weight','body_placement_parameters','body_placement_initial_m','orientation_count',
         'initialization','camera_initialization','source_hashes','official_commit','runtime_optimization'))
         and all(np.array_equal(arrays[k],old[k]) for k in ('betas','K_compute_float32','K_integer_centers','faces')),
         'Initial placement, solver, camera intrinsics or frozen model changed.')
    flags, points = {}, {}
    h = row['initial_box_xyxy'][3]-row['initial_box_xyxy'][1]
    for arm,pose,result in (('stock',old,item['report']),('candidate',arrays,report)):
        p=pose['joints_projected_native'][0]
        err=np.linalg.norm(p-A[0,:,:2],axis=1)
        flags[arm] = {}
        for label,target in (('untouched',A),('physically_supported',B)):
            flags[arm][label]=screen._candidate(row,pose,target[0],K,beta,None,result,True)[1]
            vis=np.r_[screen.BODY,screen.FEET]
            flags[arm][label+'_offenders']=vis[(target[0,vis,2]>0)&(err[vis]>.03*h)].tolist()
        points[arm]=dict(projection_integer_xy=p[LEFT_EAR].tolist(),detector_error_px=float(err[LEFT_EAR]),
                         source_ear_polygon_distance_px=polygon_distance(p[LEFT_EAR],item['ear']))
    labels,_=semantics(old['faces'])
    va,vb=old['vertices_camera'][0],arrays['vertices_camera'][0]
    protected=~np.isin(np.arange(len(va)),old['faces'][labels!=3].ravel())
    selected=np.r_[np.setdiff1d(screen.BODY,[0,15,16,17,18]),screen.FEET,np.arange(25,67)]
    supported=selected[B[0,selected,2]>0]
    delta=np.linalg.norm(arrays['joints_projected_native'][0]-old['joints_projected_native'][0],axis=1)
    vertex_m=float(np.linalg.norm(vb-va,axis=1)[protected].max())
    vertex_px=float(np.linalg.norm(project(vb,K)-project(va,K),axis=1)[protected].max())
    joints_px=float(delta[supported].max())
    passed=vertex_m<PROTECTED_METERS and vertex_px<PROTECTED_PIXELS and joints_px<PROTECTED_PIXELS
    ear_pass=points['candidate']['source_ear_polygon_distance_px']<=2. and points['candidate']['detector_error_px']<=.03*h
    return dict(frame=row['frame'], screen=flags, left_ear=points, original_gate_px=.03*h,
                all_retained_observed_nonhead_slots=supported.tolist(), max_nonhead_vertex_m=vertex_m,
                max_nonhead_vertex_native_px=vertex_px, max_observed_nonhead_joint_shift_px=joints_px,
                geometry_and_retained_protection_pass=passed, visible_ear_pass=ear_pass,
                diagnostic_accept=passed and ear_pass and not flags['candidate']['physically_supported'],
                quality_accepted=False)


def run(state):
    pins,rows,beta,latent,official,assets=state
    receipt=read(OUT/'preregistration.json')
    source_pins=receipt.pop('source_only_panel_sha256')
    need(receipt==plan(state) and not (OUT/'result.json').exists(), 'Preregistered inputs changed.')
    verify(source_pins)
    import torch
    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    body,vposer,_=seq.official.create_models(official_root=official,assets=assets,device='cpu',static_topology=False)
    pose=seq.official.decode_shared_body_pose(vposer,latent,device='cpu')
    del body,vposer
    need(array_sha(pose)==TARGET,'Frozen shared pose decode differs.')
    (OUT/'A-repeat').mkdir()
    (OUT/'B-confidence-only').mkdir()
    schedule=[(rows[0],'A'),*((r,'B') for r in rows)]
    jobs=[(item['row'],OUT/('A-repeat' if arm=='A' else 'B-confidence-only')/item['row']['frame'],
           item[arm],item['K'],beta,True,pose) for item,arm in schedule]
    start=time.monotonic()
    results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),
                             initializer=seq._init_cpu_fit,initargs=(official,assets)) as pool:
        for (item,arm),(arrays,desc) in zip(schedule,pool.map(seq._fit_cpu_job,jobs),strict=True):
            path=Path(desc['path'])
            verify({path:desc['sha256']})
            report=read(path)
            if arm=='A':
                need(report==item['report'] and arrays.keys()==item['old'].keys() and
                     all(a.dtype==item['old'][k].dtype and a.shape==item['old'][k].shape and
                         a.tobytes()==item['old'][k].tobytes() for k,a in arrays.items()),'Stock CPU parity failed.')
            else:
                result=assess(item,arrays,report,beta)
                result['candidate_receipt_sha256']=desc['sha256']
                result['candidate_winner_sha256']=sha(path.with_name('winner-camera-space.npz'))
                results.append(result)
            print(arm,item['row']['frame'],'complete',flush=True)
    verify(pins)
    verify(source_pins)
    save(OUT/'result.json',dict(state='complete_pending_visual_review', elapsed_s=time.monotonic()-start,
        stock_cpu_replay_byte_exact=True, gpu_calls=0, optimizer_edits=0, gaussian_updates=0,
        heldout_inputs_used=False, optimizer_fit_calls=3, model_promoted=False, quality_accepted=False,
        preregistration_sha256=sha(OUT/'preregistration.json'),records=results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    mode=parser.parse_args().mode
    state=preflight()
    if mode=='prepare':
        prepare(state)
    else:
        try:
            run(state)
        except Exception as exc:
            save(OUT/'failure.json',dict(state='failed',error=str(exc),gpu_calls=0))
            raise


if __name__=='__main__':
    main()
