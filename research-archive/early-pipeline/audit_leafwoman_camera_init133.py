"""Replay the two official camera-init inputs on the frozen 022 fit; no optimizer."""
import ast
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from update_leafwoman129_state import FIT, ROOT, atomic, read, sha

OUT = ROOT/'leafwoman-fit40-132/camera-init-audit.json'
OFFICIAL = ROOT.parent/'smplifyx-reference-control/official'
ADAPTER = Path(__file__).with_name('official_smplifyx_entry123.py')
HIPS = [9, 12]


def run():
    upstream_fit=OFFICIAL/'smplifyx/fit_single_frame.py'
    upstream_loss=OFFICIAL/'smplifyx/fitting.py'
    config=OFFICIAL/'cfg_files/fit_smplx.yaml'
    selected=FIT/'selected-fit.npz'
    diff=FIT/'executed-upstream-interface.diff'
    sources=[upstream_fit,upstream_loss,config,selected,diff,ADAPTER,FIT/'result.json',FIT/'plan.json',Path(__file__)]
    hashes={str(p):sha(p) for p in sources}
    assert hashes[str(selected)]==read(FIT/'result.json')['fit_sha256']

    tree=ast.parse(upstream_fit.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_single_frame')
    args=fn.args.args[-len(fn.args.defaults):]
    torso=list(ast.literal_eval(fn.args.defaults[[a.arg for a in args].index('init_joints_idxs')]))
    entry_tree=ast.parse(ADAPTER.read_text(encoding='utf-8'))
    edges=[ast.literal_eval(n.value) for n in ast.walk(entry_tree) if isinstance(n,ast.keyword) and n.arg=='body_tri_idxs']
    assert torso==[9,12,2,5] and edges==[[[5,12],[2,9]]]
    edges=np.array(edges[0])
    loss_tree=ast.parse(upstream_loss.read_text(encoding='utf-8'))
    cls=next(n for n in loss_tree.body if isinstance(n,ast.ClassDef) and n.name=='SMPLifyCameraInitLoss')
    forward=ast.unparse(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    guess=ast.unparse(next(n for n in loss_tree.body if isinstance(n,ast.FunctionDef) and n.name=='guess_init'))
    assert 'torch.index_select(gt_joints, 1, self.init_joints_idxs)' in forward
    assert 'torch.sum(joint_error)' in forward and 'joints_conf' not in forward
    assert 'est_d = focal_length * (height3d / height2d)' in guess
    assert 'joints_to_ign:\n    - 1\n    - 9\n    - 12' in config.read_text(encoding='utf-8')
    executed=diff.read_text(encoding='utf-8')
    assert '_normalized_joints(gt_joints, camera.K)' in executed and 'focal_length=1.0' in executed

    with np.load(selected,allow_pickle=False) as z:
        frames=z['frame_ids'].astype(str).tolist()
        targets=z['observed_native_keypoints'].astype(np.float64)
        points=z['joints_camera'].astype(np.float64)
        saved_uv=z['joints_projected_native'].astype(np.float64)
        K=z['K_compute_float32'].astype(np.float64)
    assert len(frames)==24 and points.shape==(24,118,3) and np.all(points[...,2]>0)
    p=points@K.T
    uv=p[...,:2]/p[...,2:3]
    assert np.max(abs(uv-saved_uv))<.001

    def normal_xy(xy):
        rays=np.concatenate((xy,np.ones_like(xy[...,:1])),axis=-1)@np.linalg.inv(K).T
        return rays[...,:2]/rays[...,2:3]

    def mean_edge(a):
        return np.linalg.norm(a[:,edges[:,0]]-a[:,edges[:,1]],axis=-1).mean(axis=1)

    # The 3D numerator is intentionally the same saved geometry in both arms.
    # The seed ratio cancels it. The real initial model used neutral/mean pose,
    # so neither replayed depth is the original start depth or a corrected camera.
    xy=targets[...,:2]
    counter=xy.copy();counter[:,HIPS]=uv[:,HIPS]
    denom=mean_edge(normal_xy(xy));counter_denom=mean_edge(normal_xy(counter))
    assert np.all(denom>0) and np.all(counter_denom>0)
    ratio=denom/counter_denom
    hip_conf_off=targets.copy();hip_conf_off[:,HIPS,2]=0
    assert np.array_equal(mean_edge(normal_xy(hip_conf_off[...,:2])),denom)
    sq=((xy-uv)**2).sum(axis=-1)
    per_hip=sq[:,HIPS].sum(axis=1)
    per_torso=sq[:,torso].sum(axis=1)
    assert np.all(per_torso>0)
    reprojection_before=sq[:,torso].sum(axis=1)
    reprojection_conf_off=((hip_conf_off[...,:2]-uv)**2).sum(axis=-1)[:,torso].sum(axis=1)
    assert np.array_equal(reprojection_conf_off,reprojection_before)
    frac=per_hip/per_torso
    rows=[dict(frame=f,original_hip_residual_native_px=np.sqrt(sq[i,HIPS]).tolist(),
        hip_fraction_of_unweighted_camera_init_2d_sse=float(frac[i]),
        original_normalized_torso_edge_mean=float(denom[i]),
        replay_replaced_hip_normalized_torso_edge_mean=float(counter_denom[i]),
        depth_ratio_replay_counterfactual_over_original=float(ratio[i])) for i,f in enumerate(frames)]
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='camera_init_hidden_hip_path_confirmed',
        source_sha256=hashes,rows=rows,init_joint_ids=torso,depth_edge_ids=edges.tolist(),
        original_camera_init_honors_joint_confidence=False,
        main_pose_ignores_hip_ids=HIPS,zero_confidence_does_not_change_depth_or_camera_init_2d_term=True,
        hip_detector_residual_median_px=float(np.median(np.sqrt(sq[:,HIPS]))),
        hip_fraction_of_init_2d_sse_median=float(np.median(frac)),
        hypothetical_depth_ratio_median=float(np.median(ratio)),
        hypothetical_depth_ratio_range=[float(ratio.min()),float(ratio.max())],
        interpretation='The stage-0 camera/body-root init ingests hip XY even when final pose loss excludes it. On saved geometry its torso 2D term is dominated by hips. Hypothetical use of saved projected hips shows sensitivity only, not observed hip truth or improved depth.',
        limits='Replay at the frozen final mesh, not the actual historical neutral start or optimizer trace. No independent scene camera/gravity; cannot isolate sole cause or assert that a seed replacement fixes shape.',
        next='Before a new fit, verify per-view support for both the original depth edges and camera-init joints, or validate an explicitly supported initializer through the original entry. Masking hip confidence alone cannot fix this. Do not use the counterfactual projected hips as observations.',
        new_fit_started=False,new_gaussian_started=False,CUDA_initialized=False,quality_accepted=False)
    assert hashes=={str(p):sha(p) for p in sources}
    atomic(OUT,result)
    print('VIDEO22_CAMERA_INIT_REPLAY',len(rows),round(float(np.median(frac)),5),[round(float(x),5) for x in (ratio.min(),np.median(ratio),ratio.max())])


if __name__=='__main__':
    run()
