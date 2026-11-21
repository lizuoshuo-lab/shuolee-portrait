"""CPU-only face/landmark boundary audit of the saved 59-frame clothing trial."""
from pathlib import Path
import time
import numpy as np
import torch
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import OUT as AUDIT
from facial_geometry_boundary123 import protected_face_vertices, clothing_only_offset

OUT = AUDIT/'face-landmark-boundary'
MODEL = io.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
CACHE = AUDIT/'original-import-teacher-contract/original-import-lbs-cache.npz'
CANDIDATE = AUDIT/'teacher-import-surface-1000/candidate-diagnostic-only.npz'
OBS = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')


def main():
    assert OUT.exists() and not (OUT/'result.json').exists()
    assert not (OUT/'plan-frame-identity.json').exists()
    files = [Path(__file__), Path(__file__).with_name('facial_geometry_boundary123.py'),
             MODEL, CACHE, CANDIDATE, OBS, Path(__file__).with_name('vhap_demo_adapter123.py'),
             OBS.parent/'inputs.json',AUDIT.parent/'data/training-inputs.json']
    plan = dict(created=io.now(), hashes={str(f):io.sha(f) for f in files},
        cpu_only=True, optimizer_updates=0, frames=59, max_seconds=60,
        rule='No Gaussian surface/colour teacher gradient to facial geometry. A separate facial landmark branch may fit shape/expression/jaw with 2D and independently validated 3D observations.',
        source_boundary='Current original Gaussian baseline is unchanged; this is the next clothing-guide interface and a saved-candidate audit, not a new facial fit.')
    io.save(OUT/'frame-order-interface-repair.json',dict(updated=io.now(),
        initial_failure='The audit assumed array order equality. Gaussian inputs intentionally place reference33 first; detector observations remain chronological.',
        correction='Join by unique frame identity and verify raw image/mask hash and fullK for all59. Do not pair arrays by slot.',
        source_geometry_or_observations_changed=False,optimizer_updates=0))
    io.save(OUT/'plan-frame-identity.json',plan)
    started=time.monotonic();torch.set_num_threads(4)
    with np.load(MODEL,allow_pickle=True) as z:
        m={k:z[k] for k in ['f','weights','lmk_faces_idx','lmk_bary_coords','dynamic_lmk_faces_idx','dynamic_lmk_bary_coords']}
    with np.load(CACHE,allow_pickle=False) as z:cache=dict(z)
    with np.load(CANDIDATE,allow_pickle=False) as z:candidate=dict(z)
    with np.load(OBS,allow_pickle=False) as z:obs=dict(z)
    assert np.array_equal(m['f'],cache['faces']) and np.array_equal(m['f'],candidate['faces'])
    assert np.array_equal(cache['frame_ids'],candidate['frame_ids'])
    ids=list(map(str,cache['frame_ids']));obsids=list(map(str,obs['frame_ids']))
    assert len(set(ids))==len(set(obsids))==59 and set(ids)==set(obsids)
    input_rows={r['frame']:r for r in io.read(OBS.parent/'inputs.json')['records']}
    training=io.read(AUDIT.parent/'data/training-inputs.json')['records']
    assert ids==[r['frame'] for r in training]
    for r in training:
        original=input_rows[r['frame']]
        for k in ['source_sha256','mask_sha256','K','source_frame_index','time_seconds']:
            assert r[k]==original[k],(r['frame'],k)
    obs_slots=[obsids.index(fid) for fid in ids]
    assert len(cache['frame_ids'])==59
    faces=torch.tensor(m['f'],dtype=torch.long);weights=torch.tensor(m['weights'])
    protected=protected_face_vertices(weights,faces,torch.tensor(m['lmk_faces_idx']),torch.tensor(m['dynamic_lmk_faces_idx']))
    clothing=torch.tensor(cache['learnable_vertices'],dtype=torch.bool)
    support=torch.unique(faces[torch.tensor(np.r_[m['lmk_faces_idx'].ravel(),m['dynamic_lmk_faces_idx'].ravel()],dtype=torch.long)])
    assert not torch.any(clothing & protected), 'Prior clothing boundary overlaps facial support'
    assert bool(protected[support].all())
    delta=torch.linspace(-.001,.001,10475*3).reshape(10475,3).requires_grad_()
    safe=clothing_only_offset(delta,clothing,protected)
    # Deliberately nonzero incoming derivatives at every vertex. Both the direct
    # displacement and a mesh-edge regularizer use the effective masked offset.
    direct=(safe*torch.linspace(.3,1.3,10475*3).reshape(10475,3)).sum()
    edges=faces[:,[0,1]]
    regularizer=(safe[edges[:,0]]-safe[edges[:,1]]).square().sum()
    g1=torch.autograd.grad(direct,delta,retain_graph=True)[0]
    g2=torch.autograd.grad(regularizer,delta)[0]
    assert not torch.any(g1[protected]) and not torch.any(g2[protected])
    assert float(g1[clothing].norm())>0 and float(g2[clothing].norm())>0
    movement=np.linalg.norm(candidate['vertices_after']-candidate['vertices_before'],axis=-1)
    assert float(movement[:,protected.numpy()].max())==0
    assert float(np.abs(candidate['additional_canonical_delta'][protected.numpy()]).max())==0
    # Verify the exact repository landmark embedding, using each posed mesh.
    def landmarks(vertices):
        return np.einsum('nlvc,lv->nlc',vertices[:,m['f'][m['lmk_faces_idx']]],m['lmk_bary_coords'])
    before=landmarks(candidate['vertices_before']);after=landmarks(candidate['vertices_after'])
    assert np.array_equal(before,after)
    bc=np.concatenate((m['lmk_bary_coords'].reshape(-1,3),m['dynamic_lmk_bary_coords'].reshape(-1,3)))
    assert np.isfinite(bc).all() and np.abs(bc.sum(-1)-1).max()<1e-5
    np.savez_compressed(OUT/'vertex-boundary.npz',protected_face_vertices=protected.numpy(),
        clothing_vertices=clothing.numpy(),landmark_support_vertices=support.numpy(),faces=m['f'])
    result=dict(updated=io.now(),state='face_excluded_from_clothing_teacher_verified',
        protected_vertices=int(protected.sum()),landmark_support_vertices=len(support),
        clothing_vertices=int(clothing.sum()),overlap_vertices=0,
        inner_landmarks=51,contour_templates=list(m['dynamic_lmk_faces_idx'].shape),
        head_protection_includes_hair_hat_proxy=True,
        direct_gradient_on_face_max_abs=float(g1[protected].abs().max()),
        mesh_edge_regularizer_gradient_on_face_max_abs=float(g2[protected].abs().max()),
        saved_520_step_all59_face_displacement_max_m=float(movement[:,protected.numpy()].max()),
        saved_520_step_inner_landmark_change_max_m=float(np.abs(after-before).max()),
        observations=dict(path=str(OBS),shape=list(obs['keypoints'].shape),
            frame_join='Unique identities plus all59 original image/mask hashes, K, PTS and source-frame index',
            observation_slots_in_candidate_order=obs_slots,
            semantic_mapping='RTMW face68 global indices23:91; inner51 global40:91 = IBUG17:68 = SMPL-X static landmark slots0:51.',
            names=[str(n) for n in obs['joint_names'][23:91]],
            independent_3d_observation_arrays=[k for k in obs if '3d' in k.lower()],
            detector_outputs_are_not_manual_truth=True),
        face_policy=dict(gaussian_teacher=False,body_offset_regularizer_teacher=False,
            face_fitting_permanently_frozen=False,facial_branch='separate native-pixel2D landmark reprojection plus qualified3D landmarks and original shape/expression priors',
            visibility='Reject detector occlusion/identity mismatches by observation evidence, not by residual trimming.',
            contour='View-dependent silhouette landmarks; do not triangulate a fixed contour index as one physical surface point.',
            genuine_3d='Only separately validated multiview/material/depth observations in a common coordinate system with uncertainty. Current model-fitted3D is a prior, not independent3D truth.',
            parameters='Keep camera/body/gauge fixed for the separate face check. No unregistered compensation through body pose or shared shape.'),
        live_baseline_changed=False,new_face_fit_run=False,quality_accepted=False,
        elapsed_seconds=time.monotonic()-started)
    assert not result['observations']['independent_3d_observation_arrays']
    assert time.monotonic()-started<plan['max_seconds'] and not torch.cuda.is_initialized()
    for f,h in plan['hashes'].items():assert io.sha(f)==h
    io.save(OUT/'result.json',result)
    print({k:result[k] for k in ['state','protected_vertices','landmark_support_vertices','clothing_vertices','overlap_vertices','saved_520_step_all59_face_displacement_max_m','new_face_fit_run']},flush=True)


if __name__=='__main__':main()
