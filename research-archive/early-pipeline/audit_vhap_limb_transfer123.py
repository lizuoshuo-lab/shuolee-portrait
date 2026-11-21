"""Original-prior evaluation of frozen all59 SMPLify/VHAP fits.

No optimizer, pose change, threshold selection, or new fitting experiment.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import fcntl
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
DEMO = ROOT / 'demo-vhap-initialization'
OUT = DEMO / 'smplify-vhap-sequential-stage/limb-transfer-audit'
PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
TRACKER = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx/vhap/model/tracker_smplx.py')
PRIOR = ROOT / 'smplifyx-reference-control/official/smplifyx/prior.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    started = time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert not OUT.exists(), 'Preserve completed diagnostics.'
    plan_path = DEMO/'smplify-vhap-limb-transfer-plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['optimizer_updates'] == 0 and plan['source_sha256'] == sha(__file__)
    for p, h in plan['input_hashes'].items():
        assert sha(p) == h, p
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(6060)
    loader = module(PROJECT/'src/avatar_core/shuolee_vposer.py', 'limb_transfer_vposer')
    loader.MODEL = ASSETS/'vposer/V02_05'
    loader.CODE = PROJECT/'third_party/human_body_prior'
    torch.backends.cuda.matmul.allow_tf32 = False
    vp, _, meta = loader.load_vposer('cuda')
    prior = module(PRIOR, 'limb_transfer_original_prior').SMPLifyAnglePrior().cuda()
    tracker_text = TRACKER.read_text()
    tree = ast.parse(tracker_text)
    regularizer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'compute_joint_L2_energy')
    regularizer_source = ast.get_source_segment(tracker_text, regularizer)
    assert 'self.pose_embedding[timesteps] ** 2' in regularizer_source
    assert 'angle_prior' not in tracker_text and 'bending_prior' not in tracker_text
    files = {
        'smplify_initial': DEMO/'fullbody-smplifyx/full-sequence/fit/fitted.npz',
        'normalized_control': DEMO/'smplify-vhap-normalized-laplacian/epoch-30/tracked-smplx-params.npz',
        'sequential_candidate': DEMO/'smplify-vhap-sequential-stage/epoch-30/tracked-smplx-params.npz',
    }
    runs = {}
    initial_parity = None
    roster = [f'frame_{i:06d}' for i in range(1, 60)]
    with torch.no_grad():
        for name, path in files.items():
            assert time.monotonic()-started < plan['max_seconds']
            with np.load(path, allow_pickle=False) as z:
                serialized_ids = list(z['frame_ids'])
                assert len(serialized_ids) == len(set(serialized_ids)) == 59
                assert set(serialized_ids) == set(roster)
                order = [serialized_ids.index(frame) for frame in roster]
                latent = torch.from_numpy(z['pose_embedding'][order].copy()).cuda()
                pose = torch.cat([vp.decode(latent[i:i+1])['pose_body'].reshape(1,63) for i in range(59)])
                if name == 'smplify_initial':
                    initial_parity = float(np.max(np.abs(pose.cpu().numpy()-z['body_pose_axis_angle'][order])))
                    print(json.dumps(dict(initial_decoded_pose_max_difference_rad=initial_parity,
                        largest_location=np.unravel_index(np.abs(pose.cpu().numpy()-z['body_pose_axis_angle'][order]).argmax(),pose.shape)),default=int),flush=True)
                    assert initial_parity < 5e-5
            penalties = prior(pose).cpu().numpy()
            pose = pose.cpu().numpy()
            runs[name] = dict(checkpoint_sha256=sha(path),
                bending_prior_unweighted={joint: dict(median=float(np.median(penalties[:, k])),
                    maximum=float(penalties[:, k].max())) for k, joint in enumerate(['left_elbow','right_elbow','left_knee','right_knee'])},
                records=[dict(frame=frame,angle_prior_components=penalties[i].tolist(),
                    selected_axis_angle_components=pose[i, [52,55,9,12]].tolist(),
                    body_axis_angle=pose[i].tolist()) for i, frame in enumerate(roster)])
    for p, h in plan['input_hashes'].items():
        assert sha(p) == h, p
    control = np.array([r['angle_prior_components'] for r in runs['normalized_control']['records']])
    candidate = np.array([r['angle_prior_components'] for r in runs['sequential_candidate']['records']])
    report = dict(updated=datetime.now(timezone.utc).isoformat(),state='complete_read_only',
        cpu_only=False,device='cuda:0',framewise_decoder_matches_saved_execution=True,
        frames=59,optimizer_updates=0,inputs_unchanged=True,
        original_prior_class_used=True,initial_decoded_pose_max_difference_rad=initial_parity,
        vposer=meta,source_sha256=sha(__file__),input_hashes=plan['input_hashes'],runs=runs,
        candidate_prior_increased_frames={joint:int((candidate[:,k]>control[:,k]).sum())
            for k,joint in enumerate(['left_elbow','right_elbow','left_knee','right_knee'])},
        source_contract=dict(smplify_initial_has_bending_prior=True,vhap_has_same_bending_prior=False,
            vhap_has_vposer_prior=True,vhap_has_hand_pca_prior=True,vhap_has_temporal_smoothing=True,
            automatic_weight_transfer_valid=False),
        seconds=time.monotonic()-started,quality_accepted=False,
        limits=['The angle prior is an exponential heuristic on four axis-angle components, not anatomical ground truth.',
            'Absence of this term in VHAP does not prove it caused the bad elbows, which already existed in SMPLify-X.',
            'SMPLify-X pixel-space sum/robust loss and VHAP NDC means have different scales; never copy numeric weights blindly.',
            'This saved-state audit cannot isolate the effect of the added sequential stage from numeric-prefix differences.'])
    OUT.mkdir()
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['runs','vposer','input_hashes']}))
    print(json.dumps({name:row['bending_prior_unweighted'] for name,row in runs.items()}))


if __name__ == '__main__':
    main()
