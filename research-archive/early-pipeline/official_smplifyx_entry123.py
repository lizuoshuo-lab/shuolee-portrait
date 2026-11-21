"""Single-frame upstream SMPLify-X entry with explicit input/output adapters.

The pinned five-stage loop, loss, FittingMonitor and vendored LBFGS are executed.
No sequence optimizer or cross-frame shape policy is implemented here. The
caller may provide one shared beta vector and explicitly freeze it for this fit.
"""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback
import types
from collections import defaultdict

import numpy as np

COMMIT = '68f8536707f43f4736cdd75a19b18ede886a4d53'
PINNED = {
    'smplifyx/fit_single_frame.py': 'd33110e81fcffe9a8806ae0a77b3f6456952a23c90b71ae61c42ec3f81bddc6f',
    'smplifyx/fitting.py': '487198f1caaae9367f4fc165db76d81a41396cbec72d64e2b080b1296ce951eb',
    'smplifyx/prior.py': 'c9eb8b97def0763e0b22bacd299b0853c2f7cfbe99a3e5180fabec6b3a4b7af9',
    'smplifyx/utils.py': '6026ab2056a5d85b9bf7fa60f04d380eabb52a3de0e2a6cbdf313a717503d945',
    'smplifyx/optimizers/optim_factory.py': '56ad18356b485c7310b10db505c311dee60786d666d1092723995f58516cf96a',
    'smplifyx/optimizers/lbfgs_ls.py': 'ace4c89b9aa6d259adf6ed88d01ab9dfe88d10045d6be70b0161c2117f6385e8',
    'cfg_files/fit_smplx.yaml': '533fbbf81ba29a60eebbb8f1383f2c10bba449a6d6c26067e170e0db7ac7edac',
}
COCO = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_official_config(official_root):
    """Apply the upstream CLI numeric types after YAML parsing, not new values.

    PyYAML retains e.g. 4.04e2 as str, whereas the upstream configargparse
    options declare type=float. Jaw triples intentionally remain strings for
    fit_single_frame's own comma-separated float conversion.
    """
    import yaml
    config = yaml.safe_load((Path(official_root)/'cfg_files/fit_smplx.yaml').read_text())
    for key in ('focal_length', 'rho', 'depth_loss_weight', 'df_cone_height',
                'side_view_thsh', 'ftol', 'gtol', 'lr'):
        if key in config:
            config[key] = float(config[key])
    for key in ('data_weights', 'body_pose_prior_weights', 'shape_weights',
                'expr_weights', 'face_joints_weights', 'hand_joints_weights',
                'hand_pose_prior_weights', 'coll_loss_weights'):
        if key in config and config[key] is not None:
            config[key] = [float(value) for value in config[key]]
    for key in ('batch_size', 'num_gaussians', 'num_pca_comps',
                'max_collisions', 'maxiters'):
        if key in config:
            config[key] = int(config[key])
    return config


def adapted_entry_source(official_root):
    """Exact replacements only; keep pristine source and expose the full diff."""
    root = Path(official_root)
    for relative, digest in PINNED.items():
        need(sha(root/relative) == digest, 'Pinned source differs: '+relative)
    raw = (root/'smplifyx/fit_single_frame.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'fit_single_frame')
    source = ast.get_source_segment(raw, node)
    original = source
    edits = [
        ("vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')", "vposer = kwargs['_vposer_instance']"),
        ("init_t = fitting.guess_init(body_model, gt_joints, edge_indices,\n                                use_vposer=use_vposer, vposer=vposer,\n                                pose_embedding=pose_embedding,\n                                model_type=kwargs.get('model_type', 'smpl'),\n                                focal_length=focal_length, dtype=dtype)",
         "init_t = fitting.guess_init(body_model, _normalized_joints(gt_joints, camera.K), edge_indices,\n                                use_vposer=use_vposer, vposer=vposer,\n                                pose_embedding=pose_embedding,\n                                model_type=kwargs.get('model_type', 'smpl'),\n                                focal_length=1.0, dtype=dtype)"),
        ("body_model.reset_params(body_pose=body_mean_pose)",
         "_reset_body(body_model, kwargs['_shared_betas'], body_pose=body_mean_pose)"),
        ("camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5",
         "_require_fixed_K(camera, kwargs['_fixed_K'])"),
        ("body_model.reset_params(**new_params)",
         "_reset_body(body_model, kwargs['_shared_betas'], **new_params)"),
    ]
    for old, new in edits:
        need(source.count(old) == 1, 'Pinned input hook does not match source.')
        source = source.replace(old, new)
    marker = '\n    if save_meshes or visualize:'
    need(source.count(marker) == 1, 'Pinned output hook does not match source.')
    # Upstream OBJ flips X and uses the last orientation's live model. Export
    # from the selected result instead; the complete optimization above remains.
    source = source.split(marker)[0] + "\n    return {'winner': results[min_idx], 'orientation_count': len(results)}\n"
    compile(source, 'pinned-fit-single-frame-input-output-adapter', 'exec')
    diff = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                                      fromfile='pinned/fit_single_frame', tofile='adapter/fit_single_frame'))
    return source, diff


def load_official(root):
    """Load headless upstream definitions without altering their AST bodies."""
    root = Path(root)
    utils = module(root/'smplifyx/utils.py', '_smplifyx123_utils')
    prior = module(root/'smplifyx/prior.py', '_smplifyx123_prior')
    fitting = types.ModuleType('_smplifyx123_fitting')
    tree = ast.parse((root/'smplifyx/fitting.py').read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if not
                 (isinstance(n, ast.ImportFrom) and n.module == 'mesh_viewer') and not
                 (isinstance(n, ast.Import) and any(a.name == 'utils' for a in n.names))]
    fitting.__dict__['utils'] = utils
    exec(compile(tree, str(root/'smplifyx/fitting.py'), 'exec'), fitting.__dict__)
    package = types.ModuleType('_smplifyx123_optimizers')
    package.__path__ = [str(root/'smplifyx/optimizers')]
    sys.modules[package.__name__] = package
    optimizer = module(root/'smplifyx/optimizers/optim_factory.py', package.__name__+'.optim_factory')
    return utils, prior, fitting, optimizer


def fit_one(*, official_root, image, keypoints, K_integer_centers, body_model,
            vposer2, result_path, shared_betas, freeze_betas,
            device, diff_path):
    """One frame only. Caller explicitly decides shared_betas/freeze_betas."""
    import cv2
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    need(isinstance(freeze_betas, bool), 'freeze_betas must be explicit bool.')
    need(not freeze_betas or shared_betas is not None, 'Frozen betas require an explicit shared vector.')
    source, diff = adapted_entry_source(official_root)
    Path(diff_path).write_text(diff, encoding='utf-8')
    utils, prior, fitting, optimizer = load_official(official_root)
    dtype = torch.float32
    dev = torch.device(device)
    K = torch.as_tensor(np.asarray(K_integer_centers), dtype=dtype, device=dev).clone()
    need(tuple(K.shape) == (3,3) and torch.isfinite(K).all().item(), 'Expected finite complete 3x3 K.')
    need(torch.equal(K[2], K.new_tensor([0,0,1])) and K[1,0].item() == 0 and
         K[0,0].item() > 0 and K[1,1].item() > 0, 'K must be positive upper triangular integer-center projection.')
    need(not hasattr(body_model, 'transl'), 'Official body must use create_transl=False; apply root translation once in camera.')
    need(not hasattr(body_model, 'body_pose'), 'Official VPoser path must use create_body_pose=False.')

    class FullKCamera(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('K', K)
            self.register_buffer('rotation', torch.eye(3, dtype=dtype, device=dev)[None])
            self.translation = nn.Parameter(torch.zeros((1,3), dtype=dtype, device=dev))
        def forward(self, points):
            p = points + self.translation[:, None]
            q = p @ self.K.T
            return q[..., :2] / q[..., 2:]

    class VPoser2API(nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def decode(self, latent, output_type='aa'):
            need(output_type == 'aa', 'Only official axis-angle decoder API is supported.')
            return self.model.decode(latent)['pose_body'].reshape(latent.shape[0], -1)

    shared = None
    if shared_betas is not None:
        value = np.asarray(shared_betas)
        need(value.shape in [(10,), (1,10)] and np.isfinite(value).all(), 'shared_betas must be one finite 10-vector.')
        shared = torch.as_tensor(value, dtype=dtype, device=dev).reshape(1,10).clone()
    body_model.betas.requires_grad_(not freeze_betas)
    # The standard entry starts at mean/zero parameters; never inherit another
    # fit through a reused body instance. Explicit shared shape is the sole
    # permitted replacement, including the geometry used by guess_init.
    body_model.reset_params(**({'betas': shared} if shared is not None else {}))
    camera = FullKCamera()

    def reset_body(body, beta, **parameters):
        if beta is not None:
            parameters['betas'] = beta
        body.reset_params(**parameters)
    def require_fixed_K(cam, frozen):
        need(torch.equal(cam.K, frozen), 'K changed during upstream initialization.')
    def normalized_joints(joints, fixed_K):
        homogeneous = torch.cat([joints, torch.ones_like(joints[..., :1])], dim=-1)
        rays = homogeneous @ torch.linalg.inv(fixed_K).T
        return rays[..., :2] / rays[..., 2:]

    scope = dict(torch=torch, np=np, time=time, pickle=pickle, osp=os.path,
                 cv2=cv2, tqdm=tqdm, defaultdict=defaultdict,
                 optim_factory=optimizer, fitting=fitting,
                 _reset_body=reset_body, _require_fixed_K=require_fixed_K, _normalized_joints=normalized_joints)
    exec(compile(source, 'pinned-fit-single-frame-input-output-adapter', 'exec'), scope)
    config = load_official_config(official_root)
    # These three are integration/runtime settings, not altered solver settings:
    # headless, custom export, and collision disabled as in the prior local run.
    config.update(interpenetration=False, visualize=False, save_meshes=False,
                  use_cuda=dev.type == 'cuda', model_type='smplx')
    jw = torch.ones((1,118), dtype=dtype, device=dev); jw[:,[1,9,12]] = 0
    l2 = lambda: prior.L2Prior().to(device=dev, dtype=dtype)
    result = scope['fit_single_frame'](
        image, keypoints, body_model, camera, jw, l2(), l2(), l2(), l2(),
        l2(), l2(), prior.SMPLifyAnglePrior().to(device=dev, dtype=dtype),
        result_fn=str(result_path), dtype=dtype,
        _vposer_instance=VPoser2API(vposer2), _shared_betas=shared,
        _fixed_K=K.detach().clone(), body_tri_idxs=[[5,12],[2,9]], **config)
    winner = result['winner']['result']
    winning_latent = torch.as_tensor(winner['body_pose'], dtype=dtype, device=dev)
    body_model.reset_params(**{name: winner[name] for name,_ in body_model.named_parameters()})
    with torch.no_grad():
        camera.translation.copy_(torch.as_tensor(winner['camera_translation'], device=dev, dtype=dtype))
        axis_angle = vposer2.decode(winning_latent)['pose_body'].reshape(1,63)
        output = body_model(body_pose=axis_angle, return_verts=True, return_full_pose=True)
        camera_vertices = output.vertices + camera.translation[:,None]
        camera_joints = output.joints + camera.translation[:,None]
        projected = camera(output.joints)
    arrays = {name: np.asarray(winner[name]) for name,_ in body_model.named_parameters()}
    arrays.update(body_pose_axis_angle=axis_angle.cpu().numpy(), pose_embedding=winning_latent.cpu().numpy(),
                  camera_translation=camera.translation.detach().cpu().numpy(),
                  vertices_camera=camera_vertices.cpu().numpy(), joints_camera=camera_joints.cpu().numpy(),
                  joints_projected_native=projected.cpu().numpy(), faces=np.asarray(body_model.faces),
                  K_integer_centers=np.asarray(K_integer_centers).copy(), K_compute_float32=K.cpu().numpy())
    need(all(np.isfinite(a).all() for a in arrays.values()), 'Winner export contains nonfinite values.')
    return arrays, dict(official_commit=COMMIT, source_hashes=PINNED,
        upstream_loss_monitor_optimizer_unchanged=True, five_stages=True,
        initialization='Official zero VPoser latent / mean pose / zero shape unless explicit shared SMPLify-X beta supplied',
        gvhmr_used=False, camera_initialization='Upstream guess_init on K-inverse normalized native 2D, focal=1; unchanged camera loss/monitor',
        interpenetration=False, collision_status='Existing local configuration; no collision assets downloaded.',
        use_shared_betas=shared_betas is not None, freeze_betas=freeze_betas,
        cross_frame_identity_policy_implemented=False, orientation_count=result['orientation_count'],
        winning_loss=float(result['winner']['loss']), coordinate_system='opencv_camera_model_meters',
        translation_applied_once=True, full_K_integer_centers=True,
        external_world_camera_used=False, physical_camera_calibration_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['official-root','data-root','runtime-root','assets-root','frame-id','output-dir','device']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--shared-betas-npz'); parser.add_argument('--shared-betas-key', default='betas')
    parser.add_argument('--freeze-betas', action='store_true')
    args = parser.parse_args()
    output = Path(args.output_dir)
    need(not output.exists(), 'Use a new output directory; never resume another fit.')
    need(args.device in ['cpu','cuda'], 'device must be cpu or cuda.')
    output.mkdir(parents=True)
    save(output/'progress.json', dict(state='preparing',pid=os.getpid(),frame_id=args.frame_id))
    try:
        adapted_entry_source(args.official_root)
        import torch
        import smplx
        from PIL import Image
        from smplify_score_contract123 import relative_confidence
        data = Path(args.data_root); assets = Path(args.assets_root); runtime = Path(args.runtime_root)
        rows = json.loads((data/'data/inputs.json').read_text())['records']
        matches = [(i,r) for i,r in enumerate(rows) if r['frame'] == args.frame_id]
        need(len(matches) == 1, 'Frame identity must exist exactly once.')
        slot, row = matches[0]
        with np.load(data/'data/observations.npz',allow_pickle=False) as f: obs = dict(f)
        need(list(obs['frame_ids']) == [r['frame'] for r in rows], 'Observation order differs from explicit frame identities.')
        image_path = Path(row['image']); need(sha(image_path) == row['source_sha256'], 'Native image hash differs.')
        with Image.open(image_path) as im:
            need(list(im.size) == row['wh'], 'Native image dimensions differ.')
            image = np.asarray(im.convert('RGB'),dtype=np.float32)/255
        confidence, score_q90 = relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
        keypoints = np.zeros((1,118,3),np.float32)
        dest = np.r_[COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
        source = np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
        keypoints[0,dest,:2] = obs['keypoints'][slot,source]
        keypoints[0,dest,2] = confidence[slot,source]
        need(np.isfinite(keypoints).all(), 'Native observation must be finite.')
        utils,_,_,_ = load_official(args.official_root)
        mapping = utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        need(len(mapping) == 118, 'Official observation mapping must contain 118 anchors.')
        loader = module(runtime/'src/avatar_core/shuolee_vposer.py','_entry123_vposer_loader')
        loader.MODEL = assets/'vposer/V02_05'; loader.CODE = runtime/'third_party/human_body_prior'
        vposer,_,vposer_meta = loader.load_vposer(args.device)
        body = smplx.SMPLX(str(assets/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,
            num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
            flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
            joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to(args.device)
        shared = None
        if args.shared_betas_npz:
            with np.load(args.shared_betas_npz,allow_pickle=False) as f: shared = f[args.shared_betas_key].copy()
        provenance = dict(frame_id=args.frame_id,slot=slot,source_frame_index=row['source_frame_index'],
            source_image_sha256=row['source_sha256'],native_wh=row['wh'],gvhmr_used=False,
            initialization='Official zero/mean SMPLify-X initialization plus VPoser2 pose prior; no image-to-3D prediction checkpoint.',
            inputs_sha256=sha(data/'data/inputs.json'),observations_sha256=sha(data/'data/observations.npz'),
            score_contract=dict(kind='relative RTMW score plus flip-consistency heuristic; not calibrated probability',
                                per_joint_q90=score_q90.tolist()),
            vposer=vposer_meta,adapter_sha256=sha(__file__),
            shared_betas_sha256=sha(args.shared_betas_npz) if args.shared_betas_npz else None)
        save(output/'input-provenance.json',provenance)
        save(output/'progress.json',dict(state='official_fit_single_frame_running',pid=os.getpid(),frame_id=args.frame_id))
        start=time.monotonic()
        arrays, report = fit_one(official_root=args.official_root,image=image,keypoints=keypoints,
            K_integer_centers=row['K_integer_centers'],body_model=body,vposer2=vposer,
            result_path=output/'official-result.pkl',shared_betas=shared,freeze_betas=args.freeze_betas,
            device=args.device,diff_path=output/'input-output-adapter.diff')
        arrays.update(frame_id=np.asarray(args.frame_id),observed_native_keypoints=keypoints,
                      source_frame=np.asarray(row['source_frame_index']),image_sha256=np.asarray(row['source_sha256']))
        np.savez_compressed(output/'winner-camera-space.npz',**arrays)
        # Original camera coordinates, no graphics-axis flip, no SfM world claim.
        import trimesh
        trimesh.Trimesh(vertices=arrays['vertices_camera'][0],faces=arrays['faces'],process=False).export(output/'winner-camera-space.obj')
        report.update(state='complete',elapsed_seconds=time.monotonic()-start,frame_id=args.frame_id,
                      image_quality_accepted=False,gaussian_training_started=False,goal_complete=False)
        save(output/'result.json',report);save(output/'progress.json',dict(state='complete',pid=None))
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as error:
        save(output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        save(output/'progress.json',dict(state='failed',pid=None))
        raise


if __name__ == '__main__':
    main()
