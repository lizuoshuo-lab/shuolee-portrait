"""One post-training ref26 mesh-facing alpha diagnostic, never a model export.

CPU plan binds the actual final18000 snapshot. A separate, <=60-second GPU
command renders new in-memory plain-PLY objects only after the trainer exits.
Official mesh winding supplies a facing proxy, not physical clothing surfaces.
"""
import argparse
import ast
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFilter
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS

H = Path(__file__).resolve().parent
ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/football-half12-paired18k126')
STEP = 18000
REFERENCE = 'frame_000026'
SNAP = ROOT/'gaussian-run/step-018000'
OUT = ROOT/'diagnostics/ref26-facing-proxy018000'


def topology_facing(vertices, faces, eye):
    assert vertices.shape == (10475,3) and faces.shape == (20908,3)
    assert np.isfinite(vertices).all() and faces.dtype.kind in 'iu'
    assert faces.min() >= 0 and faces.max() < len(vertices)
    tri = vertices[faces]
    normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    assert (np.linalg.norm(normal,axis=1)>1e-12).all()
    centroid = tri.mean(axis=1)
    dot = np.einsum('ij,ij->i',normal,np.asarray(eye)-centroid)
    front = dot >= 0  # Deterministic; no discarded grazing faces or points.
    directed = faces[:,[0,1,1,2,2,0]].reshape(-1,2)
    edge, inverse, counts = np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    signs = np.where(directed[:,0]<directed[:,1],1,-1)
    balance = np.bincount(inverse,weights=signs,minlength=len(edge))
    inconsistent = int(((counts==2)&(balance!=0)).sum())
    raw_volume = float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
    centered = vertices-vertices.mean(axis=0)
    ct = centered[faces]
    centered_volume = float(np.einsum('ij,ij->i',ct[:,0],np.cross(ct[:,1],ct[:,2])).sum()/6)
    report = dict(boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),
        inconsistent_paired_edge_winding=inconsistent,closed=bool((counts==2).all()),
        signed_volume_original_origin=raw_volume,signed_volume_centered_origin=centered_volume,
        volume_units='Model-coordinate units cubed; conditional scale, open-boundary volume is not a solid-volume proof.',
        front_faces=int(front.sum()),back_faces=int((~front).sum()),exact_grazing_faces=int((dot==0).sum()),
        rule='Unmodified official face winding: cross(v1-v0,v2-v0) dot (camera_eye-centroid)>=0 is front-facing proxy.',
        physical_clothing_surface_classification=False,topology_modified=False)
    assert inconsistent == 0 and report['nonmanifold_edges'] == 0, report
    assert raw_volume > 0 and centered_volume > 0, report
    assert front.any() and (~front).any()
    return front, dot, report


def hashes_unchanged(plan):
    for path,digest in plan['input_hashes'].items():
        assert sha(path) == digest, path


def plan():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists(), 'Exactly one registered diagnostic; inspect existing evidence, do not overwrite.'
    # These files must really exist before a plan is created. No anticipated
    # final snapshot, old1000 checkpoint or old-person object is substituted.
    final = read(ROOT/'gaussian-run/result.json')
    assert final['state'] == 'bounded_control_completed' and final['snapshot_step'] == STEP
    result = read(SNAP/'result.json')
    assert result['step'] == STEP and result['reference_frame'] == REFERENCE and result['reference_slot'] == 0
    assert result['active_sh_degree'] == 0
    parity = result['native_ply_parity']
    assert parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005
    assert sha(SNAP/'world.ply') == result['world_ply_sha256']
    rows = read(ROOT/'data/training-inputs.json')['records']
    row = rows[result['reference_slot']]
    assert len(rows) == 12 and row['frame'] == REFERENCE
    E,K = np.asarray(row['w2c'],dtype=np.float64),np.asarray(row['K'],dtype=np.float64)
    assert np.array_equal(E,np.eye(4)) and K.shape == (3,3)
    assert np.array_equal(K[2],[0,0,1]) and K[0,1] == K[1,0] == 0
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:
        ids = z['frame_ids'].astype(str).tolist()
        assert ids == [r['frame'] for r in rows]
        faces,vc = z['faces'].copy(),z['vertices_camera'][ids.index(REFERENCE)].astype(np.float64)
    vertices = (vc-E[:3,3])@E[:3,:3]
    eye = -E[:3,:3].T@E[:3,3]
    face_front,dot,topology = topology_facing(vertices,faces,eye)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:
        binding = z['triangle_ids'].copy()
        assert z['frame_ids'].astype(str).tolist() == ids
    from plyfile import PlyData
    world = PlyData.read(SNAP/'world.ply')['vertex'].data
    local = PlyData.read(SNAP/'bound-local.ply')['vertex'].data
    assert binding.shape == (len(world),) and len(world) == len(local) == result['points']
    assert binding.dtype.kind in 'iu' and binding.min() >= 0 and binding.max() < len(faces)
    assert not any(k.startswith('f_rest_') for k in world.dtype.names)
    assert all(np.isfinite(world[key]).all() for key in world.dtype.names)
    # World export and snapshot binding preserve the same Gaussian order;
    # unchanged radiance/opacity fields independently guard against reordering.
    for key in ['f_dc_0','f_dc_1','f_dc_2','opacity']:
        assert world[key].tobytes() == local[key].tobytes(), key
    point_front = face_front[binding]
    assert point_front.any() and (~point_front).any()
    assert int(point_front.sum())+int((~point_front).sum()) == len(world)
    mappings = read(ROOT/'pixel-mapping.json')['records']
    mapping = next(m for m in mappings if m['frame'] == REFERENCE)
    expected_Ki = np.asarray(mapping['integer_center_transform'])@np.asarray(mapping['source']['K_integer_centers'])
    expected_K = expected_Ki.copy();expected_K[:2,2] += .5
    assert np.array_equal(expected_K,K) and row['wh'] == mapping['derived_wh']
    inputs = [Path(__file__),SNAP/'world.ply',SNAP/'bound-local.ply',SNAP/'binding.npz',SNAP/'current-meshes.npz',
        SNAP/'result.json',SNAP/(REFERENCE+'_gaussian.png'),ROOT/'gaussian-run/result.json',
        ROOT/'data/training-inputs.json',ROOT/'pixel-mapping.json',
        Path(row['image']),Path(row['mask']),PROJECT/'src/avatar_core/gaussian_renderer/__init__.py',
        PROJECT/'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
        PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py',PROJECT/'src/shuolee_portrait/training.py']
    for file in inputs:
        assert file.is_file(), str(file)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'groups.npz',triangle_ids=binding,face_front=face_front,point_front=point_front,
        face_normal_dot_eye=dot,frame=np.asarray(REFERENCE))
    doc = dict(updated=now(),state='cpu_plan_ready',step=STEP,reference_frame=REFERENCE,
        source_world_ply_sha256=result['world_ply_sha256'],row=row,pixel_mapping=mapping,
        input_hashes={str(p):sha(p) for p in inputs},groups_sha256=sha(OUT/'groups.npz'),
        counts=dict(full=len(world),front=int(point_front.sum()),back=int((~point_front).sum())),
        topology=topology,native_snapshot_float_parity=parity,
        full_rgb_check='Same original image() encoding, compare replay PNG with actual saved snapshot PNG. This is encoded RGB replay, not invented floatRGB ground truth.',
        parity_limits=dict(mean_abs_strict_less_than=2e-6,max_abs_strict_less_than=.005),
        GPU_budget_seconds=60,GPU_started=False,optimizer_or_loss_changed=False,
        no_model_publication=True,no_point_or_topology_deletion=True,
        all_binding_points_including_face_and_protected_regions=True,quality_accepted=False,
        limitation='Independent front/back group alphas are not additive or disjoint pixel contributions. They may overlap. High full-ray alpha can be supplied by back-bound points; mesh-facing groups are not measured clothing layers.')
    save(OUT/'plan.json',doc)
    print('FOOTBALL_REF26_FACING_PLAN_READY',json.dumps(doc['counts']),flush=True)


def metrics(alpha,mask):
    values = alpha[np.asarray(mask)>0]
    return dict(pixels=int(values.size),quantile_probabilities=[0,.01,.05,.5,.95,.99,1],
        quantiles=np.quantile(values,[0,.01,.05,.5,.95,.99,1]).tolist() if values.size else None,
        fraction_below_0_5=float((values<.5).mean()) if values.size else None,
        fraction_below_0_95=float((values<.95).mean()) if values.size else None)


def run():
    import fcntl
    doc = read(OUT/'plan.json')
    assert doc['step'] == STEP and doc['reference_frame'] == REFERENCE
    assert not (OUT/'launch.json').exists(), 'One bounded run only; preserve existing diagnostics.'
    hashes_unchanged(doc)
    assert sha(OUT/'groups.npz') == doc['groups_sha256']
    progress = read(ROOT/'progress.json')
    assert progress['state'] == 'bounded_control_completed'
    pid = progress.get('pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists():
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        state = next(x.split()[1] for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('State:'))
        assert not ('train_football_half12_sh0126.py' in cmd and state != 'Z'), 'Wait for the real trainer to exit.'
    with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(OUT/'launch.json',dict(updated=now(),pid=os.getpid(),command=sys.argv,single_gpu_lock=True))
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Registered60secondGPUdiagnostic')))
        signal.alarm(60)
        started = time.monotonic()
        try:
            import torch
            torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
            sys.path.insert(0,str(PROJECT/'src'))
            from shuolee_portrait import runtime as rt
            rt.configure(ASSETS)
            from scene.gaussian_model import GaussianModel
            from gaussian_renderer import render
            from shuolee_portrait.training import camera, image
            with np.load(OUT/'groups.npz',allow_pickle=False) as z:
                front = z['point_front'].copy()
            cpu_rng = torch.get_rng_state().clone();cuda_rng = torch.cuda.get_rng_state().clone()
            py_rng = random.getstate();np_rng = np.random.get_state()
            original = GaussianModel(0);original.load_ply(str(SNAP/'world.ply'));original.active_sh_degree=0
            fields = ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation']
            saved_fields = {key:getattr(original,key).detach().clone() for key in fields}
            models = {'full':original}
            for name,keep in [('front',front),('back',~front)]:
                subset = GaussianModel(0);subset.active_sh_degree=0
                indices = torch.as_tensor(np.flatnonzero(keep),device='cuda',dtype=torch.long)
                for key in fields:
                    setattr(subset,key,torch.nn.Parameter(getattr(original,key).detach()[indices].clone(),requires_grad=False))
                models[name] = subset
            cam,projection = camera(doc['row'],0)
            pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            black = torch.zeros(3,device='cuda');white = torch.ones(3,device='cuda')
            mask = Image.open(doc['row']['mask']).convert('L')
            inner = mask.filter(ImageFilter.MinFilter(11))
            output = {};pictures = [];alpha_pictures = []
            with torch.no_grad():
                full = render(cam,original,pipe,black)['render']
                encoded = image(full)
                encoded.save(OUT/'full_rgb_encoded.png')
                expected = np.asarray(Image.open(SNAP/(REFERENCE+'_gaussian.png')).convert('RGB'),dtype=np.float64)/255
                actual = np.asarray(encoded,dtype=np.float64)/255
                difference = np.abs(actual-expected)
                parity = dict(mean_abs=float(difference.mean()),max_abs=float(difference.max()),
                    comparison='Both sides encoded by original image(), saved snapshot PNG versus new worldPLY native replay; not floatRGB GT.')
                parity['passed'] = parity['mean_abs']<2e-6 and parity['max_abs']<.005
                save(OUT/'encoded-full-replay.json',parity)
                assert parity['passed'], 'ENCODED_FULL_REPLAY_FAILED; preserve failure, do not relax thresholds: '+str(parity)
                for name,model in models.items():
                    rgb = full if name=='full' else render(cam,model,pipe,black)['render']
                    trans = render(cam,model,pipe,white,override_color=torch.zeros_like(model.get_xyz))['render']
                    channel_error = float((trans-trans[:1]).abs().max())
                    assert channel_error<2e-6
                    alpha = 1.-trans.mean(0)
                    assert torch.isfinite(alpha).all() and float(alpha.min())>=-2e-5 and float(alpha.max())<=1+2e-5
                    rgb_array = rgb.detach().cpu().numpy().astype(np.float32)
                    alpha_array = alpha.detach().cpu().numpy().astype(np.float32)
                    rgb_path,alpha_path = OUT/(name+'_rgb.npy'),OUT/(name+'_alpha.npy')
                    np.save(rgb_path,rgb_array,allow_pickle=False);np.save(alpha_path,alpha_array,allow_pickle=False)
                    picture = image(rgb);picture.save(OUT/(name+'_rgb.png'));pictures.append(picture)
                    alpha_picture = Image.fromarray(np.rint(np.clip(alpha_array,0,1)*255).astype(np.uint8)).convert('RGB')
                    alpha_picture.save(OUT/(name+'_alpha.png'));alpha_pictures.append(alpha_picture)
                    output[name] = dict(points=len(model._xyz),channel_max_abs_difference=channel_error,
                        rgb_sha256=sha(rgb_path),alpha_sha256=sha(alpha_path),
                        mask=metrics(alpha_array,mask),mask_eroded_5_derived_pixels=metrics(alpha_array,inner))
                    assert time.monotonic()-started<60
            for key,before in saved_fields.items():
                assert torch.equal(getattr(original,key).detach(),before), key
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
            assert py_rng == random.getstate()
            after_np = np.random.get_state()
            assert np_rng[0] == after_np[0] and np.array_equal(np_rng[1],after_np[1]) and np_rng[2:] == after_np[2:]
            # One same-canvas panel; no annotations alter the scientific images.
            for filename,panels in [('rgb_full_front_back.png',pictures),('alpha_full_front_back.png',alpha_pictures)]:
                sheet = Image.new('RGB',(sum(im.width for im in panels),panels[0].height))
                x = 0
                for im in panels:
                    sheet.paste(im,(x,0));x += im.width
                sheet.save(OUT/filename)
            hashes_unchanged(doc)
            save(OUT/'result.json',dict(updated=now(),state='diagnostic_completed',step=STEP,reference_frame=REFERENCE,
                groups=output,topology=doc['topology'],projection=projection,encoded_full_replay=parity,
                native_snapshot_float_parity=doc['native_snapshot_float_parity'],seconds=time.monotonic()-started,
                point_fields_unchanged=True,RNG_unchanged=True,optimizer_updates=0,no_model_publication=True,
                group_alphas_additive=False,physical_front_clothing_claim=False,
                interpretation='Front/back labels refer only to the unmodified fitted mesh face normals at ref26. Independent group renders overlap and are not a full compositing contribution decomposition.',
                quality_accepted=False))
            print('FOOTBALL_REF26_FACING_DIAGNOSTIC_COMPLETED',flush=True)
        finally:
            signal.alarm(0)


def audit():
    ast.parse(Path(__file__).read_text())
    assert not OUT.exists()
    doc = dict(updated=now(),state='static_cpu_audit_passed',helper_sha256=sha(__file__),
        step=STEP,reference=REFERENCE,GPU_budget_seconds=60,no_GPU_started=True,no_actual_final_plan_created=True,
        note='CPU plan requires actual final18000 snapshot and completed-run marker. Mesh may have32 boundary edges; no closure, topology repair or physical layer truth is assumed.')
    save(Path(__file__).with_suffix('.audit.json'),doc)
    print(json.dumps(doc),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit','plan','run'])
    parser.add_argument('--step',type=int,choices=[18000],default=18000)
    args = parser.parse_args()
    try:
        {'audit':audit,'plan':plan,'run':run}[args.mode]()
    except BaseException as error:
        if args.mode=='run' and (OUT/'launch.json').exists():
            save(OUT/'run-failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False,quality_accepted=False))
        raise
