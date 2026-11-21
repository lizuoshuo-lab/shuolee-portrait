"""Attribute the 006 face trial by replaying strict head/body Gaussian swaps in memory."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_face_photometric_006_ownership as ownership
import review_face_photometric_006 as face
import review_face_photometric_006_holdouts as held
from review_bound_alpha_ab import check_source_equivalence, delivered, pin, read, require, sha


TRIAL = face.launch
ROOT = face.LAB/'shuolee-smpl-face-photometric-006-head-hybrid'
PROJECTS = dict(baseline=face.original.PROJECT, candidate=TRIAL.PROJECT)
RUNS = dict(baseline=TRIAL.BASE, candidate=TRIAL.ROOT/'run')
COMMITS = dict(baseline=TRIAL.BASE_COMMIT, candidate=TRIAL.COMMIT)
SIDE_OTHER = dict(baseline='candidate', candidate='baseline')
FACE_REVIEW_SHA = 'a15dfa5d45c12d2150f5f2fda158216b2ce25cec11c51f2db64a6f4a40a652cf'
HELD_POSE = held.spatial.cold.prior.POSE_ARCHIVE/'evaluation-fitted.npz'
BASE_ALPHA = face.LAB/'shuolee-smpl-unified-006-spatial-alpha/new-replay/native-alpha.npz'


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def local_inputs(side, held_sha):
    project, run = PROJECTS[side], RUNS[side]
    TRIAL.rebound(TRIAL.base.common.git_state, PROJECT=project)(COMMITS[side])
    sys.path.insert(0, str(project/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    from shuolee_portrait.training import _frozen_smpl_surface

    require(rt.PROJECT == project, 'Stock replay source loaded from the wrong checkout.')
    torch.set_num_threads(4)
    rows = read(run/'data/training-inputs.json')['records']
    retained = read(run/'fit/selection.json')['retained_holdouts']
    saved = delivered(run, face.ASSET, rows, retained)
    _, _, facts = _inputs(run, saved.snapshot, face.ASSET, saved.bound)
    require(facts == saved.facts and saved.terminal['fit_reused'] is True and saved.terminal['fit_calls'] == 0
            and len(rows) == 24 and set(held.FRAMES) <= set(retained), 'Wrong fitted lineage or heldout roster.')
    bound = saved.snapshot/'bound-local.ply'
    binding, _ = inspect_bound_ply(bound, len(saved.fit['faces']))
    pin(HELD_POSE, held_sha)
    with np.load(HELD_POSE, allow_pickle=False) as archive:
        poses = dict(archive)
    faces = saved.fit['faces']
    require(np.array_equal(poses['faces'], faces) and np.array_equal(poses['shared_betas'], saved.fit['shared_betas']),
            'Frozen holdout mesh topology or shared body changed.')
    base, linear, clothing, protected, error = _frozen_smpl_surface(poses, face.ASSET)
    require(base.device.type == linear.device.type == 'cpu', 'Preflight LBS unexpectedly left CPU.')
    offset = saved.mesh['static_offset']
    require(offset.dtype == np.float32 and offset.shape == (len(protected), 3) and np.isfinite(offset).all()
            and not np.any(offset[~clothing]) and not np.any(offset[protected]), 'Saved protected offsets are nonzero.')
    strict = protected[faces].all(axis=1)
    mixed = protected[faces].any(axis=1) & ~strict
    require(strict.any() and mixed.any() and not np.any(clothing[faces[strict]]), 'Invalid protected triangle partition.')
    with np.load(saved.snapshot/'posed-replay'/saved.replay['mesh_file'], allow_pickle=False) as archive:
        vertices = archive['vertices_world']
    require(vertices.shape == (len(rows), len(protected), 3) and np.isfinite(vertices).all(), 'Invalid archived named meshes.')
    head_vertices = np.unique(faces[strict])
    receipt = dict(commit=COMMITS[side], terminal_sha256=saved.terminal_sha256,
        snapshot=str(saved.snapshot), bound_manifest_sha256=sha(saved.snapshot/'bound-replay.json'),
        bound_ply_sha256=saved.bound['bound_ply_sha256'], binding_sha256=array_sha(binding),
        named_replay_sha256=saved.replay_sha256, source_files=saved.bound['source_files_sha256'],
        fitted_sha256=facts['inputs_sha256']['fitted'], training_rows_sha256=facts['inputs_sha256']['training_rows'],
        shape_sha256=array_sha(saved.fit['shared_betas']), topology_sha256=array_sha(faces),
        protected_group_sha256=array_sha(protected), strict_group_sha256=array_sha(strict),
        protected_reference_sha256=array_sha(saved.mesh['vertices_world'][protected]),
        protected_named_meshes_sha256=array_sha(vertices[:, protected]),
        protected_eval_base_sha256=array_sha(base.numpy()[:, protected]),
        protected_eval_linear_sha256=array_sha(linear.numpy()[:, protected]),
        protected_offset_sha256=array_sha(offset[protected]), protected_offset_max_abs_m=0.,
        static_offset_sha256=array_sha(offset), frozen_holdout_cpu_replay_max_abs_m=error,
        protected_vertices=int(protected.sum()), strict_head_vertices=len(head_vertices),
        strict_head_triangles=int(strict.sum()), mixed_triangles=int(mixed.sum()),
        points=len(binding), strict_head_points=int(strict[binding].sum()), mixed_points=int(mixed[binding].sum()))
    return receipt, SimpleNamespace(rt=rt, saved=saved, base=base, linear=linear, protected=protected,
                                    strict=strict, mixed=mixed, binding=binding, offset=offset, rows=rows)


def preflight():
    spec, rois = face.preflight()
    require(spec['candidate'] is not None, 'Registered trial is unavailable.')
    path = face.OUT/'result.json'
    pin(path, FACE_REVIEW_SHA)
    previous = read(path)
    require(previous['state'] == 'face_endpoint_review_complete' and previous['optimizer_updates'] == previous['refits'] == 0,
            'Registered face comparison changed.')
    endpoint_plan = face.OUT/'plan.json'
    pin(endpoint_plan, previous['plan_sha256'])
    endpoint = read(endpoint_plan)
    require(all(endpoint.get(key) == value for key, value in spec.items()), 'Frozen endpoint camera or ROI specification changed.')
    rois_path = face.OUT/'rois.npz'
    pin(rois_path, endpoint['roi_archive_sha256'])
    with np.load(rois_path, allow_pickle=False) as cached:
        require(set(rois) == set(cached.files) and all(np.array_equal(v, cached[k]) for k,v in rois.items()),
                'Rebuilt ROIs differ from their frozen exact bitmaps.')
    held_receipt, _ = held.baseline()
    pin(ownership.REVIEW, ownership.REVIEW_SHA)
    held_review = read(ownership.REVIEW)
    require(all(held_review.get(k) == held_receipt[k] for k in ('baseline_native_alpha_sha256', 'evaluation_fit_sha256', 'frames', 'limits'))
            and held_review['candidate_terminal_sha256'] == spec['candidate']['terminal_sha256'], 'Protected replay lineage changed.')
    arms = {}
    for side in RUNS:
        result = subprocess.check_output([sys.executable, '-I', '-B', __file__, '_inspect', side,
            held_receipt['evaluation_fit_sha256']], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True)
        arms[side] = json.loads(result.splitlines()[-1])
        require(arms[side]['terminal_sha256'] == spec[side]['terminal_sha256'], 'Inspected arm is not the registered endpoint.')
    left, right = (arms[side] for side in RUNS)
    exact = ('fitted_sha256', 'training_rows_sha256', 'shape_sha256', 'topology_sha256', 'protected_group_sha256',
             'strict_group_sha256', 'protected_reference_sha256', 'protected_named_meshes_sha256',
             'protected_eval_base_sha256', 'protected_eval_linear_sha256', 'protected_offset_sha256')
    require(all(left[key] == right[key] for key in exact), 'Head attachment geometry or exact frozen fit differs.')
    require(right['bound_manifest_sha256'] == held_review['candidate_bound_sha256'], 'Candidate bound manifest changed.')
    check_source_equivalence(*(PROJECTS[side] for side in RUNS), left['source_files'], right['source_files'])
    return dict(schema='shuolee-006-strict-head-hybrid-v1', code_sha256=sha(Path(__file__)),
        arms=arms, face_receipt_sha256=FACE_REVIEW_SHA, face_plan_sha256=previous['plan_sha256'],
        face_roi_sha256=endpoint['roi_archive_sha256'], heldout_receipt=held_receipt,
        heldout_candidate_receipt_sha256=ownership.REVIEW_SHA, ownership='All three vertices in stock protected group; mixed stays body.',
        same_geometry_pins=list(exact), states=['stock', 'reciprocal_strict_head_hybrid'],
        native_pose=spec['frame'], heldouts=list(held.FRAMES), scale=1., refits=0, optimizer_updates=0,
        saves_hybrid_model=False, quality_accepted=False)


def replacement_rows(original_head, donor_head):
    """Keep both within-owner orders and the original interleaving where possible."""
    receiver_slots = np.flatnonzero(original_head)
    donor_rows = np.flatnonzero(donor_head)
    count = min(len(receiver_slots), len(donor_rows))
    require(count > 0, 'Cannot replace an empty ownership group.')
    rows = np.arange(len(original_head), dtype=np.int64)
    rows[receiver_slots[:count]] = len(original_head)+donor_rows[:count]
    keep = ~original_head.copy()
    keep[receiver_slots[:count]] = True
    pivot = receiver_slots[-1]+1
    return np.concatenate((rows[:pivot][keep[:pivot]], len(original_head)+donor_rows[count:], rows[pivot:][keep[pivot:]]))


def swap_head(receiver, donor, strict):
    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from shuolee_portrait.bound_replay import RAW_PARAMETERS

    original_head = strict[receiver.binding.cpu().numpy()]
    donor_head = strict[donor.binding.cpu().numpy()]
    self_rows = replacement_rows(original_head, original_head)
    require(np.array_equal(np.where(self_rows < len(original_head), self_rows, self_rows-len(original_head)),
                           np.arange(len(original_head))), 'Self replacement changes the exact stock raster order.')
    rows = replacement_rows(original_head, donor_head)
    expected_body = np.flatnonzero(~original_head)
    require(np.array_equal(rows[rows < len(original_head)], expected_body)
            and np.array_equal(rows[rows >= len(original_head)]-len(original_head), np.flatnonzero(donor_head)),
            'Replacement dropped or moved an owned Gaussian row.')
    gather = torch.as_tensor(rows, device=receiver.binding.device)
    hybrid = SMPLXGaussianModel(0)
    hybrid.vertices_sequence, hybrid.faces, hybrid.num_timesteps = receiver.vertices_sequence, receiver.faces, receiver.num_timesteps
    hybrid.binding = torch.cat((receiver.binding, donor.binding))[gather]
    hybrid.binding_counter = torch.bincount(hybrid.binding.long(), minlength=len(hybrid.faces)).to(torch.int32)
    require(bool((hybrid.binding_counter > 0).all()), 'Replacement lost a topology face.')
    for name in RAW_PARAMETERS:
        setattr(hybrid, name, torch.cat((getattr(receiver,name).detach(), getattr(donor,name).detach()))[gather])
    require(hybrid.optimizer is None and hybrid.active_sh_degree == 0, 'Unexpected hybrid training state.')
    return hybrid, dict(receiver_body_rows=len(expected_body), donor_head_rows=int(donor_head.sum()),
                        points=len(rows), gather_sha256=array_sha(rows), stock_self_row_order_exact=True)


def panel(output, side, view, arrays, crop, names):
    x0,y0,x1,y1 = crop
    width,height = x1-x0, y1-y0
    sheet = Image.new('RGB', (width*len(arrays), height+20), '#181818')
    draw = ImageDraw.Draw(sheet)
    for i, (array, name) in enumerate(zip(arrays,names,strict=True)):
        tile = face.original.uint8(array[y0:y1,x0:x1])
        sheet.paste(Image.fromarray(tile).convert('RGB'), (i*width,20))
        draw.text((i*width+4,4), name, fill='white')
    path = output/f'{side}-{view}.png'
    sheet.save(path)
    return dict(file=path.name, sha256=sha(path), crop_xyxy=list(crop))


def capture(output, side, lease_fd):
    require(os.path.samestat(os.fstat(lease_fd), face.GPU_LEASE.stat()), 'Global GPU lease was not inherited.')
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(output/'preregistration.json')
    pin(Path(__file__), plan['code_sha256'])
    receipt, state = local_inputs(side, plan['heldout_receipt']['evaluation_fit_sha256'])
    require(receipt == plan['arms'][side], 'Receiver stock inputs changed after preflight.')
    donor_receipt = plan['arms'][SIDE_OTHER[side]]
    donor_path = Path(donor_receipt['snapshot'])/'bound-local.ply'
    pin(donor_path, donor_receipt['bound_ply_sha256'])
    pin(face.OUT/'plan.json', plan['face_plan_sha256'])
    spec = read(face.OUT/'plan.json')
    pin(face.OUT/'result.json', plan['face_receipt_sha256'])
    expected_face = read(face.OUT/'result.json')['reports'][side]['views']
    pin(face.OUT/'rois.npz', plan['face_roi_sha256'])
    with np.load(face.OUT/'rois.npz', allow_pickle=False) as archive:
        rois = dict(archive)
    held_receipt, held_views = held.baseline()
    require(held_receipt == plan['heldout_receipt'], 'Protected cameras, masks or gates changed.')
    pin(ownership.REVIEW, plan['heldout_candidate_receipt_sha256'])
    held_previous = read(ownership.REVIEW)
    alpha_path = BASE_ALPHA if side == 'baseline' else ownership.REVIEW.with_name('candidate-native-alpha.npz')
    alpha_key = 'baseline_native_alpha_sha256' if side == 'baseline' else 'candidate_alpha_sha256'
    pin(alpha_path, held_previous[alpha_key])
    with np.load(alpha_path, allow_pickle=False) as archive:
        held_alpha = dict(archive)

    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    with gpu_lock(), torch.no_grad():
        require(torch.cuda.device_count() == 1, 'Use one CUDA device.')
        state.rt.configure(face.ASSET.parents[1])
        from gaussian_renderer import render
        receiver, _ = load_bound_snapshot(RUNS[side], state.saved.snapshot, asset_path=face.ASSET)
        require(array_sha(receiver.binding.long().cpu().numpy()) == receipt['binding_sha256'], 'Stock receiver changed local binding.')
        donor = SMPLXGaussianModel(0)
        donor.load_ply(str(donor_path))
        require(array_sha(donor.binding.long().cpu().numpy()) == donor_receipt['binding_sha256'], 'Stock donor changed local binding.')
        offset = torch.from_numpy(state.offset).cuda()[...,None]
        posed = torch.stack([v+(a@offset).squeeze(-1) for v,a in zip(state.base.cuda(),state.linear.cuda(),strict=True)])
        require(torch.equal(posed[:, state.protected].cpu(), state.base[:, state.protected]), 'Protected heldout mesh moved.')
        receiver.vertices_sequence = torch.cat((receiver.vertices_sequence,posed))
        receiver.num_timesteps = len(receiver.vertices_sequence)
        hybrid, assembly = swap_head(receiver, donor, state.strict)
        black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

        def pixels(model, row, cam, pose):
            model.select_mesh_by_timestep(pose)
            depth = model.get_xyz.new_tensor(row['w2c'][2])
            require(float((model.get_xyz@depth[:3]+depth[3]).min()) > .2, 'A Gaussian center intersects the renderer cutoff.')
            return face.scale_pixels(render, render_pose, cam, model, pipe, black, white, pose, 1.)

        face_results, held_results, panels = {}, {}, {}
        native = spec['source_row']
        pin(Path(native['image']), native['source_sha256'])
        with Image.open(native['image']) as image:
            source = np.asarray(image.convert('RGB'),dtype=np.float32)/255
        for view, row, crop in (('native',native,spec['native_crop']), ('dolly',spec['near_row'],spec['dolly_crop'])):
            cam, actual = camera(row,0,pose_index=0)
            require(actual == expected_face[view]['camera'], 'Registered face camera changed.')
            stock, stock_error = pixels(receiver,row,cam,0)
            altered, hybrid_error = pixels(hybrid,row,cam,0)
            expected = expected_face[view]
            result = {}
            for name, raster, error in (('stock',stock,stock_error), ('hybrid',altered,hybrid_error)):
                report = dict(camera=actual, alpha=face.face_alpha(raster[2],rois[view]), composition_error=error)
                if view == 'native':
                    report['appearance'] = face.appearance(raster[0],source,rois[view])
                    report['eyes'] = {k:face.appearance(raster[0],source,m) for k,m in rois.items() if k.startswith('eye_')}
                result[name] = report
            require(abs(result['stock']['alpha']['fraction_below_095']-expected['alpha']['fraction_below_095']) <= 1/rois[view].sum(),
                    'Stock face threshold does not reproduce registered control.')
            if view == 'native':
                require(abs(result['stock']['appearance']['black_rgb_mae']-expected['appearance']['black_rgb_mae']) <= 1e-6,
                        'Stock native face RGB does not reproduce registered control.')
            face_results[view] = result
            panels[view] = panel(output,side,view,[stock[0],altered[0],stock[2],altered[2]],crop,
                                 ['stock RGB','hybrid RGB','stock Alpha','hybrid Alpha'])

        for view, expected in zip(held_views,held_previous['records'],strict=True):
            frame, camera_receipt = view.spec['frame'], view.spec['camera']
            camera_index = list(held.spatial.cold.prior.HELD).index(frame)
            pose = len(state.rows)+camera_index
            row = dict(frame=frame,image=view.spec['image'],wh=camera_receipt['wh'],K=camera_receipt['K'],w2c=camera_receipt['w2c'])
            cam, actual = camera(row,camera_index,pose_index=pose)
            require(actual == camera_receipt and expected['frame'] == frame, 'Registered holdout camera changed.')
            stock, stock_error = pixels(receiver,row,cam,pose)
            cached = held_alpha[frame]
            require(float(np.abs(stock[2]-cached).max()) <= 1e-6 and np.array_equal(stock[2]>=held.ALPHA,cached>=held.ALPHA),
                    'Stock protected Alpha does not reproduce the cached native replay.')
            altered, hybrid_error = pixels(hybrid,row,cam,pose)
            limits = held_receipt['limits']
            stock_report = held.compare(view,stock[0],stock[2],limits)
            require(abs(stock_report['foreground_rgb_mae']['candidate']-expected['foreground_rgb_mae'][side]) <= 1e-6,
                    'Stock holdout RGB does not reproduce registered control.')
            held_results[frame] = dict(stock=stock_report,hybrid=held.compare(view,altered[0],altered[2],limits),
                                       composition_error=dict(stock=stock_error,hybrid=hybrid_error))
            visible = (~view.exterior)|(stock[2]>=held.ALPHA)|(altered[2]>=held.ALPHA)
            y,x = np.where(visible)
            h,w = visible.shape
            crop = [max(0,int(x.min())-24),max(0,int(y.min())-24),min(w,int(x.max())+25),min(h,int(y.max())+25)]
            panels[frame] = panel(output,side,frame,[view.source,stock[0],altered[0],stock[2],altered[2]],crop,
                                  ['source','stock RGB','hybrid RGB','stock Alpha','hybrid Alpha'])
        require(receiver.optimizer is donor.optimizer is hybrid.optimizer is None, 'Replay constructed a training optimizer.')
    face.save(output/f'{side}.json',dict(side=side,assembly=assembly,face=face_results,heldouts=held_results,panels=panels))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '_inspect':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print(json.dumps(local_inputs(sys.argv[2],sys.argv[3])[0]))
        return
    if len(sys.argv) == 5 and sys.argv[1] == '_capture':
        capture(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('--prereg-sha256')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        require(not ROOT.exists(),'Hybrid diagnostic root already exists.')
        plan = preflight()
        ROOT.mkdir(exist_ok=False)
        face.save(ROOT/'preregistration.json',plan)
        print(json.dumps(dict(state='cpu_preflight_passed',plan=str(ROOT/'preregistration.json'),
                              sha256=sha(ROOT/'preregistration.json'),gpu_calls=0,arms=plan['arms'])))
        return
    prereg = ROOT/'preregistration.json'
    require(args.prereg_sha256 is not None,'Pass the separately recorded CPU preregistration SHA.')
    pin(prereg,args.prereg_sha256)
    plan = read(prereg)
    require(plan == preflight(),'CPU preflight differs from the locked scope.')
    output = ROOT/'replay'
    with face.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        face.save(output/'preregistration.json',plan)
        for side in RUNS:
            subprocess.run([sys.executable,'-I','-B',__file__,'_capture',str(output),side,str(lease.fileno())],
                           pass_fds=(lease.fileno(),),check=True,timeout=360)
    reports = {side:read(output/f'{side}.json') for side in RUNS}
    base = {view:pair['stock'] for view,pair in reports['baseline']['face'].items()}
    gates = {side:face.gates(base,{view:pair['hybrid'] for view,pair in reports[side]['face'].items()},read(face.OUT/'plan.json'))
             for side in RUNS}
    target = output/'result.json'
    face.save(target,dict(state='hybrid_replay_complete',preregistration_sha256=args.prereg_sha256,
        reports=reports,registered_face_gates=gates,manual_eye_gray_review='pending',
        dolly_rgb_truth_available=False,quality_accepted=False,optimizer_updates=0,refits=0,saved_hybrid_model=False))
    print(json.dumps(dict(result=str(target),sha256=sha(target))))


if __name__ == '__main__':
    main()
