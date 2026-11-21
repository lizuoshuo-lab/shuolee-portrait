"""Compare two delivered bound SH0 runs on their identical native training observations."""

import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, expected):
    actual = sha(path)
    require(actual == expected, f'Changed pinned bytes: {path}')
    return actual


def matched_inputs(baseline, candidate):
    registration = read(candidate.parent/'preregistration.json')
    require(Path(registration['baseline_run']).resolve() == baseline and
            registration.get('no_old_gaussian_state_or_optimizer_reuse') is True and
            registration.get('optimizer_schedule_seed_geometry_binding_unchanged') is True,
            'The A/B runs do not declare this same frozen baseline.')
    required = {'fit/fitted.npz', 'fit/official-fit-result.json', 'fit/selection.json',
                'fit/shared-body.json', 'data/training-inputs.json'}
    pins = registration['reused_inputs_sha256']
    require(set(pins) >= required, 'Comparison preregistration omitted fitted or source inputs.')
    for name, digest in pins.items():
        pin(baseline/name, digest)
        pin(candidate/name, digest)
    official = read(baseline/'fit/official-fit-result.json')
    require(official['fit_sha256'] == pins['fit/fitted.npz'] and
            official['training_inputs_sha256'] == pins['data/training-inputs.json'], 'Invalid fit receipt.')
    observations, = [Path(p) for p in official['source']['input_files'] if Path(p).name == 'observations.npz']
    pin(observations, official['source']['input_files'][str(observations)])
    rows = read(baseline/'data/training-inputs.json')['records']
    holds = read(baseline/'fit/selection.json')['retained_holdouts']
    require(len(rows) == len({r['frame'] for r in rows}) and not set(holds) & {r['frame'] for r in rows},
            'Repeated or protected training frame.')
    for row in rows:
        pin(Path(row['image']), row['source_sha256'])
        pin(Path(row['mask']), row['mask_sha256'])
    return registration, pins, rows, observations, holds


def check_source_equivalence(left, right, a, b):
    train = 'src/shuolee_portrait/training.py'
    require(set(a) == set(b) and train in a, 'Different renderer source contracts.')
    for name in a:
        pin(left/name, a[name]); pin(right/name, b[name])
        if name != train:
            require(a[name] == b[name], f'Replay renderer or binding source changed: {name}')
    def entries(project):
        tree = ast.parse((project/train).read_text(encoding='utf-8-sig'))
        names = {'camera', 'named_pose_layout', '_frozen_smpl_surface', 'render_pose', 'image'}
        found = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
        require(set(found) == names, 'Missing exact common replay entry.')
        return found
    require(entries(left) == entries(right), 'A/B camera, pose replay or image encoding code differs.')


def delivered(root, asset, rows, holdouts):
    from shuolee_portrait.bound_replay import _inputs

    terminal_path = root/'result.json'
    terminal = read(terminal_path)
    train = read(root/'gaussian-run/result.json')
    require(terminal.get('state') == train.get('state') == 'completed' and
            terminal.get('format_passed') is True and terminal['steps'] == train['steps'] == 10000 and
            terminal['gaussian_profile'] == train['gaussian_profile'] == 'smpl_sh0' and
            terminal['sh_degree'] == train['sh_degree'] == 0 and train['final_snapshot'] == 'step-010000',
            f'No delivered 10k SH0 result: {root}')
    snapshot = root/'gaussian-run/step-010000'
    replay_path = snapshot/'posed-replay/result.json'
    require(root/terminal['bound_replay'] == replay_path, 'Unexpected posed replay location.')
    replay_sha = pin(replay_path, terminal['bound_replay_sha256'])
    pin(root/terminal['ply'], terminal['sha256']); pin(root/terminal['scene'], terminal['scene_sha256'])
    replay = read(replay_path)
    require(replay['state'] == 'completed' and replay['optimizer_updates'] == 0 and
            replay['retained_holdouts'] == holdouts, 'The terminal pose replay is incomplete or unprotected.')
    bound_path = snapshot/'bound-replay.json'
    pin(bound_path, replay['bound_replay_sha256'])
    bound = read(bound_path)
    require(bound['schema'] == 'shuolee-smplx-bound-v1' and bound['bound_ply'] == 'bound-local.ply' and
            bound['optimizer_updates'] == 0 and bound['points'] == train['points'] == terminal['points'] and
            bound['sh_degree'] == 0, 'Invalid terminal local Gaussian receipt.')
    pin(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
    pin(root/'gaussian-run/training-cameras.json', replay['training_camera_sha256'])
    mesh_path = replay_path.parent/replay['mesh_file']
    pin(mesh_path, replay['mesh_sha256'])
    fit, mesh, facts = _inputs(root, snapshot, asset)
    require(all(bound.get(key) == facts[key] for key in ('inputs_sha256', 'asset_sha256', 'frames',
            'reference_pose_index', 'reference_frame_id')), 'Bound file no longer matches fit, source roster or asset.')
    require(bound['inputs_sha256']['training_rows'] == replay['training_inputs_sha256'] and
            facts['reference_frame_id'] == terminal['reference_frame_id'] == replay['reference_frame_id'],
            'Terminal reference or source rows differ.')
    with np.load(mesh_path, allow_pickle=False) as archive:
        poses = dict(archive)
    require(np.array_equal(poses['faces'], fit['faces']) and
            poses['frame_ids'].tolist() == [x['frame'] for x in rows] and
            np.array_equal(poses['source_frames'], [x['source_frame_index'] for x in rows]) and
            np.array_equal(poses['image_sha256'], [x['source_sha256'] for x in rows]) and
            np.array_equal(poses['mask_sha256'], [x['mask_sha256'] for x in rows]) and
            np.array_equal(poses['pose_indices'], [x['pose_index'] for x in facts['frames']]),
            'Saved posed mesh identities differ from the trained fit.')
    require(len(replay['records']) == len(rows), 'Missing source-frame renders.')
    for row, item, saved in zip(rows, facts['frames'], replay['records'], strict=True):
        require(all(saved[k] == v for k, v in item.items()) and
                all(saved['camera'][k] == row[k] for k in ('K', 'w2c', 'wh', 'frame')),
                'Saved render uses a different source camera or pose.')
        pin(replay_path.parent/saved['image'], saved['image_sha256'])
    return SimpleNamespace(snapshot=snapshot, fit=fit, mesh=mesh, facts=facts, bound=bound, poses=poses,
                           terminal=terminal, terminal_sha256=sha(terminal_path), replay=replay, replay_sha256=replay_sha)


def detector_rois(rgb, mask, keypoints):
    h, w = mask.shape
    face = keypoints[23:91]
    in_view = np.isfinite(face).all(1) & (face[:, 0]>=0) & (face[:, 0]<w) & (face[:, 1]>=0) & (face[:, 1]<h)
    eye = Image.new('1', (w, h), 0)
    draw = ImageDraw.Draw(eye)
    for sl in (slice(36,42), slice(42,48)):
        if in_view[sl].all():
            draw.polygon([tuple(map(float, p)) for p in face[sl]], fill=1)
    eye = np.asarray(eye, bool) & (mask > .99)
    dark = eye & (rgb.max(axis=2) < 64)
    head = None
    if in_view.sum() >= 24:
        low, high = face[in_view].min(0), face[in_view].max(0)
        extent = high-low
        if (extent >= 4).all():
            margin = np.maximum(16, extent*np.array([.48,.85]))
            head = native_box([*(low-margin), *(high+margin)], (w,h))
    return eye, dark, head


def native_box(xyxy, wh):
    a, b, c, d = xyxy
    w, h = wh
    return max(0,int(np.floor(a))), max(0,int(np.floor(b))), min(w,int(np.ceil(c))), min(h,int(np.ceil(d)))


def mask_metrics(alpha, mask, dark):
    fg, bg = mask > .99, mask < .01
    binary, covered = mask >= .5, alpha >= .5
    require(fg.any() and bg.any(), 'Both target foreground and background are required.')
    overlap = (binary & covered).sum()
    return dict(alpha_mae=float(np.abs(alpha-mask).mean()),
                mask_iou_at_half=float(overlap/(binary|covered).sum()),
                mask_coverage_at_half=float(overlap/binary.sum()),
                mean_foreground_alpha=float(alpha[fg].mean()),
                foreground_hole_fraction_at_half=float((~covered[fg]).mean()),
                mean_background_alpha=float(alpha[bg].mean()),
                exterior_spill_pixels_at_half=int((covered & bg).sum()),
                dark_eye_foreground_pixels=int(dark.sum()),
                dark_eye_mean_alpha=float(alpha[dark].mean()) if dark.any() else None,
                dark_eye_hole_fraction_at_half=float((~covered[dark]).mean()) if dark.any() else None)


def write_panel(out, name, rgb, mask, a, b, box):
    x0,y0,x1,y1 = box
    require(x1>x0 and y1>y0, 'Empty native panel crop.')
    w,h=x1-x0,y1-y0
    u8 = lambda v: np.rint(np.clip(v,0,1)*255).astype(np.uint8)
    panels = (rgb, u8(mask), u8(a[0]), u8(a[1]), u8(a[2]), u8(b[0]), u8(b[1]), u8(b[2]))
    page = Image.new('RGB',(w*4,h*2))
    for i, item in enumerate(panels):
        page.paste(Image.fromarray(item[y0:y1,x0:x1]).convert('RGB'), ((i%4)*w,(i//4)*h))
    path = out/name
    page.save(path)
    return dict(file=name, sha256=sha(path), native_crop_xyxy=list(box))


def run(args):
    baseline, candidate, a_project, b_project, asset, output = (p.resolve() for p in (
        args.baseline,args.candidate,args.baseline_project,args.candidate_project,args.asset,args.output))
    prereg, input_pins, rows, observation_path, holdouts = matched_inputs(baseline,candidate)
    pin(b_project/'src/shuolee_portrait/training.py',prereg['source_file_sha256'])
    if args.preflight:
        print(json.dumps(dict(inputs_match=True, frames=len(rows), protected_holdouts=holdouts,
                              observations=str(observation_path), fit_sha256=input_pins['fit/fitted.npz'])))
        return
    sys.path.insert(0,str(b_project/'src'))
    import torch
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    from gaussian_renderer import render

    a,b = [delivered(root,asset,rows,holdouts) for root in (baseline,candidate)]
    require(a.terminal['sha256']==prereg['baseline_model_sha256'] and
            a.facts['reference_frame_id']==b.facts['reference_frame_id'] and a.facts['frames']==b.facts['frames'],
            'A/B terminal baseline hash, shared reference or named pose roster changed.')
    check_source_equivalence(a_project,b_project,a.bound['source_files_sha256'],b.bound['source_files_sha256'])
    a_cfg,b_cfg = [read(root/'gaussian-run/config.json') for root in (baseline,candidate)]
    for key in ('optimization','seed','native_resolution','sh_degree','initial_points','shared_clothing_offset_lr'):
        require(a_cfg[key]==b_cfg[key],f'A/B training configuration changed: {key}')
    require(len(set(args.samples))==len(args.samples) and set(args.samples)<={r['frame'] for r in rows},
            'Panel samples must be unique named training frames.')
    with np.load(observation_path,allow_pickle=False) as obs:
        landmarks = dict(zip(obs['frame_ids'].tolist(),obs['keypoints'],strict=True))

    records = []
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            models = []
            for saved in (a,b):
                model, parity = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,
                                           saved.mesh,saved.facts,asset)
                require(model.optimizer is None and model.active_sh_degree==0 and len(model.binding)==saved.terminal['points'],
                        'Readback unexpectedly has an optimizer, wrong SH or mismatched point count.')
                mesh_error = float(np.abs(model.vertices_sequence.cpu().numpy()[saved.poses['pose_indices']]-
                                          saved.poses['vertices_world']).max())
                require(mesh_error<2e-6,'Reloaded pose differs from the actual terminal mesh archive.')
                saved.model_parity = dict(**parity, all_saved_pose_mesh_max_abs_m=mesh_error)
                models.append(model)
            output.mkdir(parents=True,exist_ok=False)
            pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            black, white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
            for i,row in enumerate(rows):
                frame = row['frame']; pose=a.facts['frames'][i]['pose_index']
                cam,_ = camera(row,i,pose_index=pose)
                with Image.open(row['image']) as src, Image.open(row['mask']) as m:
                    require(list(src.size)==list(m.size)==row['wh'],'Changed native raster dimensions.')
                    rgb, mask = np.array(src.convert('RGB')),np.array(m.convert('L'),dtype=np.float32)/255.
                eye,dark,head = detector_rois(rgb,mask,landmarks[frame])
                metrics, pixels = [], []
                for model,saved in zip(models,(a,b),strict=True):
                    rb=render_pose(render,cam,model,pipe,black,pose)['render']
                    rw=render_pose(render,cam,model,pipe,white,pose)['render']
                    alpha=render_pose(render,cam,model,pipe,black,pose,
                                      override_color=torch.ones_like(model.get_xyz))['render']
                    err=float((rw-rb-(1-alpha[:1])).abs().max())
                    require(torch.isfinite(rb).all() and torch.isfinite(rw).all() and torch.isfinite(alpha).all()
                            and float(alpha.min())>=-1e-5 and float(alpha.max())<=1.00001 and
                            float((alpha-alpha[:1]).abs().max())<1e-6 and err<5e-5,
                            f'Black/white and alpha disagree for {frame}: {err}')
                    alpha_cpu=alpha[0].cpu().numpy()
                    current_black=rb.permute(1,2,0).cpu().numpy()
                    with Image.open(saved.snapshot/'posed-replay'/saved.replay['records'][i]['image']) as im:
                        prior=np.asarray(im.convert('RGB'),dtype=np.int16)
                    current=np.rint(np.clip(current_black,0,1)*255).astype(np.int16)
                    archived_error=np.abs(current-prior)
                    metrics.append(dict(**mask_metrics(alpha_cpu,mask,dark),composition_max_abs=err,
                                        black_archive_mean_error_8bit=float(archived_error.mean()),
                                        black_archive_max_error_8bit=int(archived_error.max())))
                    if frame in args.samples:
                        pixels.append((current_black,rw.permute(1,2,0).cpu().numpy(),alpha_cpu))
                item=dict(frame=frame, pose_index=pose, source_frame_index=row['source_frame_index'],
                          source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
                          eye_roi_target_mask_pixels=int(eye.sum()), baseline=metrics[0], candidate=metrics[1])
                if frame in args.samples:
                    x0,y0,x1,y1=row['initial_box_xyxy']
                    full=native_box([x0-64,y0-64,x1+64,y1+64],row['wh'])
                    item['full_panel']=write_panel(output,frame+'-full.png',rgb,mask,*pixels,full)
                    head_method='source RTMW face bounds with fixed margin'
                    if head is None:
                        head=native_box([x0-32,y0-32,x1+32,y0+.33*(y1-y0)],row['wh'])
                        head_method='upper third of source person box; no valid detector face bounds'
                    item['head_panel']=dict(**write_panel(output,frame+'-head.png',rgb,mask,*pixels,head),
                                            crop_method=head_method)
                records.append(item)
            require(all(model.optimizer is None for model in models),'Unexpected optimizer creation during readback.')
    numerical = tuple(k for k in records[0]['baseline'] if any(item['baseline'][k] is not None for item in records))
    def means(side):
        return {k:float(np.mean([item[side][k] for item in records if item[side][k] is not None])) for k in numerical}
    aa,bb = means('baseline'),means('candidate')
    result=dict(schema='shuolee-bound-alpha-ab-v1', state='completed_post_delivery_readback',
                baseline_run=str(baseline),candidate_run=str(candidate),frames=len(records),reference=a.facts['reference_frame_id'],
                baseline_result_sha256=a.terminal_sha256,candidate_result_sha256=b.terminal_sha256,
                baseline_pose_replay_sha256=a.replay_sha256,candidate_pose_replay_sha256=b.replay_sha256,
                baseline_bound_ply_sha256=a.bound['bound_ply_sha256'],candidate_bound_ply_sha256=b.bound['bound_ply_sha256'],
                baseline_stock_replay_parity=a.model_parity,candidate_stock_replay_parity=b.model_parity,
                source_observation_sha256=sha(observation_path),same_inputs_sha256=input_pins,protected_holdouts=holdouts,
                baseline_target_composition=a_cfg['target_composition'],candidate_target_composition=b_cfg['target_composition'],
                candidate_preregistered_change=prereg['comparison_variable'],candidate_density_weight=prereg.get('density_weight'),
                dark_eye_evaluable_frames=sum(item['baseline']['dark_eye_foreground_pixels']>0 for item in records),
                baseline_frame_means=aa,candidate_frame_means=bb,candidate_minus_baseline={k:bb[k]-aa[k] for k in aa},
                pixel_scope='Native same training cameras/poses/masks; alpha estimated by stock all-ones color over black. '
                            'Dark eye RGB max<64 restricts an RTMW eye-polygon diagnostic only; it is not a training rule. '
                            'Mask overlap is a training observation comparison, not physical visibility or quality truth.',
                panel_layout='Top: original | target mask | A black | A white. Bottom: A alpha | B black | B white | B alpha. '
                             'Every panel uses the same native integer crop and no resampling.',
                optimizer_updates=0,quality_accepted=False,records=records)
    path=output/'result.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(receipt=str(path), sha256=sha(path), frames=len(records),
                          baseline=aa,candidate=bb),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline','candidate','baseline-project','candidate-project','asset','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--samples',nargs='+',default=['frame_000033','frame_000012','frame_000044'])
    parser.add_argument('--preflight',action='store_true',help='Read-only common-input check, before terminal delivery.')
    run(parser.parse_args())
