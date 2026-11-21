"""Render a frozen 008 SH0 model with a single local head rotation substituted."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from audit_008_head_ownership import (ASSET, COMMIT, EVAL, FIXED_PINS, HISTORICAL, LAB, PROJECT,
    RASTER_BINARY, RASTER_ENV, RUN, SNAP, SPECS, need, one, polygon, read, save, semantics, sha, verify)


OUT = LAB/'shuolee-smpl-head-joint-008'
CONTROL_IDS = ('frame_000004', 'frame_000047')
DONOR = CONTROL_IDS[1]
TEST = {
    'frame_000042': (328, (1240, 332, 1510, 635),
        ((1321, 397), (1365, 382), (1413, 412), (1415, 457),
         (1395, 518), (1355, 528), (1320, 504), (1307, 452)),
        ((1434, 478), (1447, 478), (1452, 527), (1441, 560), (1429, 565), (1435, 528))),
    'frame_000044': (344, (1240, 344, 1510, 647),
        ((1320, 410), (1368, 396), (1414, 426), (1418, 464),
         (1395, 531), (1351, 545), (1317, 516), (1307, 457)),
        ((1432, 496), (1446, 494), (1450, 542), (1438, 578), (1429, 579), (1434, 542))),
}
LOCAL = (12-1, 15-1)
AXIS = slice(3*LOCAL[1], 3*(LOCAL[1]+1))
POSE_FIELDS = ('global_orient', 'expression', 'jaw_pose', 'leye_pose', 'reye_pose',
    'left_hand_pose', 'right_hand_pose', 'body_pose_axis_angle', 'camera_translation')
WINNER_PINS = {
    'frame_000042': '86a136ff3cc35445c2f066afac296bf602f68f5b86954593432078a24b482bc2',
    'frame_000044': '6bb4088e23d877bc96b9869bc5f68f23d68430be2193f912c7a008a809ab65b7',
    'frame_000047': '6cfdb88e94df39aa9d780b9718aa2b2d6b0146bf35c4ede23495f7d8a938333d',
}
PROTECTED_METERS = .002
PROTECTED_PIXELS = 2.


def winner(frame):
    return RUN/'fit/frames'/frame/'winner-camera-space.npz'


def rotation_matrices(row):
    return Rotation.from_rotvec(row['body_pose_axis_angle'].reshape(21,3)).as_matrix()


def magnitude(matrix):
    return float(np.rad2deg(Rotation.from_matrix(matrix).magnitude()))


def transplant(row, donor_relative):
    """Keep the existing neck joint and put the donor orientation at its child head joint."""
    neck, head = LOCAL
    matrices = rotation_matrices(row)
    target = row['body_pose_axis_angle'].copy()
    target[:,AXIS] = Rotation.from_matrix(matrices[neck].T@donor_relative).as_rotvec()
    changed = dict(row, body_pose_axis_angle=target)
    after = rotation_matrices(changed)
    need(np.array_equal(np.delete(row['body_pose_axis_angle'], np.s_[AXIS], axis=1),
                        np.delete(target, np.s_[AXIS], axis=1)) and
         np.max(np.abs(after[neck]@after[head]-donor_relative)) < 1e-6,
         'Change escaped joint 15 or failed its torso-relative orientation constraint.')
    return changed


def mesh_forward(rows, shared, faces):
    import smplx
    import torch

    torch.set_num_threads(4)
    model = smplx.SMPLX(str(ASSET), batch_size=1, num_betas=10, num_expression_coeffs=10,
        use_pca=True, num_pca_comps=12, flat_hand_mean=False, use_face_contour=False,
        create_body_pose=False, create_transl=False, dtype=torch.float32).eval().requires_grad_(False)
    need(np.array_equal(model.faces, faces) and model.parents[12] == 9 and model.parents[15] == 12,
         'Unexpected stock SMPL-X head and neck hierarchy.')
    params = POSE_FIELDS[:7]
    vertices = []
    with torch.no_grad():
        for row in rows:
            v = model(**{k: torch.from_numpy(row[k]) for k in params}, betas=torch.from_numpy(shared),
                body_pose=torch.from_numpy(row['body_pose_axis_angle']), return_verts=True).vertices.numpy()
            vertices.append(v+row['camera_translation'].reshape(1,1,3))
    return vertices


def model_axis_angle(vertices):
    nose, left, right = vertices[0,[9120,616,6]]
    ear = (left+right)*.5
    axis = nose-ear
    span = right-left
    axis = axis-span*np.dot(axis,span)/np.dot(span,span)
    angle = np.arccos(np.clip(np.dot(axis,-ear)/np.linalg.norm(axis)/np.linalg.norm(ear),-1,1))
    return float(np.rad2deg(angle))


def project(vertices, row):
    homogeneous = vertices[0].astype(np.float64)@np.asarray(row['K']).T
    need(np.min(homogeneous[:,2]) > 0, 'Non-positive vertex depth.')
    return homogeneous[:,:2]/homogeneous[:,2:]


def input_bundle():
    pointer = (PROJECT/'.git').read_text(encoding='utf-8').strip()
    need(pointer.startswith('gitdir: '), 'Expected the pinned Windows worktree.')
    gitdir = subprocess.check_output(['wslpath','-u',pointer.removeprefix('gitdir: ')], text=True).strip()
    need(subprocess.check_output(['git','--git-dir',gitdir,'rev-parse','HEAD'], text=True).strip() == COMMIT,
         'Wrong stock worktree.')
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    pins = {p:v for p,v in FIXED_PINS.items() if not p.is_relative_to(EVAL) and p != HISTORICAL}
    pins.update({winner(frame): h for frame,h in WINNER_PINS.items()})
    verify(pins)
    manifest = read(SNAP/'bound-replay.json')
    fit, mesh, _ = _inputs(RUN, SNAP, ASSET, manifest)
    pins.update({PROJECT/name: value for name,value in manifest['source_files_sha256'].items()})
    pins.update({ASSET: manifest['asset_sha256'], RUN/'fit/fitted.npz': manifest['inputs_sha256']['fitted'],
        RUN/'data/training-inputs.json': manifest['inputs_sha256']['training_rows'],
        SNAP/'reference-mesh.npz': manifest['inputs_sha256']['reference_mesh'],
        SNAP/'world.ply': manifest['inputs_sha256']['world_ply'],
        SNAP/'native-ply-parity.json': manifest['inputs_sha256']['native_ply_parity']})
    ids, _ = inspect_bound_ply(SNAP/'bound-local.ply', len(fit['faces']))
    labels, anatomical = semantics(fit['faces'])
    need(len(ids) == manifest['points'] == 58253 and manifest['sh_degree'] == 0, 'Changed SH0 snapshot.')
    protected = ~np.isin(np.arange(fit['vertices_world'].shape[1]), fit['faces'][labels != 3].ravel())
    receipt = read(SNAP/'posed-replay/result.json')
    need(set(TEST).isdisjoint(receipt['retained_holdouts']) and
         set(TEST).isdisjoint(fit['frame_ids'].tolist()), 'Candidate touched Gaussian fitting or reserved evaluation.')
    train = read(RUN/'data/training-inputs.json')['records']
    full = read(RUN/'data/inputs.json')['records']
    controls, tests = [], []
    for frame in CONTROL_IDS:
        row, recorded = one(train, frame), one(receipt['records'], frame)
        pose = fit['frame_ids'].tolist().index(frame)
        need(recorded['pose_index'] == pose and recorded['source_sha256'] == row['source_sha256'],
             'Changed control pose or source.')
        source, image = Path(row['image']), SNAP/'posed-replay'/recorded['image']
        pins.update({source: row['source_sha256'], image: recorded['image_sha256']})
        _,_,_,crop,hair = SPECS[frame]
        controls.append(SimpleNamespace(row=row, source=source, crop=crop, hair=hair, profile=None, pose=pose,
                                        archived=image, expected_camera=recorded['camera']))
    selected = []
    for frame in (*TEST,DONOR):
        row = one(train if frame == DONOR else full, frame)
        with np.load(winner(frame), allow_pickle=False) as archive:
            raw = dict(archive)
        need(raw['frame_id'].item() == frame and np.array_equal(raw['faces'], fit['faces']) and
             np.array_equal(raw['betas'], fit['shared_betas']) and
             np.array_equal(raw['K_integer_centers'], np.asarray(row['K_integer_centers'])) and
             np.array_equal(row['w2c'], np.eye(4)) and row['wh'] == [2704,1520], 'Candidate stock input differs.')
        if frame == DONOR:
            slot = fit['frame_ids'].tolist().index(frame)
            need(np.array_equal(raw['body_pose_axis_angle'][0], fit['body_pose_axis_angle'][slot]),
                 'Donor differs from the admitted rear pose.')
            donor = rotation_matrices(raw)
            continue
        index,crop,hair,profile = TEST[frame]
        need(row['source_frame_index'] == index, 'Candidate decoded source identity differs.')
        source = Path(row['image'])
        pins[source] = row['source_sha256']
        selected.append(raw)
        tests.append(SimpleNamespace(row=row, source=source, crop=crop, hair=hair, profile=profile))
    for view in controls+tests:
        with Image.open(view.source) as im:
            need(list(im.size) == view.row['wh'] and polygon(view.crop,view.hair).sum() > 1000, 'Bad original crop.')
            if view.profile:
                need(not (polygon(view.crop,view.hair)&polygon(view.crop,view.profile)).any(), 'Overlapping regions.')
    neck,head = LOCAL
    donor_relative = donor[neck]@donor[head]
    alternatives = [transplant(row,donor_relative) for row in selected]
    pair_rows = selected+alternatives
    vertices = mesh_forward(pair_rows,fit['shared_betas'],fit['faces'])
    measurements = []
    for i,(view, row) in enumerate(zip(tests,selected,strict=True)):
        stock, alternate = vertices[i], vertices[i+len(tests)]
        error = float(np.abs(stock-row['vertices_camera']).max())
        need(error < 2e-6, 'Native stock candidate mesh cannot be replayed.')
        d = np.linalg.norm(alternate[0]-stock[0],axis=1)
        px = np.linalg.norm(project(alternate,view.row)-project(stock,view.row),axis=1)
        max_m,max_px = float(d[protected].max()),float(px[protected].max())
        need(max_m < PROTECTED_METERS and max_px < PROTECTED_PIXELS,
             f'Head-only control moved protected nonhead in {view.row["frame"]}: '
             f'{max_m:.7f} m, {max_px:.4f} px.')
        m = rotation_matrices(row)
        measurements.append(dict(frame=view.row['frame'], stock_mesh_max_abs_m=error,
            model_head_camera_deg=[model_axis_angle(stock),model_axis_angle(alternate)],
            head_to_spine3_delta_deg=magnitude((m[neck]@m[head]).T@donor_relative),
            protected_nonhead_vertices=int(protected.sum()), protected_nonhead_max_m=max_m,
            protected_nonhead_max_px=max_px, active_head_axis_angle=alternatives[i]['body_pose_axis_angle'][0,AXIS].tolist()))
    bundle = {key: np.concatenate([row[key] for row in pair_rows],axis=0) for key in POSE_FIELDS}
    bundle.update(frame_ids=np.array([row['frame_id'].item() for row in pair_rows]), faces=fit['faces'],
                  vertices_world=np.concatenate(vertices,axis=0), shared_betas=fit['shared_betas'],
                  w2c=np.tile(np.eye(4),(len(pair_rows),1,1)))
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    pins[Path(sys.modules['audit_008_head_ownership'].__file__).resolve()] = sha(Path(sys.modules['audit_008_head_ownership'].__file__).resolve())
    verify(pins)
    return SimpleNamespace(pins=pins, controls=controls, tests=tests, mesh=mesh, pose_bundle=bundle,
        measurements=measurements, labels=labels[ids], anatomical=anatomical, first=len(fit['frame_ids']))


def plan(state):
    return dict(schema='shuolee-008-head-joint15-only-v1', optimizer_updates=0, fit_calls=0,
        max_render_calls=24, quality_accepted=False, target_frames=list(TEST), controls=list(CONTROL_IDS),
        donor=DONOR, heldout_inputs_used=False, changed_joint=15,
        changed_parameter='Head local axis angle = log(R_neck_target.T @ R_neck_donor @ R_head_donor).',
        protection=dict(nonhead_max_m=PROTECTED_METERS, nonhead_max_px=PROTECTED_PIXELS),
        pins={str(p):v for p,v in state.pins.items()}, measures=state.measurements,
        anatomy=state.anatomical,
        views=[dict(frame=v.row['frame'], crop_xyxy=list(v.crop), head_region=[list(p) for p in v.hair],
                    visible_profile=[list(p) for p in v.profile] if v.profile else None)
               for v in state.controls+state.tests])


def original(view, outline=False):
    with Image.open(view.source) as im:
        source = im.convert('RGB').crop(view.crop)
    if outline:
        x0,y0,_,_ = view.crop
        pen = ImageDraw.Draw(source)
        for region,color in ((view.hair,'#00f9fa'),(view.profile,'#e9a738')):
            if region:
                pts = [(x-x0,y-y0) for x,y in region]
                pen.line([*pts,pts[0]],fill=color,width=2)
    return source


def prepare(state):
    need(not OUT.exists(), 'Preserve the existing preregistration.')
    OUT.mkdir()
    evidence = {}
    for view in state.controls+state.tests:
        p = OUT/(view.row['frame']+'-source-only.png')
        original(view,True).save(p)
        evidence[str(p)] = sha(p)
    save(OUT/'preregistration.json',dict(plan(state), source_pins=evidence))
    print(json.dumps(dict(state='prepared', gpu_calls=0, measurements=state.measurements)))


def render_view(view, pose, state, model, labels):
    import torch
    from gaussian_renderer import render
    from shuolee_portrait.training import camera, image, render_pose

    cam, receipt = camera(view.row,pose,pose_index=pose)
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    black = torch.zeros(3,device='cuda')
    x0,y0,x1,y1 = view.crop
    region = (...,slice(y0,y1),slice(x0,x1))

    def draw(override=None):
        tensor = render_pose(render,cam,model,pipe,black,pose,override_color=override)['render'][region]
        need(torch.isfinite(tensor).all(), 'Nonfinite diagnostic render.')
        return tensor

    stock = draw()
    rgb = image(stock)
    if hasattr(view,'archived'):
        need(receipt == view.expected_camera, 'Control camera changed.')
        with Image.open(view.archived) as archived:
            need(np.array_equal(rgb,archived.convert('RGB').crop(view.crop)), 'Unchanged control differs from archive.')
    ones = torch.ones_like(model.get_xyz)
    alpha = draw(ones)[0]
    face = draw((labels == 0)[:,None]*ones)[0]
    mixed = draw((labels == 1)[:,None]*ones)[0]
    source = np.asarray(original(view),dtype=np.float64)/255.
    color = stock.permute(1,2,0).cpu().numpy()
    a,f,m = (t.cpu().numpy() for t in (alpha,face,mixed))
    need(float((f+m-a).max()) < 2e-5,'Invalid composited alpha split.')
    measurements = {}
    for name,pts in (('source_head_interior',view.hair),('source_ear_beard_edge',view.profile)):
        if pts:
            roi = polygon(view.crop,pts)
            mass = float(a[roi].sum())
            measurements[name] = dict(pixels=int(roi.sum()), alpha_mass=mass, face_alpha_mass=float(f[roi].sum()),
                mixed_alpha_mass=float(m[roi].sum()), face_alpha_share=float(f[roi].sum())/max(mass,1e-10),
                source_rgb_mae_black_composite=float(np.abs(source[roi]-color[roi]).mean()))
    return rgb,image(face.expand(3,-1,-1)),measurements,receipt


def panel(path,view,records):
    w,h = original(view).size
    labels = ('Native source',*records)
    sheet = Image.new('RGB',(len(labels)*w,2*h+40),'#171717')
    pen = ImageDraw.Draw(sheet)
    sheet.paste(original(view),(0,20))
    sheet.paste(original(view,True),(0,h+40))
    for i,key in enumerate(labels):
        pen.text((i*w+4,4),key,fill='white')
        pen.text((i*w+4,h+24),'Source-only polygons' if i==0 else 'Face interior alpha',fill='white')
        if i:
            rgb,face,*_ = records[key]
            sheet.paste(rgb,(i*w,20))
            sheet.paste(face,(i*w,h+40))
    sheet.save(path)


def run(state):
    data = read(OUT/'preregistration.json')
    source_pins = data.pop('source_pins')
    need(data == plan(state),'Preregistered model, source or intervention changed.')
    verify(source_pins)
    need(not (OUT/'render').exists(),'Preserve the first completed render.')
    with (Path('/tmp')/'shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from shuolee_portrait import runtime as rt
        from shuolee_portrait.bound_replay import load_bound_snapshot
        from shuolee_portrait.cli import gpu_lock
        from shuolee_portrait.training import _frozen_smpl_surface

        with gpu_lock(),torch.no_grad(),TemporaryDirectory(prefix='.render-',dir=OUT) as scratch:
            need(torch.cuda.device_count()==1,'Expose exactly one GPU.')
            rt.configure(ASSET.parent.parent)
            import diff_gaussian_rasterization as rasterizer
            from diff_gaussian_rasterization import _C
            need(Path(rasterizer.__file__).resolve()==(RASTER_ENV/'__init__.py').resolve() and
                 Path(_C.__file__).resolve()==RASTER_BINARY.resolve(),'Unexpected compiled rasterizer.')
            model,_ = load_bound_snapshot(RUN,SNAP,asset_path=ASSET)
            base,linear,clothing,_,_ = _frozen_smpl_surface(state.pose_bundle,ASSET)
            delta = state.mesh['static_offset']
            need(not np.any(delta[~clothing]),'Saved offset crosses the anatomical boundary.')
            offset = torch.from_numpy(delta).cuda()[...,None]
            poses = torch.stack([v+(a@offset).squeeze(-1) for v,a in zip(base.cuda(),linear.cuda(),strict=True)])
            model.vertices_sequence = torch.cat((model.vertices_sequence,poses))
            model.num_timesteps = len(model.vertices_sequence)
            labels = torch.from_numpy(state.labels).cuda()
            target = Path(scratch)/'render'
            target.mkdir()
            result = []
            n = len(state.tests)
            for i,view in enumerate(state.controls+state.tests):
                steps = {'Archived unchanged':view.pose} if i < len(state.controls) else {
                    'Stock fit':state.first+i-len(state.controls),'Local head only':state.first+i-len(state.controls)+n}
                pictures = {key:render_view(view,pose,state,model,labels) for key,pose in steps.items()}
                image = target/(view.row['frame']+'-head-joint.png')
                panel(image,view,pictures)
                result.append(dict(frame=view.row['frame'], image=image.name, image_sha256=sha(image),
                    measurements={k:{'regions':v[2],'camera':v[3]} for k,v in pictures.items()}))
            need(model.optimizer is None,'Read-only replay created an optimizer.')
            verify(state.pins)
            verify(source_pins)
            save(target/'result.json',dict(state='complete_pending_visual_review', render_calls=24,
                optimizer_updates=0, fit_calls=0, model_changed=False, heldout_inputs_used=False,
                quality_accepted=False, geometry=state.measurements, records=result,
                preregistration_sha256=sha(OUT/'preregistration.json')))
            target.rename(OUT/'render')
    print(OUT/'render/result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','prepare','render'))
    mode = parser.parse_args().mode
    if mode != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    state = input_bundle()
    if mode == 'render':
        run(state)
    elif mode == 'prepare':
        prepare(state)
    else:
        print(json.dumps(dict(state='preflight_passed', gpu_calls=0, measurements=state.measurements)))


if __name__=='__main__':
    main()
