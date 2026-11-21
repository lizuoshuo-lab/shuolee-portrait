"""Offline CPU reconstruction of selected signed R22 rasterizer pixels."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-splat-distance-20261111')
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round22'
REVIEW_SHA = 'c0f1b8c35c5c645deb64db1700486aedc20e7aa56d20de2abc079294242ae924'
SOURCE = PROJECT/'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer'
VIEW = 'close_yaw_p00_pitch_'
F = np.float32
LUMA = np.asarray([.2126, .7152, .0722])

# Native crop-local integer centers picked on the saved black composites before
# attribution. The controls are nearby pixels on the same row and facial area.
PICKS = {
    '016': {
        'p30': ('nose blue-gray vertical streak', [((166, 389), (182, 389)), ((165, 397), (181, 397)), ((163, 405), (179, 405))]),
        'm30': ('nose pale vertical streak', [((151, 166), (159, 166)), ((151, 175), (159, 175)), ((152, 183), (160, 183))]),
    },
    '006': {
        'p30': ('dark diagonal extending from viewer-left eye toward nose', [((145, 356), (154, 356)), ((141, 363), (151, 363)), ((137, 368), (148, 368))]),
        'm30': ('dark downward extension under viewer-right glasses rim', [((176, 165), (169, 165)), ((181, 174), (173, 174)), ((186, 181), (177, 181))]),
    },
}
# Conservatively traced on the signed 006 native source images. Each polygon
# samples visible dark rim material; lens interior and reflections are excluded.
RIMS = {
    'frame_000032': [[(1322, 482), (1335, 477), (1350, 475), (1362, 475), (1367, 480),
                      (1350, 479), (1336, 480), (1322, 484)],
                     [(1369, 482), (1383, 482), (1383, 485), (1370, 486)],
                     [(1382, 487), (1385, 485), (1386, 501), (1383, 504)]],
    'frame_000041': [[(1344, 431), (1364, 426), (1376, 423), (1385, 424), (1388, 427),
                      (1376, 427), (1364, 430), (1345, 434)],
                     [(1400, 436), (1408, 436), (1408, 440), (1400, 440)],
                     [(1398, 441), (1401, 442), (1399, 453), (1396, 455)]],
    'frame_000049': [[(1390, 402), (1402, 399), (1410, 398), (1412, 401),
                      (1402, 403), (1391, 406)],
                     [(1410, 404), (1413, 403), (1414, 421), (1411, 425), (1409, 420)]],
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, digest):
    path = Path(path)
    need(sha(path) == digest, f'SHA256 mismatch: {path}')
    return path


def checked(item):
    return pin(item['file'], item['sha256'])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ply(path):
    return PlyData.read(path, mmap=False)['vertex'].data


def fields(data, names):
    return np.column_stack([data[name] for name in names])


def numbered(data, prefix, count):
    return fields(data, [f'{prefix}_{i}' for i in range(count)])


def geometry(data):
    xyz = fields(data, 'xyz')
    # Match the stock PLY loader's exp/sigmoid/normalized wxyz getters.
    scale = torch.from_numpy(numbered(data, 'scale', 3)).exp().numpy()
    quat = torch.nn.functional.normalize(torch.from_numpy(numbered(data, 'rot', 4)), dim=1).numpy()
    opacity = torch.from_numpy(data['opacity'].copy()).sigmoid().numpy()
    rgb = np.maximum(F(.28209479177387814)*numbered(data, 'f_dc', 3)+F(.5), F(0.))
    return xyz, scale, quat, opacity, rgb


def quaternion_matrix(q):
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def preprocess(arrays, row):
    """Float32 stock forward.cu parameters, using the exact Python camera constructor."""
    from shuolee_portrait.training import camera

    xyz, scale, quat, opacity, rgb = arrays
    cam, actual = camera(row, 0)
    need(all(actual[key] == row[key] for key in ('K', 'wh', 'w2c')), 'Camera changed during replay.')
    view, proj = cam.world_view_transform.numpy(), cam.full_proj_transform.numpy()
    wh = np.asarray(row['wh'], dtype=np.int32)
    xyz1 = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float32)))
    t, h = xyz1@view[:, :3], xyz1@proj
    ndc = h[:, :2]*(1/(h[:, 3:]+F(1e-7)))
    xy = (((ndc.astype(np.float64)+1.)*wh-1.)*.5).astype(np.float32)
    tan = np.asarray([math.tan(cam.FoVx*.5), math.tan(cam.FoVy*.5)], dtype=np.float32)
    focal = wh.astype(np.float32)/(F(2.)*tan)
    valid = t[:, 2] > F(.2)
    depth = np.where(valid, t[:, 2], F(1.))
    limit = F(1.3)*tan
    clamped = np.clip(t[:, :2]/depth[:, None], -limit, limit)*depth[:, None]
    J = np.zeros((len(xyz), 2, 3), dtype=np.float32)
    J[:, 0, 0], J[:, 1, 1] = focal[:, None]/depth
    J[:, :, 2] = -focal*clamped/(depth[:, None]**2)
    axes = quaternion_matrix(quat)*scale[:, None, :]
    covariance = axes@axes.transpose(0, 2, 1)
    projection = J@view[:3, :3].T
    projected = projection@covariance@projection.transpose(0, 2, 1)
    a, b, c = projected[:, 0, 0]+F(.3), projected[:, 0, 1], projected[:, 1, 1]+F(.3)
    det = a*c-b*b
    mid = F(.5)*(a+c)
    valid &= np.isfinite(det) & (det != 0.)
    radius = np.ceil(F(3.)*np.sqrt(mid+np.sqrt(np.maximum(F(.1), mid*mid-det)))).astype(np.int32)
    grid = (wh+15)//16
    low = np.clip(np.trunc((xy-radius[:, None].astype(np.float32))/F(16.)).astype(np.int32), 0, grid)
    high = np.clip(np.trunc((xy+radius[:, None].astype(np.float32)+F(15.))/F(16.)).astype(np.int32), 0, grid)
    valid &= (high > low).all(1)
    conic = np.column_stack((c, -b, a))*(1/np.where(valid, det, F(1.)))[:, None]
    return dict(xy=xy, depth=t[:, 2], conic=conic, opacity=opacity, rgb=rgb, low=low, high=high,
                valid=valid, radius=radius, projected=projected, tiles={})


def pixel(pre, xy):
    """Mirror 16×16 binning, stable positive depth sort and early alpha exit."""
    tile = tuple(value//16 for value in xy)
    if tile not in pre['tiles']:
        possible = np.flatnonzero(pre['valid'] & ((pre['low'] <= tile) & (pre['high'] > tile)).all(1))
        pre['tiles'][tile] = possible[np.argsort(pre['depth'][possible], kind='stable')]
    rows = pre['tiles'][tile]
    x, y = (pre['xy'][rows]-np.asarray(xy, dtype=np.float32)).T
    a, b, c = pre['conic'][rows].T
    power = F(-.5)*(a*x*x+c*y*y)-b*x*y
    alpha = np.minimum(F(.99), pre['opacity'][rows]*np.exp(np.minimum(power, F(0.))))
    keep = (power <= 0.) & (alpha >= F(1./255.))
    rows, alpha = rows[keep], alpha[keep]
    T = np.cumprod(1-alpha, dtype=np.float32)
    count = int(np.searchsorted(-T, F(-.0001), side='right'))
    rows, alpha, T = rows[:count], alpha[:count], T[:count]
    weights = alpha*np.r_[F(1.), T[:-1]] if count else np.empty(0, dtype=np.float32)
    rgb = np.cumsum(pre['rgb'][rows]*weights[:, None], axis=0, dtype=np.float32)[-1] if count else np.zeros(3, np.float32)
    # The QA alpha image uses an override of ones. Its accumulation, rather
    # than 1-T, tracks the real floating-point accumulation for PNG parity.
    opacity = float(np.cumsum(weights, dtype=np.float32)[-1]) if count else 0.
    transmittance = float(T[-1]) if count else 1.
    return dict(rows=rows, weights=weights, rgb=rgb, opacity=opacity, transmittance=transmittance)


def code(value):
    return np.rint(np.clip(value, 0., 1.)*255).astype(np.int16)


def parity_panel(pre, path, crop, pairs):
    x0, y0, x1, y1 = crop
    w, h = x1-x0, y1-y0
    with Image.open(path) as image:
        source = np.asarray(image.convert('RGB'), dtype=np.int16)
    need(source.shape == (h, 4*w, 3), 'Unexpected saved review panel size.')
    xy = sorted({(x+dx, y+dy) for pair in pairs for x,y in pair for dx in (-1,0,1) for dy in (-1,0,1)})
    xy += [(5, 5), (w-6, 5), (w//2, h//3)]
    errors, traced = [], {}
    for x,y in xy:
        item = pixel(pre, (x+x0, y+y0))
        model = np.stack((code(item['rgb']), code(item['rgb']+item['transmittance']),
                          np.full(3, code(item['opacity']), dtype=np.int16)))
        actual = source[y, x+np.arange(3)*w]
        errors.append(np.abs(actual-model))
        traced[(x,y)] = item | dict(qa_black=actual[0].tolist(), qa_white=actual[1].tolist(), qa_alpha=int(actual[2,0]))
    error = np.stack(errors)
    proof = dict(pixels=len(error), black_rgb_max_code=int(error[:,0].max()),
                 white_rgb_max_code=int(error[:,1].max()), alpha_max_code=int(error[:,2].max()),
                 exact_pixels=int((error==0).all((1,2)).sum()), all_channel_mae_code=float(error.mean()),
                 passed=bool(error.max() <= 1))
    return proof, traced


def load_subject(subject, review):
    result_path = checked(review['subjects'][subject]['review'])
    result = read(result_path)
    plan_path = checked(result['plan'])
    plan = read(plan_path)
    arm = plan['static_grid']['arms']['r22']
    portable = pin(arm['ply'], arm['ply_sha256'])
    data = ply(portable)
    raw = checked(result['raw_equivalence']['r22']['raw_world'])
    raw_data = ply(raw)
    t = arm['world_to_viewer']
    R, origin = np.asarray(t['rotation']), np.asarray(t['origin_world'])
    unchanged = [name for name in data.dtype.names if name not in (*'xyz', 'nx', 'ny', 'nz', *(f'rot_{i}' for i in range(4)))]
    need(data.dtype.names == raw_data.dtype.names and t['scale'] == 1. and len(raw_data) == len(data)
         and all(np.array_equal(raw_data[name],data[name]) for name in unchanged)
         and np.array_equal(((fields(raw_data,'xyz')-origin)@R.T).astype(np.float32),fields(data,'xyz')),
         'Portable PLY lost row identity during coordinate export.')
    q = numbered(raw_data,'rot',4).astype(np.float64)
    expected_q = Rotation.from_matrix(R@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()).as_quat()[:,[3,0,1,2]]
    need(np.array_equal(expected_q.astype(np.float32), numbered(data,'rot',4)), 'Portable covariance row order differs.')
    manifest = read(checked(plan['proof']['duration_raw_18000_manifest']))
    state_path = pin(raw.with_name('state.pt'), manifest['state_sha256'])
    checkpoint = torch.load(state_path, weights_only=True, map_location='cpu')
    model = checkpoint['state']['model']
    stable, owner = (model[key]['tensor'].numpy().astype(np.int64) for key in ('densification_ids','binding'))
    local = ply(checked(arm['local']))
    need(len(data) == len(local) == len(stable) == len(owner) == manifest['points'] and (np.diff(stable)>0).all()
         and np.array_equal(local['binding_0'],owner), 'Raw IDs differ from the published bound-local row identities.')
    names = {'_xyz': fields(local,'xyz'), '_rotation':numbered(local,'rot',4), '_scaling':numbered(local,'scale',3),
             '_opacity':local['opacity'][:,None], '_features_dc':numbered(local,'f_dc',3)[:,None,:]}
    need(all(np.array_equal(checkpoint['state']['parameters'][key]['value']['tensor'].numpy(), values)
             for key,values in names.items()), 'Bound-local rows are inconsistent with the raw state.')
    pair_path = pin(raw.with_name('skin-pairs.npz'), manifest['files_sha256']['skin-pairs.npz'])
    attachment = read(pin(raw.with_name('attachment.json'),manifest['files_sha256']['attachment.json']))
    with np.load(pair_path, allow_pickle=False) as sidecar:
        pair, frames, masks = (sidecar[key] for key in ('pairs','frames','skin_mask_sha256'))
    support = attachment['support']['views']
    need(frames.tolist()==[row['frame'] for row in support]
         and masks.tolist()==[row.get('native_material_mask_sha256','') for row in support], 'Skin observations differ.')
    index = np.searchsorted(stable,pair['stable_id'])
    need(np.all(index<len(stable)) and np.array_equal(stable[index],pair['stable_id'])
         and np.array_equal(owner[index],pair['face']), 'Skin evidence does not match stable IDs and owner triangles.')
    skin = defaultdict(list)
    for row in pair:
        i = int(row['view'])
        skin[int(row['stable_id'])].append(dict(frame=frames[i].item(), bin=support[i]['angular_bin'],
             distance_mm=float(row['distance_mm']), normal_sigma_mm=float(row['normal_sigma_mm'])))
    children = {child for event in checkpoint['surface_distribution']['split_events'] for trial in event['trials']
                if trial['outcome']=='committed' for child in trial['child_ids']}
    grid = read(checked(result['static']['r22']['receipt']))
    need(grid['ply_sha256']==arm['ply_sha256'] and grid['side']=='r22', 'Wrong QA static endpoint.')
    return dict(result=result, plan=plan, arm=arm, grid=grid, arrays=geometry(data), owner=owner,
                stable=stable, skin=skin, children=children, local=local,
                source_bins={row['frame']:row['angular_bin'] for row in support},
                hashes=dict(result=sha(result_path), plan=sha(plan_path), portable=sha(portable), world=sha(raw),
                            state=sha(state_path), skin_pairs=sha(pair_path), bound_local=arm['local']['sha256']))


def material_mask(wh, polygons):
    raster = Image.new('1', tuple(wh))
    draw = ImageDraw.Draw(raster)
    for polygon in polygons:
        draw.polygon([tuple(point) for point in polygon], fill=1)
    return np.asarray(raster, dtype=bool)


def posed_model(context):
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    arm = context['arm']
    run = Path(arm['run'])
    source = context['plan']['poses']['r22']['new']
    result = read(pin(run/'result.json',source['terminal_sha256']))
    replay_path = pin(run/result['bound_replay'],result['bound_replay_sha256'])
    replay = read(replay_path)
    path = pin(replay_path.parent/replay['mesh_file'],replay['mesh_sha256'])
    with np.load(path, allow_pickle=False) as data:
        vertices, faces, frames = (data[key] for key in ('vertices_world','faces','frame_ids'))
    local = context['local']
    model = SMPLXGaussianModel(0)
    model._xyz = torch.from_numpy(fields(local,'xyz'))
    model._scaling = torch.from_numpy(numbered(local,'scale',3))
    model._rotation = torch.from_numpy(numbered(local,'rot',4))
    model.vertices_sequence = torch.from_numpy(vertices)
    model.faces = torch.as_tensor(faces, dtype=torch.long)
    model.binding = torch.from_numpy(context['owner'])
    need(frames.tolist()==list(context['source_bins']), 'The original pose replay differs from the source roster.')
    context['hashes']['native_mesh_archive'] = sha(path)
    return model, {name:i for i,name in enumerate(frames.tolist())}


def mask_evidence(subject, context, candidates):
    """Find native material ROI overlap. A supported center alone is insufficient."""
    model, slots = posed_model(context)
    pose = context['plan']['poses']['r22']
    native = context['result']['native']['r22']
    _, _, _, opacity, rgb = context['arrays']
    evidence, annotations = defaultdict(list), {}
    for pair in pose['pairs']:
        role, frame = pair['role'], pair['frame']
        registered = pose['independent_annotations'][role]
        camera = dict(pair['camera'],frame=frame)
        wh = camera['wh']
        pin(pair['source'], registered['source_sha256'])
        with Image.open(pin(pair['score_mask'],registered['score_mask_sha256'])) as im:
            foreground = np.asarray(im.convert('L'))>=253
        hair = registered['regions']['hair']
        masks = {'hair': material_mask(wh,hair['polygons'])}
        mask_sha = lambda mask: hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        need(mask_sha(masks['hair']) == hair['sha256'], 'The signed native hair ROI changed.')
        polys = {'hair':hair['polygons']}
        if subject == '006':
            polys['glasses_rim'] = RIMS[frame]
            masks['glasses_rim'] = material_mask(wh,RIMS[frame])
        need(all(foreground[mask].all() and mask.any() for mask in masks.values()),
             'A material ROI overlaps outside independent foreground.')

        model.select_mesh_by_timestep(slots[frame])
        arrays = (model.get_xyz.numpy(), model.get_scaling.numpy(), model.get_rotation.numpy(), opacity, rgb)
        pre = preprocess(arrays, camera)
        panel = native['panels']['dark_'+role]
        path = pin(Path(native['directory'])/panel['file'],panel['sha256'])
        x0,y0,x1,y1 = panel['crop_xyxy']
        w,h = x1-x0,y1-y0
        need(panel['columns']==['source_with_eye_yellow_hair_cyan','black','white','alpha'], 'Unexpected native QA panel.')
        with Image.open(path) as im:
            saved = np.asarray(im.convert('RGB'), dtype=np.int16)
        need(saved.shape==(h,4*w,3),'Unexpected native QA dimensions.')
        bins = context['source_bins'][frame]
        ann = dict(source_sha256=registered['source_sha256'], frame=frame, source_bin=bins, panel_sha256=sha(path), masks={})
        center = np.rint(pre['xy'][candidates]).astype(np.int32)
        eligible = (center[:,0]>=0)&(center[:,0]<wh[0])&(center[:,1]>=0)&(center[:,1]<wh[1])
        for name, full_mask in masks.items():
            mask = np.zeros_like(full_mask)
            mask[y0:y1,x0:x1] = full_mask[y0:y1,x0:x1]
            yy, xx = np.nonzero(mask)
            need(len(xx)>0, 'Material ROI does not intersect signed native QA crop.')
            errors, support = [], defaultdict(list)
            candidates_in_mask = {int(row) for row,(x,y),valid in zip(candidates,center,eligible,strict=True) if valid and mask[y,x]}
            centers = {int(row):tuple(map(int,xy)) for row,xy in zip(candidates,center,strict=True) if int(row) in candidates_in_mask}
            for x,y in zip(xx,yy,strict=True):
                xy = (int(x),int(y))
                item = pixel(pre,xy)
                reconstructed = np.stack((code(item['rgb']),code(item['rgb']+item['transmittance']),
                                          np.full(3,code(item['opacity']),dtype=np.int16)))
                errors.append(int(np.abs(reconstructed-saved[y-y0,x-x0+np.arange(1,4)*w]).max()))
                for row,weight in zip(item['rows'],item['weights'],strict=True):
                    if int(row) in candidates_in_mask:
                        support[int(row)].append((xy,float(weight)))
            max_error = max(errors)
            ann['masks'][name] = dict(polygons=polys[name], pixels=len(xx), sha256=mask_sha(mask),
                full_roi_pixels=int(full_mask.sum()), full_roi_sha256=mask_sha(full_mask),
                panel_max_code_error=max_error, exact_pixels=errors.count(0), validated=max_error<=1)
            if max_error>1:
                continue
            for row, observations in support.items():
                center_weight = next((weight for xy,weight in observations if xy==centers[row]),0.)
                # Require 1% at the projected center as positive visibility,
                # rather than labeling any extremely thin raster tail.
                if center_weight>=.01:
                    evidence[row].append(dict(material=name, frame=frame, bin=bins, center_pixel_xy=centers[row],
                        center_alpha_contribution=center_weight, roi_pixels_contributed=len(observations)))
        annotations[frame] = ann
    return evidence,annotations


def material_of(context, native, row):
    stable = int(context['stable'][row])
    skin = context['skin'][stable] if stable in context['skin'] else []
    evidence = native[row] if row in native else []
    names = {item['material'] for item in evidence}
    if len(names) > 1 or (skin and names):
        label = 'mixed_native_evidence'
    elif skin:
        label = 'trusted_skin_endpoint'
    elif names:
        name, = names
        label = 'native_'+name+'_roi'
    else:
        label = 'unclassified'
    return label, skin, evidence


def trace_pair(context, pre, native, traced, crop, pair):
    a,b = (traced[xy] for xy in pair)
    wa, wb = [{int(row):float(weight) for row,weight in zip(item['rows'],item['weights'],strict=True)} for item in (a,b)]
    members = set(wa)|set(wb)
    rgb_luma = pre['rgb'] @ LUMA
    groups, contributors = defaultdict(lambda:np.zeros(3)), []
    for row in members:
        first, last = wa.get(row,0.), wb.get(row,0.)
        label, skin, material = material_of(context,native,row)
        value = (first-last)*rgb_luma[row]
        groups[label] += (first,last,value)
        covariance = pre['projected'][row]+np.eye(2)*.3
        e = np.linalg.eigvalsh(covariance)
        contributors.append(dict(row=row, stable_id=int(context['stable'][row]), owner_face=int(context['owner'][row]),
            material_evidence=label, signed_luma_difference=float(value), artifact_alpha_weight=first,
            control_alpha_weight=last, depth_m=float(pre['depth'][row]), footprint_sigma_ratio=float(np.sqrt(e[-1]/e[0])),
            projected_xy_px=pre['xy'][row].tolist(), footprint_radius_px=int(pre['radius'][row]),
            color_sh0_rgb=pre['rgb'][row].tolist(), trusted_skin=skin, annotated_native_roi=material,
            exact_committed_split_child=int(context['stable'][row]) in context['children']))
    contributors.sort(key=lambda value: abs(value['signed_luma_difference']), reverse=True)
    delta = float((a['rgb']-b['rgb'])@LUMA)
    need(abs(sum(item[2] for item in groups.values())-delta)<2e-6,'The per-row luma sum does not match raster output.')
    return dict(artifact_xy_px=[pair[0][0]+crop[0],pair[0][1]+crop[1]],
        control_xy_px=[pair[1][0]+crop[0],pair[1][1]+crop[1]], artifact_qa_rgb=a['qa_black'], control_qa_rgb=b['qa_black'],
        luma_difference=delta, row_count=len(contributors), direct_split_child_rows=sum(row['exact_committed_split_child'] for row in contributors),
        group_sums={k:dict(zip(('artifact_alpha_weight','control_alpha_weight','signed_luma_difference'),v.tolist())) for k,v in groups.items()},
        artifact_alpha=a['opacity'], control_alpha=b['opacity'], contributors=contributors)


def main():
    sys.path.insert(0, str(PROJECT/'src'))
    torch.set_num_threads(2)
    review = read(pin(DELIVERY/'review-receipt.json', REVIEW_SHA))
    results = {}
    for subject in PICKS:
        context = load_subject(subject, review)
        views, candidates = {}, set()
        for suffix, (label, pairs) in PICKS[subject].items():
            row = context['grid']['records'][VIEW+suffix]
            panel = row['panel']
            pre = preprocess(context['arrays'],row['camera'])
            proof, traced = parity_panel(pre,checked(panel),panel['crop_xyxy'],pairs)
            views[suffix] = (row, panel, pre, proof, traced)
            if proof['passed']:
                candidates.update(int(row) for pair in pairs for xy in pair for row in traced[xy]['rows'])
        native, annotations = mask_evidence(subject,context,sorted(candidates)) if candidates else ({},{})
        result = dict(input_sha256=context['hashes'], native_annotation=annotations, views={})
        for suffix,(label,pairs) in PICKS[subject].items():
            row,panel,pre,proof,traced = views[suffix]
            comparisons = [trace_pair(context,pre,native,traced,panel['crop_xyxy'],pair) for pair in pairs] if proof['passed'] else []
            result['views'][VIEW+suffix] = dict(artifact=label, parity=proof, qa_panel=panel,
                camera=row['camera'], picked_crop_local_pixels=pairs, comparisons=comparisons)
            print(json.dumps(dict(subject=subject, view=suffix, parity=proof,
                delta=[pair['luma_difference'] for pair in comparisons],
                groups=[pair['group_sums'] for pair in comparisons]),ensure_ascii=False), flush=True)
        results[subject] = result
    source_names = ('forward.cu', 'rasterizer_impl.cu', 'auxiliary.h', 'config.h')
    output = DELIVERY/'input-audit/face-pixel-attribution.json'
    report = dict(schema='shuolee-r22-face-pixel-attribution/1', review_sha256=REVIEW_SHA,
        diagnostic_sha256=sha(Path(__file__)), renderer_sha256={name:sha(SOURCE/name) for name in source_names},
        gpu_used=False, renderer_or_training_changes=False, quality_accepted=False,
        comparisons='Preselected local native-raster pixels and same-row nearby controls, saved R22 18k synthetic ±30 degree close views.',
        limitations=['The saved CUDA panels have 8-bit channels. Pixel-code parity bounds the color and alpha reconstruction without exposing native internal contributor buffers.',
                     'Material ROI evidence identifies projection and actual composited weight at the row center in annotated original observations. It is not an independently triangulated material correspondence.',
                     'Unclassified rows lack positive trusted endpoint skin or the sparse independently annotated hair/rim evidence. Their physical material and past optimization support remain unknown.',
                     'Contrast attribution describes saved color/alpha at chosen rays, not a proof of the upstream cause or an exhaustive face audit.'],
        subjects=results)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(dict(output=str(output),sha256=sha(output)),flush=True)


if __name__=='__main__':
    main()
