"""Load signed native face masks and sample occupied head-relative view sectors."""

import itertools

import numpy as np
import torch

from . import runtime as rt
from .face_surface_bundle import require


SUPERVISION_R30 = dict(target_skin_alpha=.95, skin_deficit_weight=.30, opacity_lr=.004,
                       local_opacity_min=.02, local_opacity_max=.93, context_interval=12)


def _crop_mask(array, box, target):
    x0, y0, x1, y1 = box
    tx0, ty0, tx1, ty1 = target.box
    require(array.dtype == np.bool_ and array.shape == (y1-y0, x1-x0), 'Invalid signed native face mask.')
    native = torch.zeros_like(target.valid)
    lx, ly, hx, hy = max(x0, tx0), max(y0, ty0), min(x1, tx1), min(y1, ty1)
    if hx > lx and hy > ly:
        pixels = array[ly-y0:hy-y0, lx-x0:hx-x0]
        native[ly-ty0:hy-ty0, lx-tx0:hx-tx0] = torch.as_tensor(pixels, device=native.device)
    return native & target.valid


def load_face_supervision(prepared, path, coverage_path):
    receipt = rt.read(path.with_suffix('.json'))
    require(receipt['schema'] == 'shuolee-r30-native-face-supervision/1'
            and receipt['archive_sha256'] == rt.sha(path)
            and receipt['coverage_receipt_sha256'] == rt.sha(coverage_path.with_suffix('.json'))
            and receipt['baseline_bound_ply_sha256'] == prepared.source_manifest['bound_ply_sha256']
            and receipt['heldout_source_images_read'] == 0, 'Native face supervision inputs have changed.')
    views = receipt['views']
    targets = prepared.targets
    require(len(views) == len(targets), 'Face supervision must cover the exact training and auxiliary roster.')
    with np.load(path, allow_pickle=False) as archive:
        require(archive['source_slots'].tolist() == [target.slot for target in targets]
                and archive['frame_ids'].tolist() == [target.row['frame'] for target in targets],
                'Face supervision contains an unregistered or held-out view.')
        boxes = archive['boxes']
        angles = np.stack((archive['head_yaw_deg'], archive['head_pitch_deg']), axis=1)
        sectors = np.stack((archive['yaw_sector'], archive['pitch_sector']), axis=1)
        weights = archive['face_weight']
        visible = archive['face_visible']
        require(boxes.shape == (len(targets), 4) and np.issubdtype(boxes.dtype, np.integer)
                and sectors.shape == angles.shape == (len(targets), 2)
                and np.issubdtype(sectors.dtype, np.integer) and np.isfinite(angles).all()
                and visible.shape == weights.shape == (len(targets),) and visible.dtype == np.bool_
                and np.isfinite(weights).all() and np.array_equal(weights > 0, visible),
                'Invalid registered face angle or image-quality weights.')
        for column, (target, view) in enumerate(zip(targets, views, strict=True)):
            row = target.row
            require(view['slot'] == target.slot and all(view[key] == row[key]
                    for key in ('frame', 'source_sha256', 'mask_sha256', 'K', 'w2c')),
                    'Face supervision camera or original source bytes do not match.')
            box = boxes[column].tolist()
            x0, y0, x1, y1 = box
            require(0 <= x0 < x1 <= row['wh'][0] and 0 <= y0 < y1 <= row['wh'][1],
                    'Signed native crop exceeds its source image.')
            face = _crop_mask(archive[f'face_{column:03d}'], box, target)
            skin = _crop_mask(archive[f'skin_{column:03d}'], box, target)
            target.face_valid = face & bool(visible[column])
            target.skin_valid = skin & target.face_valid & ~target.dark_eye
            target.face_weight = float(weights[column]) if bool(target.skin_valid.any()) else 0.
            target.head_angles = angles[column].tolist()
            target.face_sector = tuple(sectors[column].tolist())
    return receipt


def face_training_order(targets, *, seed):
    """One visit to each occupied sector, then use native image quality within it."""
    groups = {}
    for index, target in enumerate(targets):
        if target.face_weight > 0:
            groups.setdefault(target.face_sector, []).append(index)
    roster = []
    for sector, indices in sorted(groups.items()):
        quality = np.array([targets[index].face_weight for index in indices], dtype=np.float64)
        roster.append((sector, indices, quality/quality.sum()))
    rng = np.random.default_rng(seed)

    def order():
        for _ in itertools.count():
            for group in rng.permutation(len(roster)):
                _, indices, probability = roster[group]
                yield int(rng.choice(indices, p=probability))

    report = [dict(yaw_sector=sector[0], pitch_sector=sector[1],
                   frames=[targets[index].row['frame'] for index in indices], probabilities=quality.tolist())
              for sector, indices, quality in roster]
    return report, order()


def skin_deficit(alpha, target):
    """Only current transmission on signed, opaque native skin drives local filling."""
    deficit = (SUPERVISION_R30['target_skin_alpha']-alpha).relu()
    holes = target.skin_valid & (deficit.detach() > 0)
    return (deficit*holes).sum()/holes.sum().clamp_min(1), holes
