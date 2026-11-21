"""Bounded same-prompt native SAM2 ambiguity audit on four frozen 008 witnesses."""

import argparse
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True

LAB = Path('/opt/gs-slam-lab')
RUN = LAB / 'shuolee-smpl-visibility-008/run'
OUTPUT = LAB / 'shuolee-smpl-visibility-008-sam2-alternate-tokens'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-source-visibility')
WINDOWS_GIT = Path('/mnt/c/Program Files/Git/cmd/git.exe')
SAM = PROJECT / 'third_party/sam2'
HISTORY = (Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
           / 'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
MIDPOINT = HISTORY / 'indoor-forearm-prompt275'
FRAMES = tuple(f'frame_{n:06d}' for n in (44, 48, 51, 67))
TRAINING_WITNESSES = (FRAMES[1], FRAMES[2])
SKIN_COUNTS = dict(zip(FRAMES, ((362, 0), (500, 0), (168, 0), (526, 526)), strict=True))
WINNERS = dict(zip(FRAMES, (
    '6bb4088e23d877bc96b9869bc5f68f23d68430be2193f912c7a008a809ab65b7',
    '1396a52171b22e11e479af619c05fb112f31657ae95abec6ec13e8c92cfdd9c0',
    '57e50259e7171924b17c78131b186bfb6dd1f1abe029d8ddd2643cdf271f3842',
    'db121e9a1c17a60797d491091ad7529e5a557ce788e8a544645c0e70aeb926a5',
), strict=True))
PINNED = {
    RUN / 'config.json': '6ae1e685be00858eb0083b4dbff005a7b7a62ed3ca29ee2fa13011b706d397b8',
    RUN / 'result.json': '4f2605fca0b011798308fa414de604c3c3dc00dfcb5d4779d31d5338dc89aa84',
    RUN / 'data/inputs.json': 'a0e101fb4bb546f35b1638d8786f3fde3d4a9ea9d73f797886423940abe98147',
    RUN / 'data/training-inputs.json': 'e485e3657b57d4de800623ee5f0f60da3fce8a50820afdd5f456bedf3e4443d8',
    RUN / 'data/observations.npz': 'ef9b035c8ded9eb2ccdfde70cccad5b7f90dbb39b176509e134c4b502693a8d0',
    RUN / 'fit/selection.json': '44e85e52a01642cda7c294f9afbb8ef4520f3e31b53cf218fadca0dcf5c0950d',
    RUN / 'fit/official-fit-result.json': 'ead9039816741cb6c780ecc888fc139c6f826c7580a851344e12bfca45925ef0',
    MIDPOINT / 'probe275.py': '34ccfbd46d42159c1c3abefa6bd27f854b900ee8976762bc5bf2327ee592cf9d',
    MIDPOINT / 'preregistration.json': '576792fe2baec37498227f420cba5bcc378a64ed6abb1ba1fb64ffd31c02a9a6',
    MIDPOINT / 'visual-review.json': '4d4c580e76fdf32183fb4fb1cb695e5d78e70cc5c21733e38e6738cf8ad236cc',
    HISTORY / 'sofa-sam-repeat273/probe.py': '04fe6892f791bae02808540b0c69683f1bfa4071bd0cff872a69de979886bc6b',
    HISTORY / 'indoor-accepted-arm-mask272/plan.json': 'b3f17bef23a3d62f09bb78e72eab0c7c066f149373677f511a45f9915783ffdd',
    PROJECT / 'src/shuolee_portrait/observations.py': 'ea7e6998d92583bd44e0e447ee26f91c806350e58df3d56ff4c9449f5caa35b7',
    SAM / 'sam2/modeling/sam/mask_decoder.py': 'ca3523c58365574faddf1bfb54f374e4b4beba05127185ea2b17d22b916e099a',
    SAM / 'sam2/sam2_image_predictor.py': 'f13e5f9d94e5c8d9d2c3622dab20c8f334c089ef2ee5ea8e199da7d332b029ba',
}


def prepare():
    from hashlib import sha256

    for path in (MIDPOINT / 'probe275.py', HISTORY / 'sofa-sam-repeat273/probe.py'):
        if sha256(path.read_bytes()).hexdigest() != PINNED[path]:
            raise ValueError(f'Historical helper changed: {path}')
    spec = spec_from_file_location('midpoint275', MIDPOINT / 'probe275.py')
    base = module_from_spec(spec)
    spec.loader.exec_module(base)
    require, sha, read = base.require, base.sha, base.read
    pins = {}

    def pin(path, expected=None):
        path = Path(path)
        digest = sha(path)
        require(expected is None or expected == digest, f'Input hash differs: {path}')
        pins[str(path)] = digest
        return path

    for path, digest in PINNED.items():
        pin(path, digest)
    pin(__file__)
    require(WINDOWS_GIT.is_file(), 'Windows Git is needed for this Windows-created linked worktree.')
    windows_project = subprocess.check_output(['wslpath', '-w', str(PROJECT)], text=True).strip()
    require(subprocess.check_output([str(WINDOWS_GIT), '-C', windows_project, 'rev-parse', 'HEAD'], text=True).strip()
            == 'd080021d3c9d17a986e9fb60a93a0a8243c4a210', 'Expected current immutable 008 checkout.')
    old = read(MIDPOINT / 'preregistration.json')
    source = read(HISTORY / 'indoor-accepted-arm-mask272/plan.json')
    require(tuple(r['frame'] for r in old['frames']) == tuple(r['frame'] for r in source['frames']) == FRAMES,
            'Witness roster differs.')
    require(read(MIDPOINT / 'visual-review.json')['locally_recovered_uncertainty'] == 725,
            'Prior single-midpoint comparison is required.')
    pin(base.WEIGHTS, old['weights_sha256'])
    pin(base.ASSET, '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992')
    pin(base.RASTER, '3d348ff19a50b3c7a850954508539994e24890f4994ef0590be50bdca36374df')
    for path in (SAM / 'sam2').rglob('*'):
        if path.suffix in ('.py', '.yaml', '.so') and path.is_file():
            pin(path)
    sys.path.insert(0, str(PROJECT / 'src'))
    from shuolee_portrait import official_smplifyx, stable_selection

    cfg = read(RUN / 'config.json')
    terminal = read(RUN / 'result.json')
    manifest = read(RUN / 'data/inputs.json')
    fit = read(RUN / 'fit/official-fit-result.json')
    selection = read(RUN / 'fit/selection.json')
    require(terminal['state'] == 'completed' and terminal['frames'] == 19 and terminal['steps'] == 10_000
            and fit['selection'] == selection and fit['training_inputs_sha256'] == pins[str(RUN / 'data/training-inputs.json')]
            and old['holdouts'] == selection['retained_holdouts'] and not set(FRAMES) & set(old['holdouts'])
            and tuple(f for f in FRAMES if f in selection['selected_frames']) == TRAINING_WITNESSES,
            'Current training or holdout roster changed.')
    for path, digest in fit['source']['input_files'].items():
        pin(path, digest)
    scope = stable_selection.selection_scope(dict(rows=manifest['records'], source_manifest=manifest),
                                              arc=cfg['capture_layout']['arc'], capture_layout=cfg['capture_layout'])
    rows = {r['frame']: r for r in manifest['records']}
    receipts = {r['frame_id']: r for r in fit['official_frame_results']}
    samples = []
    with np.load(RUN / 'data/observations.npz', allow_pickle=False) as obs:
        names = obs['frame_ids'].astype(str).tolist()
        fit_order = [names.index(f) for f in scope['fit_frame_ids']]
        target, confidence = official_smplifyx.adapt_rtmw_observations(
            *(obs[key][fit_order] for key in ('keypoints', 'scores', 'boxes', 'flip_disagreement')))
        require(confidence == fit['source']['confidence'], 'Relative arm confidence contract changed.')
        for w, prior in zip(source['frames'], old['frames'], strict=True):
            frame = w['frame']
            row = rows[frame]
            slot = names.index(frame)
            box, center = obs['boxes'][slot], obs['keypoints'][slot, [5, 6, 11, 12]].mean(0)[None]
            require(row['source_sha256'] == source['pins'][w['rgb']]
                    and row['source_frame_index'] == w['source_index'] == prior['source_index']
                    and row['wh'] == [2704, 1520] and row['K_integer_centers'] == w['K_integer_centers']
                    and np.array_equal(box, row['initial_box_xyxy'])
                    and np.allclose(box, prior['box'], rtol=0, atol=.0002)
                    and np.allclose(center, prior['center'], rtol=0, atol=.0002),
                    f'Native source or stock prompt changed for {frame}.')
            pin(row['image'], row['source_sha256'])
            pin(row['mask'], row['mask_sha256'])
            path = pin(RUN / 'fit/frames' / frame / 'winner-camera-space.npz', WINNERS[frame])
            with np.load(path, allow_pickle=False) as archive:
                require(str(archive['frame_id']) == frame and np.array_equal(archive['K_integer_centers'], row['K_integer_centers']),
                        'Camera-space fit source differs.')
            if frame in receipts:
                pin(receipts[frame]['path'], receipts[frame]['sha256'])
            observations = target[scope['fit_frame_ids'].index(frame)]
            samples.append(dict(frame=frame, selected=frame in TRAINING_WITNESSES, image=row['image'],
                old_mask=row['mask'], winner=str(path), box=box.tolist(), center=center.tolist(),
                box_dtype=box.dtype.str, center_dtype=center.dtype.str,
                source_index=row['source_frame_index'], wh=row['wh'],
                arm_observations=observations[:8].tolist(), skin=w['source_skin_polygon_xy'],
                background=w['source_background_polygon_xy'], sofa=prior['chair_foreground_roi'],
                native_crop=w['native_crop_xyxy']))
    plan = dict(state='preregistered', frames=samples, holdouts=old['holdouts'], pins=pins,
        prior_275_recovered_skin_unknown=725, predictor_calls=8, image_embeddings=4,
        baseline_gate='Exact token0 agreement with the current archive on every decision pixel and fixed source witness. '
                      'Full native XOR is reported separately. Ungraded disagreement is preserved in diagnostics only.',
        arm_support='Current nearest visible same-side forearm by the 275 topology, with existing endpoint confidence and '
                    '0.03 box-height protection. No midpoint test or witness labels enter this support.',
        rule='Unknown-only when archived token0 and live token0 are both background, an alternate token votes foreground '
             'and the current same-side arm support is eligible. All archive positives remain unchanged.',
        screen='All four baselines exact in decision footprints, f051 gains at least one verified off-axis skin pixel, '
               'and no verified background or sofa pixel is suppressed in any frame. Pixels outside witnesses remain unverified.',
        stock_profile={k: v for k, v in old['stock_profile'].items() if k != 'sam_source'},
        max_runtime_seconds=120, production_mask_replacement=False, optimizer_updates=0)
    return base, plan


def anatomical_support(base, item, shape, weights, topology, raster):
    """Same 275 arm ownership test, independent of centerline and witness polygons."""
    with np.load(item['winner'], allow_pickle=False) as fit:
        vertices, faces, K = fit['vertices_camera'][0].astype(float), fit['faces'], fit['K_integer_centers']
        projected = fit['joints_projected_native'][0]
    base.require(np.array_equal(faces, topology) and np.isfinite(vertices).all() and vertices[:, 2].min() > .05,
                 'Current fit topology or vertex depths differ.')
    homogeneous = vertices @ K.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:]
    obs = np.asarray(item['arm_observations'])
    height = item['box'][3] - item['box'][1]
    combined = np.zeros(shape, dtype=bool)
    sides = {}
    for side, (points, bones) in base.ARM.items():
        xy = obs[points, :2]
        errors = np.linalg.norm(projected[points] - xy, axis=-1)
        trusted = bool(((xy >= 0) & (xy < item['wh'])).all() and (obs[points, 2] >= .5).all()
                       and (errors <= .03 * height).all())
        count = 0
        if trusted:
            chosen = weights[faces][:, :, bones].sum(-1).mean(-1) > .5
            local = uv[np.unique(faces[chosen])]
            lo = np.maximum(np.floor(local.min(0)), 0).astype(int)
            hi = np.minimum(np.ceil(local.max(0)) + 1, item['wh']).astype(int)
            if (hi > lo).all():
                owner = raster(uv-lo, vertices[:, 2], faces, tuple(hi-lo))
                support = (owner >= 0) & chosen[np.maximum(owner, 0)]
                combined[lo[1]:hi[1], lo[0]:hi[0]] |= support
                count = int(support.sum())
        sides[side] = dict(trusted_endpoints=trusted, endpoint_errors_px=errors.tolist(),
                           endpoint_confidences=obs[points, 2].tolist(), visible_forearm_pixels=count)
    return combined, sides


def render_panel(path, rgb, crop, old, alternatives, unknown, polygons):
    x0, y0, x1, y1 = crop
    s = np.s_[y0:y1, x0:x1]
    source = rgb[s]
    patch = source.copy()
    patch[unknown[s]] = (patch[unknown[s]] * .45 + np.array([42, 224, 239]) * .55).astype(np.uint8)
    images = [source, source * old[s][..., None],
              *(source * mask[s][..., None] for mask in alternatives), patch]
    width, height = source.shape[1], source.shape[0]
    canvas = Image.new('RGB', (len(images) * (width + 6) - 6, height + 24), '#20262a')
    draw = ImageDraw.Draw(canvas)
    names = ('Source', 'Frozen stock mask', 'SAM2 token 1', 'SAM2 token 2', 'SAM2 token 3', 'Proposed unknown')
    for index, (name, values) in enumerate(zip(names, images, strict=True)):
        tile = Image.fromarray(values)
        marks = ImageDraw.Draw(tile)
        for points, color in polygons:
            marks.polygon([(x-x0, y-y0) for x, y in points], outline=color)
        offset = index * (width + 6)
        canvas.paste(tile, (offset, 24))
        draw.text((offset + 2, 5), name, fill='white')
    canvas.save(path)


def audit(base, plan, output):
    import torch
    from shuolee_portrait import runtime

    started = monotonic()
    runtime.configure(base.ASSETS, output)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    profile = base.profile(torch)
    source = Path(profile['sam_source']).resolve()
    base.require(source.is_relative_to(SAM.resolve()) and {k: v for k, v in profile.items() if k != 'sam_source'}
                 == plan['stock_profile'], 'Stock CUDA/SAM2 profile changed.')
    model = build_sam2(base.CFG, str(base.WEIGHTS), device='cuda', apply_postprocessing=False)
    base.require(not model.training and not model.sam_mask_decoder.dynamic_multimask_via_stability
                 and model.sam_mask_decoder.num_multimask_outputs == 3
                 and all(p.dtype == torch.float32 for p in model.parameters()), 'Stock decoder token path changed.')
    predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
    base.require(predictor.mask_threshold == 0., 'Stock SAM2 mask threshold changed.')
    with np.load(base.ASSET, allow_pickle=False) as asset:
        weights, topology = asset['weights'], asset['f']
    raster = base.rasterizer()
    records = []
    with torch.inference_mode():
        for item in plan['frames']:
            frame = item['frame']
            with Image.open(item['image']) as image, Image.open(item['old_mask']) as mask:
                rgb = np.asarray(image.convert('RGB'))
                saved = np.asarray(mask.convert('L')) > 127
            base.require(rgb.shape == (*reversed(item['wh']), 3) and saved.shape == rgb.shape[:2],
                         'Native image or mask dimensions differ.')
            support, sides = anatomical_support(base, item, saved.shape, weights, topology, raster)
            skin = base.polygon(saved.shape, item['skin'])
            background = base.polygon(saved.shape, item['background'])
            base.require((int(skin.sum()), int((skin & saved).sum())) == SKIN_COUNTS[frame],
                         'Archived current skin witness count changed.')
            sofa = np.zeros_like(saved)
            x0, y0, x1, y1 = item['sofa']
            sofa[y0:y1, x0:x1] = True
            witness = dict(skin=skin, background=background, sofa=sofa)
            stock_inputs = dict(box=np.asarray(item['box'], dtype=item['box_dtype']),
                point_coords=np.asarray(item['center'], dtype=item['center_dtype']), point_labels=np.ones(1, dtype=np.int32))
            predictor.set_image(rgb)
            embedding = predictor._features['image_embed']
            stock, score0, logits0 = predictor.predict(**stock_inputs, multimask_output=False)
            masks, scores, logits = predictor.predict(**stock_inputs, multimask_output=True)
            base.require(embedding is predictor._features['image_embed']
                         and stock.shape == (1, *saved.shape) and masks.shape == (3, *saved.shape)
                         and logits0.shape == (1, 256, 256) and logits.shape == (3, 256, 256)
                         and score0.shape == (1,) and scores.shape == (3,)
                         and all(np.isfinite(a).all() for a in (score0, scores, logits0, logits))
                         and score0[0] >= base.SCORE_THRESHOLD, 'Unexpected token outputs or stock sanity failure.')
            live, alternatives = stock[0].astype(bool), masks.astype(bool)
            changed = saved ^ live
            decision = support | skin | background | sofa
            decision_xor = int((changed & decision).sum())
            alternate_new = alternatives & ~saved[None] & ~live[None]
            raw_union = alternate_new.any(0)
            proposed = raw_union & support
            graded = proposed if decision_xor == 0 else np.zeros_like(proposed)
            base.require(not (proposed & (saved | live | ~support)).any(), 'Unknown escaped protected mask boundaries.')
            counts = {key: dict(pixels=int(region.sum()), saved_positive=int((region & saved).sum()),
                      stock_xor=int((region & changed).sum()),
                      raw_token_new=[int((region & new).sum()) for new in alternate_new],
                      raw_union_new=int((region & raw_union).sum()),
                      proposed_unknown=int((region & proposed).sum()),
                      baseline_exact_unknown=int((region & graded).sum())) for key, region in witness.items()}
            data = output / f'{frame}-token-bits.npz'
            np.savez_compressed(data, shape=saved.shape, archived=np.packbits(saved), live_token0=np.packbits(live),
                alternatives=np.stack([np.packbits(m) for m in alternatives]), arm_support=np.packbits(support),
                proposed_unknown=np.packbits(proposed), baseline_exact_unknown=np.packbits(graded))
            panel = output / f'{frame}-native-forearm.png'
            render_panel(panel, rgb, item['native_crop'], saved, alternatives, proposed,
                         ((item['skin'], '#69fb74'), (item['background'], '#ffb547')))
            records.append(dict(frame=frame, selected=item['selected'], token_scores=[float(score0[0]), *scores.tolist()],
                full_native_stock_xor=int(changed.sum()), decision_footprint_stock_xor=decision_xor,
                graded_baseline_exact=decision_xor == 0, arm_sides=sides, witnesses=counts,
                raw_new_outside_arm=int((raw_union & ~support).sum()), proposed_unknown=int(proposed.sum()),
                bits=str(data), bits_sha256=base.sha(data), panel=str(panel), panel_sha256=base.sha(panel)))
            base.require(monotonic() - started < plan['max_runtime_seconds'], 'Audit exceeded the registered budget.')
    f051 = next(r for r in records if r['frame'] == FRAMES[2])
    verified_skin = f051['witnesses']['skin']['baseline_exact_unknown']
    leakage = sum(r['witnesses'][kind]['baseline_exact_unknown'] for r in records for kind in ('background', 'sofa'))
    baseline_pass = all(row['graded_baseline_exact'] for row in records)
    for path, digest in plan['pins'].items():
        base.require(base.sha(path) == digest, f'Frozen source changed: {path}')
    return dict(state='alternate_tokens_audit_complete', records=records, profile=profile,
        f051_verified_new_skin_unknown=verified_skin, verified_bg_sofa_unknown=leakage,
        all_decision_baselines_exact=baseline_pass, numerical_screen_passed=baseline_pass and verified_skin > 0 and leakage == 0,
        image_embeddings=4, predictor_calls=8, seconds=monotonic()-started, visual_review_completed=False,
        production_mask_replacement=False, training_started=False, optimizer_updates=0, quality_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'audit'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != LAB or output.exists():
        raise ValueError('Use a fresh isolated audit output.')
    base, plan = prepare()
    if args.mode == 'check':
        print('008_ALTERNATE_TOKENS_INPUTS_VERIFIED', FRAMES, len(plan['pins']), 'gpu_calls=0')
        return
    with open('/tmp/shuolee-native122-gpu.lock', 'a+b') as gpu, open('/tmp/shuolee-body-export123.lock', 'a+b') as export:
        for lock in (gpu, export):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        base.require(not idle.strip(), 'GPU must be idle.')
        output.mkdir()
        base.save(output / 'preregistration.json', plan)
        try:
            result = audit(base, plan, output)
            base.save(output / 'result.json', dict(result, preregistration_sha256=base.sha(output / 'preregistration.json')))
            print('008_ALTERNATE_TOKENS_AUDIT_COMPLETE', result['numerical_screen_passed'])
        except BaseException as exc:
            base.save(output / 'failure.json', dict(error=repr(exc), pid=os.getpid(), training_started=False))
            raise


if __name__ == '__main__':
    main()
