"""One preregistered native photometric-localization audit. No camera fitting."""
from pathlib import Path
from datetime import datetime, timezone
import json, time
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw
from audit_grayT123_camera_support import REC, ROOT, read, save, sha, native

OUT = REC / 'grayT-skin-localization-review'


def response(p, x, y):
    cx, cy, lx, ly, amplitude, base, gx, gy = p
    blob = np.exp(-0.5 * (((x-cx)/np.exp(lx))**2 + ((y-cy)/np.exp(ly))**2))
    return amplitude * blob + base + gx*x + gy*y


def fit_patch(gray, saved_center):
    y, x = np.mgrid[-6:7, -6:7].astype(float)
    border = np.concatenate([gray[0], gray[-1], gray[1:-1, 0], gray[1:-1, -1]])
    c = np.asarray(saved_center, float)
    p0 = np.array([*c, np.log(1.3), np.log(1.3), -.12, np.median(border), 0., 0.])
    lo = np.array([*(c-3), np.log(.6), np.log(.6), -.8, 0., -.08, -.08])
    hi = np.array([*(c+3), np.log(3.5), np.log(3.5), -.005, 1., .08, .08])
    opt = least_squares(lambda p: (response(p, x, y)-gray).ravel(), p0,
                        bounds=(lo, hi), max_nfev=200, loss='linear',
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    residual = opt.fun
    variance = float(residual @ residual / (gray.size - len(p0)))
    covariance = variance * np.linalg.pinv(opt.jac.T @ opt.jac)
    interval95 = float(2.44774683*np.sqrt(max(0., np.linalg.eigvalsh(covariance[:2, :2]).max())))
    active = np.minimum(opt.x-lo, hi-opt.x) < 1e-3*(hi-lo)
    return dict(center=opt.x[:2], parameters=opt.x, success=bool(opt.success),
                nfev=opt.nfev, bound_indices=np.flatnonzero(active).tolist(),
                photometric_rmse=float(np.sqrt(np.mean(residual**2))),
                conditional_center_95_radius_px=interval95)


def synthetic_test():
    y, x = np.mgrid[-6:7, -6:7].astype(float)
    p = np.array([.31, -.42, np.log(1.1), np.log(1.7), -.18, .65, .003, -.002])
    clean = response(p, x, y)
    a = fit_patch(clean, [0., 0.])
    noisy = np.clip(clean + np.random.default_rng(122123).normal(0, .003, clean.shape), 0., 1.)
    b = fit_patch(noisy, [0., 0.])
    result = dict(clean_center_error_px=float(np.linalg.norm(a['center']-p[:2])),
                  noisy_center_error_px=float(np.linalg.norm(b['center']-p[:2])),
                  noisy_conditional95_radius_px=b['conditional_center_95_radius_px'],
                  only_synthetic_model_correctness=True)
    assert a['success'] and not a['bound_indices'] and result['clean_center_error_px'] < 1e-4
    assert b['success'] and not b['bound_indices'] and result['noisy_center_error_px'] < .5
    return result


def main():
    started = time.monotonic()
    if (OUT/'result.json').exists():
        raise RuntimeError('Registered run already completed; no rerun.')
    OUT.mkdir(exist_ok=True)
    plan_path = REC/'grayT-skin-localization-plan.json'
    plan = read(plan_path)
    assert plan['model']['native_patch_size'] == 13
    assert plan['frames'] == list(range(40,81))
    source = Path(plan['source'])
    sources = [plan_path, source, ROOT/'data/training-inputs.json', ROOT/'camera/solver-output.npz']
    hashes = [sha(p) for p in sources]
    save(OUT/'plan.json', plan)
    tests = synthetic_test()
    tracking = read(source)['observations']
    rows = read(ROOT/'data/training-inputs.json')['records']
    assert [o['frame_number'] for o in tracking] == plan['frames']
    all_rows, failures = [], []
    raw = Image.new('RGB', (8*64, 6*64), (245,245,245))
    overlay = raw.copy()
    for i, o in enumerate(tracking):
        old = np.array(o['xy'],float)
        center = np.round(old).astype(int)
        path = native(rows[o['frame_number']-1]['image'])
        with Image.open(path) as src:
            im = src.convert('RGB')
            assert center[0] >= 6 and center[1] >= 6 and center[0]+6 < im.width and center[1]+6 < im.height
            small = np.array(im.crop((center[0]-6,center[1]-6,center[0]+7,center[1]+7)),float)/255
            cropbox = [center[0]-32,center[1]-32,center[0]+32,center[1]+32]
            tile = im.crop(cropbox)
        gray = small @ np.array([.299,.587,.114])
        result = fit_patch(gray, old-center)
        new = result.pop('center')+center
        result['parameters'] = result['parameters'].tolist()
        entry = dict(frame=o['frame'],frame_number=o['frame_number'],role=o['role'],
                     saved_LK_xy=old.tolist(),xy=new.tolist(),delta=(new-old).tolist(),
                     displacement_px=float(np.linalg.norm(new-old)), **result)
        # These are model applicability checks, never reprojection-error filtering.
        if not entry['success'] or entry['bound_indices']:
            failures.append(dict(frame=o['frame'],reason='solver_or_photometric_parameter_bound',
                                 bound_indices=entry['bound_indices']))
        all_rows.append(entry)
        tile.save(OUT/f'frame_{o["frame_number"]:03d}_native.png')
        raw.paste(tile, ((i%8)*64,(i//8)*64))
        d=ImageDraw.Draw(tile)
        for xy,color in [(old,(0,255,255)),(new,(255,176,0))]:
            u,v=xy-np.array(cropbox[:2])
            d.line((u-2,v,u+2,v),fill=color,width=1)
            d.line((u,v-2,u,v+2),fill=color,width=1)
        tile.save(OUT/f'frame_{o["frame_number"]:03d}_locator.png')
        overlay.paste(tile,((i%8)*64,(i//8)*64))
        if time.monotonic()-started > 120:
            raise RuntimeError('Registered120second CPU limit; do not rerun automatically.')
    raw.save(OUT/'all41_native.png');overlay.save(OUT/'all41_locator.png')
    assert all(sha(p)==h for p,h in zip(sources,hashes))
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
                observations=all_rows,failed_frames=failures,algorithmic_model_checks_passed=not failures,
                synthetic_test=tests,geometry_comparison_executed=False,training_started=False,
                all_original_observations_retained=True,quality_accepted=False,
                conditional_uncertainty_is_not_physical_accuracy=True,
                sources=[dict(path=str(p),sha256=h) for p,h in zip(sources,hashes)])
    save(OUT/'result.json', result)
    save(OUT/'index.json',dict(images=[dict(frame='frame40–80 chronological, 8columns',
        region='原生64px皮肤标记：青十字为原光流，橙十字为一次亮度定位',columns=['native64px'],
        file='all41_native.png',locator='all41_locator.png',raw='all41_native.png')]))
    print(json.dumps(dict(seconds=result['seconds'],frames=len(all_rows),failures=failures,
        algorithmic_model_checks_passed=not failures,synthetic=tests,
        max_center_displacement_px=max(o['displacement_px'] for o in all_rows)),ensure_ascii=False))


if __name__ == '__main__': main()
