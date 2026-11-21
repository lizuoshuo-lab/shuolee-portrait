"""Read-only structure check of held-camera landmark residuals; no calibration."""
from pathlib import Path
import json
import hashlib
import numpy as np

R = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098')
p = R / 'shared-targets/shared-face-points.npz'
sha = lambda: hashlib.sha256(p.read_bytes()).hexdigest()
before = sha()
a = np.load(p)
ids = a['point_ids']
residual = a['observed_uv'] - a['held_prediction']
valid = a['observation_valid'] & np.isfinite(residual).all(-1)
rows, before_errors, after_errors = [], [], []
for i in range(len(residual)):
    for parity in (0, 1):
        fit = valid[i] & (ids % 2 == parity)
        test = valid[i] & (ids % 2 != parity)
        if fit.sum() < 6 or test.sum() < 6:
            continue
        shift = np.median(residual[i, fit], axis=0)
        raw = np.linalg.norm(residual[i, test], axis=-1)
        corrected = np.linalg.norm(residual[i, test] - shift, axis=-1)
        before_errors.extend(raw.tolist())
        after_errors.extend(corrected.tolist())
        rows.append(dict(raw=i+1, fit_parity=parity, fit_count=int(fit.sum()),
                         test_count=int(test.sum()), diagnostic_shift_px=shift.tolist(),
                         before_median_px=float(np.median(raw)), after_median_px=float(np.median(corrected))))
def numbers(x):
    x = np.asarray(x)
    return dict(n=len(x), median=float(np.median(x)), p90=float(np.quantile(x, .9)), mean=float(x.mean()))
result = dict(source=str(p), source_sha256=before,
              method='For each held camera, estimate only a 2D translation from alternating landmark IDs and test it on the other IDs. Both parities are evaluated. No camera, point, image, or model is modified.',
              caveat='A transferable common image offset cannot identify camera error versus rigid head micro-motion or systematic detector bias. This is not a replacement for full-image bundle adjustment and is not accepted calibration.',
              before=numbers(before_errors), after_diagnostic_translation=numbers(after_errors),
              rows=rows, camera_updates=0, gaussian_updates=0, diagnostic_only=True)
assert before == sha()
(R/'shared-target-residual-structure.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','method','caveat')}))
