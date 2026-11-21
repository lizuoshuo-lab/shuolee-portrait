"""Read-only SH/DC/scale statistics; no coefficient editing or rendering."""
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from run_official_gaussians123 import save, sha, now
from train_smplify_eight_reviewedfit123 import OUT, BASE


def describe(root):
    path = root/'gaussian-run/step-018000/world.ply'
    data = PlyData.read(str(path))['vertex'].data
    fields = data.dtype.names
    assert not any(k.startswith('f_rest_') for k in fields)
    dc = np.column_stack([data['f_dc_'+str(i)] for i in range(3)]).astype(np.float64)
    rgb = 0.28209479177387814*dc+0.5
    opacity = 1/(1+np.exp(-np.clip(data['opacity'].astype(np.float64),-80,80)))
    logs = np.column_stack([data['scale_'+str(i)] for i in range(3)]).astype(np.float64)
    assert np.isfinite(rgb).all() and np.isfinite(logs).all()
    largest = np.exp(logs.max(-1)); ratio = np.exp(logs.max(-1)-logs.min(-1))
    out = ((rgb<0)|(rgb>1)).any(-1)
    q = lambda values:{str(p):float(np.quantile(values,p)) for p in (.5,.9,.99,1)}
    return dict(points=len(data),world_ply_sha256=sha(path),higher_SH_fields=0,
                raw_RGB_min=rgb.min(0).tolist(),raw_RGB_max=rgb.max(0).tolist(),
                raw_RGB_outside01_point_fraction=float(out.mean()),
                raw_RGB_outside01_opacity_weighted_fraction=float(np.dot(out,opacity)/opacity.sum()),
                raw_RGB_outside01_high_opacity_point_fraction=float(out[opacity>.5].mean()),
                largest_sigma_m_quantiles=q(largest),axis_ratio_quantiles=q(ratio),
                per_Gaussian_directional_colour_variation=0)


if __name__ == '__main__':
    target=OUT/'gaussian-run/step-018000/colour-scale-audit.json'
    assert not target.exists()
    result=dict(updated=now(),source_sha256=sha(__file__),early_fit=describe(BASE),reviewed_fit=describe(OUT),
                optimization_updates=0,quality_accepted=False,
                interpretation='SH0 eliminates each Gaussian directional colour response. Rendered pixel colour still changes with visibility/compositing. Raw coefficient gamut and Gaussian axis ratio are diagnostics, not physical geometry or quality acceptance.')
    save(target,result)
    print(json.dumps(result),flush=True)
