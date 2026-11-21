from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-4k4d-source-selection-043'
BASE=LAB/'outputs/dynamic-human-4k4d-projection-control-040/original_projection/native-evaluation'
r=json.loads((OUT/'summary.json').read_text());checks=[];u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
for row in r['records']:
    prefix=f"camera{row['camera']}-frame000750";z=np.load(OUT/f'{prefix}-raw.npz');ref=np.load(BASE/f'{prefix}-raw.npz')
    assert np.isfinite(z['sample_weights']).all() and np.allclose(z['sample_weights'].sum(0),1,atol=2e-6)
    assert np.all((z['sample_support']>=0)&(z['sample_support']<=1.00001))
    assert np.max(np.abs(z['alpha']-ref['alpha']))<2e-5
    target=ref['source'].astype(np.float32)/255*ref['matte'][...,None]+.5*(1-ref['matte'][...,None]);pred={}
    for key in row['metrics']:
        assert np.isfinite(z[key]).all();array=np.clip(z[key]+.5*(1-z['alpha'][...,None]),0,1);pred[key]=array
        assert np.array_equal(np.array(Image.open(OUT/f'{prefix}-{key}.png')),u8(array))
        for label,mask in [('full_valid',ref['valid']),('foreground',ref['valid']&(ref['matte']>.5))]:
            psnr=-10*np.log10(np.mean((array[mask].astype(np.float64)-target[mask].astype(np.float64))**2))
            assert abs(psnr-row['metrics'][key][label])<1e-10
    x0,y0,x1,y1=row['bbox'];w=x1-x0
    for key in row['metrics']:
        if key=='original':continue
        panel=np.array(Image.open(OUT/f'{prefix}-{key}-comparison.png'))
        for i,array in enumerate([ref['source'],u8(pred['original']),u8(pred[key])]):assert np.array_equal(panel[32:,i*w:(i+1)*w],array[y0:y1,x0:x1])
    checks.append(dict(camera=row['camera'],native_pngs=5,full_person_panel=4,sampled_point_blend_weights_normalized=True,raw_sha256=hashlib.sha256((OUT/f'{prefix}-raw.npz').read_bytes()).hexdigest()))
(OUT/'independent-audit.json').write_text(json.dumps(dict(status='30nativePNGs_24panels_metrics_and_sampled_weight_normalization_verified',records=checks,
    scope='Independent image and metric recomputation; point-color counterfactuals checked by runtime exact baseline reconstruction and shared fragments, not separately reimplemented source-mask interpolation.',goal_complete=False),indent=2)+'\n')
print('043 passed:30nativePNGs,24panels and all metrics')
