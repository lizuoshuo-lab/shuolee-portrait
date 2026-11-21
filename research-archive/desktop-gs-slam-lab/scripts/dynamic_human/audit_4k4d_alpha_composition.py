from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-4k4d-alpha-composition-041'
BASE=LAB/'outputs/dynamic-human-4k4d-projection-control-040/original_projection/native-evaluation'
report=json.loads((OUT/'real-model-summary.json').read_text());checks=[]
u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
for r in report['records']:
    prefix=f"camera{r['camera']}-frame000750";z=np.load(OUT/f'{prefix}-raw.npz');ref=np.load(BASE/f'{prefix}-raw.npz')
    a=z['sample_pixel_alpha'].astype(np.float64);o=z['sample_point_opacity'].astype(np.float64)
    trans=np.cumprod(np.concatenate([np.ones_like(a[...,:1]),1-a[...,:-1]],axis=-1),axis=-1)
    legacy=np.sum(trans*a*o,axis=-1);standard=np.sum(trans*a,axis=-1)
    ys=z['sample_y'];xs=z['sample_x']
    le=float(np.max(np.abs(legacy-z['legacy_alpha'][ys,xs])));se=float(np.max(np.abs(standard-z['standard_alpha'][ys,xs])))
    assert le<2e-5 and se<2e-5
    delta=np.abs(z['rgb']-ref['rgb']);assert delta.mean()<2e-7 and (delta>1/255).sum()<100
    before=np.clip(z['rgb']+.5*(1-z['legacy_alpha'][...,None]),0,1);after=np.clip(z['rgb']+.5*(1-z['standard_alpha'][...,None]),0,1)
    for suffix,arr in [('source',ref['source']),('legacy',u8(before)),('standard-alpha',u8(after)),('legacy-alpha',u8(z['legacy_alpha'])),('standard-alpha-map',u8(z['standard_alpha']))]:
        assert np.array_equal(np.array(Image.open(OUT/f'{prefix}-{suffix}.png')),arr)
    target=ref['source'].astype(np.float32)/255*ref['matte'][...,None]+.5*(1-ref['matte'][...,None])
    for label,mask in [('valid',ref['valid']),('foreground',ref['valid']&(ref['matte']>.5)),('render_boundary',ref['valid']&(z['standard_alpha']>.01)&(z['standard_alpha']<.99))]:
        expected=r['metrics'][label];assert expected['pixels']==int(mask.sum())
        for key,arr in [('before_psnr',before),('after_psnr',after)]:
            val=float(-10*np.log10(np.mean((arr[mask].astype(np.float64)-target[mask].astype(np.float64))**2)))
            assert abs(val-expected[key])<1e-10
    x0,y0,x1,y1=r['bbox'];w=x1-x0;panel=np.array(Image.open(OUT/f'{prefix}-comparison.png'))
    for i,arr in enumerate([ref['source'],u8(before),u8(after)]):assert np.array_equal(panel[32:,i*w:(i+1)*w],arr[y0:y1,x0:x1])
    checks.append(dict(camera=r['camera'],native_pngs=5,comparison_panels=1,independent_fragment_samples=int(ys.size),legacy_formula_maxerr=le,standard_formula_maxerr=se,raw_sha256=hashlib.sha256((OUT/f'{prefix}-raw.npz').read_bytes()).hexdigest()))
(OUT/'independent-audit.json').write_text(json.dumps(dict(status='native_pixels_metrics_and_fragment_formulas_verified',records=checks,goal_complete=False),indent=2)+'\n')
print(json.dumps(dict(status='passed',views=len(checks),native_pngs=30,panels=6,fragment_samples=sum(r['independent_fragment_samples'] for r in checks))))
