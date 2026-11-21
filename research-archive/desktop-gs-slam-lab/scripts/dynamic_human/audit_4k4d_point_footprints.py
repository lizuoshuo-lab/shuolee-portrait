from pathlib import Path
import json,hashlib,shutil,numpy as np
from PIL import Image
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-4k4d-point-footprints-046'
BASE=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044/source_support_blending/native-evaluation'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
r=read(OUT/'summary.json');g=np.load(OUT/'geometry.npz');checks=[]
u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
for row in r['records']:
 prefix=f"camera{row['camera']}-frame000750";z=np.load(OUT/f'{prefix}-raw.npz');ref=np.load(BASE/f'{prefix}-raw.npz')
 pc=g['xyz'].astype(np.float64)@ref['w2c'][:3,:3].T+ref['w2c'][:3,3]
 rad=ref['K'][1,1]*g['radius']/pc[:,2]
 assert np.allclose(rad,z['projected_radius'],rtol=2e-6,atol=2e-6)
 q=np.quantile(rad[z['contributing_indices']],[0,.1,.5,.9,.99,1])
 assert np.allclose(q,row['statistics']['contributing_native_radius_px_quantiles'],rtol=2e-6,atol=2e-6)
 target=ref['source'].astype(np.float32)/255*ref['matte'][...,None]+.5*(1-ref['matte'][...,None]);pred={}
 for key in row['metrics']:
  array=np.clip(z[key+'_rgb']+.5*(1-z[key+'_alpha'][...,None]),0,1);pred[key]=array
  assert np.array_equal(np.array(Image.open(OUT/f'{prefix}-{key}.png')),u8(array))
  for label,mask in [('full_valid',ref['valid']),('foreground',ref['valid']&(ref['matte']>.5))]:
   psnr=-10*np.log10(np.mean((array[mask].astype(np.float64)-target[mask].astype(np.float64))**2))
   assert abs(psnr-row['metrics'][key][label])<1e-10
 x0,y0,x1,y1=row['bbox'];w=x1-x0;panel=np.array(Image.open(OUT/f'{prefix}-comparison.png'))
 for i,array in enumerate([ref['source']]+[u8(pred[key]) for key in row['metrics']]):assert np.array_equal(panel[32:,i*w:(i+1)*w],array[y0:y1,x0:x1])
 checks.append(dict(camera=row['camera'],native_pngs=3,panel=1,radius_projection_verified=True,raw_sha256=hashlib.sha256((OUT/f'{prefix}-raw.npz').read_bytes()).hexdigest()))
save(OUT/'independent-audit.json',dict(status='18nativePNGs_6panels_metrics_and_native_radius_projection_verified',records=checks,goal_complete=False))
save(OUT/'visual-review.json',dict(status='three_heldout_four_column_person_panels_visually_reviewed',cameras=['00','06','12'],
 finding='Radius reduction decreases isolated disk size but exposes holes in hair, skin and clothes. Shirt print, hands, toy and occlusion are not restored. Half radius causes clear speckled undercoverage.',
 decision='Do not adopt inference-only radius scaling as a clarity fix.',goal_complete=False))
delta={x['camera']:{v:x['metrics'][v]['foreground']-x['metrics']['original']['foreground'] for v in ['radius075','radius050']} for x in r['records']}
save(OUT/'final-summary.json',dict(status='smaller_point_radius_rejected_as_clarity_fix',native_radius_median_range_px=[min(x['statistics']['contributing_native_radius_px_quantiles'][2] for x in r['records']),max(x['statistics']['contributing_native_radius_px_quantiles'][2] for x in r['records'])],
 foreground_psnr_delta_db=delta,all_six_views_worse_at_both_scales=True,
 interpretation='Coverage and surface representation must improve together; this control does not establish that original radius is optimal after retraining or densification.',goal_complete=False))
for name in ['diagnose_4k4d_point_footprints.py','audit_4k4d_point_footprints.py']:shutil.copy2(LAB/'scripts/dynamic_human'/name,OUT/name)
print(json.dumps(read(OUT/'final-summary.json')))
