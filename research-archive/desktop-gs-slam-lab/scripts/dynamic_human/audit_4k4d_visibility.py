from pathlib import Path
import json,hashlib,shutil,numpy as np
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human';OUT=LAB/'outputs/dynamic-human-4k4d-visibility-047'
t=(S/'audit_4k4d_source_support.py').read_text(encoding='utf8')
t=t.replace('dynamic-human-4k4d-source-support-042','dynamic-human-4k4d-visibility-047')
t=t.replace('dynamic-human-4k4d-projection-control-040/original_projection','dynamic-human-4k4d-source-gate-training-044/source_support_blending')
exec(compile(t,'adapted_native_pixel_audit','exec'))
OUT=LAB/'outputs/dynamic-human-4k4d-visibility-047'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
radius=np.load(LAB/'outputs/dynamic-human-4k4d-point-footprints-046/geometry.npz')['radius'];checks=[]
for p in sorted(OUT.glob('source??-visibility.npz')):
 z=np.load(p);uv=z['point_uv'];ij=np.rint(uv).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<1920)&(ij[:,1]>=0)&(ij[:,1]<1080)
 sampled=np.zeros(len(uv),np.float32);sampled[inside]=z['front_depth'][ij[inside,1],ij[inside,0]]
 mismatch=abs(sampled-z['sampled_front'])>1e-6
 # GPU grid normalization can round a coordinate precisely at a half pixel
 # to the other neighboring sample. Preserve/count only such boundary cases.
 boundary=(abs(abs(uv-np.floor(uv))-.5)<.001).any(1)
 assert np.all(~mismatch|boundary)
 expected=np.stack([z['inside']&(z['point_depth']<=z['sampled_front']+tau*radius) for tau in [1.,2.]])
 assert np.array_equal(expected,z['visible'])
 checks.append(dict(camera=p.name[6:8],points=len(uv),nearest_sample_halfpixel_differences=int(mismatch.sum()),visibility_predicate_exact=True))
audit=read(OUT/'independent-audit.json');audit['source_depth_and_visibility_checks']=checks
audit['scope']+=' Native sampled-depth lookup and both visibility predicates independently checked; depth is model-estimated, not ground truth.'
save(OUT/'independent-audit.json',audit)
r=read(OUT/'summary.json');deltas={x['camera']:{v:x['metrics'][v]['foreground']-x['metrics']['original']['foreground'] for v in ['source_visibility1','source_visibility2']} for x in r['records']}
save(OUT/'visual-review.json',dict(status='three_heldout_four_column_panels_visually_reviewed',cameras=['00','06','12'],
 findings=['White/cyan clothes become mottled under model-derived visibility; holes/occlusion and hands remain unresolved.',
 'Some local shoe/toy edges change, but no stable recovery of fine detail or continuous surfaces.'],
 decision='Do not adopt this frozen-model visibility rule. It depends on inaccurate current geometry and leaves many points without supported source visibility.',goal_complete=False))
save(OUT/'final-summary.json',dict(status='frozen_model_visibility_not_adopted_as_clarity_solution',foreground_psnr_delta_db=deltas,
 source_depth_cameras=15,heldout_depth_sources=0,visibility_tolerances_world_radii=[1,2],
 no_visible_source_fraction_range=[min(s['no_visible_source']/x['statistics']['points'] for x in r['records'] for s in x['statistics']['visibility'].values()),max(s['no_visible_source']/x['statistics']['points'] for x in r['records'] for s in x['statistics']['visibility'].values())],
 qualification='These failures concern this inferred geometry and fixed rule; they do not establish that accurate visibility supervision or retraining cannot help.',goal_complete=False))
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['prepare_4k4d_visibility_probe.py','diagnose_4k4d_visibility.py','audit_4k4d_visibility.py','fourkd_source_support.py']:shutil.copy2(S/n,snap/n)
print(json.dumps(read(OUT/'final-summary.json')))
