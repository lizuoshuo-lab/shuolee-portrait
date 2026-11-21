from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=REC/'grayT-fixed-direction-review'
A=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
B=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
assert not (O/'plan.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prep=json.loads((O/'prepared.json').read_text(encoding='utf-8'))
crop={x['frame']:x for x in prep['crops']}
seeds={56:[([334,110],[321,230]),([363,110],[349,230]),([393,110],[378,230])],
       64:[([165,60],[148,218]),([229,60],[209,218]),([300,60],[277,218])],
       72:[([83,120],[74,265]),([175,120],[163,265]),([271,120],[257,265])],
       80:[([94,110],[90,240]),([182,110],[175,240]),([273,110],[264,240])]}
lines=[]
for n,pairs in seeds.items():
 x,y=crop[n]['crop'][:2]
 for j,(a,b) in enumerate(pairs):
  # The first45% of frame64 guide overlaps the cast-shadow tips. Exclusion is
  # frozen from raw appearance, before any line or camera consistency fit.
  t0=.45 if n==64 else 0.
  p=[a[0]+t0*(b[0]-a[0])+x,a[1]+t0*(b[1]-a[1])+y]
  q=[b[0]+x,b[1]+y]
  lines.append(dict(id=f'F{n:03d}_bar{j+1}',frame=n,role='ANCHOR' if n in [56,72] else 'HELD',
      a=p,b=q,material='Straight interior of gray railing upright; below visible cast-shadow tip',
      crop=crop[n]['crop'],physical_family='near_gray_roadside_railing_nominal_uprights'))
sources=[A/'data/inputs.json',B/'data/inputs.json',O/'prepared.json',O/'seed-row-appearance.json']+[Path(crop[n]['image']) for n in seeds]
plan=dict(updated=datetime.now(timezone.utc).isoformat(),previous_goal_turn='progress: completed and published two-run support/BA-membership audit',
 hypothesis='Given the existing conditional K, frozen camera rotations should map a common fixed railing-upright direction consistently across views.',
 novelty='Line-direction observation independent of SIFT material-point identity; distinct from whiteT residual-distortion D1-D4. No distortion or camera coefficient fit.',
 no_camera_updates=True,no_translation_estimation=True,no_new_SIFT=True,no_human_training=True,
 evidence_boundary='Railing uprights are assumed approximately parallel, not measured with a level. Direction disagreement can reflect K, residual image warp, physical bar nonparallelism or rotation. It is not a unique external-pose error. One direction cannot constrain yaw about itself or any translation.',
 input='1520x2704 decoded MP4, already rectified/cropped, unchanged first-matrix conditional K; no re-undistortion or output image warping.',
 sources=[dict(path=str(p),sha256=sha(p)) for p in sources],lines=lines,
 pre_image_only_exclusions={'40':'Gray roadside rail too small, do not substitute another dark-fence family','48':'Thin projected bars and visible aliasing on the lower rail; excluded during raw image feasibility, before any geometric calculation'},
 measurement=dict(count_per_line=80,horizontal_search_halfwidth_px=4,step_px=.25,
   tangent_rows=[-1,0,1],minimum_positive_gradient=4,secondary_peak_min_separation_px=1.5,secondary_to_primary_limit=.7,
   max_abs_offset_px=3.5,minimum_valid_fraction=.8,minimum_valid_samples=48,
   subpixel='Quadratic interpolation at strongest positive derivative; ambiguity/gradient/boundary exclusion only, never geometric-residual removal'),
 computation=dict(camera_runs={'research':str(A),'company00':str(B)},fit='One unit world direction per frozen camera run from six anchor-line backprojection planes by SVD; no camera/point optimization',
   validation='Held frames64,80 never estimate world direction. Predicted image line passes measured centroid; evaluate all retained samples. This is a direction test, not absolute reprojection or face admission.',
   measurement_diagnostics='TLS perpendicular residual, line linearity and first-half/second-half slope disagreement reported for every line. No line omitted after geometry.',
   uncertainty='Predetermined worst-case endpoint jitter +/-0.5 native pixel; report angular sensitivity, not a calibrated statistical confidence interval.',
   synthetic_controls=['Known R/K/nonzero principal point direction recovery','2degree held pitch negative control','Arbitrary rotation around world direction remains unobservable'],
   cpu_seconds=120,max_runs=1),
 stop=['Any source hash change: stop','Any line fails photometric coverage: retain measurements and stop before camera comparison; no replacement or threshold sweep','One direction cannot admit camera or training, even if all held lines appear consistent'],
 physical_quality_accepted=False)
(O/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(lines=len(lines),anchors=6,held=6,plan=str(O/'plan.json')),ensure_ascii=False))
