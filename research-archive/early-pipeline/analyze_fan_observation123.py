"""Describe the frozen FAN observations without fitting or selecting points."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D = T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis'
F = D/'fan-observation'

def longpath(p):
    return Path('\\\\?\\'+str(Path(p).resolve()))

def read(p):
    return json.loads(longpath(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(longpath(p).read_bytes()).hexdigest()

def stats(x):
    x = np.asarray(x)
    return dict(n=x.size, median=float(np.median(x)) if x.size else None,
                max=float(x.max()) if x.size else None,
                mean=float(x.mean()) if x.size else None)

result = read(F/'result.json')
assert result['network_forwards_completed'] == 6 and result['frames'] == 3
assert not result['quality_accepted'] and result['optimizer_steps'] == 0
plan = read(D/'fan-observation-plan.json')
review_path = F/'frozen-review-snapshot.json'
review = read(review_path)['records']
assert sha(review_path) == next(r['sha256'] for r in plan['immutable_sources']
                              if r['role'] == 'frozen_semantic_visibility_and_FIT_HELD')
with np.load(F/'observations.npz', allow_pickle=False) as z:
    arrays = {k: z[k] for k in z.files}
rows = []
for i, frame in enumerate(plan['immutable_frames']):
    records = [r for r in review if r['frame'] == frame['frame']]
    assert len(records) == 51
    groups = dict(all_static51=list(range(17, 68)),
                  observed_fit=[r['ibug68_index'] for r in records if r['visibility']=='observed' and r['split']=='FIT'],
                  observed_held=[r['ibug68_index'] for r in records if r['visibility']=='observed' and r['split']=='HELD'],
                  unknown=[r['ibug68_index'] for r in records if r['visibility']=='unknown'])
    metrics = {}
    values = {'original_vs_flip': arrays['flip_disagreement_native_px'][i]}
    for branch in ('original', 'flip_restored', 'sum'):
        values[branch+'_vs_RTMw'] = np.linalg.norm(arrays[branch+'_continuous'][i]-arrays['rtmw_native68'][i],axis=1)
        values[branch+'_integer_cast_displacement'] = np.linalg.norm(arrays[branch+'_continuous'][i]-arrays[branch+'_official_integer'][i],axis=1)
    for name, indices in groups.items():
        metrics[name] = {metric:stats(value[indices]) for metric,value in values.items()}
    rows.append(dict(frame=frame['frame'], source_frame=frame['source_frame_index'],
                     native_face_ROI_wh=(np.array(frame['roi_native_xyxy'][2:])-frame['roi_native_xyxy'][:2]).tolist(),
                     groups=metrics,
                     all68_per_point={metric:value.tolist() for metric,value in values.items()}))
report = dict(updated=datetime.now(timezone.utc).isoformat(), state='descriptive_observation_differences_only',
              scope='Same native images and frozen semantic visibility/FIT/HELD groups. No reference 3D projection or new fit.',
              rows=rows, source_observations_sha256=sha(F/'observations.npz'),
              frozen_review_sha256=sha(review_path), measured_absolute_accuracy=False,
              measured_physical_reprojection=False, flip_difference_is_localization_uncertainty=False,
              missing_uncertainty_remains_missing=True, point_selection_or_review_changes=False,
              fit_or_training_admitted=False, quality_accepted=False,
              caveats=['RTMW to FAN changes model and crop context together.',
                       'Continuous versus integer changes only the final cast, not the underlying sampled raster.',
                       'A 42-62px native face crop upsampled to 256px gains no new native detail.',
                       'The three frames do not represent all94 or a new person.'])
longpath(F/'descriptive-metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(rows=[dict(source_frame=r['source_frame'], observed_fit=r['groups']['observed_fit'],
                                observed_held=r['groups']['observed_held']) for r in rows]),ensure_ascii=False))
