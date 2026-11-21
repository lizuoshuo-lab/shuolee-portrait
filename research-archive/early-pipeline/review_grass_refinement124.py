"""Same native comparison for every completed frame; no pose selection."""
import argparse
import inspect
from pathlib import Path
import run_grass_temporal124 as c
import review_grass_temporal124 as review
from run_grass_refinement124 import OUT,PREVIOUS

parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=int,choices=[13,27,94],required=True)
checkpoint=parser.parse_args().checkpoint
if checkpoint==94:
    progress=c.read(OUT/'temporal-progress.json');assert progress['state']=='completed' and progress['frames']==94
    request=dict(frames=c.read(OUT/'plan.json')['ordered_fit_frames'],last_frame=None)
else:request=c.read(OUT/('review-request.json' if checkpoint==13 else 'review-request-027.json'))
original=inspect.getsource(review.main)
hook="    request = c.read(c.OUT/('review-request'+suffix+'.json'))"
assert original.count(hook)==1
source=original.replace(hook,'    request = provided_request')
source=source.replace("        assert np.array_equal(old['camera_translation'],new['camera_translation'])",
    "        assert np.isfinite(new['camera_translation']).all()")
source=source.replace('草地连续帧124','草地连续精修124').replace('独立拟合_连续约束','独立拟合_连续精修')
source=source.replace("'temporal'","'refinement'")
source=source.replace('same_K_shape_camera_translation=True','same_K_shape=True,camera_translation_free_in_new=True')
c.OUT=OUT;review.provided_request=request
exec(compile(source,'<grass-native-refinement-review>','exec'),review.__dict__)
review.main(checkpoint)
path=OUT/f'review-{checkpoint:03d}/result.json';result=c.read(path)
prior={r['frame']:r for r in c.read(PREVIOUS/'review-013/result.json')['records']}
result['comparison_with_previous13']=[dict(frame=r['frame'],previous=prior[r['frame']]['metrics']['free_translation'],
    current=r['metrics']['refinement']) for r in result['records'] if r['frame'] in prior]
result['adapter_source_sha256']=c.sha(__file__)
result['comparison']='Immutable independent94 baseline for all frames; separate matched previous13 free-placement control. No mixture or selection of new output poses.'
c.save(path,result)
