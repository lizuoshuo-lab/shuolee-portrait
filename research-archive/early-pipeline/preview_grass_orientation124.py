"""Read-only audit at the first original dual-orientation branch transition."""
import inspect
from pathlib import Path
import run_grass_temporal124 as c
import review_grass_temporal124 as review

c.OUT=c.ROOT/'grass-temporal-refinement'
out=c.OUT/'orientation-preview'
assert not out.exists()
frames=[f'frame_{n:06d}' for n in [76,77,78]]
receipts=[c.read(c.OUT/'official-fit/frames'/f/'result.json') for f in frames]
assert [r['orientation_count'] for r in receipts]==[1,2,2]
sources=[c.OUT/'official-fit/frames'/f/'winner-camera-space.npz' for f in frames]
c.save(c.OUT/'orientation-preview-plan.json',dict(updated=c.now(),
    scope='First1->2 original orientation branch transition, read-only inspection of76,77,78 chosen from logs before looking at images.',
    frames=frames,optimizer_updates=0,training_modified=False,review_only=True,
    one_preview=True,gaussian_admitted=False,sources={str(f):c.sha(f) for f in sources}))
source=inspect.getsource(review.main)
hook="    request = c.read(c.OUT/('review-request'+suffix+'.json'))"
assert source.count(hook)==1
source=source.replace(hook,'    request = provided_request')
source=source.replace("out = c.OUT/f'review-{checkpoint:03d}'", "out = c.OUT/'orientation-preview'")
source=source.replace("        assert np.array_equal(old['camera_translation'],new['camera_translation'])", "        assert np.isfinite(new['camera_translation']).all()")
source=source.replace('草地连续帧124','草地连续精修124').replace('独立拟合_连续约束','独立拟合_连续精修')
source=source.replace("'temporal'","'refinement'")
source=source.replace('same_K_shape_camera_translation=True','same_K_shape=True,camera_translation_free_in_new=True')
review.provided_request=dict(frames=frames,last_frame=None)
exec(compile(source,'<grass-dual-orientation-readonly-review>','exec'),review.__dict__)
review.main(3)
for f in sources:assert c.sha(f)==c.read(c.OUT/'orientation-preview-plan.json')['sources'][str(f)]
