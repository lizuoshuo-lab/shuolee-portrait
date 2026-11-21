"""Reuse the native/side reviewer, comparing frozen and free body placement."""
import inspect
import run_grass_temporal124 as c
import review_grass_temporal124 as review
from run_grass_translation124 import OUT,PREVIOUS

original=inspect.getsource(review.main)
source=original.replace("        assert np.array_equal(old['camera_translation'],new['camera_translation'])",
    "        assert np.isfinite(new['camera_translation']).all()")
assert source!=original
hook="        with np.load(oldfile,allow_pickle=False) as z: old=dict(z)"
assert source.count(hook)==1
source=source.replace(hook,"        if not oldfile.exists():\n            assert name=='frame_000026'\n            oldfile=previous/'review-pause-pose.npz'\n"+hook)
source=source.replace('草地连续帧124','草地连续平移124').replace('独立拟合_连续约束','固定平移_联合平移')
source=source.replace("'independent'","'frozen_translation'").replace("'temporal'","'free_translation'")
source=source.replace('same_K_shape_camera_translation=True','same_K_shape=True,camera_translation_free_in_new=True')
review.previous=PREVIOUS;c.OUT=OUT;c.BASE=PREVIOUS/'official-fit'
exec(compile(source,'<grass-native-review-frozen-vs-free-placement>','exec'),review.__dict__)
review.main(13)
path=OUT/'review-013/result.json';result=c.read(path)
result['adapter_source_sha256']=c.sha(__file__)
result['comparison']='Previous stopped temporal13 with frozen translation vs new temporal13 with3 translation scalars in original optimizer; neither result is ground truth.'
c.save(path,result)
