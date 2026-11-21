"""Same native/synthetic-side reviewer on the winterhat candidate."""
import inspect
import review_grass_selected_views124 as original
source=inspect.getsource(original.main)
changes={
    "'grass-temporal-refinement/official-fit/frames'":"'winterhat-temporal/official-fit/frames'",
    "'/mnt/d/00/记录/005'":"'/mnt/d/00/记录/001'",
    '草地八视角联合124':'冬帽八视角联合124',
}
for a,b in changes.items():
    assert a in source,a
    source=source.replace(a,b)
scope=original.__dict__.copy();scope['OUT']=original.ROOT/'winterhat-selected-view-pose124'
exec(compile(source,'<same-native-review-winterhat-candidate>','exec'),scope)
if __name__=='__main__':scope['main']()
