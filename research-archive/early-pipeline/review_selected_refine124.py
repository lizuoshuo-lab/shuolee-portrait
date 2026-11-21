"""Reuse the saved mesh reviewer for this result, comparing its coherent seed."""
import inspect
import review_grass_selected_views124 as original
source=inspect.getsource(original.main)
changes={
    "'grass-temporal-refinement/official-fit/frames'":"'grass-selected-view-pose124/frames'",
    '草地八视角联合124':'草地八视角精修124',
    '旧连续约束_共享局部姿态':'联合初值_逐帧精修',
    "'old_temporal'":"'shared_seed'",
    "'shared_eight'":"'refined'",
    'EIGHT_VIEW_NATIVE_REVIEW_READY':'EIGHT_VIEW_REFINED_REVIEW_READY',
}
for a,b in changes.items():
    assert a in source,a
    source=source.replace(a,b)
scope=original.__dict__.copy();scope['OUT']=original.ROOT/'grass-selected-view-refine124'
exec(compile(source,'<existing-native-review-new-saved-candidate>','exec'),scope)
if __name__=='__main__':scope['main']()
