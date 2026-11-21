"""Reuse the existing native and four-side renderer on the coherent sofa fit."""
import inspect
import review_grass_selected_views124 as original

scope=original.__dict__.copy();scope['OUT']=original.ROOT/'sofa-selected-view-pose124'
code=inspect.getsource(original.main)
changes=[
    ("ROOT/'grass-temporal-refinement/official-fit/frames'","ROOT/'sofa-temporal-warmstart-v2/official-fit/frames'"),
    ("assert np.array_equal(old['betas'],new['betas']) and np.array_equal(old['faces'],new['faces'])","assert np.array_equal(old['faces'],new['faces'])"),
    ("Path('/mnt/d/00/记录/005')","Path('/mnt/d/00/记录/006')"),
    ("        images={}\n","        hand=new['observed_native_keypoints'][0,25:67]\n        hand=hand[hand[:,2]>0,:2]\n        boxes['hands']=crop_box(np.r_[hand.min(0),hand.max(0)],raw.size,55)\n        images={}\n"),
    ('草地八视角联合124','沙发七帧联合124'),
    ('原图_旧连续约束_共享局部姿态','原图_旧连续约束_共享体型姿态'),
    ('相同四方向_旧连续约束_共享局部姿态','相同四方向_旧连续约束_共享体型姿态'),
    ('shared_eight','shared_selected'),
    ('same_K_shape_pixels=True','same_K_pixels=True,shared_shape_refitted=True')]
for old,new in changes:
    assert old in code,old
    code=code.replace(old,new)
exec(compile(code,'<same-native-side-sofa-review>','exec'),scope)

if __name__=='__main__':scope['main']()
