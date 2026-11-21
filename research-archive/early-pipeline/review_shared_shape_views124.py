"""Reuse the native-crop review with fixed versus jointly estimated shape."""
import inspect
import review_grass_selected_views124 as original
scope=original.__dict__.copy()
scope['OUT']=original.ROOT/'grass-selected-shape124'
code=inspect.getsource(original.main)
code=code.replace("ROOT/'grass-temporal-refinement/official-fit/frames'", "ROOT/'grass-selected-view-pose124/frames'")
code=code.replace("assert np.array_equal(old['betas'],new['betas']) and np.array_equal(old['faces'],new['faces'])", "assert np.array_equal(old['faces'],new['faces'])")
code=code.replace('草地八视角联合124','草地共享体型124').replace('原图_旧连续约束_共享局部姿态','原图_固定体型_联合体型').replace('相同四方向_旧连续约束_共享局部姿态','相同四方向_固定体型_联合体型')
code=code.replace('old_temporal','fixed_shared_shape').replace('shared_eight','joint_shared_shape')
code=code.replace('same_K_shape_pixels=True','same_K_pixels=True, shared_shape_updated=True')
exec(compile(code,'<same-native-review-shared-shape>','exec'),scope)
if __name__=='__main__':scope['main']()
