"""Reuse the native comparison renderer for the single input-roster control."""
import inspect
import review_grass_selected_views124 as original
scope=original.__dict__.copy();scope['OUT']=original.ROOT/'grass-screened-seven124'
code=inspect.getsource(original.main)
code=code.replace("ROOT/'grass-temporal-refinement/official-fit/frames'","ROOT/'grass-selected-shape124/frames'")
code=code.replace("assert np.array_equal(old['betas'],new['betas']) and np.array_equal(old['faces'],new['faces'])","assert np.array_equal(old['faces'],new['faces'])")
code=code.replace('草地八视角联合124','草地源头筛选七帧124').replace('原图_旧连续约束_共享局部姿态','原图_八帧_筛选七帧').replace('相同四方向_旧连续约束_共享局部姿态','相同四方向_八帧_筛选七帧')
code=code.replace('old_temporal','eight_input_views').replace('shared_eight','seven_source_supported_views')
code=code.replace('same_K_shape_pixels=True','same_K_pixels=True,shared_shape_refitted=True')
exec(compile(code,'<same-native-review-input-roster>','exec'),scope)
if __name__=='__main__':scope['main']()
