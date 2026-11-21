"""Validate native reviewed observation transport and conditional fixed K only."""
from pathlib import Path
import time,re,json
import numpy as np
import match_video10_temporal_bridge123 as b
U=b.R/'video10-remaining-material-review'; P=U/'global-input-preflight'; P.mkdir(exist_ok=True)
assert not (P/'result.json').exists()
start=time.monotonic(); plan=b.read(U/'global-initialization-preflight-plan.json')
for f,h in plan['source_sha256'].items(): assert b.sha(Path(f))==h
runtime=b.read(P/'runtime.json'); assert runtime['camera_estimates']==0
support=b.read(U/'material-support.json')['combined_rows']; sec=b.read(b.R/'video10-semantic-role/secondary-observations.json')
source=b.read(b.O/'run/observations.json'); raw={t['track']:t for t in source['tracks']}; secondary={t['track']:t for t in sec['tracks']}
families={f['family']:f for f in b.read(b.O/'run/families.json')['rows']}
anchors=set(b.read(b.O/'plan.json')['anchor_frames']); frames=b.read(b.O/'plan.json')['frames']
feature_paths={f:(b.B if f in anchors else b.O)/'run'/f'features_{f:06d}.npz' for f in frames}
features={}
for f,p in feature_paths.items():
    with np.load(p,allow_pickle=False) as data: features[f]=data['uv'].copy()
grouped={'fit':[],'heldout':[]}; seen=set(); observations=0
for rep in support:
    tid=rep['track'];fam=rep['family'];split=rep['split'];t=secondary[tid];old=raw[tid]
    assert fam not in seen;seen.add(fam)
    assert t['family']==fam==old['family'] and t['split']==split==families[fam]['split']
    assert fam not in [182,1275,1338] and not old['duplicate_frame_conflict']
    assert len({o['frame'] for o in t['observations']})==len(t['observations'])
    assert sorted(rep['frames'])==sorted(o['frame'] for o in t['observations'])
    for o,orig in zip(t['observations'],old['observations']):
        assert (o['frame'],o['feature'],o['uv'])==(orig['frame'],orig['feature'],orig['uv'])
        assert np.array_equal(features[o['frame']][o['feature']],np.asarray(o['uv']))
        assert np.isfinite(o['uv']).all() and 0<=o['uv'][0]<2704 and 0<=o['uv'][1]<1520
    observations+=len(t['observations'])
    grouped[split].append(dict(family=fam,track=tid,split=split,observations=[dict(frame=o['frame'],feature=o['feature'],uv=o['uv']) for o in t['observations']],
        material_status='visually_supported_coarse_candidate_not_subpixel_truth',physical_independence_verified=False))
assert [len(grouped[k]) for k in ['fit','heldout']]==[162,45]
fit_nodes={(o['frame'],o['feature']) for t in grouped['fit'] for o in t['observations']}
held_nodes={(o['frame'],o['feature']) for t in grouped['heldout'] for o in t['observations']}
assert not fit_nodes&held_nodes
for split,rows in grouped.items():b.save(P/(split+'-observations.json'),dict(source_sha256=b.sha(b.O/'run/observations.json'),frames=frames,tracks=rows,
    coordinates_unchanged=True,source_family_split_unchanged=True,training_admitted=False))
calib=Path('D:/00/工作/数据集/mono_main_calib.yaml'); txt=calib.read_text(encoding='utf-8-sig')
m=re.search(r'^camera_matrix:.*?data:\s*\[(.*?)\]',txt,re.S|re.M); assert m
K0=np.fromstring(m.group(1),sep=',').reshape(3,3)
assert re.search(r'^image_width:\s*4000\s*$',txt,re.M) and re.search(r'^image_height:\s*3000\s*$',txt,re.M)
s=2704/4000; A=np.array([[s,0,(s-1)/2],[0,s,(s-1)/2-254],[0,0,1.]])
K=A@K0
legacy_plan=b.R.parent/'raw-domain-121/global-fixedk-plan.json'
reference=np.asarray(b.read(legacy_plan)['subjects']['indoor']['K_integer_centers'])
assert np.allclose(K,reference,atol=1e-12,rtol=0)
calibration=dict(source=str(calib),source_sha256=b.sha(calib),K_source=K0.tolist(),A_source_to_video=A.tolist(),K_integer_centers=K.tolist(),
 source_dimensions=[4000,3000],assumed_resize_dimensions=[2704,2028],assumed_center_crop_xy=[0,254],video_dimensions=[2704,1520],
 resampling_convention='Integer pixel centers; half-pixel resize, followed by integer center crop. Same conditional assumption as the existing landscape contract.',
 camera_model='PINHOLE',distortion=[0.]*12,K_optimized=False,physical_export_verified=False,
 evidence_level='User-provided processing order and4000x3000 working assumption; actual exporterK/rounding not independently measured.',
 no_second_fisheye_undistortion=True,legacy_K_exactly_reproduced=True)
b.save(P/'conditional-camera.json',calibration)
legacy=Path(__file__).parent/'global_fixedk121.py'; code=legacy.read_text(encoding='utf-8-sig')
assert 'seedpairs=' in code and 'poselib.estimate_absolute_pose' in code and 'poses,points,ba=bundle(' in code
new_next='现有名为global_fixedk121的脚本实际使用双视图起步、逐帧PnP，再对已注册视角做联合BA；它不是全局相机初始化。下一步只实现并检验固定K的共享轨迹全局初始化最小原型，先用本轮已知合成场景验证位姿/尺度和退化拒绝，再登记真实33帧候选。不得把改名或放宽顺序点数门槛当作新方法。'
elapsed=time.monotonic()-start; assert elapsed<120
for f,h in plan['source_sha256'].items(): assert b.sha(Path(f))==h
result=dict(updated=b.now(),state='native_input_and_fixedK_verified_global_initializer_implementation_gap',elapsed_seconds=elapsed,
  plan_sha256=b.sha(U/'global-initialization-preflight-plan.json'),frames=33,fit_tracks=162,heldout_tracks=45,total_observations=observations,
  fit_observations=sum(len(t['observations']) for t in grouped['fit']),heldout_observations=sum(len(t['observations']) for t in grouped['heldout']),
  original_pixels_bit_exact=True,original_fit_heldout_preserved=True,no_shared_exact_feature_between_splits=True,
  K_assumption_verified_algebra_only=True,K_physical_calibration_verified=False,
  legacy_initializer=dict(source=str(legacy),sha256=b.sha(legacy),method='Relative E seed -> sequential PnP -> joint BA on registered subset',global_initializer=False),
  runtime_evidence=str(P/'runtime.json'),installed_global_initializer_verified=False,
  available_building_blocks='PoseLib2.0.5 calibrated relative/absolute pose; SciPy1.14.1; PyCOLMAP3.10.0 exposes only incremental mapping. No GLOMAP executable on the inspected PATH.',
  heldout_is_not_certified_physical_truth=True,real_camera_estimates=0,new_training=0,GPU_tasks=0,next=new_next,
  source_sha256={str(p):b.sha(p) for p in [U/'material-support.json',b.O/'run/observations.json',b.R/'video10-semantic-role/secondary-observations.json',calib,legacy_plan,legacy,P/'runtime.json']},
  feature_file_sha256={str(p):b.sha(p) for p in feature_paths.values()})
b.save(P/'result.json',result)
print(json.dumps({k:result[k] for k in ['state','fit_tracks','heldout_tracks','fit_observations','heldout_observations','original_pixels_bit_exact','real_camera_estimates','next']},ensure_ascii=False))
