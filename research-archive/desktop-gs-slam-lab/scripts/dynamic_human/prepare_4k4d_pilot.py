"""Freeze train-only visual-hull/IBR inputs for a single-frame official-model pilot."""
from pathlib import Path
import json, yaml, hashlib, datetime
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-4k4d-official-035'
SOURCE=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
DATA=Path('/opt/gs-slam-lab/4k4d/actor5_6-pilot035'); DATA.mkdir(exist_ok=True)
for name in ['images','bgmtv2','optimized']:
    p=DATA/name
    if not p.exists(): p.symlink_to(SOURCE/name,target_is_directory=True)
    assert p.resolve()==(SOURCE/name).resolve()
train=[i for i in range(18) if i not in [0,6,12]]
common=dict(type='ImageBasedDataset',data_root=str(DATA),frame_sample=[750,751,1],ratio=1.,
    bounds=[[-3.,-3.,-2.],[3.,3.,2.]],dist_opt_K=False,use_aligned_cameras=False,
    use_masks=True,use_vhulls=True,vhulls_dir='train-vhulls',count_thresh=15,vhull_thresh=1.,
    vhull_voxel_size=.005,immask_crop=True,imbound_crop=True,immask_fill=True,
    cache_raw=True,supply_decoded=True,encode_ext='.png',n_srcs_list=[8],n_srcs_prob=[1.],
    extra_src_pool=1,append_gt_prob=.1)
cfg=dict(configs=['configs/exps/4k4d/4k4d_actor5_6_r4_fg.yaml'],exp_name='enerf_actor5_6_train15_frame750_pilot035',
    fix_random=True,benchmark=False,ignore_breakpoint=True,hide_progress=True,
    dataloader_cfg=dict(num_workers=2,dataset_cfg=dict(**common,view_sample=train,src_view_sample=[0,None,1]),
                        sampler_cfg=dict(view_sample=[0,None,1])),
    val_dataloader_cfg=dict(num_workers=1,dataset_cfg=dict(common,view_sample=[0,18,1],src_view_sample=train,
                           extra_src_pool=0,append_gt_prob=1.),sampler_cfg=dict(view_sample=[0,18,6])),
    model_cfg=dict(sampler_cfg=dict(use_diffgl=False,use_cudagl=False,use_pulsar=False,render_gs=False,
                    points_dir='train-surfs',points_expanded=False,points_aligned=True,n_points=262144,
                    ibr_embedder_cfg=dict(n_track=2))),
    runner_cfg=dict(epochs=20,ep_iter=50,eval_ep=20,save_ep=20,save_latest_ep=1,save_lim=30,log_interval=10,
                    record_images_to_tb=False,ignore_eval_error=False,
                    trained_model=str(OUT/'model'),
                    visualizer_cfg=dict(result_dir=str(OUT/'official-render'),types=['RENDER','ALPHA']),
                    recorder_cfg=dict(record_dir=str(OUT/'tensorboard'))))
path=OUT/'pilot.yaml'
text=yaml.safe_dump(cfg,sort_keys=False)
if path.exists(): assert path.read_text()==text
else:path.write_text(text)
sources=[]
for c in train:
    for folder in ['images','bgmtv2']:
        p=SOURCE/folder/f'{c:02d}'/'000750.jpg'
        sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
protocol=dict(status='fixed_before_data_preparation_and_optimization',date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    method='Official4K4D R4DVSampler: learned point radius/opacity, K-plane residual position and feature fields, source-image feature/RGB blending plus SH residual; point renderer, not anisotropic3DGS',
    dataset='actor5_6',frames=[750],purpose='Native-resolution single-frame clarity and implementation pilot before continuous walking fit; not temporal validation',
    train_cameras=train,appearance_heldout_cameras=[0,6,12],geometry_masks=train,source_texture_cameras=train,
    author_surfs_used=False,calibration='Released optimized cameras unchanged; supplied calibration may have used all cameras',
    native_resolution=[1920,1080],resize=False,cropping='Official mask/bounds crop with K principal-point shift; native pixel density retained',
    nominal_training_steps=1000,model_selection='Fixed final budget; pilot engineering failures may require separately recorded recovery, no heldout model selection',
    implementation_adjustments=['Official PyTorch3D point-renderer path selected explicitly for headless WSL','Only2of8source CNN inputs retain gradient to control VRAM;8sources used for appearance','Train-only visual hull at0.005releasedunits; no author surface points','Same native K after undistortion to make image audit interpretable'],
    sources=sources,config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),goal_complete=False)
pp=OUT/'pilot-protocol.json'
if not pp.exists():pp.write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps(dict(config=str(path),train=train,heldout=[0,6,12],source_files=len(sources))),flush=True)
