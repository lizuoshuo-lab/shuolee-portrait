"""Use official data loaders; record and assert actual source-view membership."""
from pathlib import Path
import json,torch,hashlib,shutil,os
from easyvolcap.engine import cfg,DATASETS
from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
OUT=Path(os.environ.get('FOURKD_RUN_DIR','/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-4k4d-official-035'))
torch.set_num_threads(8)
train=DATASETS.build(cfg.dataloader_cfg.dataset_cfg)
records=[]
assert train.n_views==15 and train.n_latents==1
for i in range(len(train)):
    batch=train[i]
    target=int(train.view_inds[int(batch.view_index)])
    src=[int(train.view_inds[j]) for j in batch.src_inds]
    assert target not in [0,6,12] and not set(src)&{0,6,12}
    records.append(dict(target=target,source_cameras=src,H=int(batch.meta.H),W=int(batch.meta.W),
                       rgb_shape=list(batch.rgb.shape),K=batch.K.tolist(),w2c=batch.w2c.tolist(),
                       rgb_finite=bool(torch.isfinite(batch.rgb).all())))
report=dict(status='official_train_loader_pass',views=train.n_views,frames=train.n_latents,records=records,
    hull_files=[dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in train.vhs])
# Official fallback writes to vhulls when the configured directory is initially
# absent; make that exact train-only hull available at the requested path.
dest=Path(train.data_root,'train-vhulls');dest.mkdir(exist_ok=True)
for name in train.vhs:
    p=Path(name); d=dest/p.name
    if p!=d:
        if d.exists():assert d.read_bytes()==p.read_bytes()
        else:shutil.copy2(p,d)
(OUT/'official-data-probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
