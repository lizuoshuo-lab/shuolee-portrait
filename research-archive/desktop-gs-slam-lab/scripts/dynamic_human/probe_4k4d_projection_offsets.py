"""Verify real decoded batches and camera aliases without training or rendering."""
from pathlib import Path
import json,torch,hashlib,copy,numpy as np
from easyvolcap.engine import cfg,DATASETS
from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
from easyvolcap.utils.data_utils import default_collate
from fourkd_projection_offsets import apply_projection_offsets
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-enerf-static-calibration-037'
calibration=json.loads((OUT/'calibration.json').read_text())
offsets={r['camera']:r['shift_xy_px'] for r in calibration['rows']}
assert len(offsets)==18 and cfg.model_cfg.let_user_handle_input and cfg.model_cfg.camera_cfg.type=='NoopCamera'
torch.set_num_threads(4)
ds=DATASETS.build(cfg.dataloader_cfg.dataset_cfg)
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
records=[]
for i in range(len(ds)):
    item=ds[i];target=int(ds.view_inds[int(item.view_index)]);sources=[int(ds.view_inds[int(j)]) for j in item.src_inds]
    assert target not in [0,6,12] and not set(sources)&{0,6,12}
    batch=default_collate([item]);before={key:batch[key].clone() for key in ['K','R','T','src_ixts','src_exts','tar_ixt']}
    image_hashes=[sha(batch.rgb),sha(batch.msk)]+[sha(t) for t in batch.src_inps]+[sha(t) for t in batch.src_msks]
    metaK=batch.meta.K.clone();metaSrc=batch.meta.src_ixts.clone()
    apply_projection_offsets(batch,target,sources,offsets)
    after_hashes=[sha(batch.rgb),sha(batch.msk)]+[sha(t) for t in batch.src_inps]+[sha(t) for t in batch.src_msks]
    assert image_hashes==after_hashes
    for key in ['R','T','src_exts']:assert torch.equal(before[key],batch[key])
    expected=torch.zeros_like(before['K']);expected[...,0,2]=offsets[f'{target:02d}'][0];expected[...,1,2]=offsets[f'{target:02d}'][1]
    for value,old in [(batch.K,before['K']),(batch.tar_ixt,before['tar_ixt']),(batch.meta.K,metaK)]:assert torch.allclose(value-old,expected,atol=1e-4,rtol=0)
    source_expected=torch.zeros_like(before['src_ixts'])
    source_expected[...,0,2]=torch.tensor([offsets[f'{c:02d}'][0] for c in sources]);source_expected[...,1,2]=torch.tensor([offsets[f'{c:02d}'][1] for c in sources])
    assert torch.allclose(batch.src_ixts-before['src_ixts'],source_expected,atol=1e-4,rtol=0)
    assert torch.allclose(batch.meta.src_ixts-metaSrc,source_expected,atol=1e-4,rtol=0)
    # Independently project actual camera-coordinate positions; only pixel offsets may differ.
    points=torch.tensor([[.2,.3,3.],[-.5,.4,5.],[.1,-.7,4.]])
    a=points@before['K'][0].T;b=points@batch.K[0].T
    measured=b[:,:2]/b[:,2:]-a[:,:2]/a[:,2:]
    assert torch.allclose(measured,torch.tensor(offsets[f'{target:02d}']).expand(3,2),atol=1e-4,rtol=0)
    try:apply_projection_offsets(batch,target,sources,offsets)
    except AssertionError:guard=True
    else:raise AssertionError('Repeated offset was accepted')
    records.append(dict(target=target,sources=sources,target_and_source_pixels_unchanged=True,pose_unchanged=True,all_projection_aliases_once=True,duplicate_guard_pass=guard,input_hashes=image_hashes))
result=dict(status='15real_training_batches_projection_only_contract_pass',records=records,
    calibration_sha256=hashlib.sha256((OUT/'calibration.json').read_bytes()).hexdigest(),
    scope='CPU loader/metadata intervention only. All native decoded source/target RGB and matte tensors exactly preserved; all source views remain training-only. No model training or quality improvement.',
    active_training_035_changed=False,goal_complete=False)
(OUT/'projection-adapter-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],batches=len(records))),flush=True)
