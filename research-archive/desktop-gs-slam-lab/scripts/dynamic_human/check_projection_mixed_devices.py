from pathlib import Path
import json,torch
from easyvolcap.utils.base_utils import dotdict
from fourkd_projection_offsets import apply_projection_offsets
k=torch.eye(3)[None];src=k[:,None].repeat(1,2,1,1)
batch=dotdict(K=k.cuda(),tar_ixt=k.cuda(),src_ixts=src.cuda(),meta=dotdict(K=k.clone(),src_ixts=src.clone()))
offsets={'01':[.3,-1.2],'02':[-.1,.6],'03':[.4,.5]}
before={key:value.clone() for key,value in batch.items() if isinstance(value,torch.Tensor)}
apply_projection_offsets(batch,1,[2,3],offsets)
assert torch.equal(batch.K.cpu(),batch.meta.K) and torch.equal(batch.K,batch.tar_ixt)
assert torch.equal(batch.src_ixts.cpu(),batch.meta.src_ixts)
assert torch.allclose(batch.K.cpu()[0,:2,2],torch.tensor([.3,-1.2]))
assert torch.allclose(batch.src_ixts.cpu()[0,:,:2,2],torch.tensor([[-.1,.6],[.4,.5]]))
assert torch.equal(k,torch.eye(3)[None]) and torch.equal(src,k[:,None].repeat(1,2,1,1))
assert batch.meta.K.device.type=='cpu' and batch.K.device.type=='cuda'
try:apply_projection_offsets(batch,1,[2,3],offsets)
except AssertionError:pass
else:raise AssertionError('Duplicate application must be rejected')
report=dict(status='mixed_GPU_tensors_CPU_metadata_projection_aliases_verified',initial_training_attempt_updates=0,recovery='Exact original parent and fixed random seed; no partially-trained model reused.')
p=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-4k4d-projection-control-040/mixed-device-recovery.json')
p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
