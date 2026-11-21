"""Projection-only control for R4DV/NoopCamera, after original image decoding.

This empirical image-plane correction is not a new physical camera calibration.
Never modify the dataset's decoding K/D or its cached images with these offsets.
"""
import torch

def apply_projection_offsets(batch,target_camera,source_cameras,offsets):
    assert not batch.get('projection_offset_applied',False),'Do not apply twice'
    assert batch.K.shape[-2:]==(3,3)
    assert batch.src_ixts.shape[-3:]==(len(source_cameras),3,3)
    if batch.K.ndim==3:assert batch.K.shape[0]==1
    delta=torch.as_tensor(offsets[f'{int(target_camera):02d}'],dtype=batch.K.dtype,device=batch.K.device)
    src_delta=torch.as_tensor([offsets[f'{int(c):02d}'] for c in source_cameras],dtype=batch.src_ixts.dtype,device=batch.src_ixts.device)
    assert bool(torch.isfinite(delta).all()) and bool(torch.isfinite(src_delta).all())
    # Replace each alias independently from its original value to avoid double shifts.
    for obj,key in [(batch,'K'),(batch,'tar_ixt'),(batch.meta,'K')]:
        if key in obj:
            adjusted=obj[key].clone()
            local_delta=delta.to(device=adjusted.device,dtype=adjusted.dtype)
            adjusted[...,0,2]+=local_delta[0];adjusted[...,1,2]+=local_delta[1]
            obj[key]=adjusted
    for obj,key in [(batch,'src_ixts'),(batch.meta,'src_ixts')]:
        if key in obj:
            adjusted=obj[key].clone()
            local_delta=src_delta.to(device=adjusted.device,dtype=adjusted.dtype)
            adjusted[...,0,2]+=local_delta[:,0];adjusted[...,1,2]+=local_delta[:,1]
            obj[key]=adjusted
    batch.projection_offset_applied=True
    return batch
