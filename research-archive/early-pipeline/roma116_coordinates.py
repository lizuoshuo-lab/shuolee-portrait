"""RoMa normalized maps -> original raster-center coordinates. No image warping."""
import torch
import torch.nn.functional as F

def raw_centers_to_normalized(xy, hw):
    h,w=hw
    return (xy+0.5)/xy.new_tensor([w,h])*2-1

def normalized_to_raw_centers(xy, hw):
    h,w=hw
    return (xy+1)*xy.new_tensor([w,h])/2-0.5

def directional_map(warp,certainty,direction):
    if warp.ndim==4:
        assert warp.shape[0]==1,'One image pair per inference'
        warp=warp[0]
    if certainty.ndim==3:
        assert certainty.shape[0]==1
        certainty=certainty[0]
    assert warp.ndim==3 and warp.shape[-1]==4
    assert tuple(certainty.shape)==tuple(warp.shape[:2])
    h,w2,_=warp.shape
    assert w2%2==0,'Only the preregistered symmetric RoMa output is supported'
    w=w2//2
    # Official symmetric output: left=(A_grid,A_to_B), right=(B_to_A,B_grid).
    if direction=='AtoB':return warp[:,:w,2:],certainty[:,:w]
    if direction=='BtoA':return warp[:,w:,:2],certainty[:,w:]
    raise ValueError(direction)

def sample_direction(warp,certainty,xy,source_hw,target_hw,direction):
    flow,cert=directional_map(warp,certainty,direction)
    grid=raw_centers_to_normalized(xy,source_hw)
    mapped=F.grid_sample(flow.permute(2,0,1)[None],grid[None,None],mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,0].mT
    confidence=F.grid_sample(cert[None,None],grid[None,None],mode='bilinear',padding_mode='zeros',align_corners=False)[0,0,0]
    h,w=flow.shape[:2]
    valid=(grid.abs()<=grid.new_tensor([1-1/w,1-1/h])).all(-1)&(mapped.abs()<=1).all(-1)
    return normalized_to_raw_centers(mapped,target_hw),confidence,valid
