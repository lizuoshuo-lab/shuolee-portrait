"""Raster support is necessary for a measured2D observation, not visibility proof."""
import numpy as np


def mask_outside_raster(mapped_keypoints, wh):
    points=np.asarray(mapped_keypoints)
    width,height=wh
    if points.ndim!=3 or points.shape[-1]!=3 or not np.isfinite(points).all():
        raise ValueError('Expected finite batched x,y,confidence observations.')
    if width<=0 or height<=0 or int(width)!=width or int(height)!=height:
        raise ValueError('Positive integer native raster dimensions are required.')
    if np.any(points[...,2]<0) or np.any(points[...,2]>1):
        raise ValueError('Pass explicitly normalized relative confidence.')
    # Integer-center pixels have outer boundaries at-.5 and W/H-.5.
    inside=((points[...,0]>=-.5)&(points[...,0]<width-.5)&
            (points[...,1]>=-.5)&(points[...,1]<height-.5))
    result=points.copy()
    result[...,2]=np.where(inside,points[...,2],0)
    return result,inside


def verify():
    a=np.array([[[-.5,0,.8],[9,7,.7],[9.5,4,.6],[1,7.5,.5],[-.501,2,.9]]],np.float32)
    old=a.copy();b,valid=mask_outside_raster(a,(10,8))
    checks=dict(integer_center_edges=valid.tolist()==[[True,True,False,False,False]],
                raw_observations_preserved=np.array_equal(a,old),
                coordinates_preserved=np.array_equal(a[...,:2],b[...,:2]),
                valid_confidence_unchanged=np.array_equal(a[...,2][valid],b[...,2][valid]),
                invalid_weight_zero=bool(np.all(b[...,2][~valid]==0)),
                idempotent=np.array_equal(mask_outside_raster(b,(10,8))[0],b))
    assert all(checks.values()),checks
    return checks
