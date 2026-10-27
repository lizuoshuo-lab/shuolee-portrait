"""Full-image support of measured landmarks; not a 3D visibility estimator."""
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def image_support(keypoints, mask, margin_px=4.):
    """Suppress off-image/outside-person targets without moving any landmark.

    A four-pixel boundary allowance matches the existing observation diagnostic
    margin. Being inside the mask does not prove anatomical identity or visibility.
    Detector scores remain heuristic weights, not calibrated probabilities.
    """
    xy=np.asarray(keypoints,float);mask=np.asarray(mask)
    if xy.ndim!=2 or xy.shape[1]!=2 or mask.ndim!=2:
        raise ValueError('Expected N x 2 full-image pixels and one H x W mask.')
    if not np.isfinite(xy).all() or not np.isfinite(margin_px) or margin_px<0:
        raise ValueError('Finite pixels and a nonnegative margin are required.')
    H,W=mask.shape
    bounds=(xy[:,0]>=0)&(xy[:,0]<=W-1)&(xy[:,1]>=0)&(xy[:,1]<=H-1)
    foreground=mask>0
    if not foreground.any():return np.zeros(len(xy),np.float32)
    outside_distance=distance_transform_edt(~foreground)
    distance=map_coordinates(outside_distance,[xy[:,1],xy[:,0]],order=1,mode='constant',cval=float('inf'))
    return (bounds&(distance<=margin_px)).astype(np.float32)
