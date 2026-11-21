"""Check lens inversion in camera coordinates, without world-pose rounding.

The unchanged rigid-matrix validator still rejects scale/reflection. A separate
world-ray closure diagnostic is reported but is not mislabeled as lens error.
"""
import numpy as np
from native_residual_camera122 import validate_camera,undistort_normalized,distort,rays_world,project_world

def sampled_domain_check(camera):
    validate_camera(camera)
    x,y=np.meshgrid(np.linspace(.5,camera['width']-.5,17),np.linspace(.5,camera['height']-.5,17))
    uv=np.column_stack([x.ravel(),y.ravel()]);k=np.asarray(camera['K'],dtype=float)
    f=np.array([k[0,0],k[1,1]]);obs=(uv-k[:2,2])/f
    ideal=undistort_normalized(obs,camera['residual_projection'])
    back=distort(ideal,camera['residual_projection'])*f+k[:2,2]
    error=float(np.max(np.linalg.norm(back-uv,axis=-1)))
    if error>1e-6:raise ValueError('Sampled sensor lens roundtrip failed')
    origin,rays=rays_world(uv,camera)
    world_error=float(np.max(np.linalg.norm(project_world(origin[None]+rays*3,camera)-uv,axis=-1)))
    rot=np.asarray(camera['w2c'],dtype=float)[:3,:3]
    return dict(samples=len(uv),max_pixel_roundtrip_error=error,global_injectivity_proved=False,
        world_ray_closure_px=world_error,rotation_orthogonality_max_abs=float(np.max(np.abs(rot@rot.T-np.eye(3)))),
        scope='Lens check in normalized camera coordinates; world float rounding reported separately')
