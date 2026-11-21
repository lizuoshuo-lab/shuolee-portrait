"""Output-pixel Brown camera contract shared by observations and native rendering.

This describes residual distortion of the existing processed video, not its raw
fisheye lens. No image resampling, coefficient fitting, or pixel-origin conversion
is implicit. Coefficients map ideal normalized coordinates to observed ones.
"""
from dataclasses import dataclass
import copy
import numpy as np
from native_scene_package122 import validate_camera as validate_base, keys, array

SCHEMA = 'output-residual-brown12-v1'
ORDER = ['k1','k2','k3','k4','k5','k6','p1','p2','s1','s2','s3','s4']


@dataclass(frozen=True)
class ResidualRenderContract:
    input_domain: str
    pixel_convention: str = 'pixel_corners'

    def validate(self):
        if self.input_domain not in ('synthetic_known_camera', 'processed_output_pixels'):
            raise ValueError('Explicit processed-output domain required; no raw fisheye substitution')
        if self.pixel_convention != 'pixel_corners':
            raise ValueError('Convert pixel origin explicitly, for both K and observations')

    def manifest(self):
        self.validate()
        return dict(schema=SCHEMA, input_domain=self.input_domain, pixels=self.pixel_convention,
                    projection='opencv_brown12_output', coefficient_order=ORDER,
                    mapping='ideal_normalized_to_observed_normalized', units='normalized_camera',
                    image_resampled=False, gaussian_evaluation='ray-3d', camera_optimization=False)


def projection(radial, tangential, thin_prism, *, input_domain):
    contract = ResidualRenderContract(input_domain); contract.validate()
    p = dict(schema=SCHEMA, input_domain=input_domain, pixels='pixel_corners',
             mapping='ideal_normalized_to_observed_normalized', units='normalized_camera',
             radial=np.asarray(radial).tolist(), tangential=np.asarray(tangential).tolist(),
             thin_prism=np.asarray(thin_prism).tolist())
    validate_projection(p)
    return p


def validate_projection(p):
    keys(p, ['schema','input_domain','pixels','mapping','units','radial','tangential','thin_prism'], 'residual projection')
    if p['schema'] != SCHEMA or p['mapping'] != 'ideal_normalized_to_observed_normalized' or p['units'] != 'normalized_camera':
        raise ValueError('Explicit output-residual model, direction and normalized units required')
    ResidualRenderContract(p['input_domain'], p['pixels']).validate()
    for k, n in [('radial',6),('tangential',2),('thin_prism',4)]:
        array(p[k], (n,), k)


def base_camera(c):
    return {k:copy.deepcopy(v) for k,v in c.items() if k != 'residual_projection'}


def validate_camera(c):
    if not isinstance(c, dict) or 'residual_projection' not in c:
        raise ValueError('Camera-specific residual projection is required; no zero-distortion default')
    validate_base(base_camera(c)); validate_projection(c['residual_projection'])


def with_projection(c, p):
    validate_base(c); validate_projection(p)
    return dict(copy.deepcopy(c), residual_projection=copy.deepcopy(p))


def cv_coefficients(p):
    validate_projection(p)
    r,t,s = (np.asarray(p[n], dtype=np.float64) for n in ('radial','tangential','thin_prism'))
    return np.array([r[0],r[1],t[0],t[1],r[2],r[3],r[4],r[5],*s])


def distort(xy, p, *, jacobian=False):
    validate_projection(p)
    q = np.asarray(xy, dtype=np.float64)
    if q.shape[-1:] != (2,) or not np.isfinite(q).all():
        raise ValueError('Finite normalized XY required')
    x,y = q[...,0], q[...,1]; r2 = x*x+y*y
    k1,k2,k3,k4,k5,k6 = p['radial']; p1,p2 = p['tangential']; s1,s2,s3,s4 = p['thin_prism']
    num = 1+r2*(k1+r2*(k2+r2*k3)); den = 1+r2*(k4+r2*(k5+r2*k6))
    if np.any(den <= 1e-12) or np.any(num <= 0):
        raise ValueError('Nonpositive rational radial mapping; no folded-ray fallback')
    radial = num/den
    out = np.stack([x*radial+2*p1*x*y+p2*(r2+2*x*x)+s1*r2+s2*r2*r2,
                    y*radial+p1*(r2+2*y*y)+2*p2*x*y+s3*r2+s4*r2*r2], axis=-1)
    if not jacobian: return out
    slope = ((k1+r2*(2*k2+3*k3*r2))*den-num*(k4+r2*(2*k5+3*k6*r2)))/den**2
    dx,dy = 2*x*slope, 2*y*slope
    j = np.stack([radial+x*dx+2*p1*y+6*p2*x+2*x*(s1+2*s2*r2),
                  x*dy+2*p1*x+2*p2*y+2*y*(s1+2*s2*r2),
                  y*dx+2*p2*y+2*p1*x+2*x*(s3+2*s4*r2),
                  radial+y*dy+2*p2*x+6*p1*y+2*y*(s3+2*s4*r2)],axis=-1).reshape(q.shape[:-1]+(2,2))
    return out,j


def undistort_normalized(observed, p):
    """Bounded Newton inverse. Invalid/nonconverged rays fail, never become pinhole."""
    target = np.asarray(observed, dtype=np.float64)
    q = target.copy()
    for _ in range(20):
        value,j = distort(q,p,jacobian=True)
        if np.any(np.linalg.det(j) <= 1e-8): raise ValueError('Noninvertible or reversed local camera mapping')
        delta = np.linalg.solve(j, (value-target)[...,None])[...,0]
        q -= delta
        if np.max(np.abs(delta), initial=0) < 1e-12: break
    if not np.isfinite(q).all() or np.max(np.abs(distort(q,p)-target), initial=0) > 1e-10:
        raise ValueError('Residual camera inverse did not converge; no pinhole fallback')
    return q


def project_world(points, camera):
    validate_camera(camera)
    xyz = np.asarray(points,dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all(): raise ValueError('Finite world XYZ required')
    e,k = np.asarray(camera['w2c']),np.asarray(camera['K'])
    pc = xyz@e[:3,:3].T+e[:3,3]
    if np.any(pc[:,2] <= 0): raise ValueError('Projection of behind-camera points is not an observed match')
    xy = distort(pc[:,:2]/pc[:,2:],camera['residual_projection'])
    return xy*np.array([k[0,0],k[1,1]])+k[:2,2]


def rays_world(uv, camera):
    validate_camera(camera)
    pixels = np.asarray(uv,dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all(): raise ValueError('Finite observed pixels required')
    e,k = np.asarray(camera['w2c']),np.asarray(camera['K'])
    xy = undistort_normalized((pixels-k[:2,2])/np.array([k[0,0],k[1,1]]),camera['residual_projection'])
    d = np.column_stack([xy,np.ones(len(xy))])@e[:3,:3]
    d /= np.linalg.norm(d,axis=-1,keepdims=True)
    return -e[:3,:3].T@e[:3,3],d


def sampled_domain_check(camera):
    """17x17 sensor check, explicitly not a proof of global lens injectivity."""
    validate_camera(camera)
    x,y = np.meshgrid(np.linspace(.5,camera['width']-.5,17),np.linspace(.5,camera['height']-.5,17))
    uv = np.column_stack([x.ravel(),y.ravel()]); origin,rays = rays_world(uv,camera)
    back = project_world(origin[None]+rays*3,camera)
    err = float(np.max(np.linalg.norm(back-uv,axis=-1)))
    if err > 1e-6: raise ValueError('Sampled sensor ray roundtrip failed')
    return dict(samples=len(uv),max_pixel_roundtrip_error=err,global_injectivity_proved=False)
