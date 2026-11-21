"""Coarse initialization gate and full-native nonlinear mesh review overlays."""
from pathlib import Path
import numpy as np
from PIL import Image
from native_residual_fit122 import TriangleBVH,cameras_from_fit_rows
from native_residual_camera122 import rays_world


def mesh_mask(bvh,camera):
    """Bounded working memory, all native pixels, no pinhole proxy depth."""
    width,height=camera['width'],camera['height']
    mask=np.zeros((height,width),bool)
    for start in range(0,height,16):
        x,y=np.meshgrid(np.arange(width)+.5,np.arange(start,min(start+16,height))+.5)
        uv=np.column_stack([x.ravel(),y.ravel()]);origin,dirs=rays_world(uv,camera)
        _,triangle=bvh.trace(origin,dirs)
        mask[start:start+len(y)]=(triangle>=0).reshape(y.shape)
    return mask


def evaluate(root,cfg):
    from shuolee_portrait import runtime as rt
    fit=dict(np.load(root/'fit/fitted.npz'));obs=dict(np.load(root/'data/observations.npz'));weights=dict(np.load(root/'fit/observation-weights.npz'))
    h=obs['boxes'][:,3]-obs['boxes'][:,1]
    error=np.linalg.norm(fit['projected_coco17']-weights['body_target'],axis=-1)/h[:,None]
    reliable=weights['body']*weights['visibility']>.25
    if reliable.sum()<len(h)*4:raise RuntimeError('Insufficient reliable anatomical observations for coarse fit gate')
    median=float(np.median(error[reliable]));p90=float(np.quantile(error[reliable],.9))
    passed=median<=cfg['max_fit_median_body_fraction'] and p90<=cfg['max_fit_p90_body_fraction']
    result=dict(median_body_fraction=median,p90_body_fraction=p90,coarse_initialization_gate_passed=passed,
        quality_accepted=False,from_scratch=True,physical_accuracy_proven=False,thresholds='Unchanged coarse anatomical limits, never physical material subpixel acceptance',
        fit_sha256=rt.sha(root/'fit/fitted.npz'),camera_projection='opencv_brown12_output',review_projection='continuous native pixel ray/mesh',images_resampled=False)
    rt.save(root/'fit/geometry-gate.json',result)
    rows=rt.read(root/'data/training-inputs.json')['records'];cameras=cameras_from_fit_rows(rows)
    if not np.array_equal(fit['vertices_world'],np.broadcast_to(fit['vertices_world'][0],fit['vertices_world'].shape)):raise ValueError('One shared mesh required')
    bvh=TriangleBVH(fit['vertices_world'][0],fit['faces']);out=root/'reviews/fit';out.mkdir(parents=True)
    for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
        row=rows[i];camera=cameras[i]
        if rt.sha(row['image'])!=row['source_sha256']:raise ValueError('Fit review image changed')
        im=Image.open(row['image']).convert('RGB')
        if im.size!=(camera['width'],camera['height']):raise ValueError('Fit review native camera/image size differs')
        rgb=np.asarray(im);mask=mesh_mask(bvh,camera);overlay=rgb.astype(float)
        overlay[mask]=overlay[mask]*.6+np.array([15,190,210])*.4
        im.save(out/f'{row["frame"]}_input.png');Image.fromarray(overlay.astype(np.uint8)).save(out/f'{row["frame"]}_mesh_overlay.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(out/f'{row["frame"]}_mesh_mask.png')
    if not passed:raise RuntimeError('New coarse fit failed. No Gaussian training was started')
