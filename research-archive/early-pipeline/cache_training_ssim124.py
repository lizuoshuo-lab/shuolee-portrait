"""Optional camera-bound SSIM cache for new adapters; originals stay frozen."""
import ast
import sys
from cache_training_targets124 import adapt as cache_targets
from fixed_target_ssim124 import FixedTargetSSIM


class CameraTargetSSIM:
    def __init__(self, original_ssim, cameras, targets, frames):
        owner = getattr(original_ssim, '__self__', None)
        module = getattr(owner, 'module', None)
        if module is None:
            module = sys.modules[original_ssim.__module__]
        if len(cameras) != len(frames) or len({id(cam) for cam in cameras}) != len(frames):
            raise ValueError('Require one distinct camera per target frame')
        self.cameras = tuple(cameras)
        self.frames = {id(cam): frame for cam, frame in zip(cameras, frames)}
        self.cache = FixedTargetSSIM(module, targets, frames,
            [getattr(cam, 'supervision_validity', None) for cam in cameras])
        self.current_frame = None

    def use_camera(self, camera):
        if id(camera) not in self.frames:
            self.current_frame = None
            raise ValueError('Unregistered rendered camera; cannot select target statistics')
        self.current_frame = self.frames[id(camera)]

    def ssim(self, prediction, target):
        if self.current_frame is None:
            raise RuntimeError('Render a registered camera before computing cached SSIM')
        return self.cache.ssim(prediction, target, frame_id=self.current_frame)


def adapt(source):
    result = cache_targets(source)
    marker = '    class Images:\n'
    render = '    def render(cam, pc, pipe, background, **kwargs):\n'
    assert result.count(marker) == 1 and result.count(render) == 1
    result = result.replace(marker,
        "    from cache_training_ssim124 import CameraTargetSSIM\n"
        "    ssim_camera_cache = CameraTargetSSIM(ssim, cams, target_cache.targets, [r['frame'] for r in rows])\n"
        "    ssim = ssim_camera_cache.ssim\n" + marker)
    result = result.replace(render, render + '        ssim_camera_cache.use_camera(cam)\n')
    ast.parse(result)
    return result
