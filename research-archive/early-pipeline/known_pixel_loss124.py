"""Bind the existing optional validity loss to the actual rendered camera."""


class KnownPixelLoss:
    def __init__(self, loss_module):
        self.module = loss_module
        self.current = None
        self.frame = None

    def use_camera(self, camera):
        valid = camera.supervision_validity
        if valid.ndim != 3 or valid.shape[0] != 1:
            raise ValueError('Expected one camera with a 1HW validity tensor')
        self.current = valid
        self.frame = camera.supervision_frame

    def l1(self, prediction, target):
        if self.current is None:
            raise RuntimeError('Loss requires an explicitly rendered camera')
        return self.module.l1_loss(prediction, target, valid_mask=self.current)

    def ssim(self, prediction, target):
        if self.current is None:
            raise RuntimeError('Loss requires an explicitly rendered camera')
        return self.module.ssim(prediction, target, valid_mask=self.current)
