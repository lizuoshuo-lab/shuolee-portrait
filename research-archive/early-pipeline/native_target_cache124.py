"""Optional cache of the existing CPU-composited targets; no loss changes."""
import torch


class NativeTargetCache:
    def __init__(self, pixels, masks, frame_ids, device='cuda'):
        if not (len(pixels) == len(masks) == len(frame_ids)):
            raise ValueError('One immutable image and mask per named frame required')
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError('Duplicate frame identities')
        self.frame_ids = tuple(frame_ids)
        self.targets = []
        for rgb, mask in zip(pixels, masks):
            if rgb.device.type != 'cpu' or mask.device.type != 'cpu':
                raise ValueError('Preserve the original CPU composition arithmetic')
            if rgb.dtype != torch.uint8 or mask.dtype != torch.uint8:
                raise ValueError('Expected original uint8 pixels and mask')
            if rgb.ndim != 3 or rgb.shape[0] != 3 or mask.shape != (1, *rgb.shape[1:]):
                raise ValueError('Expected native CHW RGB and 1HW mask')
            # Exactly the existing Images.__getitem__ expression. Do not move
            # division/multiplication to CUDA, resize or threshold the mask.
            target = rgb.float().div(255) * mask.float().div(255)
            self.targets.append(target.to(device))
        self.bytes = sum(x.numel()*x.element_size() for x in self.targets)

    def get(self, index, frame_id):
        if self.frame_ids[index] != frame_id:
            raise ValueError('Frame/target identity mismatch')
        return self.targets[index]
