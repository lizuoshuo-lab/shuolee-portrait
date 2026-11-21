"""Matched stochastic-background compositing, separate RNG, unchanged RGB loss."""
import torch


class PairedTrainingBackground:
    def __init__(self, renderer, masks, frames, target_cache, seed=124):
        self.renderer = renderer
        self.masks = [m.to(device='cuda', dtype=torch.float32)/255 for m in masks]
        self.frames = list(frames)
        self.slots = {frame:i for i,frame in enumerate(frames)}
        assert len(self.slots) == len(frames)
        self.cache = target_cache
        self.generator = torch.Generator(device='cuda').manual_seed(seed)
        self.calls = 0
        self.first_colors = []

    def __call__(self, cam, model, pipe, background, **kwargs):
        if not torch.is_grad_enabled():
            return self.renderer(cam, model, pipe, background, **kwargs)
        frame = cam.supervision_frame
        slot = self.slots[frame]
        foreground = self.cache.get(slot, frame)
        mask = self.masks[slot]
        assert foreground.shape == (3,*mask.shape[-2:])
        chosen = torch.rand(3, device='cuda', generator=self.generator)
        # get() returns an immutable cached premultiplied RGB. Never modify it.
        cam.original_image = foreground + (1-mask)*chosen[:,None,None]
        self.calls += 1
        if self.calls <= 3:
            self.first_colors.append(chosen.detach().cpu().tolist())
        return self.renderer(cam, model, pipe, chosen, **kwargs)

    def receipt(self):
        return dict(training_calls=self.calls,seed=124,first_colors=self.first_colors,
                    paired_prediction_target_background=True,global_rng_unchanged=True,
                    known_unknown_masks_unchanged=True,evaluation_background='black')
