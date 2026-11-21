"""Alpha-only adapter for the existing VHAP NVDiffRenderer; no new projection."""
import nvdiffrast.torch as dr


def native_alpha(renderer, vertices, faces, RT, K, image_size):
    rast = renderer.rasterize(vertices, faces, RT, K, image_size, use_cache=False)
    foreground = (rast['rast_out'][..., -1:] > 0).to(vertices.dtype)
    alpha = dr.antialias(foreground, rast['rast_out'], rast['verts_clip'], faces.int())
    return alpha.flip(1), rast


def foreground_normalized_mask_loss(predicted, target):
    """Local explicit normalization; the upstream commented mask term is unnormalized."""
    return (predicted-target).abs().sum()/target.sum().detach().clamp_min(1)
