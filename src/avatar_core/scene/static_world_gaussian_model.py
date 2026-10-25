"""Unbound ordinary 3DGS used as a static control, not an avatar method."""
import numpy as np
from .gaussian_model import GaussianModel

class StaticWorldGaussianModel(GaussianModel):
    representation='shared_static_world'
    def __init__(self,sh_degree=3):
        super().__init__(sh_degree)
        self.anchor_ids=None;self.anchor_weights=None;self.control_ids=None;self.control_weights=None;self.clothing_control=None
    def load_motion(self,path):
        z=np.load(path)
        assert str(z['representation'])==self.representation
        assert not any(k in z for k in ['vertices_world','vertex_affine','delta_world'])
        self.world_scale=float(z['world_scale']);self.num_timesteps=int(z['num_timesteps']);self.reference_frame=int(z['reference_frame'])
    def select_mesh_by_timestep(self,timestep,original=False):
        assert 0<=timestep<self.num_timesteps
        self.timestep=int(timestep) # Camera association only; XYZ and covariance never change with time.
    def refresh_anchors(self):
        assert self.binding is None
    def export_world_ply(self,path,timestep=0):
        self.select_mesh_by_timestep(timestep);self.save_ply(str(path))

def load_free_model(root,degree=3):
    from pathlib import Path
    path=Path(root)/'clothing-init/motion.npz'
    with np.load(path) as meta:static='representation' in meta and str(meta['representation'])=='shared_static_world'
    if static:model=StaticWorldGaussianModel(degree)
    else:
        from .clothing_gaussian_model import ClothingGaussianModel
        model=ClothingGaussianModel(degree)
    model.load_motion(path)
    return model
