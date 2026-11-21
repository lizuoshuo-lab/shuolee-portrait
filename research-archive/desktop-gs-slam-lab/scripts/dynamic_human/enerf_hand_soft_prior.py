"""Frozen uncertain articulated targets; penalty only, no geometry replacement."""
import numpy as np
import torch

class SoftHandPrior:
    def __init__(self,path,scene_scale,device='cuda'):
        z=np.load(path);self.lookup={(str(c),int(t)):i for i,(c,t) in enumerate(zip(z['camera'],z['frame']))}
        self.ids=torch.as_tensor(z['selected_parent_ids'],dtype=torch.long,device=device)
        self.weight=torch.as_tensor(z['weight'],device=device);self.scene_scale=float(scene_scale)
        self.target=torch.as_tensor(z['target'],device=device)
        self.surface=torch.as_tensor(z['surface'],device=device)
        self.normal=torch.as_tensor(z['normal'],device=device)
    def loss(self,positions,camera,frame):
        i=self.lookup[camera,frame];p=positions[self.ids]
        motion=torch.log1p(((p-self.target[i])/(.015*self.scene_scale)).square().sum(-1))
        signed=((p-self.surface[i])*self.normal[i]).sum(-1)
        surface=torch.log1p((torch.relu(signed.abs()-.003*self.scene_scale)/(.006*self.scene_scale)).square())
        total=(self.weight*(motion+surface)*.5).sum()/self.weight.sum()
        return total,dict(motion=float((self.weight*motion).sum().detach()/self.weight.sum()),surface=float((self.weight*surface).sum().detach()/self.weight.sum()))
