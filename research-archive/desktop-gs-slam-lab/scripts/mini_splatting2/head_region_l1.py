"""Prepared training-only spatial weighting; not used by resolution experiments.

L = sum_p w_p v_p |I_p - G_p| / (3 sum_p w_p v_p),
w_p = 1 + (coefficient - 1) H_p.
H is a frozen training-photo rectangle, not segmentation or generated pixels.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import torch


class TrainingHeadL1:
    def __init__(self,region_root,coefficient=1.,cache_bound=4):
        root=Path(region_root)
        manifest=json.loads((root/'manifest.json').read_text())
        review=json.loads((root/'visual-review.json').read_text())
        assert review['approved_for_training_attention_experiment']
        assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==review['manifest_sha256']
        self.boxes={row['frame']:row['box'] for row in manifest['rows']}
        assert set(self.boxes)==set(review['reviewed_frames']) and len(self.boxes)==51
        assert manifest['training_only'] and not manifest['heldout_images_read']
        assert all(int(name)%8!=0 for name in self.boxes)
        self.coefficient=float(coefficient);assert 1.<=self.coefficient<=4.
        self.cache=OrderedDict();self.cache_bound=cache_bound;self.used=set()

    def region(self,name,height,width,device,dtype):
        assert name in self.boxes,'Evaluation frames cannot supply training attention'
        key=(name,height,width,str(device),dtype)
        if key not in self.cache:
            while len(self.cache)>=self.cache_bound:self.cache.popitem(last=False)
            x0,y0,x1,y1=[round(value*scale) for value,scale in zip(self.boxes[name],[width/757,height/1347,width/757,height/1347])]
            assert 0<=x0<x1<=width and 0<=y0<y1<=height
            mask=torch.zeros((1,height,width),device=device,dtype=dtype);mask[:,y0:y1,x0:x1]=1
            self.cache[key]=mask
        self.cache.move_to_end(key)
        return self.cache[key]

    def __call__(self,name,image,reference,valid):
        assert image.shape==reference.shape and image.shape[0]==3
        assert name in self.boxes
        self.used.add(name)
        # Exact original expression for the neutral control; no extra operations.
        if self.coefficient==1.:
            return ((image-reference).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
        region=self.region(name,image.shape[-2],image.shape[-1],image.device,image.dtype)
        weights=(valid*(1+(self.coefficient-1)*region)).detach()
        return ((image-reference).abs()*weights).sum()/(3*weights.sum().clamp_min(1))
