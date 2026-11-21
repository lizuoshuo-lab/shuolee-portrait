"""Local foreground-support adapter; retains the official model parameters."""
from types import MethodType
import torch
from easyvolcap.utils.ibr_utils import compute_src_inps,sample_geometry_feature_image
from easyvolcap.utils.image_utils import pad_image
from easyvolcap.utils.sh_utils import eval_sh

def normalize_supported_weights(weights,support,threshold=.5):
    gated=weights*(support>=threshold)
    total=gated.sum(-3,keepdim=True)
    return torch.where(total>1e-7,gated/total.clamp_min(1e-7),weights)

def enable_source_foreground_support(sampler):
    assert not getattr(sampler,'_source_foreground_support_enabled',False)
    original_embed=sampler.ibr_embedder.forward
    reg=sampler.ibr_regressor
    assert not reg.blend_shs and not reg.skip_shs and not reg.skip_eval_shs and not reg.manual_chunking
    def embed(self,xyz,batch,optimize_cnn=True):
        features=original_embed(xyz,batch,optimize_cnn=optimize_cnn)
        masks=compute_src_inps(batch,'src_msks')
        masks=pad_image(masks,size=batch.persistent.src_feat_rgb.shape[-2:])
        batch.output.source_foreground_support=sample_geometry_feature_image(xyz,masks,batch.src_exts,batch.src_ixts,batch.persistent.src_scale,padding_mode='zeros')
        return features
    def blend(self,xyz_feat_dir,batch,return_rgb_sh=False):
        xyz_feat,direction=xyz_feat_dir[...,:-3],xyz_feat_dir[...,-3:]
        features=batch.output.src_feat_rgb
        expanded=xyz_feat[:,None].expand(features.shape[:-1]+(xyz_feat.shape[-1],))
        weights=self.rgb_mlp(torch.cat([expanded,features],dim=-1)).softmax(-3)
        support=batch.output.source_foreground_support
        assert support.shape==weights.shape
        weights=normalize_supported_weights(weights,support)
        rgb=(features[...,-3:]*weights).sum(-3)
        sh=self.sh_mlp(xyz_feat).reshape(*xyz_feat.shape[:-1],self.out_dim,self.sh_dim//self.out_dim)
        if return_rgb_sh:return rgb,sh
        return (rgb+eval_sh(self.sh_deg,sh,direction).tanh()*self.resd_limit).clamp(0,1)
    sampler.ibr_embedder.forward=MethodType(embed,sampler.ibr_embedder)
    sampler.ibr_regressor.forward=MethodType(blend,reg)
    sampler._source_foreground_support_enabled=True
