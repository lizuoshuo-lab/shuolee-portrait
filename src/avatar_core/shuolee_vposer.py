"""Load verified VPoser weights with the upstream model and strict tensor keys."""
import hashlib
import sys
from pathlib import Path

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
MODEL=Path('/mnt/d/GS-SLAM-Data/models/vposer/V02_05')
CODE=LAB/'repos/human_body_prior-shuolee'
WEIGHT_NAME='V02_05_epoch=08_val_loss=0.03.ckpt'
WEIGHT_SHA='a914aef21a5c1f4e4d1a616f28def84bc2550c4e7312a689a64e9a35eb7ebee3'

def load_vposer(device='cuda'):
    import torch
    from omegaconf import OmegaConf
    if str(CODE) not in sys.path:sys.path.insert(0,str(CODE))
    from human_body_prior.models.vposer_model import VPoser
    path=MODEL/'snapshots'/WEIGHT_NAME
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    assert h.hexdigest()==WEIGHT_SHA,'VPoser checkpoint differs from recorded download.'
    cfg=OmegaConf.load(MODEL/'V02_05.yaml')
    assert cfg.general.expr_id=='V02_05' and cfg.model_params.latentD==32
    # The mirror file uses the legacy torch serialization format. Restricted
    # weights-only loading works, so no arbitrary pickle globals are enabled.
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    prefix='vp_model.'
    state={k[len(prefix):]:v for k,v in checkpoint['state_dict'].items() if k.startswith(prefix)}
    assert state and all(torch.is_tensor(v) for v in state.values())
    model=VPoser(cfg)
    model.load_state_dict(state,strict=True)
    model=model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    meta={'model':'VPoser','version':'V02_05','latent_dim':32,'body_joints':21,
        'checkpoint_name':WEIGHT_NAME,'checkpoint_sha256':h.hexdigest(),
        'checkpoint_epoch_field':checkpoint.get('epoch'),'state_tensor_count':len(state),
        'other_checkpoint_tensor_keys':[k for k in checkpoint['state_dict'] if not k.startswith(prefix)],
        'strict_state_loading':True,'weights_only_deserialization':True,
        'original_model_code_modified':False,'source_repository':'https://github.com/nghorbani/human_body_prior',
        'model_repository_commit':'6b883ecc3000697409a8caa2744523afe041e48a',
        'weight_source_is_official_host':False,
        'weight_source':'https://huggingface.co/lithiumice/vposer_v02_05/tree/d881a28005876bc890768ab673448e07a91b3003',
        'source_integrity_scope':'SHA256 agrees with the public mirror LFS object; official weight hash was unavailable.'}
    return model,cfg,meta

def verify():
    import numpy as np
    import shuolee_common as c
    torch=c.runtime()
    model,cfg,meta=load_vposer()
    torch.manual_seed(6032)
    latent=torch.nn.Parameter(torch.randn(4,32,device='cuda')*.1)
    out=model.decode(latent)
    pose=out['pose_body'];R=out['pose_body_matrot'].reshape(4,21,3,3)
    assert tuple(pose.shape)==(4,21,3)
    assert torch.isfinite(pose).all() and torch.isfinite(R).all()
    error=(R.transpose(-1,-2)@R-torch.eye(3,device='cuda')).abs().max()
    det_error=(torch.linalg.det(R)-1).abs().max()
    assert error<2e-5 and det_error<2e-5
    loss=pose.square().mean();loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().max()>0
    with torch.no_grad():
        import smplx
        body=smplx.SMPLX(str(c.MODEL),batch_size=4,num_betas=10,num_expression_coeffs=10,
            use_pca=False,flat_hand_mean=False).cuda()
        result=body(body_pose=pose.detach().reshape(4,63))
        assert tuple(result.vertices.shape)==(4,10475,3) and torch.isfinite(result.vertices).all()
        raw=np.concatenate([np.load(c.BASE/f'current/smplx-whole-body-prior-fresh-006/predictions/{i:06d}.npz')['smplx_body_pose'] for i in range(59)])
        encoding=model.encode(torch.as_tensor(raw,dtype=torch.float32,device='cuda'))
        rec=model.decode(encoding.mean)
        assert encoding.mean.shape==(59,32) and torch.isfinite(rec['pose_body']).all()
    meta.update(created=c.stamp(),state='downloaded_strictly_loaded_cuda_and_smplx_forward_verified',
        gpu=torch.cuda.get_device_name(),rotation_orthogonality_max_abs=float(error),
        rotation_determinant_max_error=float(det_error),latent_gradient_max_abs=float(latent.grad.abs().max()),
        smplx_vertex_shape=list(result.vertices.shape),official_detector_pose_encoding_shape=list(encoding.mean.shape),
        personalized_fitting_updates=0,gaussian_updates=0,body_geometry_accepted=False,
        limit='A plausible-pose prior does not establish scene alignment or true subject articulation. Next experiment must refit from original inputs.')
    c.save(MODEL/'verification.json',meta)
    print(meta,flush=True)

if __name__=='__main__':verify()
